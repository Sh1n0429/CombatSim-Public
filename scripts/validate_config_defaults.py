"""Check configuration precedence across entry points and reusable components."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from combatsim.config_defaults import APPLICATION_DEFAULTS, ENVIRONMENT_DEFAULTS, RULE_DEFAULTS
from combatsim.env import Config
from combatsim.factory import load_settings, make_env
from combatsim.policies import RulePolicy
from combatsim.rllib_experiment import load_experiment
from scripts.train import parse_args
from combatsim.reward_hooks import OriginalRewardHooks, RewardEvent
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class ConfigDefaultsChecks(unittest.TestCase):
    def test_base_configuration_and_mutable_state_are_independent(self):
        self.assertEqual(json.loads(json.dumps(asdict(Config()))), ENVIRONMENT_DEFAULTS)
        first, second = Config(), Config()
        first.rule_policy['test'] = True
        self.assertNotIn('test', second.rule_policy)
        self.assertEqual(RulePolicy().parameters, RULE_DEFAULTS)

    def test_removed_motion_modes_are_not_serialized_or_silently_migrated(self):
        from combatsim.motion import AerodynamicAircraftMotion, AerodynamicMissileMotion
        settings = load_settings(ROOT/'configs/scenario-default.json')
        current = Config(**settings)
        old = Config(**settings, aircraft_motion='aerodynamic', missile_motion='aerodynamic')
        self.assertEqual(asdict(current), asdict(old))
        self.assertNotIn('aircraft_motion', asdict(old))
        self.assertNotIn('missile_motion', asdict(old))
        env = make_env(current)
        self.assertIsInstance(env.aircraft_motion, AerodynamicAircraftMotion)
        self.assertIsInstance(env.missile_motion, AerodynamicMissileMotion)
        for name in ('aircraft_motion', 'missile_motion'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'removed'):
                Config(**{name: 'constant'})

    def test_removed_speed_and_mix_are_rejected_and_absent_from_observations(self):
        from combatsim.observation_schema import PERFORMANCE_KEYS, FEATURE_NAMES
        from combatsim.rllib_env import CombatSimFlightEnv
        self.assertNotIn('missile_speed', PERFORMANCE_KEYS)
        for group in ('self', 'allies'):
            self.assertNotIn('performance_missile_speed', FEATURE_NAMES[group])
        for settings in ({'missile_speed': 2.3}, {'rule_policy_mix': {'fix': 1}}):
            with self.assertRaises(TypeError):
                Config(**settings)
        env = CombatSimFlightEnv({'settings': {}, 'rule_ratio': 1.})
        env.reset(seed=7)
        self.assertEqual(env.rule_profile, 'fix')
        with self.assertRaisesRegex(ValueError, 'random_rate'):
            CombatSimFlightEnv({'settings': {}, 'random_rate': None})

    def test_default_factory_uses_packaged_scenario(self):
        expected = Config(**load_settings(ROOT/'configs/scenario-default.json'))
        self.assertEqual(asdict(make_env().config), asdict(expected))
        changed = deepcopy(load_settings(ROOT/'configs/scenario-default.json'))
        changed['aero_thrust'] = expected.aero_thrust * 1.1
        env = make_env(Config(**changed))
        for own in env.aircraft_settings.values():
            self.assertEqual(own.aero_thrust, changed['aero_thrust'])

    def test_partial_yaml_inherits_standard_then_cli_wins(self):
        baseline = load_experiment(ROOT/'experiments/default.yaml')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'partial.yaml'
            path.write_text('training:\n  iterations: 7\n', encoding='utf-8')
            args = parse_args(['--experiment', str(path), '--iterations', '9'])
            self.assertEqual(args.iterations, 9)
            self.assertEqual(args.lr, baseline['lr'])
            self.assertEqual(args.num_epochs, baseline['num_epochs'])
            self.assertEqual(args.checkpoint_dir, baseline['checkpoint_dir'])
            self.assertEqual(args.max_seq_len, baseline.get(
                'max_seq_len', APPLICATION_DEFAULTS['training']['max_seq_len']))

    def test_selected_yaml_paths_and_sequence_override(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'custom.yaml'
            path.write_text('training:\n  max_seq_len: 8\noutput:\n  checkpoint_dir: local-model\n', encoding='utf-8')
            args = parse_args(['--experiment', str(path)])
            self.assertEqual(args.max_seq_len, 8)
            self.assertEqual(args.checkpoint_dir, str((Path(folder)/'local-model').resolve()))

    def test_live_checkpoint_default_follows_training_output(self):
        from scripts import live
        expected = load_experiment(ROOT/'experiments/default.yaml')['checkpoint_dir']
        with patch('sys.argv', ['live', '--no-browser']), \
                patch.object(live, 'EvaluationSession') as session, patch.object(live, 'serve_live'):
            live.main()
        self.assertEqual(session.call_args.args[3], [expected])

    def test_baseline_hook_reads_reward_config(self):
        data = json.loads((ROOT/'configs/rewards.json').read_text(encoding='utf-8'))
        next(item for item in data['rewards'] if item['type']=='victory')['win'] = .75
        with patch('combatsim.config_defaults.read_defaults', return_value=data):
            hooks = OriginalRewardHooks()
        signal = hooks.on_win(SimpleNamespace(actor='blue_wing0'), None)
        self.assertEqual(signal.value, .75)


if __name__ == '__main__':
    unittest.main()
