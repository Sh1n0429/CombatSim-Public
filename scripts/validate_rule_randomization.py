"""Checks for seeded, episode-constant NPC variation around a fixed baseline."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from combatsim.factory import load_settings
from combatsim.policies import RulePolicy
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.rllib_experiment import load_experiment
from combatsim.rllib_rating import RatingCoordinator
from combatsim.evaluation import EvaluationModels
from scripts.train import parse_args, restore_opponent_parameters

ROOT = Path(__file__).resolve().parents[1]


class RuleRandomizationChecks(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(ROOT/'configs/scenario-default.json')

    def env(self, rate=None, rule_ratio=1., settings=None):
        env = CombatSimFlightEnv({'settings': settings or self.settings,
                             'rule_ratio': rule_ratio, 'random_rate': rate})
        self.addCleanup(env.close)
        return env

    def test_legacy_checkpoint_names_are_normalized(self):
        from combatsim.rule_profiles import normalize_rule_settings, normalize_rule_statistics
        old = copy.deepcopy(self.settings)
        old['rule_policy_profiles']['medium'] = old['rule_policy_profiles'].pop('fix')
        old['rule_policy_profiles']['medium']['fire_range'] = 271.
        env = self.env(0., settings=old)
        env.reset(seed=123)
        self.assertEqual(env.rule_profile, 'fix')
        self.assertEqual(env.rule_policy.parameters['fire_range'], 271.)
        self.assertIn('medium', old['rule_policy_profiles'])
        self.assertNotIn('medium', normalize_rule_settings(old)['rule_policy_profiles'])
        state = SimpleNamespace(rule_profile_results={'medium': {'win': 2}, 'fix': {'win': 1}},
            rule_profile_ratings={'medium': 1020.},
            matchup_statistics={'rule:medium:blue': {'games': 2, 'wins': 2, 'return_sum': 3.}})
        normalize_rule_statistics(state)
        self.assertEqual(state.rule_profile_results, {'fix': {'win': 3}})
        self.assertEqual(state.rule_profile_ratings, {'fix': 1020.})
        self.assertEqual(state.matchup_statistics['rule:fix:blue']['games'], 2)

    def test_rate_endpoints_and_baseline_inheritance(self):
        for rate, profile in ((0., 'fix'), (1., 'random')):
            env = self.env(rate)
            for seed in (1, 2, 3):
                env.reset(seed=seed)
                self.assertEqual(env.rule_profile, profile)
                if profile == 'fix':
                    self.assertEqual(env.rule_policy.parameters['fire_range'], 265.)
                    self.assertEqual(env.rule_policy.parameters['fire_interval_ticks'], 35)
        settings = copy.deepcopy(self.settings)
        settings['rule_policy_profiles']['fix'] = {'fire_range': 270., 'fire_interval_ticks': 23}
        # Values outside the requested randomization still inherit the baseline.
        for key in ('fire_range', 'fire_interval_ticks'):
            settings['rule_policy_profiles']['random']['randomize'].pop(key)
        env = self.env(1., settings=settings)
        env.reset(seed=123)
        self.assertEqual(env.rule_policy.parameters['fire_range'], 270.)
        self.assertEqual(env.rule_policy.parameters['fire_interval_ticks'], 23)

    def test_samples_are_bounded_and_only_intended_fields_vary(self):
        env = self.env(1.)
        overrides = {k: v for k, v in self.settings['rule_policy_profiles']['random'].items()
                     if k != 'randomize'}
        baseline = RulePolicy({**self.settings['rule_policy'],
            **self.settings['rule_policy_profiles']['fix'], **overrides}).parameters
        bounds = self.settings['rule_policy_profiles']['random']['randomize']
        samples = []
        for seed in range(20):
            env.reset(seed=seed)
            params = dict(env.rule_policy.parameters)
            for key, value in params.items():
                if key in bounds:
                    self.assertLessEqual(bounds[key][0], value)
                    self.assertLessEqual(value, bounds[key][1])
                else:
                    self.assertEqual(value, baseline[key], key)
            self.assertIs(type(params['egress_ticks']), int)
            self.assertIs(type(params['fire_interval_ticks']), int)
            samples.append(params)
        self.assertGreater(len({json.dumps(v, sort_keys=True) for v in samples}), 1)
        for key in ('fire_range', 'fire_interval_ticks'):
            self.assertGreater(len({v[key] for v in samples}), 1)

    def test_sampled_firing_range_and_interval_control_actions(self):
        settings = copy.deepcopy(self.settings)
        ranges = settings['rule_policy_profiles']['random']['randomize']
        ranges.update(fire_range=[275., 275.], fire_interval_ticks=[17, 17])
        env = self.env(1., settings=settings)
        env.reset(seed=42)
        policy = env.rule_policy
        obs = {'tick': 100, 'self': {'id': 'test_0', 'team': 'blue',
                'x': .5, 'y': .5, 'z': .5, 'heading': 0., 'pitch': 0.,
                'ammo': 6, 'shoot_cooldown': 0},
               'airspace': {'x': 1000., 'y': 1000.},
               'performance': {'radar_range': 280.},
               'contacts': [{'id': 'target', 'distance': .276, 'bearing': 0.,
                             'heading': 0., 'elevation': 0., 'detected_by_self': True}]}
        self.assertEqual(policy.act(obs)[2], 0)
        obs['contacts'][0]['distance'] = .193
        self.assertEqual(policy.act(obs)[2], 0)
        # The rear limit is 70% of the sampled 275, not the fixed profile's range.
        obs['contacts'][0]['distance'] = .192
        self.assertEqual(policy.act(obs)[2], 1)
        obs['tick'] = 116
        self.assertEqual(policy.act(obs)[2], 0)
        obs['tick'] = 117
        self.assertEqual(policy.act(obs)[2], 1)

    def test_same_seed_reproduces_parameters_and_rollout(self):
        first, second = self.env(1.), self.env(1.)
        for env in (first, second):
            env.reset(seed=502)
        params = dict(first.rule_policy.parameters)
        self.assertEqual(params, second.rule_policy.parameters)
        for _ in range(15):
            first.step({})
            second.step({})
            self.assertEqual(first.env.snapshot(), second.env.snapshot())
            self.assertEqual(first.rule_policy.parameters, params)

    def test_rate_changes_only_rule_variant_not_spawn_or_matchup_side(self):
        first, second = self.env(0., .5), self.env(1., .5)
        saw_rule = saw_selfplay = False
        for seed in range(30):
            first.reset(seed=seed)
            second.reset(seed=seed)
            self.assertEqual(first.rule_side, second.rule_side)
            self.assertEqual(first.env.snapshot(), second.env.snapshot())
            saw_rule |= first.rule_side is not None
            saw_selfplay |= first.rule_side is None
        self.assertTrue(saw_rule and saw_selfplay)

    def test_half_rate_samples_both_and_is_reproducible(self):
        first, second = self.env(.5), self.env(.5)
        profiles = []
        for seed in range(200):
            first.reset(seed=seed)
            second.reset(seed=seed)
            self.assertEqual(first.rule_profile, second.rule_profile)
            profiles.append(first.rule_profile)
        self.assertTrue(70 <= profiles.count('random') <= 130)

    def test_explicit_evaluation_and_live_catalog_offer_fix_and_random(self):
        env = self.env(1.)
        env.reset(seed=502, options={'rule_side': 'red', 'rule_profile': 'fix'})
        self.assertEqual(env.rule_profile, 'fix')
        expected = RulePolicy(self.settings['rule_policy']).parameters
        self.assertEqual(env.rule_policy.parameters, expected)
        self.assertEqual(set(EvaluationModels(ROOT).entries), {'rule:fix', 'rule:random'})
        for removed in ('weak', 'strong'):
            with self.assertRaises(ValueError):
                env.reset(seed=502, options={'rule_side': 'red', 'rule_profile': removed})

    def test_rate_validation_and_yaml_cli(self):
        for value in (-.1, 1.1, float('nan'), float('inf'), True, '0.5'):
            with self.assertRaises(ValueError):
                self.env(value)
        self.assertEqual(load_experiment(ROOT/'experiments/default.yaml')['random_rate'], .5)
        self.assertEqual(parse_args(['--random-rate', '1']).random_rate, 1.)
        for value in (-.1, 1.1, True, '0.5'):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'experiment.yaml'
                path.write_text('opponents:\n  random_rate: '+json.dumps(value))
                with self.assertRaises(ValueError):
                    load_experiment(path)

    def test_resume_preserves_rate_unless_explicitly_overridden(self):
        saved = dict(pool_size=5, k=24., snapshot_ratio=.25, interval=500,
                     rule_ratio=.25, random_rate=.5)
        args = SimpleNamespace(snapshot_pool_size=1, rating_k=10., snapshot_ratio=0.,
            snapshot_interval_matches=100, rule_ratio=1., random_rate=1.,
            resume_override_opponents=False)
        restore_opponent_parameters(args, saved)
        self.assertEqual(args.random_rate, .5)
        args.resume_override_opponents = True
        args.random_rate = 1.
        updated = restore_opponent_parameters(args, saved)
        self.assertEqual(updated['random_rate'], 1.)
        self.assertEqual(saved['random_rate'], .5)
        args.resume_override_opponents = False
        old = dict(saved)
        old.pop('random_rate')
        restore_opponent_parameters(args, old)
        self.assertEqual(args.random_rate, 1.)

    def test_sampled_parameters_are_written_in_match_log(self):
        env = self.env(1.)
        env.reset(seed=123)
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'matches.jsonl'
            coordinator = RatingCoordinator.__ray_metadata__.modified_class(str(log), 24, 0, 1, 10)
            coordinator.report({'episode_id': 1, 'winner': None, 'seed': 123,
                'rule_side': 'red', 'rule_profile': 'random',
                'rule_parameters': dict(env.rule_policy.parameters),
                'snapshot_side': None, 'snapshot_policy_id': None,
                'simulation_steps': 100})
            record = next(r for r in map(json.loads, log.read_text().splitlines())
                          if r.get('type') == 'match')
            self.assertEqual(record['rule_profile'], 'random')
            self.assertEqual(record['rule_parameters'], env.rule_policy.parameters)


if __name__ == '__main__':
    unittest.main()
