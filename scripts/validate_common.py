"""Regression checks for shared physics, checkpoints and inference utilities."""
import contextlib
from dataclasses import asdict, astuple, replace
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numba import njit
import torch

from combatsim.atmosphere import density, thrust_fraction
from combatsim.checkpoints import load_checkpoint_network, read_policy_checkpoint, save_torch
from combatsim.env import Config
from combatsim.factory import load_settings
from combatsim.inference import rule_match_generator, select_actions
from combatsim.motion import AerodynamicAircraftMotion, FlightControl, Pose
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.rllib_evaluation import evaluate_network
from combatsim.set_transformer import SetTransformerActorCritic
from combatsim.observation_schema import OBSERVATION_VERSION
from scripts import visualize


ROOT = Path(__file__).resolve().parents[1]


class CommonChecks(unittest.TestCase):
    def test_aircraft_thrust_lapse_and_legacy_setting(self):
        c = Config(aero_thrust_density_exponent=1., aero_cd0=0.,
                   aero_induced_drag=0., aero_gravity=0.)
        mover = AerodynamicAircraftMotion()
        control = FlightControl(0., 0.)
        accelerations = [mover.acceleration((0., 0., h), (250., 0., 0.), control, c)[0]
                         for h in (0., 5000., 10000.)]
        self.assertAlmostEqual(accelerations[0], c.aero_thrust/c.aero_mass)
        self.assertGreater(accelerations[0], accelerations[1])
        self.assertGreater(accelerations[1], accelerations[2])
        legacy = replace(c, aero_thrust_density_exponent=0.)
        self.assertAlmostEqual(mover.acceleration((0., 0., 10000.), (250., 0., 0.), control, legacy)[0], accelerations[0])
        self.assertEqual(Config().aero_thrust_density_exponent, 0.)
        self.assertEqual(thrust_fraction(density(0.), 1.), 1.)
        for invalid in (-1., float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                Config(aero_thrust_density_exponent=invalid)

    def test_aircraft_lapse_python_and_compiled_integration_agree(self):
        class ReferenceMotion(AerodynamicAircraftMotion):
            pass
        for exponent in (0., 1.):
            c = Config(aero_thrust_density_exponent=exponent)
            for altitude in (1., 24., 44.):
                pose = Pose(100., 100., altitude, .2, .05, 1.)
                control = FlightControl(.4, .1)
                reference = ReferenceMotion().advance(pose, control, c)
                compiled = AerodynamicAircraftMotion().advance(pose, control, c)
                np.testing.assert_allclose(astuple(reference), astuple(compiled), rtol=1e-11, atol=1e-11)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        with torch.random.fork_rng():
            torch.manual_seed(502)
            cls.network = SetTransformerActorCritic().eval()
        cls.weights = cls.network.state_dict()
        policy = cls.folder/'policies'/'main'
        policy.mkdir(parents=True)
        cls.policy_bytes = pickle.dumps({
            'weights': {'network.'+key: value.numpy() for key, value in cls.weights.items()},
            'num_grad_updates': 7,
        })
        (policy/'policy_state.pkl').write_bytes(cls.policy_bytes)
        save_torch(cls.folder/'run-initial-weights.pth', cls.weights)
        (cls.folder/'training-state.json').write_text(json.dumps(
            {'observation_version': OBSERVATION_VERSION}))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_checkpoint_roundtrip_for_main_and_initial(self):
        for policy_id in ('main', 'initial'):
            weights, state, digest = read_policy_checkpoint(self.folder, policy_id)
            self.assertEqual(set(weights), set(self.weights))
            for key in weights:
                torch.testing.assert_close(weights[key], self.weights[key], rtol=0, atol=0)
            source = (self.policy_bytes if policy_id == 'main' else
                      (self.folder/'run-initial-weights.pth').read_bytes())
            self.assertEqual(digest, hashlib.sha256(source).hexdigest())
            self.assertEqual(state.get('num_grad_updates'), 7 if policy_id == 'main' else None)
            network, _, _ = load_checkpoint_network(self.folder, policy_id)
            self.assertFalse(network.training)
            for key, value in network.state_dict().items():
                torch.testing.assert_close(value, self.weights[key], rtol=0, atol=0)

    def test_preflight_accepts_model_reward_additions(self):
        from scripts import validate_rewards as preflight
        from combatsim.reward_hooks import CustomRewardHooks, RewardSignal

        class ShapedReward(CustomRewardHooks):
            def on_step(self, event, state):
                return (*preflight.normalized(super().on_step(event, state)),
                        RewardSignal(0.00001, ('blue_0',), 'lesson_bonus'))

        with patch.object(preflight, 'CustomRewardHooks', ShapedReward):
            preflight.validate_all_hooks()
            preflight.validate_original_is_preserved()
            preflight.validate_normal_reward_parity()

    def test_preflight_rejects_wrong_kill_loss_allocation_and_value(self):
        from scripts import validate_rewards as preflight
        from combatsim.reward_hooks import CustomRewardHooks, OriginalRewardHooks, RewardSignal

        for name in ('on_kill', 'on_friendly_loss'):
            # Team allocation must not satisfy the individual reward contract.
            wrong_team = type('WrongTeam', (CustomRewardHooks,), {
                name: getattr(OriginalRewardHooks, name)})
            with self.subTest(hook=name, error='team'), \
                    patch.object(preflight, 'CustomRewardHooks', wrong_team), \
                    self.assertRaises(AssertionError):
                preflight.validate_original_is_preserved()

            reason = 'enemy_kill' if name == 'on_kill' else 'friendly_loss'

            def wrong_value(self, event, state, reason=reason):
                return RewardSignal(123., (event.actor,), reason)

            wrong_amount = type('WrongAmount', (CustomRewardHooks,), {name: wrong_value})
            with self.subTest(hook=name, error='amount'), \
                    patch.object(preflight, 'CustomRewardHooks', wrong_amount), \
                    self.assertRaises(AssertionError):
                preflight.validate_original_is_preserved()

    def test_replay_uses_saved_decision_interval(self):
        metadata_path = self.folder/'training-state.json'
        original = metadata_path.read_text()
        settings = load_settings(ROOT/'configs'/'scenario-default.json')
        settings['max_steps'] = 4
        try:
            for interval, expected in ((1, [0, 1, 2, 3, 4]), (3, [0, 3, 4])):
                with self.subTest(decision_interval=interval):
                    settings['decision_interval_steps'] = interval
                    metadata_path.write_text(json.dumps({
                        'observation_version': OBSERVATION_VERSION,
                        'scenario_settings': settings}))
                    output = self.folder/f'interval-{interval}.html'
                    argv = ['visualize', '--checkpoint', str(self.folder), '--output', str(output)]
                    with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                        visualize.main()
                    replay = json.loads(output.with_suffix('.json').read_text())
                    self.assertEqual([frame['tick'] for frame in replay['frames']], expected)
                    self.assertEqual(replay['config']['decision_interval_steps'], interval)
        finally:
            metadata_path.write_text(original)

    def test_atmosphere_reference_and_compiled_aircraft(self):
        compiled_density = njit(density)
        for altitude in (-100, 0, 5000, 11000, 11000.01, 20000):
            self.assertAlmostEqual(density(altitude), compiled_density(altitude), places=14)
        self.assertAlmostEqual(density(0), 1.2250122659906946, places=12)
        self.assertGreater(density(5000), density(11000))
        self.assertGreater(density(11000), density(20000))

        class ReferenceAircraft(AerodynamicAircraftMotion):
            pass

        config = Config()
        for pitch in (-.5, 0, .3):
            pose = Pose(50, 50, 30, .1, pitch, 1.2)
            control = FlightControl(vertical=.2, target_heading=.7)
            compiled = AerodynamicAircraftMotion().advance(pose, control, config)
            reference = ReferenceAircraft().advance(pose, control, config)
            np.testing.assert_allclose(list(asdict(compiled).values()),
                                       list(asdict(reference).values()), rtol=1e-11, atol=1e-11)

    def test_action_modes_and_explicit_generators(self):
        env = CombatSimFlightEnv({'settings': asdict(Config(max_steps=10)), 'rule_ratio': 0})
        observations, _ = env.reset(seed=8)
        agents = sorted(observations)
        first, second = rule_match_generator(8, 'blue'), rule_match_generator(8, 'blue')
        self.assertNotEqual(first.initial_seed(), rule_match_generator(8, 'red').initial_seed())
        for _ in range(3):
            np.testing.assert_array_equal(
                select_actions(self.network, observations, agents, 'sample', first),
                select_actions(self.network, observations, agents, 'sample', second))
        np.testing.assert_array_equal(
            select_actions(self.network, observations, agents, 'greedy', first),
            select_actions(self.network, observations, agents, 'greedy', second))
        with self.assertRaises(ValueError):
            select_actions(self.network, observations, agents, 'invalid')

    def test_evaluation_and_replay_produce_identical_frames(self):
        captured = []

        class RecordingEnv(CombatSimFlightEnv):
            def reset(self, **kwargs):
                result = super().reset(**kwargs)
                self.frames = [self.env.snapshot()]
                captured.append(self)
                return result

            def step(self, actions):
                result = super().step(actions)
                self.frames.append(self.env.snapshot())
                return result

        settings = load_settings(ROOT/'configs'/'scenario-default.json')
        settings['max_steps'] = 12
        scenario = self.folder/'scenario.json'
        scenario.write_text(json.dumps(settings), encoding='utf-8')
        for mode in ('sample', 'greedy'):
            captured.clear()
            with patch('combatsim.rllib_evaluation.CombatSimFlightEnv', RecordingEnv):
                evaluate_network(self.network, settings, [2001], mode)
            for side, evaluated in zip(('blue', 'red'), captured):
                output = self.folder/f'{mode}-{side}.html'
                argv = ['visualize', '--checkpoint', str(self.folder), '--scenario', str(scenario),
                        '--seed', '2001', '--learned-side', side, '--action-mode', mode,
                        '--output', str(output)]
                with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    visualize.main()
                replay = json.loads(output.with_suffix('.json').read_text())
                self.assertEqual(replay['frames'], evaluated.frames)


if __name__ == '__main__':
    unittest.main(verbosity=2)
