"""Checks for the toy self-warning observation and model compatibility.

Run with: uv run --frozen python -m scripts.validate_mws
"""
import json
import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from combatsim.checkpoints import validate_checkpoint_schema
from combatsim.env import Config, Missile, ToyEnv
from combatsim.env import TURN_TABLE, PITCH_TABLE
from combatsim.observation_schema import ENTITY_DIMS, OBSERVATION_VERSION, SELF_INDEX
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.policies import RulePolicy
from combatsim.set_transformer import entity_features, observation_tensor, SetTransformerActorCritic


class MWSChecks(unittest.TestCase):
    def fixture(self, **settings):
        env = ToyEnv(Config(blue=2, red=1, radar_range=1000,
                            radar_fov=2*math.pi, **settings))
        env.reset(seed=12)
        for ident, aircraft in env.aircraft.items():
            aircraft.x, aircraft.y, aircraft.z = (50, 50, 20)
            aircraft.heading = 0
        env.aircraft['blue_1'].x = 70
        env.aircraft['red_0'].x = 90
        return env

    def add(self, env, ident='warning', **values):
        params = dict(id=ident, owner='red_0', team='red', target='blue_0',
                      x=60, y=60, z=30, heading=math.pi, aim_x=50, aim_y=50,
                      phase='active', speed=2)
        params.update(values)
        missile = Missile(**params)
        env.missiles.append(missile)
        return missile

    def test_no_warning_and_eligibility(self):
        env = self.fixture()
        for values in ({'phase': 'midcourse'}, {'phase': 'coast'},
                       {'target': 'blue_1'}, {'team': 'blue', 'owner': 'blue_1'},
                       {'alive': False}):
            env.missiles.clear()
            self.add(env, **values)
            own = env.observations()['blue_0']['self']
            self.assertFalse(own['mws_active'], values)
            self.assertEqual(own['mws_direction'], (0., 0., 0.))
            self.assertNotIn('mws_distance', own)

    def test_nearest_warning_direction_and_no_distance_features(self):
        env = self.fixture()
        self.add(env, 'far', x=80, y=50, z=20)
        self.add(env, 'near')
        self.add(env, 'closer-but-not-a-warning', x=51, y=50, z=20, phase='midcourse')
        env.aircraft['blue_0'].heading = math.pi/2
        raw = env.observations()['blue_0']
        own = raw['self']
        self.assertTrue(own['mws_active'])
        np.testing.assert_allclose(own['mws_direction'], np.array([1, -1, 1])/math.sqrt(3))
        self.assertNotIn('mws_distance', own)
        features = entity_features(raw)
        self.assertEqual(features['self'].shape, (1, ENTITY_DIMS['self']))
        row = features['self'][0]
        self.assertEqual(row[SELF_INDEX['mws_active']], 1.)
        np.testing.assert_allclose([row[SELF_INDEX[k]] for k in
                                   ('mws_forward', 'mws_right', 'mws_up')], own['mws_direction'])
        self.assertNotIn('mws_distance', SELF_INDEX)

    def test_warning_is_independent_of_radar_but_not_enemy_tracks(self):
        env = self.fixture(share_tracks=False)
        env.config.radar_fov = math.pi/2
        missile = self.add(env, x=40, y=50, z=20)
        raw = env.observations()['blue_0']
        self.assertTrue(raw['self']['mws_active'])
        self.assertEqual(raw['missiles'], [])
        env.config.radar_range = .01
        self.assertTrue(env.observations()['blue_0']['self']['mws_active'])
        env.aircraft['blue_0'].heading = math.pi
        self.assertTrue(env.observations()['blue_0']['self']['mws_active'])
        missile.alive = False
        raw = env.observations()['blue_0']
        self.assertFalse(raw['self']['mws_active'])
        self.assertEqual(raw['missiles'], [])

    def test_warning_does_not_depend_on_shared_detection(self):
        env = self.fixture(share_tracks=True)
        env.config.radar_fov = math.pi/2
        self.add(env, x=40, y=50, z=20)
        env.aircraft['blue_1'].heading = math.pi
        self.assertTrue(env.observations()['blue_0']['self']['mws_active'])
        env.config.share_tracks = False
        self.assertTrue(env.observations()['blue_0']['self']['mws_active'])

    def test_mws_inputs_do_not_encode_range(self):
        env = self.fixture(share_tracks=False)
        env.config.radar_range = .01
        missile = self.add(env, x=60, y=50, z=20)
        first = entity_features(env.observations()['blue_0'])['self'].copy()
        missile.x = 500
        raw = env.observations()['blue_0']
        self.assertTrue(raw['self']['mws_active'])
        self.assertEqual(raw['missiles'], [])
        np.testing.assert_array_equal(first, entity_features(raw)['self'])

    def test_rule_responds_to_warning_without_radar_and_releases(self):
        env = self.fixture(share_tracks=False)
        env.config.radar_range = .01
        missile = self.add(env, x=40, y=50, z=30)
        raw = env.observations()['blue_0']
        self.assertEqual(raw['missiles'], [])
        self.assertEqual(raw['contacts'], [])
        policy = RulePolicy(seed=1)
        action = policy.act(raw)
        self.assertEqual(policy.mode('blue_0'), 'evade')
        self.assertEqual(action[2], 0)
        # Continuing warnings do not require a range field or a radar contact.
        raw['tick'] += 1
        self.assertEqual(action, policy.act(raw))
        missile.alive = False
        policy.act(env.observations()['blue_0'])
        self.assertEqual(policy.mode('blue_0'), 'search')

    def test_rule_warning_priority_and_direction_dependence(self):
        env = self.fixture()
        raw = env.observations()['blue_0']
        raw['self'].update(mws_active=True, mws_direction=(1., 1., 1.))
        # A conflicting radar track cannot replace the MWS direction.
        first = RulePolicy(seed=1).act(raw)
        changed = copy.deepcopy(raw)
        changed['missiles'] = [{'distance': 0.001, 'bearing': -1., 'elevation': -1.}]
        policy = RulePolicy(seed=1)
        self.assertEqual(first, policy.act(changed))
        self.assertEqual(policy.mode('blue_0'), 'evade')
        changed['self']['mws_direction'] = (1., -1., -1.)
        second = RulePolicy(seed=1).act(changed)
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])
        # Existing boundary recovery keeps precedence.
        changed['self']['x'] = .001
        policy = RulePolicy(seed=1)
        policy.act(changed)
        self.assertEqual(policy.mode('blue_0'), 'boundary')

    def test_rule_evasion_strength_scales_mws_response(self):
        env = self.fixture()
        raw = env.observations()['blue_0']
        raw['self'].update(mws_active=True, mws_direction=(0.0, 1.0, .5))
        profiles = {
            'small': {'evade_turn_scale': .35, 'evade_vertical_degrees': 5.0},
            'middle': {'evade_turn_scale': .65, 'evade_vertical_degrees': 15.0},
            'large': {'evade_turn_scale': 1.0, 'evade_vertical_degrees': 30.0},
        }
        actions = [RulePolicy(settings, seed=1).act(raw) for settings in profiles.values()]
        self.assertLess(abs(TURN_TABLE[actions[0][0]]), abs(TURN_TABLE[actions[1][0]]))
        self.assertLess(abs(TURN_TABLE[actions[1][0]]), abs(TURN_TABLE[actions[2][0]]))
        self.assertLess(abs(PITCH_TABLE[actions[0][1]]), abs(PITCH_TABLE[actions[1][1]]))
        self.assertLess(abs(PITCH_TABLE[actions[1][1]]), abs(PITCH_TABLE[actions[2][1]]))
        # Warning changes maneuver strength, not the independent legal shot.
        self.assertEqual([action[2] for action in actions], [1, 1, 1])

    def test_rule_evasion_parameters_are_bounded(self):
        for parameters in ({'evade_turn_scale': 1.1}, {'evade_vertical_degrees': 46.0}):
            with self.assertRaises(ValueError):
                RulePolicy(parameters)

    def test_model_adapter_and_absorbing_state(self):
        torch.set_num_threads(1)
        adapter = CombatSimFlightEnv({'settings': {'blue': 1, 'red': 1}, 'rule_ratio': 0})
        adapter.reset(seed=5)
        own = adapter.env.aircraft['blue_0']
        self.add(adapter.env, x=own.x+1, y=own.y, z=own.z)
        adapter.env.config.radar_range = 1000
        adapter.env.config.radar_fov = 2*math.pi
        raw = adapter.env.observations()['blue_0']
        features = adapter._features('blue_0', raw)
        self.assertEqual(features['self'][0, SELF_INDEX['mws_active']], 1.)
        with torch.inference_mode():
            network = SetTransformerActorCritic().eval()
            logits, value, _ = network.forward_recurrent(observation_tensor(features))
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(all(torch.isfinite(head).any() for head in logits))
        absorbing = adapter._absorbing_observation()['self'][0]
        for name in ('mws_active', 'mws_forward', 'mws_right', 'mws_up'):
            self.assertEqual(absorbing[SELF_INDEX[name]], 0.)
        adapter.reset(seed=5)
        self.assertFalse(adapter.env.observations()['blue_0']['self']['mws_active'])

    def test_previous_checkpoint_version_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'training-state.json'
            for version in range(1, OBSERVATION_VERSION):
                path.write_text(json.dumps({'observation_version': version}))
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    validate_checkpoint_schema(folder)
            path.write_text(json.dumps({'observation_version': OBSERVATION_VERSION}))
            validate_checkpoint_schema(folder)


if __name__ == '__main__':
    unittest.main(verbosity=2)
