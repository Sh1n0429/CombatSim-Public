"""Exact airborne limits, environment enforcement and policy action masks."""
import math
import unittest
import torch

from combatsim.entities import shot_choices, selected_entities
from combatsim.env import Config, ToyEnv, Missile, encode_action
from combatsim.factory import load_settings
from combatsim.inference import select_actions
from combatsim.observation_schema import SELF_INDEX
from combatsim.set_transformer import SetTransformerActorCritic, entity_features, observation_tensor
from pathlib import Path


class FrozenMotion:
    def advance(self, pose, command, config):
        return pose


class MissileLimitChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, limit=2):
        env = ToyEnv(Config(blue=2, red=1, ammunition=6, radar_range=1000,
                            radar_fov=2*math.pi, shoot_cooldown=1, max_in_flight_missiles=limit,
                            missile_lifetime=1000, seeker_range=5),
                     aircraft_motion=FrozenMotion(), missile_motion=FrozenMotion())
        env.reset(seed=17)
        for ident, x in (('blue_0', 10), ('blue_1', 20), ('red_0', 90)):
            a = env.aircraft[ident]
            a.x, a.y, a.z, a.heading = x, 50, 20, 0
        return env

    def fire(self, env, ident='blue_0'):
        env.aircraft[ident].shoot_cooldown = 0
        return env.step({ident: encode_action(fire=1)})

    def test_environment_blocks_third_shot_without_spending_ammo(self):
        env = self.fixture()
        self.fire(env)
        self.fire(env)
        ammo = env.aircraft['blue_0'].ammo
        self.fire(env)
        self.assertEqual(env.own_missiles_in_flight('blue_0'), 2)
        self.assertEqual(env.aircraft['blue_0'].ammo, ammo)
        self.assertFalse(any(e['type'] == 'launch' for e in env.events))
        self.fire(env, 'blue_1')
        self.assertEqual(env.own_missiles_in_flight('blue_1'), 1)
        env.missiles[0].alive = False
        self.assertTrue(env.missile_capacity_available('blue_0'))
        self.fire(env)
        self.assertEqual(env.own_missiles_in_flight('blue_0'), 2)
        self.assertEqual(env.aircraft['blue_0'].ammo, ammo-1)

    def test_no_tracking_missiles_still_count_and_zero_disables_limit(self):
        env = self.fixture()
        self.fire(env)
        self.fire(env)
        for missile in env.missiles:
            missile.phase = 'coast'
            env._missile_tracking[missile.id] = False
        self.assertFalse(env.missile_capacity_available('blue_0'))
        unlimited = self.fixture(limit=0)
        for _ in range(4):
            self.fire(unlimited)
        self.assertEqual(unlimited.own_missiles_in_flight('blue_0'), 4)

    def test_exact_count_survives_entity_truncation_and_masks_training_and_inference(self):
        env = self.fixture()
        self.fire(env)
        self.fire(env)
        env.aircraft['blue_0'].shoot_cooldown = 0
        for missile in env.missiles:
            missile.x = 500
        # Nearer allied projectiles fill the actor's entity budget.
        for i in range(40):
            env.missiles.append(Missile(f'ally_{i}', 'blue_1', 'blue', 'red_0',
                                       11, 50, 0, 90, 50, z=20, aim_z=20))
        raw = env.observations()['blue_0']
        self.assertFalse(any(m['is_own'] for m in selected_entities(raw)['friendly_missiles']))
        self.assertEqual(raw['self']['own_missiles_in_flight'], 2)
        self.assertFalse(raw['self']['missile_capacity_available'])
        self.assertEqual(shot_choices(raw), [0])
        features = entity_features(raw)
        self.assertAlmostEqual(float(features['self'][0, SELF_INDEX['own_missiles_in_flight']]), 2/3)
        tokens = observation_tensor(features)
        model = SetTransformerActorCritic().eval()
        heads, _ = model(tokens)
        torch.testing.assert_close(heads[2].softmax(-1)[0], torch.tensor([1.]+[0.]*31))
        (heads[0].log_softmax(-1)[0, 0] + heads[2].log_softmax(-1)[0, 0]).backward()
        for p in model.fire_gate.parameters():
            self.assertTrue(p.grad is None or torch.count_nonzero(p.grad) == 0)
        for mode in ('sample', 'greedy'):
            self.assertEqual(select_actions(model, {'blue_0': features}, ['blue_0'], mode)[0, 2], 0)

    def test_configuration_and_default_scenario(self):
        for invalid in (-1, True, 1.5, None):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                Config(max_in_flight_missiles=invalid)
        self.assertEqual(Config().max_in_flight_missiles, 0)
        path = Path(__file__).resolve().parents[1]/'configs/scenario-default.json'
        self.assertEqual(Config(**load_settings(path)).max_in_flight_missiles, 2)


if __name__ == '__main__':
    unittest.main()
