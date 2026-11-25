"""MAPPO checks: privileged value inputs, actor isolation and export parity."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from combatsim.central_state import AIRCRAFT_STATE_FIELDS, central_state_size, missile_capacity
from combatsim.env import ACTION_SHAPE, Config, Missile
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.rllib_model import CombatSimSetTransformerModel
from combatsim.rllib_experiment import load_experiment
from combatsim.set_transformer import SetTransformerActorCritic
from scripts.train import restore_training_parameters


class MAPPOChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        env = CombatSimFlightEnv({'algorithm': 'mappo', 'settings': {
            'blue': 2, 'red': 1, 'max_steps': 12, 'decision_interval_steps': 3}})
        obs, _ = env.reset(seed=17)
        model = CombatSimSetTransformerModel(env.get_observation_space(), env.get_action_space(),
            sum(ACTION_SHAPE), {'custom_model_config': {'algorithm': 'mappo'}}, 'test')
        return env, obs, model

    def tensors(self, obs):
        return {k: torch.as_tensor(v[None].copy()) for k, v in obs.items()}

    def test_global_state_tracks_opponents_and_absorbing_agents(self):
        env, obs, _ = self.fixture()
        self.assertEqual(obs['blue_0']['central_state'].shape, (central_state_size(3),))
        for sample in obs.values():
            self.assertTrue(env.get_observation_space().contains(sample))
        before = obs['blue_0']['central_state'].copy()
        env.env.aircraft['red_0'].x += 1
        after = env._obs('blue_0')['central_state']
        self.assertFalse(np.array_equal(before, after))
        env.env.aircraft['blue_0'].alive = False
        obs, *_ = env.step({a: (0, 0, 0) for a in obs})
        state = obs['blue_0']['central_state']
        self.assertEqual(state[1 + AIRCRAFT_STATE_FIELDS.index('alive')], 0)
        self.assertGreater(state[0], 0)
        self.assertTrue(env.get_observation_space().contains(obs['blue_0']))

    def test_actor_and_memory_ignore_privileged_state_but_value_uses_it(self):
        _, obs, model = self.fixture()
        model.eval()
        tokens = self.tensors(obs['blue_0'])
        tokens['central_state'].requires_grad_()
        logits, memory = model.forward({'obs': tokens}, [], torch.tensor([1]))
        value = model.value_function().clone()
        value.sum().backward()
        self.assertGreater(tokens['central_state'].grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.network.parameters()))
        changed = {**tokens, 'central_state': tokens['central_state'].detach() + 10}
        logits2, memory2 = model.forward({'obs': changed}, [], torch.tensor([1]))
        torch.testing.assert_close(logits, logits2, rtol=0, atol=0)
        for a, b in zip(memory[:2], memory2[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(torch.equal(value, model.value_function()))

    def test_actor_export_has_original_shapes_and_identical_logits(self):
        _, obs, model = self.fixture()
        exported = SetTransformerActorCritic()
        exported.load_state_dict(model.network.state_dict(), strict=True)
        model.eval()
        exported.eval()
        tokens = self.tensors(obs['blue_0'])
        logits, _ = model.forward({'obs': tokens}, [], torch.tensor([1]))
        local = {k: v for k, v in tokens.items() if k != 'central_state'}
        heads, _, _ = exported.forward_sequence(local, None, torch.tensor([1]))
        torch.testing.assert_close(logits, torch.cat(heads, -1))

    def missile_fixture(self):
        env, _, model = self.fixture()
        # Place an undetected enemy missile far outside actor sensors.
        env.env.missiles.append(Missile('hidden', 'red_0', 'red', 'blue_0',
            10000., 10000., 0., 20., 20., z=20., aim_z=20., phase='active', speed=2.))
        env.env._missile_tracking['hidden'] = True
        env.env._missile_guidance_strength['hidden'] = 1.
        return env, self.tensors(env._obs('blue_0')), model.eval()

    def test_unseen_missile_and_exact_relations_reach_critic_only(self):
        env, tokens, model = self.missile_fixture()
        self.assertEqual(tokens['missiles_mask'].sum(), 0)
        self.assertEqual(tokens['central_missiles_mask'].sum(), 1)
        self.assertEqual(tokens['central_missile_owners'][0, 0].argmax(), 2)
        self.assertEqual(tokens['central_missile_targets'][0, 0].argmax(), 0)
        tokens['central_missiles'].requires_grad_()
        logits, memory = model.forward({'obs': tokens}, [], torch.tensor([1]))
        value = model.value_function().clone()
        value.sum().backward()
        self.assertGreater(tokens['central_missiles'].grad[0, 0].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.network.parameters()))
        changed = {k: v.detach().clone() for k, v in tokens.items()}
        changed['central_missiles_mask'][:] = 0
        logits2, memory2 = model.forward({'obs': changed}, [], torch.tensor([1]))
        torch.testing.assert_close(logits, logits2, rtol=0, atol=0)
        for x, y in zip(memory[:2], memory2[:2]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertFalse(torch.equal(value, model.value_function()))
        self.assertTrue(env.get_observation_space().contains(env._obs('blue_0')))

    def test_critic_is_invariant_to_entity_order_and_ignores_padding(self):
        _, tokens, model = self.missile_fixture()
        expected = model.central_critic(tokens)
        changed = {k: v.clone() for k, v in tokens.items()}
        n = (tokens['central_state'].shape[-1]-1)//len(AIRCRAFT_STATE_FIELDS)
        order = torch.arange(n-1, -1, -1)
        changed['central_state'][:, 1:] = tokens['central_state'][:, 1:].reshape(
            1, n, -1)[:, order].reshape(1, -1)
        for key in ('central_missile_owners', 'central_missile_targets'):
            changed[key] = changed[key][:, :, order]
        torch.testing.assert_close(expected, model.central_critic(changed), atol=1e-6, rtol=1e-5)
        for key in ('central_missiles', 'central_missile_owners', 'central_missile_targets'):
            changed[key][:, 1:] = 1e10
        torch.testing.assert_close(expected, model.central_critic(changed), atol=1e-6, rtol=1e-5)
        for key in ('central_missiles', 'central_missiles_mask',
                    'central_missile_owners', 'central_missile_targets'):
            changed[key] = changed[key].flip(1)
        torch.testing.assert_close(expected, model.central_critic(changed), atol=1e-6, rtol=1e-5)

    def test_owner_target_edges_affect_value(self):
        _, tokens, model = self.missile_fixture()
        expected = model.central_critic(tokens)
        changed = {k: v.clone() for k, v in tokens.items()}
        changed['central_missile_targets'][0, 0, 0] = 0
        changed['central_missile_targets'][0, 0, 1] = 1
        self.assertFalse(torch.equal(expected, model.central_critic(changed)))

    def test_capacity_covers_roster_overrides_and_randomization(self):
        c = Config(blue=2, red=1, ammunition=0)
        self.assertEqual(missile_capacity(c), 1)
        c = Config(blue=2, red=1, ammunition=2, performance_randomization={
            'enabled': True, 'randomize': {'ammunition': [3, 8]}})
        self.assertEqual(missile_capacity(c), 24)
        c = Config(aircraft=[{'id': 'blue_0', 'team': 'blue', 'performance': {'ammunition': 50}},
                             {'id': 'red_0', 'team': 'red'}])
        self.assertEqual(missile_capacity(c), 50+c.ammunition)

    def test_ppo_observations_stay_local(self):
        env = CombatSimFlightEnv({'settings': {'blue': 1, 'red': 1}})
        obs, _ = env.reset(seed=17)
        self.assertNotIn('central_state', obs['blue_0'])

    def test_algorithm_validation_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'experiment.yaml')
            path.write_text('training:\n  algorithm: mappo\n', encoding='utf-8')
            self.assertEqual(load_experiment(path)['algorithm'], 'mappo')
            path.write_text('training:\n  algorithm: invalid\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_experiment(path)
        with self.assertRaisesRegex(ValueError, 'different algorithm'):
            restore_training_parameters(SimpleNamespace(algorithm='mappo'), {}, [])
        args = SimpleNamespace(algorithm='mappo')
        restore_training_parameters(args, {'training_parameters': {'algorithm': 'mappo'}}, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
