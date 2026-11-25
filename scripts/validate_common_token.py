"""Common context participates in set attention with its own embedding."""
import unittest
import torch

from combatsim.observation_schema import COMMON_INDEX, SELF_INDEX
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.set_transformer import SetTransformerActorCritic, observation_tensor


class CommonTokenChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_time_is_shared_and_separate_from_self(self):
        env = CombatSimFlightEnv({'settings': {'blue': 1, 'red': 1, 'max_steps': 20}})
        obs, _ = env.reset(seed=17)
        obs, *_ = env.step({a: (5, 5, 0) for a in obs})
        self.assertNotIn('time', SELF_INDEX)
        for sample in obs.values():
            self.assertAlmostEqual(float(sample['common'][0, COMMON_INDEX['time']]), .05)
            self.assertEqual(sample['common_mask'][0], 1)
            self.assertTrue(env.get_observation_space().contains(sample))

    def test_common_has_own_embedding_and_reaches_actor(self):
        env = CombatSimFlightEnv({'settings': {'blue': 1, 'red': 1}})
        obs, _ = env.reset(seed=17)
        net = SetTransformerActorCritic()
        tokens = observation_tensor(obs['blue_0'])
        tokens['common'][:] = .5
        tokens['common'].requires_grad_()
        heads, _ = net(tokens)
        heads[0][0, 0].backward()
        self.assertGreater(tokens['common'].grad.abs().sum().item(), 0)
        self.assertGreater(net.entity_mlps['common'][0].weight.grad.abs().sum().item(), 0)

    def test_full_entity_budget_keeps_common_and_all_entities(self):
        env = CombatSimFlightEnv({'settings': {'blue': 1, 'red': 1}})
        obs, _ = env.reset(seed=17)
        tokens = observation_tensor(obs['blue_0'])
        # 31 other entities plus self and common: no valid token is dropped.
        tokens['allies_mask'][:] = 1
        tokens['contacts_mask'][:] = 0
        tokens['missiles_mask'][:] = 0
        tokens['friendly_missiles_mask'][:] = 0
        compact = SetTransformerActorCritic(compact_attention=True).eval()
        full = SetTransformerActorCritic(compact_attention=False).eval()
        full.load_state_dict(compact.state_dict())
        with torch.no_grad():
            a, av = compact(tokens)
            b, bv = full(tokens)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)
        torch.testing.assert_close(av, bv)


if __name__ == '__main__':
    unittest.main()
