"""CPU fast paths preserve actor outputs, gradients and Adam updates."""
from copy import deepcopy
import unittest
import numpy as np
import torch

from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.set_transformer import SetTransformerActorCritic


class CPUOptimizationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        env = CombatSimFlightEnv({'settings': {'blue': 2, 'red': 2, 'radar_range': 1000, 'radar_fov': 6.28}})
        obs, _ = env.reset(seed=17)
        sample = obs['blue_0']
        tokens = {k: torch.tensor(np.stack([v]*6)) for k, v in sample.items()}
        # Empty types and non-prefix masks must preserve entity associations.
        tokens['allies_mask'][:, 4] = tokens['allies_mask'][:, 0]
        tokens['allies'][:, 4] = tokens['allies'][:, 0]
        tokens['allies_mask'][:, 0] = 0
        return tokens

    def loss(self, heads, value):
        return value.square().sum() + sum(
            (head.softmax(-1)*torch.arange(head.shape[-1], dtype=head.dtype)).sum() for head in heads)

    def test_fast_attention_and_sparse_mlp_preserve_outputs_and_gradients(self):
        torch.manual_seed(23)
        baseline = SetTransformerActorCritic()
        baseline.compact_entity_mlp = False
        baseline.attention_need_weights = True
        tokens = self.fixture()
        lengths = torch.tensor([3, 2])
        heads, value, state = baseline.forward_sequence(tokens, None, lengths)
        self.loss(heads, value).backward()
        for sparse in (False, True):
            candidate = deepcopy(baseline)
            candidate.zero_grad(set_to_none=True)
            candidate.compact_entity_mlp = sparse
            candidate.attention_need_weights = False
            actual, av, ast = candidate.forward_sequence(tokens, None, lengths)
            for a, b in zip(heads, actual):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(value, av, atol=3e-6, rtol=3e-5)
            for a, b in zip(state, ast):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
            self.loss(actual, av).backward()
            for (name, a), (_, b) in zip(baseline.named_parameters(), candidate.named_parameters()):
                self.assertEqual(a.grad is None, b.grad is None, name)
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, atol=5e-6, rtol=5e-4, msg=name)

    def test_adam_momentum_for_empty_entity_types_is_preserved(self):
        torch.manual_seed(24)
        baseline = SetTransformerActorCritic()
        baseline.compact_entity_mlp = False
        candidate = deepcopy(baseline)
        candidate.compact_entity_mlp = True
        optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in (baseline, candidate)]
        tokens = self.fixture()
        # First build momentum with valid missiles, then remove the whole type.
        tokens['missiles_mask'][:, 0] = 1
        tokens['missiles'][:, 0] = .2
        for present in (True, False):
            tokens['missiles_mask'][:, 0] = float(present)
            for model, optimizer in zip((baseline, candidate), optimizers):
                optimizer.zero_grad(set_to_none=True)
                heads, value, _ = model.forward_sequence(tokens, None, torch.tensor([3, 2]))
                self.loss(heads, value).backward()
                optimizer.step()
        for (name, a), (_, b) in zip(baseline.named_parameters(), candidate.named_parameters()):
            if name.startswith('entity_mlps.missiles.'):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5, msg=name)
        a, av, _ = baseline.forward_sequence(tokens, None, torch.tensor([3, 2]))
        b, bv, _ = candidate.forward_sequence(tokens, None, torch.tensor([3, 2]))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(av, bv, atol=1e-5, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
