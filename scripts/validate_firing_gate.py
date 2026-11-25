"""Learned firing gate probabilities, masks, PPO gradients and greedy semantics."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from gymnasium import spaces

from combatsim.env import ACTION_SHAPE
from combatsim.inference import greedy_fire_action
from combatsim.rllib_distribution import GatedMultiCategorical
from combatsim.set_transformer import gated_fire_logits


class FiringGateChecks(unittest.TestCase):
    def test_gate_probability_does_not_increase_with_candidate_count(self):
        for count in (1, 2, 31):
            gate = torch.tensor([.4]).logit()
            logits = gated_fire_logits(gate, torch.zeros(1, count), torch.ones(1, count, dtype=torch.bool))
            probabilities = logits.softmax(-1)
            torch.testing.assert_close(probabilities[:, 0], torch.tensor([.6]))
            torch.testing.assert_close(probabilities[:, 1:].sum(-1), torch.tensor([.4]))

    def test_masks_and_gradients_for_hold_and_fire(self):
        gate = torch.zeros(2, requires_grad=True)
        targets = torch.tensor([[.1, .5, .3], [.1, .5, .3]], requires_grad=True)
        valid = torch.tensor([[True, False, True], [False, False, False]])
        logits = gated_fire_logits(gate, targets, valid)
        probabilities = logits.softmax(-1)
        self.assertEqual(float(probabilities[0, 2].detach()), 0)
        torch.testing.assert_close(probabilities[1], torch.tensor([1., 0, 0, 0]))
        (-torch.distributions.Categorical(logits=logits).log_prob(torch.zeros(2, dtype=torch.long)).sum()).backward()
        self.assertGreater(gate.grad[0].abs().item(), 0)
        self.assertEqual(gate.grad[1].item(), 0)
        torch.testing.assert_close(targets.grad, torch.zeros_like(targets), atol=1e-6, rtol=0)
        gate.grad = None
        targets.grad = None
        (-gated_fire_logits(gate, targets, valid)[0, 1]).backward()
        self.assertGreater(gate.grad[0].abs().item(), 0)
        self.assertGreater(targets.grad[0, 0].abs().item(), 0)
        self.assertEqual(targets.grad[0, 1].item(), 0)

    def test_greedy_threshold_and_rllib_distribution_agree(self):
        gate = torch.tensor([.3, .7]).logit()
        fire_logits = gated_fire_logits(gate, torch.zeros(2, 31), torch.ones(2, 31, dtype=torch.bool))
        # With 31 targets, joint argmax would always hold even for gate=0.7.
        torch.testing.assert_close(greedy_fire_action(fire_logits), torch.tensor([0, 1]))
        outputs = torch.cat((torch.zeros(2, ACTION_SHAPE[0]+ACTION_SHAPE[1]), fire_logits), -1)
        model = SimpleNamespace(action_space=spaces.MultiDiscrete(np.array(ACTION_SHAPE)))
        distribution = GatedMultiCategorical(outputs, model)
        torch.testing.assert_close(distribution.deterministic_sample()[:, 2], torch.tensor([0, 1]))
        actions = torch.tensor([[0, 0, 0], [0, 0, 1]])
        expected = -torch.log(torch.tensor(float(ACTION_SHAPE[0]*ACTION_SHAPE[1]))) + torch.log(torch.tensor([.7, .7/31]))
        torch.testing.assert_close(distribution.logp(actions), expected)
        expected_entropy = (torch.distributions.Bernoulli(probs=torch.tensor([.3, .7])).entropy()
                            + torch.tensor([.3, .7])*np.log(31))
        torch.testing.assert_close(distribution.multi_entropy()[:, 2], expected_entropy)


if __name__ == '__main__':
    unittest.main()
