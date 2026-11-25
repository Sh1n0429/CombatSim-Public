"""Independent recurrent critic: sequence parity, history gradients and isolation."""
import unittest
import numpy as np
import torch

from combatsim.env import ACTION_SHAPE
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.rllib_model import CombatSimSetTransformerModel
from combatsim.set_transformer import EMBED_SIZE


class CriticMemoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, size=6):
        env = CombatSimFlightEnv({'algorithm': 'mappo', 'settings': {'blue': 1, 'red': 1}})
        obs, _ = env.reset(seed=19)
        model = CombatSimSetTransformerModel(env.get_observation_space(), env.get_action_space(),
            sum(ACTION_SHAPE), {'custom_model_config': {'algorithm': 'mappo'}}, 'test').eval()
        tokens = {k: torch.tensor(np.stack([v]*size)) for k, v in obs['blue_0'].items()}
        tokens['central_state'][:, 0] = torch.arange(size)/10
        return model, tokens

    def test_stream_matches_padded_sequences(self):
        model, tokens = self.fixture()
        lengths = torch.tensor([3, 2])
        expected, updated = model.central_critic.forward_sequence(tokens, None, lengths)
        for batch, length in enumerate(lengths.tolist()):
            hidden = None
            for time in range(length):
                row = batch*3+time
                sample = {k: v[row:row+1] for k, v in tokens.items()}
                actual, hidden = model.central_critic.forward_sequence(sample, hidden, torch.tensor([1]))
                torch.testing.assert_close(expected[row], actual[0], atol=2e-6, rtol=2e-5)
            for x, y in zip(updated, hidden):
                torch.testing.assert_close(x[batch], y[0], atol=2e-6, rtol=2e-5)

    def test_value_gradient_reaches_earlier_privileged_observation(self):
        model, tokens = self.fixture(3)
        tokens['central_state'].requires_grad_()
        values, _ = model.central_critic.forward_sequence(tokens, None, torch.tensor([3]))
        values[-1].backward()
        self.assertGreater(tokens['central_state'].grad[0].abs().sum().item(), 0)
        self.assertGreater(model.central_critic.lstm.weight_hh_l0.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.network.parameters()))

    def test_separate_critic_memory_never_changes_actor(self):
        model, tokens = self.fixture(1)
        state = [s[None] for s in model.get_initial_state()]
        self.assertEqual(len(state), 4)
        self.assertTrue(all(s.shape == (1, EMBED_SIZE) for s in state))
        logits, updated = model.forward({'obs': tokens}, state, torch.tensor([1]))
        value = model.value_function().clone()
        changed = [state[0], state[1], torch.randn_like(state[2]), torch.randn_like(state[3])]
        logits2, updated2 = model.forward({'obs': tokens}, changed, torch.tensor([1]))
        torch.testing.assert_close(logits, logits2, rtol=0, atol=0)
        for x, y in zip(updated[:2], updated2[:2]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertFalse(torch.equal(value, model.value_function()))


if __name__ == '__main__':
    unittest.main()
