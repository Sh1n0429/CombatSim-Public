"""Multi-slot PMA shape, masking, invariance and gradient checks."""
import unittest
import torch

from combatsim.set_transformer import (
    EMBED_SIZE, PMA_SEEDS, PoolingByMultiheadAttention, SetTransformerActorCritic,
)


class PMAChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_queries_start_orthogonal_and_produce_distinct_slots(self):
        torch.manual_seed(25)
        pma = PoolingByMultiheadAttention()
        queries = pma.seed[0].detach()
        torch.testing.assert_close(queries @ queries.T, torch.eye(PMA_SEEDS)*EMBED_SIZE,
                                   atol=2e-5, rtol=1e-5)
        pooled = pma(torch.randn(2, 7, EMBED_SIZE), torch.zeros(2, 7, dtype=torch.bool))
        slots = pooled.reshape(2, PMA_SEEDS, EMBED_SIZE)
        cosine = torch.nn.functional.cosine_similarity(slots[:, :, None], slots[:, None, :], -1)
        self.assertLess(float(cosine[:, ~torch.eye(PMA_SEEDS, dtype=torch.bool)].detach().max()), .95)

    def test_query_initialization_is_reproducible(self):
        torch.manual_seed(26)
        a = PoolingByMultiheadAttention().seed.detach().clone()
        torch.manual_seed(26)
        b = PoolingByMultiheadAttention().seed.detach().clone()
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_slots_match_recurrent_input_and_all_receive_gradients(self):
        torch.manual_seed(23)
        model = SetTransformerActorCritic()
        tokens = torch.randn(2, 7, EMBED_SIZE, requires_grad=True)
        mask = torch.zeros(2, 7, dtype=torch.bool)
        pooled = model.pma(tokens, mask)
        self.assertEqual(EMBED_SIZE, 128)
        self.assertEqual(PMA_SEEDS, 4)
        self.assertEqual(pooled.shape, (2, 512))
        self.assertEqual(model.lstm.input_size, 512)
        self.assertEqual(model.lstm.hidden_size, 128)
        recurrent, _ = model.lstm(pooled[:, None])
        model.turn_actor(recurrent[:, 0]).square().sum().backward()
        self.assertTrue((model.pma.seed.grad.abs().sum(-1) > 0).all())
        self.assertGreater(tokens.grad.abs().sum().item(), 0)

    def test_permutation_invariance_and_padding_is_ignored(self):
        torch.manual_seed(24)
        pma = PoolingByMultiheadAttention().eval()
        tokens = torch.randn(2, 7, EMBED_SIZE)
        mask = torch.tensor([[False]*5+[True]*2, [True]*7])
        expected = pma(tokens, mask)
        order = torch.tensor([6, 2, 4, 0, 5, 1, 3])
        torch.testing.assert_close(expected, pma(tokens[:, order], mask[:, order]))
        changed = tokens.clone()
        changed[mask] = 1e10
        torch.testing.assert_close(expected, pma(changed, mask))
        self.assertTrue(torch.isfinite(expected).all())
        torch.testing.assert_close(expected[1], torch.zeros(512))


if __name__ == '__main__':
    unittest.main()
