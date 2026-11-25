import unittest
from unittest.mock import Mock
import torch
from combatsim.factory import load_settings
from combatsim.set_transformer import SetTransformerActorCritic
from combatsim.rllib_evaluation import evaluate_network
from combatsim.rllib_logging import rule_opponent_results
from scripts.train import write_tensorboard_evaluation
from pathlib import Path

class SplitEvaluationChecks(unittest.TestCase):
    def test_split_reproducible_counts_and_tags(self):
        torch.set_num_threads(1)
        settings = load_settings(Path(__file__).resolve().parents[1]/'configs/scenario-default.json')
        settings['max_steps'] = 3
        network = SetTransformerActorCritic()
        result = evaluate_network(network, settings, [4001, 4002], rule_profile='both')
        repeat = evaluate_network(network, settings, [4001, 4002], rule_profile='both')
        self.assertTrue(network.training)
        self.assertEqual(result['episodes'], repeat['episodes'])
        self.assertEqual(result['games'], 8)
        self.assertEqual(set(result['by_opponent']), {'fix', 'random'})
        for group in result['by_opponent'].values():
            self.assertEqual(group['games'], 4)
            self.assertEqual(group['wins'] + group['draws'] + group['losses'], 4)
            self.assertEqual(group['win_rate'], group['wins']/4)
        episodes = result['episodes']
        self.assertEqual(episodes[4]['rule_parameters'], episodes[5]['rule_parameters'])
        self.assertNotEqual(episodes[4]['rule_parameters'], episodes[6]['rule_parameters'])
        self.assertEqual(episodes[0]['rule_parameters']['fire_range'], 265)
        writer = Mock()
        write_tensorboard_evaluation(writer, dict(result, iteration=2))
        tags = {call.args[0] for call in writer.add_scalar.call_args_list}
        self.assertIn('evaluation/fix/win_rate', tags)
        self.assertIn('evaluation/random/win_rate', tags)

    def test_training_denominator_includes_draws(self):
        result = rule_opponent_results({'fix': {'win': 1, 'draw': 2, 'loss': 1}, 'random': {}})
        self.assertEqual(result['fix']['win_rate'], .25)
        self.assertIsNone(result['random']['win_rate'])

if __name__ == '__main__':
    unittest.main()
