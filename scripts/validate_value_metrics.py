"""Value reporting excludes RNN padding without changing the PPO loss."""
import unittest
import torch

from combatsim.env import ACTION_SHAPE
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.rllib_model import CombatSimSetTransformerModel
from combatsim.rllib_logging import iteration_record


class ValueMetricChecks(unittest.TestCase):
    def model(self):
        env = CombatSimFlightEnv({'settings': {'blue': 1, 'red': 1}})
        return CombatSimSetTransformerModel(env.get_observation_space(), env.get_action_space(),
                                        sum(ACTION_SHAPE), {}, 'test')

    def test_padding_does_not_affect_reported_variance_or_loss(self):
        model = self.model()
        model._value = torch.tensor([1., 2., 100., 3., 100., 100.], requires_grad=True)
        target = torch.tensor([1., 2., 0., 3., 0., 0.])
        losses = [torch.tensor(2., requires_grad=True)]
        returned = model.custom_loss(losses, {'value_targets': target, 'seq_lens': torch.tensor([2, 1])})
        self.assertIs(returned, losses)
        self.assertEqual(float(model.tower_stats['vf_explained_var']), 1.)
        self.assertLess(float(model.metrics()['vf_explained_var_with_padding']), 0.)
        self.assertEqual(float(model.metrics()['padding_fraction']), .5)
        self.assertEqual(float(model.metrics()['value_error_variance_valid']), 0.)
        returned[0].backward()
        self.assertIsNone(model._value.grad)

    def test_nonrecurrent_and_one_sample_metrics_remain_finite(self):
        model = self.model()
        model._value = torch.tensor([1., 2., 3.])
        loss = torch.tensor(0.)
        self.assertIs(model.custom_loss(loss, {'value_targets': model._value.clone()}), loss)
        self.assertEqual(float(model.metrics()['vf_explained_var_valid']), 1.)
        model._value = torch.tensor([1.])
        self.assertIs(model.custom_loss(loss, {'value_targets': torch.tensor([1.]), 'seq_lens': [1]}), loss)

    def test_new_metrics_are_exposed_in_iteration_log(self):
        record = iteration_record({'training_iteration': 1, 'info': {'learner': {'main': {
            'learner_stats': {'cur_kl_coeff': .002}, 'model': {
                'vf_explained_var_valid': .4, 'padding_fraction': .1,
                'fire_probability_ready': .2, 'turn_entropy_alive': 2.1}}}}},
            {'completed': 1, 'main_rating': 1000}, 'test')
        self.assertEqual(record['learner']['cur_kl_coeff'], .002)
        self.assertEqual(record['model']['vf_explained_var_valid'], .4)
        self.assertEqual(record['model']['fire_probability_ready'], .2)


if __name__ == '__main__':
    unittest.main()
