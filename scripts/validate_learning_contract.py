"""Collect real RLlib trajectories and verify terminal/absorbing PPO targets."""
import json
import argparse
import os
from pathlib import Path
import secrets

from combatsim.ray_runtime import configure_ray_runtime
configure_ray_runtime()

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.train._internal.storage import StorageContext

from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.rllib_model import CombatSimSetTransformerModel
from combatsim.rllib_distribution import GatedMultiCategorical
from combatsim.rllib_logging import iteration_record


class ScriptedDeathEnv(CombatSimFlightEnv):
    """One aircraft dies after decision 1; teammates finish a 4-decision draw."""
    def __init__(self, config=None):
        super().__init__({'settings': {'blue': 2, 'red': 1, 'ammunition': 0,
            'max_steps': 12, 'decision_interval_steps': 3,
            'rewards': [{'type': 'victory', 'draw': 1.}]}, 'seed': 17,
            'algorithm': (config or {}).get('algorithm', 'ppo')})

    def step(self, actions):
        if self.env.tick == 3:
            self.env.aircraft['blue_0'].alive = False
        return super().step(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=('ppo', 'mappo'), default='ppo')
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(f'artifacts/consistency-validation/learning-contract-{args.algorithm}').resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('RAY_AUTH_TOKEN', secrets.token_urlsafe(32))
    ray.init(include_dashboard=False, num_cpus=2, _temp_dir=str(root/'ray'))
    ModelCatalog.register_custom_model('consistency_model', CombatSimSetTransformerModel)
    ModelCatalog.register_custom_action_dist('combatsim_gated', GatedMultiCategorical)
    algorithm = None
    try:
        config = (PPOConfig().api_stack(enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False)
                  .environment(ScriptedDeathEnv, env_config={'algorithm': args.algorithm}).framework('torch')
                  .env_runners(num_env_runners=0, create_env_on_local_worker=True,
                               batch_mode='complete_episodes', rollout_fragment_length=4)
                  .learners(num_learners=0, num_gpus_per_learner=0)
                  .training(train_batch_size=16, minibatch_size=16, num_epochs=1,
                            gamma=.9, lambda_=1., model={'custom_model': 'consistency_model', 'max_seq_len': 16,
                                'custom_action_dist': 'combatsim_gated',
                                'custom_model_config': {'algorithm': args.algorithm}})
                  .multi_agent(policies={'main'}, policies_to_train=['main'],
                               policy_mapping_fn=lambda agent_id, episode, **kw: 'main'))
        storage = StorageContext(storage_path=str(root/'results'), experiment_dir_name='contract', trial_dir_name='run')
        algorithm = config.algo_class(config=config, storage=storage)
        batch = algorithm.env_runner_group.local_env_runner.sample().policy_batches['main']
        groups = {}
        for i, key in enumerate(zip(batch[SampleBatch.EPS_ID], batch[SampleBatch.AGENT_INDEX])):
            groups.setdefault(key, []).append(i)
        dead_paths = 0
        for indices in groups.values():
            rows = sorted(indices, key=lambda i: batch[SampleBatch.T][i])
            assert len(rows) == 4, ('compressed/missing decisions', len(rows))
            np.testing.assert_array_equal(batch[SampleBatch.T][rows], np.arange(4))
            np.testing.assert_array_equal(batch[SampleBatch.REWARDS][rows], [0, 0, 0, 1])
            np.testing.assert_array_equal(batch[SampleBatch.TERMINATEDS][rows], [False, False, False, True])
            assert not batch[SampleBatch.TRUNCATEDS][rows].any()
            # With lambda=1, the terminal draw must propagate through all four
            # real decision intervals, including the destroyed plane's no-ops.
            np.testing.assert_allclose(batch[Postprocessing.VALUE_TARGETS][rows], [.9**3, .9**2, .9, 1.], atol=2e-5)
            if np.all(np.abs(batch[SampleBatch.ACTION_LOGP][rows[-2:]]) < 1e-6):
                np.testing.assert_array_equal(batch[SampleBatch.ACTIONS][rows[-2:]], np.zeros((2, 3)))
                dead_paths += 1
        assert dead_paths, 'must exercise an absorbing trajectory'
        model = algorithm.get_policy('main').model
        if args.algorithm == 'mappo':
            before = {k: v.detach().clone() for k, v in model.central_critic.state_dict().items()}
        trained = algorithm.train()
        if args.algorithm == 'mappo':
            assert any(not torch.equal(before[k], v) for k, v in model.central_critic.state_dict().items())
        stats = trained['info']['learner']['main']['learner_stats']
        record = iteration_record(trained, {'completed': 1, 'main_rating': 1000}, 'contract')
        assert record['model']['alive_entropy'] is not None, trained['info']['learner']['main']
        assert 0 < record['model']['alive_sample_fraction'] < 1
        for key in ('policy_loss', 'vf_loss', 'entropy', 'kl'):
            assert np.isfinite(stats[key]), (key, stats[key])
        report = {'passed': True, 'algorithm': args.algorithm,
                  'trajectories': len(groups), 'absorbing_trajectories': dead_paths,
                  'model_metrics': record['model'],
                  'discounted_draw_targets': [.9**3, .9**2, .9, 1.],
                  'learner': {key: float(stats[key]) for key in ('policy_loss', 'vf_loss', 'entropy', 'kl')}}
        (root/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


if __name__ == '__main__':
    main()
