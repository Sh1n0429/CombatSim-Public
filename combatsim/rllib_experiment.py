"""Load an RLlib experiment with paths relative to its YAML file."""
from pathlib import Path
import math

import yaml

SECTION_FIELDS = {
    'training': {'head_hidden_sizes', 'max_seq_len', 'checkpoint_interval', 'compact_attention', 'iterations', 'batch_size', 'minibatch_size', 'num_epochs', 'seed', 'resume',
                 'resume_override_opponents',
                 'lr', 'gamma', 'gae_lambda', 'clip_param', 'entropy_coeff',
                 'vf_loss_coeff', 'vf_clip_param', 'grad_clip', 'kl_coeff', 'kl_target'},
    'opponents': {'rating_k', 'rule_ratio', 'random_rate', 'snapshot_ratio', 'snapshot_pool_size',
                  'snapshot_interval_matches'},
    'workers': {'num_env_runners', 'num_envs_per_runner', 'cpus_per_env_runner',
                'ray_address', 'ray_temp_dir'},
    'evaluation': {'enabled', 'interval', 'seeds', 'action_mode', 'log', 'policy_ids', 'num_workers',
                   'rule_profile', 'stop_win_rate', 'stop_consecutive', 'confirm_seeds'},
    'output': {'checkpoint_dir', 'rating_log', 'metrics_log', 'tensorboard_dir', 'profile'},
}
SECTION_FIELDS['training'].add('algorithm')
SECTION_FIELDS['workers'].add('learner_threads')
PATH_FIELDS = {'eval_log', 'config', 'resume', 'checkpoint_dir', 'rating_log',
               'metrics_log', 'tensorboard_dir', 'ray_temp_dir', 'profile'}
INTEGER_FIELDS = {'max_seq_len', 'checkpoint_interval', 'eval_num_workers', 'eval_interval', 'iterations', 'batch_size', 'minibatch_size', 'num_epochs', 'seed', 'snapshot_pool_size',
                  'snapshot_interval_matches', 'num_env_runners', 'num_envs_per_runner'}
FLOAT_FIELDS = {'lr', 'gamma', 'gae_lambda', 'clip_param', 'entropy_coeff',
                'vf_loss_coeff', 'vf_clip_param', 'grad_clip', 'kl_coeff', 'kl_target', 'rating_k', 'rule_ratio', 'random_rate', 'snapshot_ratio', 'cpus_per_env_runner'}
NULLABLE = {'grad_clip', 'seed', 'resume', 'profile', 'ray_address'}
INTEGER_FIELDS.add('eval_stop_consecutive')
INTEGER_FIELDS.add('learner_threads')
FLOAT_FIELDS.add('eval_stop_win_rate')
NULLABLE.update(('eval_stop_win_rate', 'eval_confirm_seeds'))


def load_experiment(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Experiment YAML must contain a mapping')
    unknown = set(data) - {'scenario', *SECTION_FIELDS}
    if unknown:
        raise ValueError(f'Unknown experiment sections: {sorted(unknown)}')
    values = {}
    if 'scenario' in data:
        values['config'] = data['scenario']
    for section, fields in SECTION_FIELDS.items():
        block = data.get(section, {})
        if not isinstance(block, dict):
            raise ValueError(f'{section} must be a mapping')
        unknown = set(block) - fields
        if unknown:
            raise ValueError(f'Unknown {section} options: {sorted(unknown)}')
        values.update({f'eval_{key}': value for key, value in block.items()}
                      if section == 'evaluation' else block)
    for key, value in values.items():
        if value is None and key in NULLABLE:
            continue
        if key == 'algorithm':
            valid = value in ('ppo', 'mappo')
        elif key == 'head_hidden_sizes':
            valid = isinstance(value, list) and all(type(size) is int and size > 0 for size in value)
        elif key == 'eval_policy_ids':
            import re
            valid = (isinstance(value, list) and bool(value) and
                     all(isinstance(policy, str) and re.fullmatch(r'initial|main|snapshot_[0-9]+', policy) for policy in value) and
                     len(set(value)) == len(value))
        elif key in ('eval_seeds', 'eval_confirm_seeds'):
            valid = (isinstance(value, list) and bool(value) and
                     all(type(seed) is int and 0 <= seed < 2**32 for seed in value) and
                     len(set(value)) == len(value))
        elif key == 'eval_action_mode':
            valid = value in ('sample', 'greedy')
        elif key == 'eval_rule_profile':
            valid = value in ('fix', 'random', 'both')
        elif key in ('compact_attention', 'eval_enabled', 'resume_override_opponents'):
            valid = type(value) is bool
        elif key in INTEGER_FIELDS:
            valid = type(value) is int
        elif key in FLOAT_FIELDS:
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise ValueError(f'Invalid value for {key}: {value!r}')
        if key == 'random_rate' and not 0 <= value <= 1:
            raise ValueError('random_rate must be between 0 and 1')
        if key in PATH_FIELDS:
            values[key] = str((path.parent / value).resolve())
    return values
