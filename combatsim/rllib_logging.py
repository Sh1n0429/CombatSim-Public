"""Readable console summaries and machine-readable experiment metrics."""
import json
import math
from collections import Counter
from numbers import Real
from datetime import datetime, timezone
from pathlib import Path

from .config_defaults import APPLICATION_DEFAULTS


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def _number(value):
    return float(value) if isinstance(value, Real) and math.isfinite(value) else None


def _custom_metric_mean(custom, name):
    """Read a custom-metric mean across supported RLlib result layouts."""
    for key in (name, f'{name}_mean'):
        value = custom.get(key)
        if isinstance(value, dict):
            value = value.get('mean')
        value = _number(value)
        if value is not None:
            return value
    return None


def win_rate_window():
    value = APPLICATION_DEFAULTS['training']['win_rate_window']
    if type(value) is not int or value < 1:
        raise ValueError('training.win_rate_window must be a positive integer')
    return value


def opponent_results(counts):
    games = sum(counts.get(key, 0) for key in ('win', 'draw', 'loss'))
    return {'games': games, 'wins': counts.get('win', 0),
            'draws': counts.get('draw', 0), 'losses': counts.get('loss', 0),
            'win_rate': counts.get('win', 0) / games if games else None}


def rule_opponent_results(profiles):
    result = {}
    for profile, counts in profiles.items():
        name = profile
        result[name] = opponent_results(counts)
    return result


def combine_boundary_reasons(reasons):
    """Combine boundary costs for reporting without changing reward accounting."""
    combined = dict(reasons)
    if 'boundary_override' in combined or 'out_of_bounds' in combined:
        combined['boundary_penalty'] = (
            combined.get('boundary_penalty', 0.0)
            + combined.pop('boundary_override', 0.0)
            + combined.pop('out_of_bounds', 0.0))
    return combined


def iteration_record(result, state, run_id):
    timers = result.get('timers', {})
    env = result.get('env_runners', {})
    learner = result.get('info', {}).get('learner', {}).get('main', {}).get('learner_stats', {})
    model_stats = result.get('info', {}).get('learner', {}).get('main', {}).get('model', {})
    custom = env.get('custom_metrics', result.get('custom_metrics', {})) or {}
    learning_return = _custom_metric_mean(custom, 'learning_return')
    reasons = {}
    for key in custom:
        base = key
        for suffix in ('_mean', '_min', '_max'):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        if base.startswith('reward_reason/'):
            reason = base.split('/', 1)[1]
            value = _custom_metric_mean(custom, base)
            if value is not None:
                reasons[reason] = value
    return {
        'type': 'iteration', 'run_id': run_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'iteration': result['training_iteration'],
        'matches': state['completed'], 'main_rating': state['main_rating'],
        'outcomes': state.get('outcomes', {}),
        'reference_results': state.get('reference_results', {}),
        'opponent_results': opponent_results(state.get('reference_results', {})),
        'recent_opponent_results': {
            **opponent_results(Counter(state.get('recent_reference_results', []))),
            'window': win_rate_window(),
        },
        'rule_profile_results': state.get('rule_profile_results', {}),
        'rule_opponent_results': rule_opponent_results(state.get('rule_profile_results', {})),
        'rule_profile_ratings': state.get('rule_profile_ratings', {}),
        'matchup_statistics': state.get('matchup_statistics', {}),
        'rating_semantics': 'mixed_opponent_index_not_absolute_strength',
        'boundary_override_fraction': _custom_metric_mean(custom, 'boundary_override_fraction'),
        'learner': {key: _number(learner.get(key)) for key in
                    ('policy_loss', 'vf_loss', 'vf_explained_var', 'entropy', 'kl', 'cur_lr', 'cur_kl_coeff')},
        'model': {key: _number(model_stats.get(key)) for key in (
            'alive_entropy', 'alive_sample_fraction', 'turn_entropy_alive', 'pitch_entropy_alive',
            'fire_probability_ready', 'fire_gate_above_half_ready', 'fire_ready_fraction',
            'vf_explained_var_with_padding', 'vf_explained_var_valid', 'padding_fraction',
            'value_target_variance_valid', 'value_error_variance_valid')},
        'mean_return': (learning_return if learning_return is not None else
                        _number(env.get('episode_return_mean',
                                        env.get('episode_reward_mean', result.get('episode_reward_mean'))))),
        'mean_return_unit': ('mean_per_learning_aircraft_per_episode' if learning_return is not None
                             else 'legacy_episode_return'),
        'reward_breakdown': {
            'unit': 'mean_per_learning_aircraft_per_episode',
            'original': _custom_metric_mean(custom, 'reward/original'),
            'additional': _custom_metric_mean(custom, 'reward/additional'),
            'reasons': dict(sorted(combine_boundary_reasons(reasons).items())),
            'clipping_events': _custom_metric_mean(custom, 'reward_clipping/events'),
            'clipped_value_abs': _custom_metric_mean(custom, 'reward_clipping/discarded_abs'),
        },
        'env_steps': _number(result.get('num_env_steps_sampled_this_iter')),
        'iteration_seconds': _number(result.get('time_this_iter_s')),
        'timer_aggregation': 'rolling_mean',
        'sample_seconds': _number(timers.get('sample_time_ms', 0) / 1000),
        'learn_seconds': _number(timers.get('learn_time_ms', 0) / 1000),
    }


def format_iteration(record):
    outcomes, reference = record['outcomes'], record['reference_results']
    cumulative = record.get('opponent_results', opponent_results(reference))
    recent = record.get('recent_opponent_results', {
        **opponent_results({}), 'window': win_rate_window()})
    def percentage(value):
        return '-' if value is None else f'{value:.1%}'
    learner = record.get('learner', {})
    breakdown = record.get('reward_breakdown', {})
    def number(value, suffix=''):
        return '-' if value is None else f'{value:.2f}{suffix}'
    def statistic(value):
        return '-' if value is None else f'{value:.4g}'
    return (
        '\n' + '=' * 78 + '\n'
        f"[TRAIN] iteration {record['iteration']:>5} | matches {record['matches']:>7} "
        f"| mixed-opponent index {record['main_rating']:>8.1f}\n"
        f"  Match results     BLUE {outcomes.get('blue_win', 0):>5}  "
        f"RED {outcomes.get('red_win', 0):>5}  DRAW {outcomes.get('draw', 0):>5}\n"
        f"  MAIN vs opponents WIN  {reference.get('win', 0):>5}  "
        f"LOSS {reference.get('loss', 0):>5}  DRAW {reference.get('draw', 0):>5}  "
        f"WIN RATE {percentage(cumulative['win_rate'])}\n"
        f"  MAIN recent {recent['games']}/{recent['window']:<7} "
        f"WIN  {recent['wins']:>5}  LOSS {recent['losses']:>5}  "
        f"DRAW {recent['draws']:>5}  WIN RATE {percentage(recent['win_rate'])}\n"
        f"  Time (avg)        sample {number(record['sample_seconds'], 's'):>8}  "
        f"learn {number(record['learn_seconds'], 's'):>8}\n"
        f"  Wall time         this update {number(record['iteration_seconds'], 's'):>8}\n"
        f"  Mixed mean return {number(record['mean_return'])}\n"
        f"  Rule-specific indices {record.get('rule_profile_ratings', {})}\n"
        f"  Boundary override {number(record.get('boundary_override_fraction'))}\n"
        f"  Reward mean/plane original {number(breakdown.get('original')):>8}  "
        f"additional {number(breakdown.get('additional')):>8}\n"
        f"  Reward reasons    " + (', '.join(
            f"{key}={value:+.3f}" for key, value in breakdown.get('reasons', {}).items())
            or '-') + "\n"
        f"  Reward clipping   events {number(breakdown.get('clipping_events')):>8}  "
        f"discarded |value| {number(breakdown.get('clipped_value_abs')):>8}\n"
        f"  Loss              policy {statistic(learner.get('policy_loss')):>8}  "
        f"value {statistic(learner.get('vf_loss')):>8}  "
        f"entropy {statistic(learner.get('entropy')):>8}\n"
        f"  Optimizer         LR {statistic(learner.get('cur_lr')):>12}  "
        f"KL {statistic(learner.get('kl')):>11}\n" + '=' * 78
    )


def format_match(result):
    if result['matchup'] == 'selfplay':
        outcome = result['result'].upper()
        opponent = 'SELFPLAY'
    else:
        outcome = ('DRAW' if result['winner'] is None else
                   'WIN' if result['winner'] == result['learning_side'] else 'LOSS')
        opponent = ((f"RULE:{result.get('rule_profile') or 'fix'}") if result['matchup'] == 'rule' else
                    f"{result['snapshot_policy_id']}@g{result['snapshot_generation']}")
        opponent += f" MAIN={result['learning_side'].upper()}"
    return (f"[MATCH {result['completed']:06d}] {opponent:<29} {outcome:<8} "
            f"ticks={result['simulation_steps']:>4} | "
            f"rating {result['before']:.1f} -> {result['rating']:.1f} "
            f"({result['rating']-result['before']:+.1f})")
