"""Cluster-safe Elo rating and snapshot coordination for RLlib."""
import json
from pathlib import Path
import random
import uuid

import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from .rule_profiles import normalize_rule_statistics
from .rating import update_rating, RATING_DEFAULTS
from .reward_hooks import ORIGINAL_REWARD_REASONS
from .rllib_logging import combine_boundary_reasons, format_match, win_rate_window


def update_rating_against_rule(rating, learning_side, winner, k=RATING_DEFAULTS['k']):
    score = 0.5 if winner is None else 1.0 if winner == learning_side else 0.0
    return update_rating(rating, score, reference=RATING_DEFAULTS['reference'], k=k)


def update_rating_against_reference(rating, learning_side, winner, reference, k=RATING_DEFAULTS['k']):
    score = 0.5 if winner is None else 1.0 if winner == learning_side else 0.0
    return update_rating(rating, score, reference=reference, k=k)


@ray.remote
class RatingCoordinator:
    """Own rating and snapshot metadata once for the entire Ray cluster."""
    def __init__(self, path, k, snapshot_ratio, pool_size, interval, saved_state=None, run_id=None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.k, self.snapshot_ratio = float(k), float(snapshot_ratio)
        self.pool_size, self.interval = int(pool_size), int(interval)
        self.main_rating, self.completed = RATING_DEFAULTS['reference'], 0
        self.capture_count, self.next_capture_at, self.pending = 1, self.interval, 0
        self.active = ['snapshot_0']
        self.ratings, self.generations = {'snapshot_0': RATING_DEFAULTS['reference']}, {'snapshot_0': 0}
        self.outcomes = {'blue_win': 0, 'red_win': 0, 'draw': 0}
        self.reference_results = {'win': 0, 'loss': 0, 'draw': 0}
        self.recent_reference_results = []
        self.rule_profile_results = {}
        self.rule_profile_ratings = {}
        self.matchup_statistics = {}
        # Not persisted: resumed workers start fresh episodes. Occupied slots
        # must retain their weights until every match using them has finished.
        self.in_flight = {}
        if saved_state is not None:
            if saved_state.get('version') != 1:
                raise ValueError('Unsupported rating state version')
            if saved_state['pool_size'] != self.pool_size:
                raise ValueError('snapshot pool size differs from checkpoint')
            for key in self.export_state():
                if key != 'version':
                    value = (saved_state.get(key, getattr(self, key))
                             if key in ('outcomes', 'reference_results', 'rule_profile_results',
                                        'rule_profile_ratings', 'matchup_statistics', 'recent_reference_results') else saved_state[key])
                    setattr(self, key, value)
        self.recent_reference_results = list(self.recent_reference_results)[
            -win_rate_window():]
        normalize_rule_statistics(self)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'type': 'resume' if saved_state else 'start',
                'main_rating': self.main_rating, 'completed': self.completed, 'run_id': self.run_id,
                'rule_rating': RATING_DEFAULTS['reference'], 'k': self.k}) + '\n')

    def export_state(self):
        keys = ('k', 'snapshot_ratio', 'pool_size', 'interval', 'main_rating',
                'completed', 'capture_count', 'next_capture_at', 'pending',
                'active', 'ratings', 'generations', 'outcomes', 'reference_results',
                'rule_profile_results', 'rule_profile_ratings', 'matchup_statistics',
                'recent_reference_results')
        return {'version': 1, **{key: getattr(self, key) for key in keys}}


    def choose_snapshot(self, episode_id):
        if episode_id in self.in_flight:
            return dict(self.in_flight[episode_id])
        rng = random.Random(f'combatsim-snapshot:{episode_id}')
        if rng.random() >= self.snapshot_ratio:
            return None
        retiring = f'snapshot_{self.capture_count % self.pool_size}' if self.pending else None
        candidates = [slot for slot in self.active if slot != retiring]
        if not candidates:
            return None  # Let the occupied slot drain before refreshing it.
        choice = {'side': rng.choice(('blue', 'red')), 'policy_id': rng.choice(candidates)}
        self.in_flight[episode_id] = choice
        return dict(choice)

    def report(self, data):
        self.in_flight.pop(data.get('episode_id'), None)
        winner, rule_side = data['winner'], data['rule_side']
        rule_profile = data.get('rule_profile') or 'fix'
        snapshot_side, snapshot_id = data['snapshot_side'], data['snapshot_policy_id']
        matchup = 'rule' if rule_side else 'snapshot' if snapshot_side else 'selfplay'
        opponent_side = rule_side or snapshot_side
        learning_side = None if opponent_side is None else ('red' if opponent_side == 'blue' else 'blue')
        before, opponent_rating = self.main_rating, None
        if rule_side:
            opponent_rating = RATING_DEFAULTS['reference']
            self.main_rating = update_rating_against_rule(before, learning_side, winner, self.k)
            self.rule_profile_ratings[rule_profile] = update_rating_against_rule(
                self.rule_profile_ratings.get(rule_profile, RATING_DEFAULTS['reference']), learning_side, winner, self.k)
        elif snapshot_side:
            opponent_rating = self.ratings[snapshot_id]
            self.main_rating = update_rating_against_reference(
                before, learning_side, winner, opponent_rating, self.k)
        self.completed += 1
        for side, value in data.get('learning_returns_by_side', {}).items():
            label = f'{matchup}:{rule_profile if rule_side else snapshot_id if snapshot_side else "main"}:{side}'
            totals = self.matchup_statistics.setdefault(label, {'games': 0, 'wins': 0, 'draws': 0,
                                                                'losses': 0, 'return_sum': 0.})
            totals['games'] += 1
            totals['wins' if winner == side else 'draws' if winner is None else 'losses'] += 1
            totals['return_sum'] += value
        while self.completed >= self.next_capture_at:
            self.pending += 1
            self.next_capture_at += self.interval
        result = 'blue_win' if winner == 'blue' else 'red_win' if winner == 'red' else 'draw'
        self.outcomes[result] += 1
        if learning_side is not None:
            reference_result = 'draw' if winner is None else 'win' if winner == learning_side else 'loss'
            self.reference_results[reference_result] += 1
            self.recent_reference_results.append(reference_result)
            self.recent_reference_results = self.recent_reference_results[
                -win_rate_window():]
            if rule_side:
                profile_results = self.rule_profile_results.setdefault(
                    rule_profile, {'win': 0, 'loss': 0, 'draw': 0})
                profile_results[reference_result] += 1
        record = {'run_id': self.run_id, 'type': 'match', 'completed': self.completed, 'result': result,
            'seed': data.get('seed'),
            'winner': winner, 'matchup': matchup, 'rule_side': rule_side,
            'rule_profile': rule_profile if rule_side else None,
            'rule_parameters': data.get('rule_parameters') if rule_side else None,
            'snapshot_side': snapshot_side, 'snapshot_policy_id': snapshot_id,
            'snapshot_generation': self.generations.get(snapshot_id),
            'learning_side': learning_side, 'opponent_rating': opponent_rating,
            'simulation_steps': data['simulation_steps'], 'main_rating_before': before,
            'main_rating': self.main_rating, 'main_rating_delta': self.main_rating-before}
        record['learning_returns_by_side'] = data.get('learning_returns_by_side', {})
        record['rating_semantics'] = 'mixed_opponent_index_not_absolute_strength'
        record['rule_profile_ratings'] = dict(self.rule_profile_ratings)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return {'completed': self.completed, 'rating': self.main_rating,
                'before': before, 'matchup': matchup, 'result': result,
                **{key: record[key] for key in ('winner', 'learning_side',
                   'rule_profile', 'snapshot_policy_id', 'snapshot_generation',
                   'simulation_steps')}}

    def state(self):
        return {'completed': self.completed, 'main_rating': self.main_rating,
                'pending_captures': self.pending, 'outcomes': dict(self.outcomes),
                'reference_results': dict(self.reference_results),
                'recent_reference_results': list(self.recent_reference_results),
                'rule_profile_results': {name: dict(results)
                                         for name, results in self.rule_profile_results.items()},
                'rule_profile_ratings': dict(self.rule_profile_ratings),
                'matchup_statistics': {key: {**value, 'mean_return': value['return_sum']/value['games']}
                                      for key, value in self.matchup_statistics.items()}}


    def commit_snapshot(self):
        if self.pending < 1:
            return None
        slot = f'snapshot_{self.capture_count % self.pool_size}'
        if any(choice['policy_id'] == slot for choice in self.in_flight.values()):
            return None
        generation = self.capture_count
        self.ratings[slot], self.generations[slot] = self.main_rating, generation
        if slot not in self.active:
            self.active.append(slot)
        self.pending -= 1
        self.capture_count += 1
        record = {'run_id': self.run_id, 'type': 'snapshot', 'completed': self.completed, 'policy_id': slot,
            'generation': generation, 'snapshot_rating': self.main_rating,
            'active_snapshots': list(self.active)}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        return record


class RLLibRatingCallbacks(DefaultCallbacks):
    """Send episode decisions and results to the central Ray actor."""
    actor_name = None

    @classmethod
    def configure(cls, actor_name):
        cls.actor_name = actor_name

    def _actor(self, actor_name=None):
        if not hasattr(self, '_actor_handle'):
            self._actor_handle = ray.get_actor(actor_name or type(self).actor_name)
        return self._actor_handle

    def on_episode_start(self, *, episode, base_env, env_index, **kwargs):
        del kwargs
        environment = base_env.get_sub_environments()[env_index]
        choice = None if environment.rule_side else ray.get(
            self._actor(environment.rating_actor_name).choose_snapshot.remote(
                episode.episode_id))
        episode.user_data['snapshot_side'] = choice['side'] if choice else None
        if choice:
            episode.user_data['snapshot_policy_id'] = choice['policy_id']

    def on_episode_end(self, *, episode, base_env, env_index, **kwargs):
        del kwargs
        environment = base_env.get_sub_environments()[env_index]
        winner = environment.env.winner
        snapshot_side = episode.user_data.get('snapshot_side')
        learning_returns = {}
        for side in ('blue', 'red'):
            agents = [agent for agent in environment._episode_agents
                      if environment.env.aircraft[agent].team == side and side != snapshot_side]
            if agents:
                learning_returns[side] = sum(environment.env.episode_returns[a] for a in agents)/len(agents)
        result = ray.get(self._actor(environment.rating_actor_name).report.remote({'winner': winner,
            'seed': environment.episode_seed,
            'episode_id': episode.episode_id,
            'rule_side': environment.rule_side,
            'rule_profile': environment.rule_profile,
            'rule_parameters': (dict(environment.rule_policy.parameters)
                                if environment.rule_policy else None),
            'snapshot_side': episode.user_data.get('snapshot_side'),
            'snapshot_policy_id': episode.user_data.get('snapshot_policy_id'),
            'learning_returns_by_side': learning_returns,
            'simulation_steps': environment.env.tick}))
        episode.custom_metrics['main_rating'] = result['rating']
        episode.custom_metrics['blue_win'] = float(winner == 'blue')
        episode.custom_metrics['red_win'] = float(winner == 'red')
        episode.custom_metrics['draw'] = float(winner is None)
        episode.custom_metrics['rule_match'] = float(environment.rule_side is not None)
        episode.custom_metrics['snapshot_match'] = float(episode.user_data.get('snapshot_side') is not None)
        # Log mean reward per learning aircraft so different team sizes remain
        # easier to compare and avoids counting rule-controlled opponents.
        snapshot_side = episode.user_data.get('snapshot_side')
        learning_agents = {agent for agent in environment._episode_agents
                           if environment.env.aircraft[agent].team != snapshot_side}
        divisor = max(1, len(learning_agents))
        episode.custom_metrics['learning_return'] = sum(
            environment.env.episode_returns[agent] for agent in learning_agents) / divisor
        counts = environment.episode_control_counts
        episode.custom_metrics['boundary_override_fraction'] = sum(counts[a]['overrides'] for a in learning_agents) / max(
            1, sum(counts[a]['ticks'] for a in learning_agents))
        reason_means = {
            reason: sum(values.get(agent, 0.0) for agent in learning_agents) / divisor
            for reason, values in environment.episode_reward_reasons.items()
        }
        # Include zero-event episodes in every baseline reason's mean.
        for reason in ORIGINAL_REWARD_REASONS:
            reason_means.setdefault(reason, 0.0)
        original = sum(value for reason, value in reason_means.items()
                       if reason in ORIGINAL_REWARD_REASONS)
        model = sum(value for reason, value in reason_means.items()
                      if reason not in ORIGINAL_REWARD_REASONS)
        episode.custom_metrics['reward/original'] = original
        episode.custom_metrics['reward/additional'] = model
        for field in ('events', 'discarded_abs'):
            episode.custom_metrics[f'reward_clipping/{field}'] = sum(
                environment.episode_reward_clipping_by_agent[agent][field] for agent in learning_agents)
        for reason, value in combine_boundary_reasons(reason_means).items():
            episode.custom_metrics[f'reward_reason/{reason}'] = value
        print(format_match(result), flush=True)
