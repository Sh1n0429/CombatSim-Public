"""Paired-seed evaluation against fixed and randomized rule opponents, without weight updates."""
from collections import Counter
import math
import time

import torch

from .rllib_env import CombatSimFlightEnv
from .rllib_logging import append_record
from .inference import rule_match_generator, select_actions


def _evaluate_profile(network, settings, seeds, action_mode='sample', rule_profile='fix'):
    if action_mode not in ('sample', 'greedy'):
        raise ValueError('action_mode must be sample or greedy')
    seeds = list(seeds)
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('seeds must contain integers between 0 and 4294967295')
    if len(set(seeds)) != len(seeds):
        raise ValueError('seeds must not contain duplicates')
    was_training = network.training
    device = next(network.parameters()).device
    episodes = []
    started = time.perf_counter()
    network.eval()
    try:
        with torch.inference_mode():
            for seed in seeds:
                for learning_side in ('blue', 'red'):
                    env = CombatSimFlightEnv({'settings': settings, 'rule_ratio': 1})
                    try:
                        opponent_side = 'red' if learning_side == 'blue' else 'blue'
                        observations, _ = env.reset(
                            seed=seed, options={'rule_side': opponent_side, 'rule_profile': rule_profile})
                        generator = rule_match_generator(seed, learning_side, device)
                        recurrent_states = {}
                        outside = contacts = overrides = decisions = nonzero = 0
                        turn_hist, pitch_hist = Counter(), Counter()
                        while True:
                            agents = sorted(observations)
                            actions = select_actions(network, observations, agents, action_mode, generator,
                                                     states=recurrent_states)
                            for agent, action in zip(agents, actions):
                                aircraft = env.env.aircraft[agent]
                                if not aircraft.alive:
                                    continue
                                outside += not (0 <= aircraft.x <= env.config.arena_x and
                                                0 <= aircraft.y <= env.config.arena_y and
                                                0 <= aircraft.z <= env.config.ceiling)
                                contacts += bool(observations[agent]['contacts_mask'].any())
                                overrides += env._decision_overrides.get(agent, 0.) > 0
                                decisions += 1
                                turn_hist[int(action[0])] += 1
                                pitch_hist[int(action[1])] += 1
                            observations, rewards, terminated, truncated, _ = env.step(dict(zip(agents, actions)))
                            nonzero += sum(abs(value) > 1e-12 for value in rewards.values())
                            if terminated['__all__'] or truncated['__all__']:
                                break
                        aircraft = list(env.env.aircraft.values())
                        own = [a for a in aircraft if a.team == learning_side]
                        opponents = [a for a in aircraft if a.team == opponent_side]
                        result = 'draw' if env.env.winner is None else 'win' if env.env.winner == learning_side else 'loss'
                        missiles = env.env.missiles
                        episodes.append({
                            'seed': seed, 'learning_side': learning_side, 'result': result,
                            'opponent': rule_profile,
                            'rule_profile': rule_profile,
                            'rule_parameters': dict(env.rule_policy.parameters),
                            'winner': env.env.winner, 'ticks': env.env.tick,
                            'team_return': sum(env.env.episode_returns[a.id] for a in own),
                            'survivors': sum(a.alive for a in own),
                            'opponent_survivors': sum(a.alive for a in opponents),
                            'launches': sum(m.team == learning_side for m in missiles),
                            'opponent_launches': sum(m.team == opponent_side for m in missiles),
                            'kills': len({m.target for m in missiles if m.team == learning_side and m.phase == 'hit'}),
                            'crashes': sum(not a.alive and a.z <= 0 for a in own),
                            'outside_decision_fraction': outside / max(1, decisions),
                            'contact_decision_fraction': contacts / max(1, decisions),
                            'override_decision_fraction': overrides / max(1, decisions),
                            'override_tick_fraction': sum(v['overrides'] for v in env.episode_control_counts.values()) /
                                                      max(1, sum(v['ticks'] for v in env.episode_control_counts.values())),
                            'nonzero_reward_decisions': nonzero,
                            'turn_histogram': dict(turn_hist), 'pitch_histogram': dict(pitch_hist),
                        })
                    finally:
                        env.close()
    finally:
        network.train(was_training)
    return summarize_episodes(episodes, time.perf_counter() - started)


def summarize_episodes(episodes, seconds):
    results = Counter(episode['result'] for episode in episodes)
    count = len(episodes)
    def mean(key):
        return sum(episode[key] for episode in episodes) / count
    by_side = {}
    for side in ('blue', 'red'):
        games = [e for e in episodes if e['learning_side'] == side]
        outcomes = Counter(e['result'] for e in games)
        by_side[side] = {'games': len(games), 'wins': outcomes['win'], 'draws': outcomes['draw'],
                         'losses': outcomes['loss'], 'win_rate': outcomes['win']/len(games),
                         'mean_team_return': sum(e['team_return'] for e in games)/len(games)}
    # Descriptive uncertainty assumes independent games; side-swapped games
    # sharing a seed are paired, so this is deliberately not a significance test.
    return {'episodes': episodes, 'games': count, 'by_side': by_side,
            'win_rate': results['win']/count,
            'wins': results['win'], 'draws': results['draw'], 'losses': results['loss'],
            'score': (results['win'] + .5 * results['draw']) / count,
            'mean_team_return': mean('team_return'), 'mean_survivors': mean('survivors'),
            'mean_opponent_survivors': mean('opponent_survivors'),
            'mean_launches': mean('launches'), 'mean_opponent_launches': mean('opponent_launches'),
            'mean_kills': mean('kills'), 'mean_crashes': mean('crashes'),
            'outside_decision_fraction': mean('outside_decision_fraction'),
            'contact_decision_fraction': mean('contact_decision_fraction'),
            'override_decision_fraction': mean('override_decision_fraction'),
            'override_tick_fraction': mean('override_tick_fraction'),
            'nonzero_reward_decisions': sum(e['nonzero_reward_decisions'] for e in episodes),
            'evaluation_seconds': seconds}


def evaluate_network(network, settings, seeds, action_mode='sample', rule_profile='fix'):
    profiles = {'fix': ('fix',),
                'random': ('random',), 'both': ('fix', 'random')}
    if rule_profile not in profiles:
        raise ValueError('rule_profile must be fix, random or both')
    seeds = list(seeds)
    groups = {}
    for profile in profiles[rule_profile]:
        groups[profile] = _evaluate_profile(network, settings, seeds, action_mode, profile)
    episodes = [episode for group in groups.values() for episode in group['episodes']]
    result = summarize_episodes(episodes, sum(g['evaluation_seconds'] for g in groups.values()))
    result['by_opponent'] = {name: {k: v for k, v in group.items() if k != 'episodes'}
                             for name, group in groups.items()}
    return result


def format_opponent_results(result):
    return ' | '.join(f"{name}: {group['win_rate']:.1%} "
                     f"W/D/L={group['wins']}/{group['draws']}/{group['losses']}"
                     for name, group in result['by_opponent'].items())


def evaluate_algorithm(algorithm, settings, seeds, action_mode, log_path, run_id, initial_weights,
                       rule_profile='fix'):
    network = algorithm.get_policy('main').model.network
    result = evaluate_network(network, settings, seeds, action_mode, rule_profile)
    weights = network.state_dict()
    delta_squared = sum((value.detach().cpu().double() - initial_weights[key].double()).square().sum().item()
                        for key, value in weights.items())
    base_squared = sum(value.double().square().sum().item() for value in initial_weights.values())
    result.update(type='evaluation', run_id=run_id, iteration=algorithm.iteration,
                  action_mode=action_mode, seeds=seeds,
                  rule_profile=rule_profile,
                  weight_delta_l2=math.sqrt(delta_squared),
                  relative_weight_delta=math.sqrt(delta_squared / max(base_squared, 1e-30)))
    append_record(log_path, result)
    print(f"\n[EVAL iter={algorithm.iteration:04d}] {format_opponent_results(result)} | "
          f"W/D/L={result['wins']}/{result['draws']}/{result['losses']} | "
          f"return={result['mean_team_return']:+.3f} | "
          f"launches={result['mean_launches']:.2f} kills={result['mean_kills']:.2f} "
          f"crashes={result['mean_crashes']:.2f} | "
          f"outside={result['outside_decision_fraction']:.1%} | "
          f"weight_delta={result['relative_weight_delta']:.2%}", flush=True)
    return result
