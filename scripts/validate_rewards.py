"""Run the no-training checks required before an simulator session."""
from dataclasses import asdict, replace
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

from combatsim.env import Config, Missile
from combatsim.rewards import RewardContext, RewardSystem
from combatsim.factory import load_settings, make_env
from combatsim.reward_hooks import (AircraftSnapshot, EmptyRewardHooks,
                                CustomRewardHooks, OriginalRewardHooks,
                                RewardEvent, RewardSignal, RewardState)
from combatsim.tournament import settings_for_team_sizes, model_team_assignments


ROOT = Path(__file__).resolve().parents[1]


def check(name, function):
    function()
    print(f'[PASS] {name}')


def sample_state():
    aircraft = (
        AircraftSnapshot('blue_0', 'blue', 20, 40, 15, 0, 0, 1, 4, True),
        AircraftSnapshot('blue_1', 'blue', 22, 45, 15, 0, 0, 1, 4, True),
        AircraftSnapshot('red_0', 'red', 80, 40, 15, math.pi, 0, 1, 4, True),
        AircraftSnapshot('red_1', 'red', 78, 45, 15, math.pi, 0, 1, 4, True),
    )
    return RewardState(10, 400, False, None, False, aircraft, ())


def hook_event(name):
    kind = name.removeprefix('on_')
    return RewardEvent(kind, actor='blue_0', target='red_0', team='blue',
                       missile='missile_0')


def normalized(value):
    if value is None:
        return ()
    if isinstance(value, RewardSignal):
        return (value,)
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, RewardSignal) for v in value):
        raise AssertionError('hook must return RewardSignal, list/tuple, or None')
    return tuple(value)


def validate_all_hooks():
    custom, state = CustomRewardHooks(), sample_state()
    methods = [name for name, _ in inspect.getmembers(EmptyRewardHooks, inspect.isfunction)
               if name.startswith('on_')]
    assert len(methods) == 18, f'expected 18 hooks, found {len(methods)}'
    for name in methods:
        for signal in normalized(getattr(custom, name)(hook_event(name), state)):
            assert signal.recipients
            assert math.isfinite(signal.value)


def validate_original_is_preserved():
    """Preserve baseline values, with individual kill/loss allocation for interns."""
    original, custom, state = OriginalRewardHooks(), CustomRewardHooks(), sample_state()
    for name in ('on_step', 'on_boundary_override', 'on_out_of_bounds', 'on_win', 'on_loss', 'on_draw', 'on_kill', 'on_friendly_loss'):
        event = hook_event(name)
        expected = normalized(getattr(original, name)(event, state))
        if name in ('on_kill', 'on_friendly_loss'):
            expected = tuple(replace(signal, recipients=(event.actor,)) for signal in expected)
        actual = normalized(getattr(custom, name)(event, state))
        for signal in expected:
            assert signal in actual, f'{name} no longer preserves {signal.reason}'


def validate_signal_guards():
    invalid = (
        lambda: RewardSignal(float('nan'), ('blue_0',), 'bad'),
        lambda: RewardSignal(1.0, ['blue_0'], 'bad'),
        lambda: RewardSignal(1.0, ('',), 'bad'),
        lambda: RewardSignal(1.0, ('blue_0',), ''),
    )
    for make in invalid:
        try:
            make()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid RewardSignal was accepted')


def validate_normal_reward_parity():
    """Compare the fixed baseline through kills, time caps, endings and reset."""
    normal_specs = json.loads((ROOT/'configs'/'rewards.json').read_text())['rewards']
    # Intern rewards may intentionally add shaping; their totals need not match.
    for profile in ('original',):
        for winner in (None, 'blue', 'red'):
            normal = RewardSystem(normal_specs)
            hooked = RewardSystem([{'type': 'hooks', 'profile': profile}])
            aircraft = {a.id: SimpleNamespace(**asdict(a)) for a in sample_state().aircraft}
            config = Config(max_steps=1305)
            for tick in range(1, 1306):
                previous = {ident: (a.x, a.y, a.z) for ident, a in aircraft.items() if a.alive}
                shot_down, events, missiles = (), (), ()
                if tick == 200:
                    aircraft['red_0'].alive = False
                    shot_down = ('red_0',)
                    missiles = (Missile('m0', 'blue_0', 'blue', 'red_0', 0, 0, 0, 0, 0, phase='hit'),)
                    events = ({'type': 'hit', 'missile': 'm0', 'target': 'red_0'},)
                context = RewardContext(config, aircraft, previous, tick == 1305,
                                        winner if tick == 1305 else None,
                                        shot_down=shot_down, events=events, missiles=missiles,
                                        tick=tick, max_steps=config.max_steps,
                                        timed_out=tick == 1305 and winner is None)
                expected, actual = normal.calculate(context), hooked.calculate(context)
                for ident in aircraft:
                    assert math.isclose(expected.get(ident, 0), actual.get(ident, 0), abs_tol=1e-12), (
                        profile, winner, tick, ident, expected, actual)
                if tick == 1300:
                    assert hooked.last_components.get('HookReward:time_penalty')
            normal.reset()
            hooked.reset()
            context = replace(context, tick=1, done=False, winner=None, timed_out=False)
            assert normal.calculate(context) == hooked.calculate(context)
            from combatsim.tactical_rewards import TimePenalty
            time_spec = next(spec for spec in normal_specs if spec['type'] == 'time_penalty')
            component = TimePenalty(**{k:v for k,v in time_spec.items() if k != 'type'})
            assert math.isclose(hooked.last_components['HookReward:time_penalty']['blue_0'],
                                component.per_tick(config.max_steps))


def validate_reward_accounting():
    settings = load_settings(ROOT / 'configs' / 'scenario-default.json')
    env = make_env(Config(**settings))
    observations, _ = env.reset(seed=123)
    accumulated = {agent: 0.0 for agent in observations}
    for _ in range(25):
        actions = {agent: (5, 5, 0) for agent in observations}
        observations, rewards, _, _, info = env.step(actions)
        reasons = {name.split(':', 1)[1]: values for name, values
                   in info['reward_components'].items() if name.startswith('HookReward:')}
        for agent, reward in rewards.items():
            explained = sum(values.get(agent, 0.0) for values in reasons.values())
            assert math.isclose(reward, explained, abs_tol=1e-12), (agent, reward, explained)
            accumulated[agent] += explained
    for agent, total in accumulated.items():
        assert math.isclose(total, env.episode_returns[agent], abs_tol=1e-12)


def validate_clipping_is_visible():
    class HugeReward(EmptyRewardHooks):
        def on_step(self, event, state):
            return RewardSignal(1.0, state.alive_team_members('blue'), 'too_large')

    settings = load_settings(ROOT / 'configs' / 'scenario-default.json')
    settings['rewards'] = [{'type': 'hooks', 'profile': 'empty',
                            'max_step_total': 0.1, 'max_episode_total': 1.0}]
    env = make_env(Config(**settings))
    component = env.reward_system.components[0]
    component.hooks = HugeReward()
    observations, _ = env.reset(seed=321)
    _, rewards, _, _, info = env.step({agent: (5, 5, 0) for agent in observations})
    assert all(math.isclose(rewards[agent], 0.1) for agent in rewards if agent.startswith('blue'))
    clipped = info['reward_clipping']['too_large']
    assert sum(row['events'] for row in clipped.values()) == 2
    assert math.isclose(sum(row['discarded_abs'] for row in clipped.values()), 1.8)


def validate_exchange_configs():
    automatic = json.loads((ROOT / 'examples' / 'teams.json').read_text(encoding='utf-8'))
    first = model_team_assignments(automatic['team_alpha'], automatic['team_bravo'], 0, seed=4102)
    second = model_team_assignments(automatic['team_alpha'], automatic['team_bravo'], 0, seed=4102)
    assert first == second
    explicit = json.loads((ROOT / 'examples' / 'teams-explicit.json').read_text(encoding='utf-8'))
    alpha, bravo = explicit['team_alpha'], explicit['team_bravo']
    assigned = model_team_assignments(alpha['models'], bravo['models'], 0, seed=4102,
                                        blue_lineup=alpha['lineup'], red_lineup=bravo['lineup'])
    assert len(assigned) == 10 and set(assigned.values()) == set(alpha['models'] + bravo['models'])
    settings = load_settings(ROOT / 'configs' / 'scenario-tournament.json')
    assert len(settings_for_team_sizes(settings, 5, 5)['aircraft']) == 10


def main():
    checks = (
        ('reward hooks return valid values', validate_all_hooks),
        ('CustomRewardHooks preserves baseline values and individual kill/loss rewards', validate_original_is_preserved),
        ('standard and hook rewards match through time caps, deaths, endings and reset', validate_normal_reward_parity),
        ('invalid RewardSignal values are rejected', validate_signal_guards),
        ('reason totals equal actual episode returns', validate_reward_accounting),
        ('reward clipping is visible with discarded value', validate_clipping_is_visible),
        ('automatic and explicit exchange teams are valid', validate_exchange_configs),
    )
    for name, function in checks:
        check(name, function)
    print(f'\nOK: all {len(checks)} simulator preflight checks passed (no training performed).')


if __name__ == '__main__':
    main()
