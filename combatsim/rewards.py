"""Composable, JSON-selected rewards. These classes never modify the world."""
from dataclasses import dataclass, field, replace
import math
from .config_defaults import REWARD_DEFAULTS
from typing import Protocol
from .tactical_rewards import TACTICAL_REWARD_TYPES, validate_tactical_reward
from .reward_hooks import (AircraftSnapshot, EmptyRewardHooks, CustomRewardHooks,
                           MissileSnapshot, OriginalRewardHooks, RewardEvent,
                           RewardSignal, RewardState)


@dataclass(frozen=True)
class RewardContext:
    config: object
    aircraft: dict
    previous_positions: dict
    done: bool
    winner: str | None
    shot_down: tuple[str, ...] = ()
    crashed: tuple[str, ...] = ()
    missed_missile_owners: tuple[str, ...] = ()
    evaded_missiles: tuple[tuple[str, bool], ...] = ()
    bite_owners: tuple[str, ...] = ()
    track_loss_owners: tuple[str, ...] = ()
    radar_detection_fractions: dict = field(default_factory=dict)
    previous_poses: dict = field(default_factory=dict)
    aircraft_settings: dict = field(default_factory=dict)
    events: tuple[dict, ...] = ()
    radar_detected_targets: dict = field(default_factory=dict)
    tick: int = 0
    max_steps: int = 0
    timed_out: bool = False
    missiles: tuple = ()
    missile_telemetry: dict = field(default_factory=dict)


class Reward(Protocol):
    def calculate(self, context: RewardContext) -> dict[str, float]: ...


@dataclass(frozen=True)
class VictoryReward:
    win: float = REWARD_DEFAULTS['VictoryReward']['win']
    loss: float = REWARD_DEFAULTS['VictoryReward']['loss']
    draw: float = REWARD_DEFAULTS['VictoryReward']['draw']

    def calculate(self, context):
        if not context.done:
            return {}
        # Destroyed aircraft remain on the team's timeline as absorbing no-ops.
        return {ident: self.draw if context.winner is None else
                self.win if a.team == context.winner else self.loss
                for ident, a in context.aircraft.items()}


@dataclass(frozen=True)
class OutOfBoundsPenalty:
    penalty: float = REWARD_DEFAULTS['OutOfBoundsPenalty']['penalty']
    mode: str = REWARD_DEFAULTS['OutOfBoundsPenalty']['mode']
    include_altitude: bool = REWARD_DEFAULTS['OutOfBoundsPenalty']['include_altitude']
    max_total: float | None = REWARD_DEFAULTS['OutOfBoundsPenalty']['max_total']

    def outside(self, position, config):
        x, y, z = position
        return (not 0 <= x <= config.arena_x or not 0 <= y <= config.arena_y
                or (self.include_altitude and not 0 <= z <= config.ceiling))

    def calculate(self, context):
        result = {}
        for ident, previous in context.previous_positions.items():
            a = context.aircraft[ident]
            if self.outside((a.x, a.y, a.z), context.config) and (
                    self.mode == 'per_step' or not self.outside(previous, context.config)):
                result[ident] = self.penalty
        return result


@dataclass(frozen=True)
class BoundaryOverridePenalty:
    """Charge safety interventions once per tick, with a per-aircraft cap."""
    penalty: float = REWARD_DEFAULTS['BoundaryOverridePenalty']['penalty']
    max_total: float | None = REWARD_DEFAULTS['BoundaryOverridePenalty']['max_total']

    def calculate(self, context):
        affected = {event['aircraft'] for event in context.events
                    if event['type'] == 'boundary_override'
                    and event.get('aircraft') in context.previous_positions}
        return {ident: self.penalty for ident in affected}


@dataclass(frozen=True)
class ShootdownReward:
    kill: float = REWARD_DEFAULTS['ShootdownReward']['kill']
    loss: float = REWARD_DEFAULTS['ShootdownReward']['loss']

    def calculate(self, context):
        # Victims are transitions from this tick, not all currently dead bodies.
        # A salvo hitting the same aircraft therefore earns only one reward.
        victims = {ident for ident in context.shot_down if ident in context.previous_positions}
        losses = victims | {ident for ident in context.crashed if ident in context.previous_positions}
        if not losses:
            return {}
        return {ident: sum(self.loss if a.team == context.aircraft[victim].team else
                           self.kill if victim in victims else 0.0
                           for victim in losses)
                for ident, a in context.aircraft.items()}


@dataclass(frozen=True)
class CrashPenalty:
    penalty: float = REWARD_DEFAULTS['CrashPenalty']['penalty']

    def calculate(self, context):
        return {i:self.penalty for i in context.crashed if i in context.previous_positions}


@dataclass(frozen=True)
class MissileMissPenalty:
    penalty: float = REWARD_DEFAULTS['MissileMissPenalty']['penalty']

    def calculate(self, context):
        result = {}
        for owner in context.missed_missile_owners:
            result[owner] = result.get(owner, 0.0) + self.penalty
        return result


class HookReward:
    """Dispatch environment events to the user-configurable reward hooks."""

    def __init__(self, profile='custom', max_step_total=REWARD_DEFAULTS['HookReward']['max_step_total'],
                 max_episode_total=REWARD_DEFAULTS['HookReward']['max_episode_total']):
        profiles = {'empty': EmptyRewardHooks, 'original': OriginalRewardHooks,
                    'custom': CustomRewardHooks}
        if profile not in profiles:
            raise ValueError('hooks profile must be empty, original or custom')
        for name, value in (('max_step_total', max_step_total),
                            ('max_episode_total', max_episode_total)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        self.max_step_total = float(max_step_total)
        self.max_episode_total = float(max_episode_total)
        self.profile = profile
        self.hooks = profiles[profile]()
        self.reset()

    def reset(self):
        self.detected = {}
        self.seen_targets = set()
        self.previous_state = None
        self.pending_start = ()
        self.episode_totals = {}
        self.started = False
        self.last_reasons = {}
        self.last_clipping = {}
        reset = getattr(self.hooks, 'reset', None)
        if reset:
            reset()

    @staticmethod
    def _state(context):
        aircraft = tuple(AircraftSnapshot(
            a.id, a.team, a.x, a.y, a.z, a.heading, a.pitch, a.speed,
            a.ammo, a.alive) for a in sorted(context.aircraft.values(), key=lambda a: a.id))
        missiles = tuple(MissileSnapshot(
            m.id, m.owner, m.team, m.target, m.phase, m.age, m.alive,
            **{k: getattr(m, k) for k in ('x', 'y', 'z', 'heading', 'pitch', 'speed')},
            **context.missile_telemetry.get(m.id, {}))
            for m in sorted(context.missiles, key=lambda m: m.id))
        return RewardState(context.tick, context.max_steps or context.config.max_steps, context.done,
                           context.winner, context.timed_out, aircraft, missiles,
                           arena_x=context.config.arena_x, arena_y=context.config.arena_y,
                           ceiling=context.config.ceiling, tick_seconds=context.config.aero_tick_seconds,
                           radar_contacts=tuple((i, tuple(sorted(targets))) for i, targets
                                                in sorted(context.radar_detected_targets.items())))

    def begin_episode(self, context):
        state = self._state(context)
        self.previous_state = state
        self.started = True
        self.pending_start = self._call(self.hooks, 'on_episode_start', RewardEvent('episode_start'), state)

    @staticmethod
    def _call(hooks, name, event, state):
        signals = getattr(hooks, name)(event, state)
        if signals is None:
            return ()
        if isinstance(signals, RewardSignal):
            return (signals,)
        if not isinstance(signals, (tuple, list)) or any(
                not isinstance(signal, RewardSignal) for signal in signals):
            raise TypeError(f'{name} must return RewardSignal, a list/tuple of them, or None')
        return tuple(signals)

    def _events(self, context, state):
        dispatched = [("on_step", RewardEvent('step'))]
        if not self.started:
            dispatched.insert(0, ('on_episode_start', RewardEvent('episode_start')))
            self.started = True
        for raw in context.events:
            kind = raw['type']
            actor = raw.get('aircraft')
            missile = raw.get('missile')
            target = raw.get('target')
            if actor is None and missile:
                body = next((m for m in context.missiles if m.id == missile), None)
                actor = body.owner if body else None
                target = target or (body.target if body else None)
            team = context.aircraft[actor].team if actor in context.aircraft else None
            event = RewardEvent(kind, actor, target, team, missile,
                                reason=raw.get('reason'), tracking_source=raw.get('tracking_source'))
            mapping = {
                'boundary_override': 'on_boundary_override', 'launch': 'on_fire',
                'bite': 'on_seeker_acquired', 'missile_track_loss': 'on_missile_track_lost',
                'hit': 'on_missile_hit',
            }
            if kind in mapping:
                dispatched.append((mapping[kind], event))
            if kind == 'expired':
                reason = raw.get('reason')
                if reason == 'target_lost':
                    reason = 'target_destroyed'
                dispatched.append(('on_missile_expired', RewardEvent(
                    kind, actor, target, team, missile, reason=reason)))
        for ident, previous in context.previous_positions.items():
            aircraft = context.aircraft[ident]
            c = context.config
            horizontal = lambda x, y: not (0 <= x <= c.arena_x and 0 <= y <= c.arena_y)
            outside_h = horizontal(aircraft.x, aircraft.y)
            outside_v = not 0 <= aircraft.z <= c.ceiling
            if outside_h or outside_v:
                dispatched.append(('on_out_of_bounds', RewardEvent(
                    'out_of_bounds', ident, team=aircraft.team,
                    outside_horizontal=outside_h, outside_vertical=outside_v,
                    previous_outside_horizontal=horizontal(previous[0], previous[1]),
                    previous_outside_vertical=not 0 <= previous[2] <= c.ceiling)))
        current = {ident: frozenset(targets)
                   for ident, targets in context.radar_detected_targets.items()}
        for ident in sorted(set(current) | set(self.detected)):
            team = context.aircraft[ident].team
            for target in sorted(current.get(ident, frozenset()) - self.detected.get(ident, frozenset())):
                dispatched.append(('on_target_detected', RewardEvent(
                    'target_detected', ident, target, team,
                    first_occurrence=(ident, target) not in self.seen_targets)))
                self.seen_targets.add((ident, target))
            for target in sorted(self.detected.get(ident, frozenset()) - current.get(ident, frozenset())):
                dispatched.append(('on_target_lost', RewardEvent(
                    'target_lost', ident, target, team, reason=(
                        'observer_destroyed' if not context.aircraft[ident].alive else
                        'target_destroyed' if not context.aircraft[target].alive else 'radar_contact_lost'))))
        self.detected = current
        for victim in context.shot_down:
            victim_team = context.aircraft[victim].team
            hit = next((e for e in reversed(context.events)
                        if e['type'] == 'hit' and e.get('target') == victim), None)
            missile = next((m for m in context.missiles
                            if hit and m.id == hit.get('missile')), None)
            killer = missile.owner if missile else None
            killer_team = context.aircraft[killer].team if killer else None
            dispatched.extend([
                ('on_kill', RewardEvent('kill', killer, victim, killer_team,
                                        missile.id if missile else None)),
                ('on_friendly_loss', RewardEvent('friendly_loss', victim, killer,
                                                 victim_team, missile.id if missile else None,
                                                 reason='shot_down')),
            ])
        for victim in sorted(set(context.crashed) - set(context.shot_down)):
            if victim not in context.previous_positions:
                continue
            team = context.aircraft[victim].team
            dispatched.extend([
                ('on_friendly_loss', RewardEvent('friendly_loss', victim, team=team, reason='crash')),
            ])
        if context.done:
            for a in state.aircraft:
                name = ('on_draw' if context.winner is None else
                        'on_win' if a.team == context.winner else 'on_loss')
                dispatched.append((name, RewardEvent(name[3:], a.id, team=a.team)))
            if context.timed_out:
                dispatched.append(('on_timeout', RewardEvent('timeout')))
            dispatched.append(('on_episode_end', RewardEvent('episode_end')))
        enriched = []
        for name, event in dispatched:
            missile = state.missile_by_id(event.missile)
            previous = next((m for m in state.previous_missiles if m.id == event.missile), None)
            shooter = event.target if name == 'on_friendly_loss' else (
                event.actor if name in ('on_fire', 'on_kill', 'on_missile_hit', 'on_missile_expired',
                                       'on_seeker_acquired', 'on_missile_track_lost') else None)
            victim = event.actor if name == 'on_friendly_loss' else (
                event.target if name in ('on_kill', 'on_missile_hit') else None)
            target = state.aircraft_by_id(missile.target if missile else event.target)
            event = replace(event, shooter=shooter, victim=victim,
                observer=event.actor if name in ('on_target_detected', 'on_target_lost') else None,
                reason=event.reason,
                tracking_source=event.tracking_source or (missile.tracking_source if missile else None),
                supporters=missile.supporters if missile else (),
                previous_supporters=previous.supporters if previous else (),
                target_alive=target.alive if target else None,
                ever_seeker_acquired=missile.ever_seeker_acquired if missile else False)
            enriched.append((name, event))
        return enriched

    def calculate(self, context):
        state = self._state(context)
        if self.previous_state:
            state = replace(state, previous_aircraft=self.previous_state.aircraft,
                            previous_missiles=self.previous_state.missiles)
        result = {}
        self.last_reasons = {}
        self.last_clipping = {}
        step_totals = {}
        calls = [('on_episode_start', self.pending_start)] if self.pending_start else []
        self.pending_start = ()
        calls.extend((name, self._call(self.hooks, name, event, state))
                     for name, event in self._events(context, state))
        for name, signals in calls:
            for signal in signals:
                for ident in signal.recipients:
                    if ident not in context.aircraft:
                        raise ValueError(f'{name} returned unknown recipient {ident!r}')
                    remaining_step = (math.inf if signal.budget == 'base' else
                                      max(0.0, self.max_step_total-step_totals.get(ident, 0.0)))
                    remaining_episode = (math.inf if signal.budget == 'base' else
                                         max(0.0, self.max_episode_total-self.episode_totals.get(ident, 0.0)))
                    requested = float(signal.value)
                    value = math.copysign(min(abs(requested), remaining_step,
                                              remaining_episode), requested)
                    discarded = abs(requested) - abs(value)
                    if discarded > 1e-12:
                        clipping = self.last_clipping.setdefault(signal.reason, {}).setdefault(
                            ident, {'events': 0, 'discarded_abs': 0.0})
                        clipping['events'] += 1
                        clipping['discarded_abs'] += discarded
                    result[ident] = result.get(ident, 0.0) + value
                    reasons = self.last_reasons.setdefault(signal.reason, {})
                    reasons[ident] = reasons.get(ident, 0.0) + value
                    if signal.budget == 'additional':
                        step_totals[ident] = step_totals.get(ident, 0.0) + abs(value)
                        self.episode_totals[ident] = self.episode_totals.get(ident, 0.0) + abs(value)
        self.previous_state = state
        return result


REWARD_TYPES = {'victory': VictoryReward, 'out_of_bounds': OutOfBoundsPenalty,
                'boundary_override': BoundaryOverridePenalty,
                'shootdown': ShootdownReward, 'crash': CrashPenalty,
                'missile_miss': MissileMissPenalty, 'hooks': HookReward,
                **TACTICAL_REWARD_TYPES}


class RewardSystem:
    def __init__(self, specifications):
        if not isinstance(specifications, list):
            raise ValueError('rewards must be a list')
        self.components = []
        for spec in specifications:
            if not isinstance(spec, dict) or not isinstance(spec.get('type'), str) or spec['type'] not in REWARD_TYPES:
                raise ValueError('Unknown reward type')
            try:
                reward = REWARD_TYPES[spec['type']](**{k: v for k, v in spec.items() if k != 'type'})
            except TypeError as error:
                raise ValueError(f'Invalid reward parameters: {spec}') from error
            tactical = spec['type'] in TACTICAL_REWARD_TYPES
            if tactical:
                validate_tactical_reward(reward)
            numeric = (() if tactical or isinstance(reward, HookReward) else ('win', 'loss', 'draw') if isinstance(reward, VictoryReward) else
                       ('kill', 'loss') if isinstance(reward, ShootdownReward) else ('penalty',))
            if any(type(getattr(reward, k)) not in (float, int) or not math.isfinite(getattr(reward, k)) for k in numeric):
                raise ValueError('Reward values must be finite numbers')
            if isinstance(reward, ShootdownReward) and (reward.kill < 0 or reward.loss > 0):
                raise ValueError('shootdown requires kill >= 0 and loss <= 0')
            if isinstance(reward, OutOfBoundsPenalty) and (
                    reward.penalty > 0 or reward.mode not in ('per_step', 'on_exit')
                    or type(reward.include_altitude) is not bool):
                raise ValueError('Invalid out_of_bounds reward settings')
            if getattr(reward, 'max_total', None) is not None:
                if type(reward.max_total) not in (int,float) or not math.isfinite(reward.max_total) or reward.max_total <= 0:
                    raise ValueError('max_total must be positive and finite')
            if isinstance(reward, BoundaryOverridePenalty) and reward.penalty > 0:
                raise ValueError('boundary_override penalty must be nonpositive')
            if isinstance(reward, CrashPenalty) and reward.penalty > 0:
                raise ValueError('crash penalty must be nonpositive')
            if isinstance(reward, MissileMissPenalty) and reward.penalty > 0:
                raise ValueError('missile_miss penalty must be nonpositive')
            self.components.append(reward)
        self.reset()

    def reset(self):
        self.spent = {}
        self.last_components = {}
        self.last_clipping = {}
        for component in self.components:
            reset = getattr(component, 'reset', None)
            if reset:
                reset()

    def begin_episode(self, context):
        for component in self.components:
            if isinstance(component, HookReward):
                component.begin_episode(context)

    def calculate(self, context):
        result = {ident: 0.0 for ident in context.previous_positions}
        self.last_components = {}
        self.last_clipping = {}
        for index, component in enumerate(self.components):
            values = component.calculate(context)
            for ident, value in values.items():
                if getattr(component, 'max_total', None) is not None:
                    used = self.spent.get((index,ident),0)
                    value = math.copysign(min(abs(value),max(0,component.max_total-used)), value)
                    self.spent[index,ident] = used+abs(value)
                self.last_components.setdefault(type(component).__name__, {})[ident] = self.last_components.get(type(component).__name__,{}).get(ident,0)+value
                result[ident] = result.get(ident, 0.0) + value
            if isinstance(component, HookReward):
                for reason, reason_values in component.last_reasons.items():
                    self.last_components[f'HookReward:{reason}'] = dict(reason_values)
                self.last_clipping.update({reason: {ident: dict(values)
                                                    for ident, values in agents.items()}
                                           for reason, agents in component.last_clipping.items()})
        return result
