"""Optional tactical rewards using measured environment state, not action indices."""
from dataclasses import dataclass
import math
from .config_defaults import REWARD_DEFAULTS


@dataclass(frozen=True)
class MissileEvasionReward:
    reward: float = REWARD_DEFAULTS['MissileEvasionReward']['reward']
    require_bite: bool = REWARD_DEFAULTS['MissileEvasionReward']['require_bite']

    def calculate(self, context):
        result = {}
        for target, bitten in context.evaded_missiles:
            if not self.require_bite or bitten:
                result[target] = result.get(target, 0.0) + self.reward
        return result


@dataclass(frozen=True)
class SurvivalReward:
    reward: float = REWARD_DEFAULTS['SurvivalReward']['reward']

    def calculate(self, context):
        return ({ident: self.reward for ident, a in context.aircraft.items() if a.alive}
                if context.done else {})


@dataclass(frozen=True)
class BiteReward:
    reward: float = REWARD_DEFAULTS['BiteReward']['reward']

    def calculate(self, context):
        result = {}
        for owner in context.bite_owners:
            result[owner] = result.get(owner, 0.0) + self.reward
        return result


@dataclass(frozen=True)
class MissileTrackLossPenalty:
    penalty: float = REWARD_DEFAULTS['MissileTrackLossPenalty']['penalty']

    def calculate(self, context):
        result = {}
        for owner in context.track_loss_owners:
            result[owner] = result.get(owner, 0.0) + self.penalty
        return result


@dataclass(frozen=True)
class RadarDetectionReward:
    reward: float = REWARD_DEFAULTS['RadarDetectionReward']['reward']
    max_total: float | None = REWARD_DEFAULTS['RadarDetectionReward']['max_total']

    def calculate(self, context):
        return {ident: self.reward * fraction
                for ident, fraction in context.radar_detection_fractions.items()}


@dataclass(frozen=True)
class TimePenalty:
    """Per-tick cost, optionally spreading an episode budget over its horizon."""
    penalty: float = REWARD_DEFAULTS['TimePenalty']['penalty']
    max_total: float | None = REWARD_DEFAULTS['TimePenalty']['max_total']
    mode: str = REWARD_DEFAULTS['TimePenalty']['mode']

    def per_tick(self, max_steps):
        if self.mode == 'episode_budget':
            if max_steps <= 0:
                raise ValueError('episode_budget requires positive max_steps')
            return -self.max_total / max_steps
        return self.penalty

    def calculate(self, context):
        return {ident: self.per_tick(context.max_steps or context.config.max_steps) for ident, aircraft in context.aircraft.items()
                if aircraft.alive}


@dataclass(frozen=True)
class ExcessiveManeuverPenalty:
    penalty: float = REWARD_DEFAULTS['ExcessiveManeuverPenalty']['penalty']
    free_rate_degrees: float = REWARD_DEFAULTS['ExcessiveManeuverPenalty']['free_rate_degrees']
    max_total: float | None = REWARD_DEFAULTS['ExcessiveManeuverPenalty']['max_total']

    def calculate(self, context):
        result = {}
        for ident, previous in context.previous_poses.items():
            a = context.aircraft[ident]
            if not a.alive:
                continue
            c = context.aircraft_settings[ident]
            # Angle between velocity directions: avoids heading wrap and the
            # artificial yaw singularity near vertical flight. No roll model.
            dot = (math.sin(previous.pitch)*math.sin(a.pitch)
                   + math.cos(previous.pitch)*math.cos(a.pitch)
                   * math.cos(a.heading-previous.heading))
            rate = math.degrees(math.acos(max(-1.0, min(1.0, dot)))) / c.aero_tick_seconds
            excess = max(0.0, rate/self.free_rate_degrees-1.0)
            result[ident] = self.penalty * min(1.0, excess*excess) * c.aero_tick_seconds
        return result


@dataclass(frozen=True)
class ForwardProgressReward:
    scale: float = REWARD_DEFAULTS['ForwardProgressReward']['scale']

    def calculate(self, context):
        width = context.config.arena_x
        result = {}
        for ident, previous in context.previous_positions.items():
            a = context.aircraft[ident]
            if not a.alive:
                continue
            before = max(0.0, min(width, previous[0]))
            after = max(0.0, min(width, a.x))
            result[ident] = self.scale * (after-before)/width * (1 if a.team == 'blue' else -1)
        return result


@dataclass(frozen=True)
class MechanicalEnergyReward:
    scale: float = REWARD_DEFAULTS['MechanicalEnergyReward']['scale']
    energy_scale: float = REWARD_DEFAULTS['MechanicalEnergyReward']['energy_scale']  # Specific energy, J/kg.

    def calculate(self, context):
        result = {}
        for ident, previous in context.previous_poses.items():
            a = context.aircraft[ident]
            if not a.alive:
                continue
            c = context.aircraft_settings[ident]
            velocity_scale = c.aero_length_scale/c.aero_tick_seconds
            before_speed = c.aircraft_speed if previous.speed is None else previous.speed
            before = .5*(before_speed*velocity_scale)**2 + c.aero_gravity*previous.z*c.aero_length_scale
            after = .5*(a.speed*velocity_scale)**2 + c.aero_gravity*a.z*c.aero_length_scale
            result[ident] = self.scale*(after-before)/self.energy_scale
        return result


TACTICAL_REWARD_TYPES = {
    'missile_evasion': MissileEvasionReward,
    'survival': SurvivalReward,
    'bite': BiteReward,
    'missile_track_loss': MissileTrackLossPenalty,
    'radar_detection': RadarDetectionReward,
    'time_penalty': TimePenalty,
    'excessive_maneuver': ExcessiveManeuverPenalty,
    'forward_progress': ForwardProgressReward,
    'mechanical_energy': MechanicalEnergyReward,
}


def validate_tactical_reward(reward):
    for name in ('reward', 'penalty', 'scale', 'energy_scale', 'free_rate_degrees'):
        if not hasattr(reward, name):
            continue
        value = getattr(reward, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
        if ((name == 'penalty' and value > 0)
                or (name in ('reward', 'scale') and value < 0)
                or (name in ('energy_scale', 'free_rate_degrees') and value <= 0)):
            raise ValueError(f'Invalid {name}: {value}')
    if isinstance(reward, TimePenalty):
        if reward.mode not in ('per_step', 'episode_budget'):
            raise ValueError('Invalid time penalty mode')
        if reward.mode == 'episode_budget' and (type(reward.max_total) not in (int, float)
                or not math.isfinite(reward.max_total) or reward.max_total <= 0):
            raise ValueError('episode_budget requires a positive finite max_total')
    if isinstance(reward, MissileEvasionReward) and type(reward.require_bite) is not bool:
        raise ValueError('require_bite must be boolean')
