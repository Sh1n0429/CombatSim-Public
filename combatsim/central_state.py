"""Privileged aircraft and missile state for the MAPPO value function only."""
import math
import numpy as np


AIRCRAFT_STATE_FIELDS = (
    'self', 'ally', 'alive', 'x', 'y', 'z', 'sin_heading', 'cos_heading',
    'sin_pitch', 'cos_pitch', 'speed', 'ammo', 'shoot_cooldown',
)
MISSILE_STATE_FIELDS = (
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'elapsed', 'age_fraction',
    'midcourse', 'active', 'coast', 'tracking', 'guidance_strength',
    'reacquisition_fraction', 'aim_x', 'aim_y', 'aim_z',
)


def missile_capacity(config):
    """All initial ammunition can be airborne; never truncate live missiles."""
    upper = (config.performance_randomization.get('randomize', {}).get('ammunition', [0, 0])[1]
             if config.performance_randomization.get('enabled', True) else 0)
    roster = config.aircraft or [{} for _ in range(config.blue + config.red)]
    return max(1, sum(max(upper, spec.get('performance', {}).get('ammunition', config.ammunition))
                      for spec in roster))


def central_missiles(env, capacity):
    """All teams' live projectiles and exact owner/target relations to aircraft."""
    ids = {ident: index for index, ident in enumerate(sorted(env.aircraft))}
    missiles = [m for m in env.missiles if m.alive]
    if len(missiles) > capacity:
        raise ValueError('Live missiles exceed the scenario ammunition capacity')
    values = np.zeros((capacity, len(MISSILE_STATE_FIELDS)), dtype=np.float32)
    mask = np.zeros(capacity, dtype=np.float32)
    owners = np.zeros((capacity, len(ids)), dtype=np.float32)
    targets = np.zeros_like(owners)
    c = env.config
    for row, m in enumerate(missiles):
        own = env.aircraft_settings[m.owner]
        speed = own.aircraft_speed if m.speed is None else m.speed
        horizontal = speed * math.cos(m.pitch)
        values[row] = (
            m.x/c.arena_x, m.y/c.arena_y, m.z/c.ceiling,
            horizontal*math.cos(m.heading), horizontal*math.sin(m.heading), speed*math.sin(m.pitch),
            m.elapsed/(own.missile_lifetime*own.aero_tick_seconds), m.age/own.missile_lifetime,
            m.phase == 'midcourse', m.phase == 'active', m.phase == 'coast',
            env._missile_tracking.get(m.id, False), env._missile_guidance_strength.get(m.id, 0.),
            env._missile_reacquisition_ticks.get(m.id, 0)/own.missile_reacquisition_confirm_ticks,
            m.aim_x/c.arena_x, m.aim_y/c.arena_y, m.aim_z/c.ceiling,
        )
        mask[row] = 1
        owners[row, ids[m.owner]] = 1
        targets[row, ids[m.target]] = 1
    return {'central_missiles': values, 'central_missiles_mask': mask,
            'central_missile_owners': owners, 'central_missile_targets': targets}


def central_state_size(aircraft_count):
    return 1 + aircraft_count * len(AIRCRAFT_STATE_FIELDS)


def central_state(env, agent):
    """Stable slots with all aircraft, including undetected and dead aircraft.

    This global state is deliberately separate from actor sensors.
    Agent/team indicators allow one shared critic to predict individual returns.
    """
    c = env.config
    team = env.aircraft[agent].team
    values = [env.tick / c.max_steps]
    for ident in sorted(env.aircraft):
        a = env.aircraft[ident]
        values.extend((
            float(ident == agent), float(a.team == team), float(a.alive),
            a.x / c.arena_x, a.y / c.arena_y, a.z / c.ceiling,
            math.sin(a.heading), math.cos(a.heading),
            math.sin(a.pitch), math.cos(a.pitch), a.speed / c.aircraft_speed,
            float(a.ammo), float(a.shoot_cooldown),
        ))
    return np.asarray(values, dtype=np.float32)
