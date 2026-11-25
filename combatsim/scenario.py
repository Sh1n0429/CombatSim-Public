"""Validation and per-aircraft configuration for explicit scenario rosters."""
from copy import deepcopy
from dataclasses import replace
import math


PERFORMANCE_FIELDS = {
    'aircraft_motion', 'aircraft_speed', 'turn_rate', 'pitch_rate', 'max_pitch', 'pitch_error_scale',
    'ammunition', 'shoot_cooldown', 'max_in_flight_missiles', 'radar_range', 'radar_fov',
    'aero_mass', 'aero_wing_area', 'aero_thrust', 'aero_cd0', 'aero_induced_drag',
    'aero_cl_max', 'aero_max_load', 'aero_max_bank',
    'missile_motion', 'missile_lifetime', 'seeker_delay',
    'seeker_range', 'seeker_fov', 'hit_radius', 'aero_missile_mass',
    'aero_missile_area', 'aero_missile_thrust', 'aero_missile_burn_seconds',
    'aero_missile_boost_thrust', 'aero_missile_boost_seconds',
    'aero_missile_sustain_thrust', 'aero_missile_sustain_seconds',
    'aero_missile_drag', 'aero_missile_gravity', 'aero_missile_turn_rate',
    'aero_missile_turn_drag',
    'missile_reacquisition_ramp_seconds',
    'missile_reacquisition_confirm_ticks',
}


def aircraft_config(config, spec):
    return replace(config, aircraft=None, **spec.get('performance', {}))


def validate_aircraft(config):
    roster = config.aircraft
    if not isinstance(roster, list) or not roster:
        raise ValueError('aircraft must be a nonempty list')
    ids = set()
    counts = {'blue': 0, 'red': 0}
    for spec in roster:
        if not isinstance(spec, dict) or set(spec) - {'id', 'team', 'spawn', 'heading', 'pitch', 'performance'}:
            raise ValueError('Invalid aircraft specification')
        ident, team = spec.get('id'), spec.get('team')
        if not isinstance(ident, str) or not ident.strip() or ident in ids:
            raise ValueError('Aircraft IDs must be unique nonempty strings')
        if not isinstance(team, str) or team not in counts:
            raise ValueError('Aircraft team must be blue or red')
        performance = spec.get('performance', {})
        if not isinstance(performance, dict) or set(performance) - PERFORMANCE_FIELDS:
            raise ValueError(f'Invalid performance fields for {ident}')
        own = aircraft_config(config, spec)
        if 'spawn' in spec:
            replace(own, **{f'{team}_spawn': spec['spawn']})
        for key in ('heading', 'pitch'):
            value = spec.get(key, 0)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite angle')
        if abs(spec.get('pitch', 0)) > own.max_pitch:
            raise ValueError('Initial pitch must be within max_pitch')
        ids.add(ident)
        counts[team] += 1
    if not all(counts.values()):
        raise ValueError('A scenario requires at least one aircraft on each team')
    config.aircraft = deepcopy(roster)
    config.blue, config.red = counts['blue'], counts['red']
