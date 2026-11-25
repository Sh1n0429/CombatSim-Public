"""Environment-side boundary override; uses only own state and configured limits."""
from dataclasses import replace
import math

from .config_defaults import APPLICATION_DEFAULTS
BOUNDARY_DEFAULTS = APPLICATION_DEFAULTS["boundary"]

from .motion import AerodynamicAircraftMotion


def validate_boundary_control(settings):
    if not isinstance(settings, dict):
        raise ValueError('boundary_override must be an object')
    if set(settings)-{'enabled', 'horizontal_margin', 'vertical_margin', 'release_ratio'}:
        raise ValueError('Unknown boundary_override setting')
    if type(settings.get('enabled', BOUNDARY_DEFAULTS['enabled'])) is not bool:
        raise ValueError('boundary_override.enabled must be boolean')
    for name in ('horizontal_margin', 'vertical_margin', 'release_ratio'):
        default = BOUNDARY_DEFAULTS[name]
        value = settings.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= (1 if name == 'release_ratio' else 0):
            raise ValueError(f'Invalid boundary_override.{name}')


def boundary_control(pose, control, config, previous=None):
    settings = config.boundary_override
    if not settings.get('enabled', BOUNDARY_DEFAULTS['enabled']):
        return control, {}
    c = config
    previous = previous or {}
    release = settings.get('release_ratio', BOUNDARY_DEFAULTS['release_ratio'])
    speed = c.aircraft_speed if pose.speed is None else pose.speed
    horizontal_speed = speed*abs(math.cos(pose.pitch))
    omega = c.turn_rate
    velocity = speed*c.aero_length_scale/c.aero_tick_seconds
    lift = min(.5*AerodynamicAircraftMotion.density(pose.z*c.aero_length_scale)
               * velocity**2*c.aero_wing_area*c.aero_cl_max/c.aero_mass,
               c.aero_max_load*9.80665)
    lateral = min(c.aero_gravity*math.tan(c.aero_max_bank),
                  math.sqrt(max(0., lift**2-c.aero_gravity**2)))
    omega = min(omega, lateral/max(velocity, 1.)*c.aero_tick_seconds)
    # Allow for pitch/turn sharing, acceleration during the maneuver and a
    # reversal toward the center, rather than only a steady 90-degree turn.
    radius = horizontal_speed/max(omega, .01)
    margin = max(settings.get('horizontal_margin', BOUNDARY_DEFAULTS['horizontal_margin']), 2.5*radius+5*horizontal_speed)
    margin = min(margin, .45*min(c.arena_x, c.arena_y)/release)
    heading = (pose.heading+control.turn*math.pi/4 if control.target_heading is None
               else control.target_heading)
    distances = (pose.x, c.arena_x-pose.x, pose.y, c.arena_y-pose.y)
    current = (math.cos(pose.heading), -math.cos(pose.heading),
               math.sin(pose.heading), -math.sin(pose.heading))
    requested = (math.cos(heading), -math.cos(heading), math.sin(heading), -math.sin(heading))
    horizontal = (min(distances) < 0
                  or any(d < margin and (v < -1e-6 or r < -1e-6)
                         for d, v, r in zip(distances, current, requested))
                  or (previous.get('horizontal', False) and min(distances) < margin*release))
    if horizontal:
        control = replace(control, target_heading=math.atan2(c.arena_y/2-pose.y, c.arena_x/2-pose.x),
                          vertical=0, rear_turn_direction=0)
    pitch_response = max(abs(pose.pitch), c.pitch_error_scale)/max(c.pitch_rate, .001)
    # Simultaneous turns may leave only a quarter of unloading authority.
    pitch_response *= 4 if horizontal else 1
    vertical_margin = max(settings.get('vertical_margin', BOUNDARY_DEFAULTS['vertical_margin']),
                          speed*abs(math.sin(pose.pitch))*(pitch_response+2))
    vertical_margin = min(vertical_margin, .45*c.ceiling/release)
    lower = (pose.z <= 0 or (pose.z < vertical_margin and (pose.pitch < 0 or control.vertical < 0))
             or (previous.get('lower', False) and pose.z < vertical_margin*release))
    upper = (pose.z >= c.ceiling or (c.ceiling-pose.z < vertical_margin and (pose.pitch > 0 or control.vertical > 0))
             or (previous.get('upper', False) and c.ceiling-pose.z < vertical_margin*release))
    if lower:
        control = replace(control, vertical=1.)
    elif upper:
        control = replace(control, vertical=-10/45)
    if lower or upper:
        unsafe_vertical_motion = (lower and pose.pitch < 0) or (upper and pose.pitch > 0)
        if not horizontal or unsafe_vertical_motion:
            # Arrest motion toward the altitude limit before spending lift on
            # a turn; resume horizontal recovery once the vertical motion is safe.
            control = replace(control, target_heading=pose.heading, rear_turn_direction=0)
        else:
            control = replace(control, vertical=0.)
    return control, {'horizontal': horizontal, 'lower': lower, 'upper': upper,
                     'horizontal_margin': margin, 'vertical_margin': vertical_margin}
