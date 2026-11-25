"""Versioned neural feature layout; no dependency on environment or torch."""
OBSERVATION_VERSION = 8
PERFORMANCE_KEYS = (
    'aircraft_speed', 'turn_rate', 'pitch_rate', 'max_pitch', 'radar_range',
    'radar_fov', 'ammunition', 'shoot_cooldown', 'missile_lifetime',
    'aero_max_load', 'aero_max_bank', 'pitch_error_scale',
    'aero_mass', 'aero_wing_area', 'aero_thrust', 'aero_cd0',
    'aero_induced_drag', 'aero_cl_max',
)
# Avoid saturating squash with dimensional SI parameters (e.g. 10,000 kg).
PERFORMANCE_SCALES = dict(aero_mass=10000., aero_wing_area=30., aero_thrust=30000.,
                          aero_cd0=.025, aero_induced_drag=.06, aero_cl_max=1.5)
KINEMATIC_FIELDS = ('position_x','position_y','position_z','velocity_x','velocity_y','velocity_z')
PERFORMANCE_FIELDS = tuple('performance_'+key for key in PERFORMANCE_KEYS)
FEATURE_NAMES = {
    'common': ('time',),
    'self': KINEMATIC_FIELDS + ('ammo','shoot_cooldown','previous_turn','previous_vertical','previous_fire')
            + PERFORMANCE_FIELDS + ('boundary_home','boundary_enemy','boundary_y_min','boundary_y_max',
                                    'boundary_ground','boundary_ceiling')
            + ('alive', 'boundary_override_fraction')
            + ('mws_active', 'mws_forward', 'mws_right', 'mws_up')
            + ('own_missiles_in_flight', 'missile_capacity_available'),
    'allies': KINEMATIC_FIELDS + ('ammo','shoot_cooldown') + PERFORMANCE_FIELDS,
    'contacts': KINEMATIC_FIELDS + ('detected_by_self','detector_count','selected_last_step'),
    'missiles': KINEMATIC_FIELDS,
    'friendly_missiles': KINEMATIC_FIELDS + ('elapsed','is_own','midcourse','active','tracking'),
}
ENTITY_DIMS = {key:len(fields) for key,fields in FEATURE_NAMES.items()}
SELF_INDEX = {field:i for i,field in enumerate(FEATURE_NAMES['self'])}
COMMON_INDEX = {field:i for i,field in enumerate(FEATURE_NAMES['common'])}
CONTACT_INDEX = {field:i for i,field in enumerate(FEATURE_NAMES['contacts'])}
