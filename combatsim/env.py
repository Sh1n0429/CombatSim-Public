from dataclasses import InitVar, asdict, dataclass, field, fields, replace
import math
import random

from .config_defaults import environment_default
from .atmosphere import density
from .geometry import distance, wrap
from .entities import TOKEN_COUNT, selected_entities, shot_choices
from .rewards import RewardContext, RewardSystem
from .scenario import aircraft_config, validate_aircraft
from .observation_schema import PERFORMANCE_KEYS
from .boundary_control import boundary_control, validate_boundary_control
from .motion import (AircraftMotion, FlightControl, MissileMotion,
                     Pose, Waypoint, AerodynamicAircraftMotion,
                     AerodynamicMissileMotion)


TURN_TABLE = tuple(index * 360 / 11 for index in range(-5, 6))
PITCH_TABLE = (-45, -30, -15, -10, -5, 0, 5, 10, 15)
TURN_ACTION_COUNT = len(TURN_TABLE)
VERTICAL_ACTION_COUNT = len(PITCH_TABLE)
FIRE_ACTION_COUNT = TOKEN_COUNT
ACTION_SHAPE = (TURN_ACTION_COUNT, VERTICAL_ACTION_COUNT, FIRE_ACTION_COUNT)
PERFORMANCE_INTEGER_FIELDS = {
    'ammunition', 'shoot_cooldown', 'missile_lifetime',
    'missile_reacquisition_confirm_ticks',
}


def validate_performance_randomization(settings):
    if not isinstance(settings, dict):
        raise ValueError('performance_randomization must be an object')
    if type(settings.get('enabled', True)) is not bool:
        raise ValueError('performance_randomization.enabled must be boolean')
    if set(settings) - {'enabled', 'mode', 'randomize'} or settings.get('mode', 'mirrored_by_slot') != 'mirrored_by_slot':
        raise ValueError('performance_randomization supports enabled, mode mirrored_by_slot and randomize')
    ranges = settings.get('randomize', {})
    if not isinstance(ranges, dict) or set(ranges) - set(PERFORMANCE_KEYS):
        raise ValueError('performance_randomization contains an unsupported performance field')
    for name, bounds in ranges.items():
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or bounds[0] > bounds[1]):
            raise ValueError(f'performance_randomization.randomize.{name} must be [min, max]')
        if name in PERFORMANCE_INTEGER_FIELDS and any(type(v) is not int for v in bounds):
            raise ValueError(f'performance_randomization.randomize.{name} requires integer bounds')


def sample_performance(settings, seed, slots):
    validate_performance_randomization(settings)
    if not settings.get('enabled', True):
        return []
    rng = random.Random(f'combatsim-performance:{seed}')
    result = []
    for _ in range(slots):
        sampled = {}
        for name, bounds in settings.get('randomize', {}).items():
            sampled[name] = (rng.randint(*bounds) if name in PERFORMANCE_INTEGER_FIELDS
                             else rng.uniform(*bounds))
        result.append(sampled)
    return result


def encode_action(turn=0.0, vertical=0.0, fire=False):
    """Encode team heading / 180 degrees, normalized pitch and shot choice."""
    if not -1 <= turn <= 1 or not -1 <= vertical <= 1:
        raise ValueError("turn and vertical must be within -1 and 1")
    if type(fire) not in (bool,int) or not 0 <= fire < FIRE_ACTION_COUNT:
        raise ValueError("fire must be a shot choice 0..31 (False/True mean 0/1)")
    turn_degrees = math.degrees(wrap(turn*math.pi))
    pitch_degrees = vertical*(45 if vertical < 0 else 15)
    return (min(range(TURN_ACTION_COUNT), key=lambda i: abs(wrap(math.radians(TURN_TABLE[i]-turn_degrees)))),
            min(range(VERTICAL_ACTION_COUNT), key=lambda i: abs(PITCH_TABLE[i]-pitch_degrees)), int(fire))


def decode_action(action):
    if type(action) is int:
        if not 0 <= action < 18:
            raise ValueError("Legacy actions must be integers 0..17")
        return action % 3 - 1, (0, -1, 1)[action // 6], action % 6 >= 3
    if not isinstance(action, (tuple, list)) or len(action) != 3:
        raise ValueError("Actions must be (turn, vertical, fire) tuples")
    turn, vertical, fire = action
    if type(turn) is not int or not 0 <= turn < TURN_ACTION_COUNT:
        raise ValueError("Turn action must be an integer 0..10")
    if type(vertical) is not int or not 0 <= vertical < VERTICAL_ACTION_COUNT:
        raise ValueError("Vertical action must be an integer 0..8")
    if type(fire) is not int or not 0 <= fire < FIRE_ACTION_COUNT:
        raise ValueError("Shot action must be an integer 0..31")
    pitch = PITCH_TABLE[vertical]
    return TURN_TABLE[turn]/180, pitch/(45 if pitch < 0 else 15), fire


@dataclass
class Config:
    blue: int = environment_default('blue')
    red: int = environment_default('red')
    arena_x: float = environment_default('arena_x')
    arena_y: float = environment_default('arena_y')
    ceiling: float = environment_default('ceiling')
    # Normalized (min, max) for X, Y, Z; Z is scaled by ceiling.
    blue_spawn: tuple = environment_default('blue_spawn')
    red_spawn: tuple = environment_default('red_spawn')
    spawn_min_horizontal_distance: float = environment_default('spawn_min_horizontal_distance')
    paired_spawn: bool = environment_default('paired_spawn')
    max_steps: int = environment_default('max_steps')
    decision_interval_steps: int = environment_default('decision_interval_steps')
    aircraft_speed: float = environment_default('aircraft_speed')
    aircraft_motion: InitVar[str | None] = None  # Accept old aerodynamic snapshots only.
    # Physical model: metres per world unit, seconds per tick, then SI values.
    aero_length_scale: float = environment_default('aero_length_scale')
    aero_tick_seconds: float = environment_default('aero_tick_seconds')
    aero_substeps: int = environment_default('aero_substeps')
    aero_mass: float = environment_default('aero_mass')
    aero_wing_area: float = environment_default('aero_wing_area')
    aero_thrust: float = environment_default('aero_thrust')
    # Shared engine model; zero preserves saved scenarios predating thrust lapse.
    aero_thrust_density_exponent: float = environment_default('aero_thrust_density_exponent')
    aero_cd0: float = environment_default('aero_cd0')
    aero_induced_drag: float = environment_default('aero_induced_drag')
    aero_cl_max: float = environment_default('aero_cl_max')
    aero_max_load: float = environment_default('aero_max_load')
    aero_max_bank: float = environment_default('aero_max_bank')
    aero_gravity: float = environment_default('aero_gravity')
    turn_rate: float = environment_default('turn_rate')
    pitch_rate: float = environment_default('pitch_rate')
    max_pitch: float = environment_default('max_pitch')
    pitch_error_scale: float = environment_default('pitch_error_scale')
    ammunition: int = environment_default('ammunition')
    shoot_cooldown: int = environment_default('shoot_cooldown')
    max_in_flight_missiles: int = environment_default('max_in_flight_missiles')
    radar_range: float = environment_default('radar_range')
    radar_fov: float = environment_default('radar_fov')
    share_tracks: bool = environment_default('share_tracks')
    missile_motion: InitVar[str | None] = None  # Accept old aerodynamic snapshots only.
    # SI point-mass missile parameters. Positions still use game units and
    # are converted with aero_length_scale/aero_tick_seconds by the motion.
    aero_missile_mass: float = environment_default('aero_missile_mass')
    aero_missile_area: float = environment_default('aero_missile_area')
    aero_missile_thrust: float = environment_default('aero_missile_thrust')
    aero_missile_burn_seconds: float = environment_default('aero_missile_burn_seconds')
    aero_missile_boost_thrust: float = environment_default('aero_missile_boost_thrust')
    aero_missile_boost_seconds: float = environment_default('aero_missile_boost_seconds')
    aero_missile_sustain_thrust: float = environment_default('aero_missile_sustain_thrust')
    aero_missile_sustain_seconds: float = environment_default('aero_missile_sustain_seconds')
    aero_missile_drag: float = environment_default('aero_missile_drag')
    aero_missile_gravity: float = environment_default('aero_missile_gravity')
    aero_missile_turn_rate: float = environment_default('aero_missile_turn_rate')
    aero_missile_max_lateral_acceleration: float = environment_default('aero_missile_max_lateral_acceleration')
    aero_missile_control_speed: float = environment_default('aero_missile_control_speed')
    aero_missile_acquisition_time: float = environment_default('aero_missile_acquisition_time')
    # Time to restore full steering authority after confirmed reacquisition.
    missile_reacquisition_ramp_seconds: float = environment_default('missile_reacquisition_ramp_seconds')
    missile_reacquisition_confirm_ticks: int = environment_default('missile_reacquisition_confirm_ticks')
    aero_missile_turn_drag: float = environment_default('aero_missile_turn_drag')
    missile_navigation_constant: float = environment_default('missile_navigation_constant')
    missile_lifetime: int = environment_default('missile_lifetime')
    seeker_delay: int = environment_default('seeker_delay')
    seeker_range: float = environment_default('seeker_range')
    seeker_fov: float = environment_default('seeker_fov')
    hit_radius: float = environment_default('hit_radius')

    aircraft: list | None = environment_default('aircraft')
    rewards: list = field(default_factory=lambda: environment_default('rewards'))
    rule_policy: dict = field(default_factory=lambda: environment_default('rule_policy'))
    rule_policy_profiles: dict = field(default_factory=lambda: environment_default('rule_policy_profiles'))
    performance_randomization: dict = field(default_factory=lambda: environment_default('performance_randomization'))
    boundary_override: dict = field(default_factory=lambda: environment_default('boundary_override'))

    def __post_init__(self, aircraft_motion, missile_motion):
        if type(self.paired_spawn) is not bool:
            raise ValueError('paired_spawn must be boolean')
        validate_boundary_control(self.boundary_override)
        for name, value in (('aircraft_motion', aircraft_motion), ('missile_motion', missile_motion)):
            if value not in (None, 'aerodynamic'):
                raise ValueError(f'{name}: constant motion has been removed; only aerodynamic is supported')
        if type(self.aero_substeps) is not int or not 1 <= self.aero_substeps <= 1000:
            raise ValueError('aero_substeps must be an integer in [1, 1000]')
        if not 0 < self.aero_max_bank < math.pi/2:
            raise ValueError('aero_max_bank must be in (0, pi/2)')
        for name in ("blue", "red", "max_steps", "decision_interval_steps", "ammunition", "shoot_cooldown", "missile_lifetime", "seeker_delay", "missile_reacquisition_confirm_ticks", "max_in_flight_missiles"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("ammunition", "seeker_delay", "max_in_flight_missiles") else 1):
                raise ValueError(f"Invalid {name}: {value}")
        for field in fields(self):
            if field.type is float:
                value = getattr(self, field.name)
                allow_zero = field.name in ('spawn_min_horizontal_distance', 'aero_thrust', 'aero_thrust_density_exponent', 'aero_cd0', 'aero_induced_drag',
                                            'aero_cl_max', 'aero_gravity',
                                            'aero_missile_area',
                                            'aero_missile_thrust', 'aero_missile_burn_seconds',
                                            'aero_missile_boost_thrust', 'aero_missile_boost_seconds',
                                            'aero_missile_sustain_thrust', 'aero_missile_sustain_seconds',
                                            'aero_missile_drag', 'aero_missile_gravity',
                                            'aero_missile_turn_rate', 'aero_missile_turn_drag')
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                    raise ValueError(f"Invalid {field.name}: {value}")
        if not 0 < self.radar_fov <= 2 * math.pi or not 0 < self.seeker_fov <= 2 * math.pi:
            raise ValueError("FOV must be in (0, 2*pi]")
        if type(self.share_tracks) is not bool:
            raise ValueError("share_tracks must be boolean")
        if not isinstance(self.rule_policy, dict):
            raise ValueError("rule_policy must be an object")
        if not isinstance(self.rule_policy_profiles, dict) or any(
                not isinstance(name, str) or not name or not isinstance(settings, dict)
                for name, settings in self.rule_policy_profiles.items()):
            raise ValueError("rule_policy_profiles must map names to objects")
        validate_performance_randomization(self.performance_randomization)
        if self.max_pitch >= math.pi/2:
            raise ValueError("max_pitch must be less than pi/2")
        for name in ("blue_spawn", "red_spawn"):
            box = getattr(self, name)
            if not isinstance(box, (tuple, list)) or len(box) != 3:
                raise ValueError(f"{name} must contain X, Y, Z ranges")
            for bounds in box:
                if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                        or not 0 <= bounds[0] <= bounds[1] <= 1):
                    raise ValueError(f"{name} ranges must satisfy 0 <= min <= max <= 1")
            setattr(self, name, tuple(tuple(bounds) for bounds in box))

        RewardSystem(self.rewards)
        if self.aircraft is not None:
            validate_aircraft(self)


@dataclass
class Aircraft:
    id: str
    team: str
    x: float
    y: float
    heading: float
    ammo: int
    alive: bool = True
    shoot_cooldown: int = 0
    z: float = 20.0
    pitch: float = 0.0
    speed: float = 1.0


@dataclass
class Missile:
    id: str
    owner: str
    team: str
    target: str
    x: float
    y: float
    heading: float
    aim_x: float
    aim_y: float
    age: int = 0
    phase: str = "midcourse"
    alive: bool = True
    z: float = 20.0
    pitch: float = 0.0
    aim_z: float = 20.0
    speed: float | None = None
    elapsed: float = 0.0


class ToyEnv:
    """Parallel multi-agent API with independent turn/pitch/shot action components.

    reset(seed) -> (observations, info)
    step(actions) -> (observations, rewards, terminated, truncated, info)
    Terminal flags are global booleans. This is not a Gymnasium/PettingZoo adapter.
    """

    def __init__(self, config=None, *, aircraft_motion: AircraftMotion | None = None,
                 missile_motion: MissileMotion | None = None):
        self.config = config or Config()
        self.aircraft_motion = aircraft_motion if aircraft_motion is not None else AerodynamicAircraftMotion()
        self.missile_motion = missile_motion if missile_motion is not None else AerodynamicMissileMotion()
        self._custom_missile_motion = missile_motion is not None
        self.reward_system = RewardSystem(self.config.rewards)
        self.reset()

    def reset(self, seed=None, *, initial_heading=None, randomize_performance=False):
        self.reward_system.reset()
        rng = random.Random(seed)
        pair_seed = rng.getrandbits(64) if seed is None else seed
        team_rngs = {side: random.Random(pair_seed) for side in ('blue', 'red')}
        c = self.config
        self.aircraft = {}
        self.aircraft_settings = {}
        roster = c.aircraft if c.aircraft is not None else [
            {"id": f"{team}_{i}", "team": team}
            for team, count in (("blue", c.blue), ("red", c.red)) for i in range(count)]
        slot_by_id = {}
        team_slots = {'blue': 0, 'red': 0}
        for spec in roster:
            slot_by_id[spec['id']] = team_slots[spec['team']]
            team_slots[spec['team']] += 1
        randomized = (sample_performance(c.performance_randomization, seed, max(team_slots.values()))
                      if randomize_performance and c.performance_randomization else [])
        for spec in roster:
            ident, team = spec['id'], spec['team']
            if c.paired_spawn:
                rng = team_rngs[team]
            own = aircraft_config(c, spec) if c.aircraft is not None else c
            if randomized:
                own = replace(own, aircraft=None, **randomized[slot_by_id[ident]])
            self.aircraft_settings[ident] = own
            box = spec.get('spawn', getattr(c, f'{team}_spawn'))
            def sample_position(axis):
                u = rng.random()
                if c.paired_spawn and team == 'red' and axis < 2:
                    u = 1-u
                lo, hi = box[axis]
                return (lo+(hi-lo)*u)*(c.arena_x, c.arena_y, c.ceiling)[axis]
            x, y, z = (sample_position(axis) for axis in range(3))
            allies = [a for a in self.aircraft.values() if a.team == team]
            if c.spawn_min_horizontal_distance > 0 and allies:
                # Bounded rejection keeps custom cramped/many-aircraft scenarios
                # usable. If the target cannot be reached, keep the best sample.
                def clearance(position):
                    return min(math.hypot(position[0]-a.x, position[1]-a.y) for a in allies)
                best = (x, y)
                best_gap = clearance(best)
                for _ in range(128):
                    if best_gap >= c.spawn_min_horizontal_distance:
                        break
                    candidate = (sample_position(0), sample_position(1))
                    gap = clearance(candidate)
                    if gap > best_gap:
                        best, best_gap = candidate, gap
                x, y = best
            self.aircraft[ident] = Aircraft(ident, team, x, y, spec.get('heading', 0 if team == 'blue' else math.pi),
                                           own.ammunition, z=z, pitch=spec.get('pitch', 0), speed=own.aircraft_speed)
        self.initial_heading_mode = initial_heading.apply(self.aircraft, seed) if initial_heading is not None else 'front'
        self.team_slots = slot_by_id
        self.episode_returns = {ident: 0.0 for ident in self.aircraft}
        self.previous_targets = {ident:None for ident in self.aircraft}
        self.previous_actions = {ident: (0.0, 0.0, False) for ident in self.aircraft}
        self.previous_heading_targets = {ident: None for ident in self.aircraft}
        self.applied_controls = {}
        self._boundary_override_state = {}
        self.control_diagnostics = {}
        self._sensor_tracks = {}
        self._missile_bitten = set()
        self._missile_tracking = {}
        self._missile_supporters = {}
        self._missile_guidance_strength = {}
        self._missile_reacquisition_ticks = {}
        self._missile_track_loss_penalized = set()
        self._next_sensor_track = {}
        self.missiles = []
        self.tick = 0
        self.done = False
        self.winner = None
        self.events = []
        self.reward_system.begin_episode(RewardContext(
            self.config, self.aircraft, {}, False, None, tick=0,
            max_steps=self.config.max_steps, missiles=()))
        return self.observations(), self.info()

    def _visible(self, observer, target, radius, fov):
        dx, dy, dz = target.x-observer.x, target.y-observer.y, target.z-observer.z
        length = distance(observer, target)
        if not target.alive or length > radius:
            return False
        if length == 0:
            return True
        dot = (dx*math.cos(observer.heading)*math.cos(observer.pitch)
               + dy*math.sin(observer.heading)*math.cos(observer.pitch) + dz*math.sin(observer.pitch))/length
        return max(-1, min(1, dot)) >= math.cos(fov/2)

    def radar_visible(self, observer, target):
        own = self.aircraft_settings[observer.id]
        return self._visible(observer, target, own.radar_range, own.radar_fov)

    def aircraft_mover(self, config):
        return self.aircraft_motion

    def missile_mover(self, config):
        return self.missile_motion

    def tracks(self, aircraft, *, radar_visible=None):
        c = self.config
        radar_visible = self.radar_visible if radar_visible is None else radar_visible
        observers = [a for a in self.aircraft.values() if a.alive and a.team == aircraft.team and (c.share_tracks or a.id == aircraft.id)]
        return [target for target in self.aircraft.values() if target.team != aircraft.team and any(radar_visible(a, target) for a in observers)]

    @staticmethod
    def _team_vector(a, x, y, z):
        # Fixed team axes: towards the opponent, lateral, world up.
        sign = 1 if a.team == 'blue' else -1
        return (sign*x, sign*y, z)

    def _team_position(self, observer, body):
        c = self.config
        origin_x = 0 if observer.team == 'blue' else c.arena_x
        x, y, z = self._team_vector(observer, body.x-origin_x, body.y-c.arena_y/2, body.z)
        horizontal_scale = max(c.arena_x, c.arena_y)
        return (x/horizontal_scale, y/horizontal_scale, z/c.ceiling)

    def _velocity(self, body):
        speed = body.speed
        if speed is None:
            speed = self.aircraft_settings[body.owner].aircraft_speed
        return (speed*math.cos(body.pitch)*math.cos(body.heading),
                speed*math.cos(body.pitch)*math.sin(body.heading), speed*math.sin(body.pitch))

    def _kinematics(self, a, body, *, velocity_lookup=None):
        c = self.config
        scale = max(c.arena_x, c.arena_y)
        delta = (body.x-a.x, body.y-a.y, body.z-a.z)
        local = self._team_vector(a, *delta)
        velocity_lookup = self._velocity if velocity_lookup is None else velocity_lookup
        velocity, own_velocity = velocity_lookup(body), velocity_lookup(a)
        relative = tuple(v-u for v, u in zip(velocity, own_velocity))
        length = math.sqrt(sum(d*d for d in delta))
        return dict(zip(('dx', 'dy', 'dz'), (local[0]/scale, local[1]/scale, local[2]/c.ceiling)),
                    team_position=self._team_position(a, body),
                    team_heading=wrap(body.heading-(0 if a.team == 'blue' else math.pi)),
                    flight_pitch=body.pitch,
                    velocity=self._team_vector(a, *velocity),
                    relative_velocity=self._team_vector(a, *relative),
                    closing_speed=-sum(d*v for d, v in zip(delta, relative))/length if length else 0.0)

    def _track_ids(self, observer, kind, bodies):
        """Local track labels, valid only for uninterrupted observed contacts.

        Internal IDs remain private association keys in this ideal sensor model;
        this does not simulate noisy measurement association or reidentification.
        """
        key = (observer.id, kind)
        previous = self._sensor_tracks.get(key, {})
        current = {}
        for body in bodies:
            if body.id in previous:
                current[body.id] = previous[body.id]
            else:
                number = self._next_sensor_track.get(key, 0)
                current[body.id] = f'{kind}_track_{number}'
                self._next_sensor_track[key] = number+1
        self._sensor_tracks[key] = current
        return current

    def own_missiles_in_flight(self, ident):
        return sum(m.alive and m.owner == ident for m in self.missiles)

    def missile_capacity_available(self, ident):
        limit = self.aircraft_settings[ident].max_in_flight_missiles
        return limit == 0 or self.own_missiles_in_flight(ident) < limit

    def observations(self):
        c = self.config
        result = {}
        # These caches live for this call only. Public bodies/configs remain
        # mutable, so neither radar results nor velocities may survive a call.
        visibility, velocities = {}, {}

        def radar_visible(observer, target):
            key = (id(observer), id(target))
            if key not in visibility:
                visibility[key] = self.radar_visible(observer, target)
            return visibility[key]

        def velocity(body):
            key = id(body)
            if key not in velocities:
                velocities[key] = self._velocity(body)
            return velocities[key]

        for a in self.aircraft.values():
            if not a.alive:
                continue
            contacts = sorted(self.tracks(a, radar_visible=radar_visible), key=lambda t: (distance(a, t), t.id))
            allies = sorted((ally for ally in self.aircraft.values()
                             if ally.alive and ally.team == a.team and ally.id != a.id),
                            key=lambda ally: (distance(a, ally), ally.id))
            observers = [o for o in self.aircraft.values()
                         if o.alive and o.team == a.team and (c.share_tracks or o.id == a.id)]
            visible_missiles = sorted(
                (m for m in self.missiles if m.alive and m.team != a.team
                 and any(radar_visible(o, m) for o in observers)),
                key=lambda m: (distance(a, m), m.id))
            contact_ids = self._track_ids(a, 'aircraft', contacts)
            missile_ids = self._track_ids(a, 'missile', visible_missiles)
            result[a.id] = {
                "self": {"id": a.id, "team": a.team, "x": a.x/c.arena_x, "y": a.y/c.arena_y, "z": a.z/c.ceiling, "heading": a.heading, "pitch": a.pitch, "ammo": a.ammo, "shoot_cooldown": a.shoot_cooldown},
                "contacts": [{"id": contact_ids[t.id], "distance": distance(a, t)/max(c.arena_x, c.arena_y),
                              "bearing": wrap(math.atan2(t.y-a.y, t.x-a.x)-a.heading),
                              "elevation": math.atan2(t.z-a.z, math.hypot(t.x-a.x, t.y-a.y)),
                              "altitude_delta": (t.z-a.z)/c.ceiling,
                              "dx": (t.x-a.x)/c.arena_x, "dy": (t.y-a.y)/c.arena_y,
                              "dz": (t.z-a.z)/c.ceiling} for t in contacts],
                "allies": [{"id": ally.id, "dx": (ally.x-a.x)/c.arena_x,
                            "dy": (ally.y-a.y)/c.arena_y,
                            "dz": (ally.z-a.z)/c.ceiling,
                            "speed": ally.speed, "ammo": ally.ammo,
                            "shoot_cooldown": ally.shoot_cooldown,
                            "performance": {key:getattr(self.aircraft_settings[ally.id],key) for key in PERFORMANCE_KEYS},
                            "heading": wrap(ally.heading-a.heading), "pitch": ally.pitch} for ally in allies],
                "time": self.tick/c.max_steps,
                "airspace": {"x": c.arena_x, "y": c.arena_y},
            }
            # Speeds are world units per tick, matching Pose.speed. Only sensed
            # kinematics are exposed for enemy missiles. Friendly telemetry below
            # never reveals an unobserved enemy position.
            result[a.id]["self"]["speed"] = a.speed
            for contact, target in zip(result[a.id]["contacts"], contacts):
                detected_by = sorted(o.id for o in observers if radar_visible(o, target))
                contact.update(speed=target.speed, heading=wrap(target.heading-a.heading),
                               pitch=target.pitch, detected_by=detected_by,
                               detected_by_self=a.id in detected_by,
                               detector_count=len(detected_by), selected_last_step=self.previous_targets.get(a.id)==target.id)
            result[a.id]["missiles"] = [
                {"id": missile_ids[m.id], "distance": distance(a, m)/max(c.arena_x, c.arena_y),
                 "dx": (m.x-a.x)/c.arena_x, "dy": (m.y-a.y)/c.arena_y,
                 "dz": (m.z-a.z)/c.ceiling,
                 "bearing": wrap(math.atan2(m.y-a.y, m.x-a.x)-a.heading),
                 "elevation": math.atan2(m.z-a.z, math.hypot(m.x-a.x, m.y-a.y)),
                 "speed": self.aircraft_settings[m.owner].aircraft_speed if m.speed is None else m.speed}
                for m in visible_missiles]
            # Idealized game warning independent of radar visibility. Only the
            # nearest live, active enemy projectile assigned to this aircraft
            # supplies a direction; range is never emitted as an MWS feature.
            warning = min((m for m in self.missiles
                           if m.alive and m.team != a.team
                           and m.phase == 'active' and m.target == a.id),
                          key=lambda m: (distance(a, m), m.id), default=None)
            if warning is None:
                result[a.id]['self'].update(mws_active=False,
                                            mws_direction=(0.0, 0.0, 0.0))
            else:
                dx, dy, dz = warning.x-a.x, warning.y-a.y, warning.z-a.z
                length = math.sqrt(dx*dx+dy*dy+dz*dz)
                heading = a.heading
                direction = ((dx*math.cos(heading)+dy*math.sin(heading))/length,
                             (-dx*math.sin(heading)+dy*math.cos(heading))/length,
                             dz/length) if length else (0.0, 0.0, 0.0)
                result[a.id]['self'].update(mws_active=True, mws_direction=direction)
            own = self.aircraft_settings[a.id]
            observation = result[a.id]
            observation['self'].update(alive=True, slot=self.team_slots[a.id],
                own_missiles_in_flight=self.own_missiles_in_flight(a.id),
                missile_capacity_available=self.missile_capacity_available(a.id))
            observation['tick'] = self.tick
            observation['boundary_override_fraction'] = float(
                self.control_diagnostics.get(a.id, {}).get('boundary_override', False))
            observation['coordinate_frame'] = 'team'
            observation['self'].update(self._kinematics(a, a, velocity_lookup=velocity))
            observation['previous_action'] = dict(zip(('turn', 'vertical', 'fire'), self.previous_actions[a.id]))
            previous_heading = self.previous_heading_targets[a.id]
            observation['previous_action']['turn'] = (0.0 if previous_heading is None else
                                                      wrap(previous_heading-a.heading)/math.pi)
            observation['performance'] = {name: getattr(own, name) for name in PERFORMANCE_KEYS}
            # Signed perpendicular distances plus local inward normals identify each wall.
            observation['boundaries'] = [
                {'distance': d/scale, 'normal': self._team_vector(a, *normal)}
                for d, scale, normal in (
                    (a.x, c.arena_x, (1, 0, 0)), (c.arena_x-a.x, c.arena_x, (-1, 0, 0)),
                    (a.y, c.arena_y, (0, 1, 0)), (c.arena_y-a.y, c.arena_y, (0, -1, 0)),
                    (a.z, c.ceiling, (0, 0, 1)), (c.ceiling-a.z, c.ceiling, (0, 0, -1)))]
            if a.team == 'red':
                observation['boundaries'] = [observation['boundaries'][i] for i in (1, 0, 3, 2, 4, 5)]
            for key, bodies in (('contacts', contacts), ('allies', allies), ('missiles', visible_missiles)):
                for record, body in zip(observation[key], bodies):
                    record.update(self._kinematics(a, body, velocity_lookup=velocity))
            observation['friendly_missiles'] = [
                dict(id=m.id, owner=m.owner, is_own=m.owner == a.id, age=m.age,
                     tracking=bool(self._missile_tracking.get(m.id, False)),
                     target_track=contact_ids.get(m.target),
                     elapsed=m.age*self.aircraft_settings[m.owner].aero_tick_seconds,
                     phase=m.phase, distance=distance(a, m)/max(c.arena_x, c.arena_y),
                     speed=self._velocity_speed(m), heading=wrap(m.heading-a.heading), pitch=m.pitch,
                     **self._kinematics(a, m, velocity_lookup=velocity))
                for m in sorted(self.missiles, key=lambda m: (distance(a, m), m.id))
                if m.alive and m.team == a.team]
        return result

    def _velocity_speed(self, missile):
        return self.aircraft_settings[missile.owner].aircraft_speed if missile.speed is None else missile.speed

    def info(self):
        return {"tick": self.tick, "winner": self.winner, "events": list(self.events),
                "reward_components": {name: dict(values) for name, values in self.reward_system.last_components.items()},
                "reward_clipping": {reason: {ident: dict(values) for ident, values in agents.items()}
                                    for reason, agents in self.reward_system.last_clipping.items()},
                "episode_returns": dict(self.episode_returns), "controls": dict(self.control_diagnostics)}

    def step(self, actions, *, decision_observations=None):
        """Advance one tick, optionally reusing a caller-owned raw observation.

        The RLlib adapter supplies its private observation from the immediately
        preceding reset/step. Callers editing world state must omit this option
        or obtain fresh observations after those edits.
        """
        if self.done:
            raise RuntimeError("Episode ended; call reset()")
        active = {a.id for a in self.aircraft.values() if a.alive}
        if set(actions) - active:
            raise ValueError("Actions contain unknown or destroyed aircraft")
        previous_positions = {ident: (self.aircraft[ident].x, self.aircraft[ident].y, self.aircraft[ident].z) for ident in active}
        previous_poses = {ident: Pose.from_body(self.aircraft[ident]) for ident in active}
        decoded = {ident: decode_action(value) for ident, value in actions.items()}
        # Bind the selected observed track BEFORE movement can reorder contacts.
        selected_targets = {}
        if decision_observations is None:
            decision_observations = self.observations()
        for ident, action in actions.items():
            if type(action) is int:  # Legacy inputs retain automatic own-radar selection.
                continue
            shot = decoded[ident][2]
            observation = decision_observations[ident]
            if shot and shot in shot_choices(observation):
                track = selected_entities(observation)['contacts'][shot-1]['id']
                selected_targets[ident] = next((target_id for target_id, local_track in self._sensor_tracks.get((ident,'aircraft'),{}).items()
                                               if local_track == track), None)
        c = self.config
        self.events = []
        self.previous_targets.update({ident:selected_targets.get(ident) for ident in active})
        self.previous_actions.update({ident: (*decoded.get(ident,(0.0,0.0,False))[:2],bool(decoded.get(ident,(0,0,False))[2])) for ident in active})
        self.tick += 1
        self.control_diagnostics = {}
        # Both teams move before any launch or impact is resolved.
        for ident in sorted(active):
            a = self.aircraft[ident]
            c = self.aircraft_settings[ident]
            turn, vertical, _ = decoded.get(ident, (0, 0, False))
            # Tuple actions choose an absolute heading in the team's fixed axes.
            # Legacy scalar actions retain their original relative-turn meaning.
            legacy = type(actions.get(ident)) is int
            target_heading = (None if legacy else
                              wrap(turn*math.pi + (0 if a.team == 'blue' else math.pi)))
            if ident not in actions:
                target_heading = a.heading
            requested = FlightControl(turn, vertical, target_heading)
            control, override = boundary_control(Pose.from_body(a), requested, c,
                                                  self._boundary_override_state.get(ident))
            self._boundary_override_state[ident] = override
            overridden = any(override.get(key, False) for key in ('horizontal', 'lower', 'upper'))
            if overridden:
                self.events.append({'type': 'boundary_override', 'aircraft': ident, **override})
                self.previous_actions[ident] = (turn, control.vertical, self.previous_actions[ident][2])
            self.previous_heading_targets[ident] = (wrap(a.heading+control.turn*math.pi/4)
                                                   if control.target_heading is None else control.target_heading)
            control = control.stabilized(a.heading, self.applied_controls.get(ident))
            self.applied_controls[ident] = control
            pose = self.aircraft_mover(c).advance(Pose.from_body(a), control, c)
            self.control_diagnostics[ident] = {
                'boundary_override': overridden, 'boundary_override_details': override,
                'requested_heading': wrap(a.heading+turn*math.pi/4) if target_heading is None else target_heading,
                'requested_pitch': requested.target_pitch,
                'target_heading': self.previous_heading_targets[ident],
                'heading_before': a.heading, 'heading_after': pose.heading,
                'heading_error': wrap(self.previous_heading_targets[ident]-a.heading),
                'turn_command': control.heading_command(a.heading),
                'heading_change': wrap(pose.heading-a.heading),
                'rear_turn_direction': control.rear_turn_direction,
                'target_pitch': max(-c.max_pitch, min(c.max_pitch, control.target_pitch)),
                'pitch_before': a.pitch, 'pitch_after': pose.pitch,
                'speed': pose.speed if pose.speed is not None else c.aircraft_speed,
                'altitude': pose.z,
                'boundary_distance': min(pose.x,c.arena_x-pose.x,pose.y,c.arena_y-pose.y),
            }
            pose.apply_to(a)
            if a.z <= 0:
                a.alive = False
                self.events.append({'type': 'crash', 'aircraft': ident})
            a.shoot_cooldown = max(0, a.shoot_cooldown-1)
        radar_detection_fractions = {}
        radar_detected_targets = {}
        for ident in sorted(active):
            a = self.aircraft[ident]
            c = self.aircraft_settings[ident]
            if not a.alive:
                continue
            # Shared tracks remain observable, but launch requires own detection.
            contacts = [target for target in self.aircraft.values()
                        if target.team != a.team and self.radar_visible(a, target)]
            enemy_count = sum(target.alive and target.team != a.team for target in self.aircraft.values())
            radar_detection_fractions[ident] = len(contacts)/enemy_count if enemy_count else 0.0
            radar_detected_targets[ident] = tuple(target.id for target in contacts)
            if (decoded.get(ident, (0, 0, False))[2] and a.ammo and not a.shoot_cooldown
                    and contacts and self.missile_capacity_available(ident)):
                if type(actions.get(ident)) is int:
                    target = min(contacts, key=lambda t: (distance(a, t), t.id))
                else:
                    target = next((t for t in contacts if t.id == selected_targets.get(ident)), None)
                    if target is None:
                        continue
                self.previous_targets[ident] = target.id
                heading = math.atan2(target.y-a.y, target.x-a.x)
                m = Missile(f"m{len(self.missiles)}", a.id, a.team, target.id, a.x, a.y, heading, target.x, target.y)
                m.z, m.aim_z = a.z, target.z
                m.pitch = math.atan2(target.z-a.z, math.hypot(target.x-a.x, target.y-a.y))
                m.heading, m.pitch, m.speed = a.heading, a.pitch, a.speed
                self.missiles.append(m)
                self._missile_tracking[m.id] = True
                self._missile_guidance_strength[m.id] = 1.0
                self._missile_reacquisition_ticks[m.id] = c.missile_reacquisition_confirm_ticks
                a.ammo -= 1
                a.shoot_cooldown = c.shoot_cooldown
                self.events.append({"type": "launch", "aircraft":a.id, "missile": m.id, "target": target.id})
        hits = set()
        missed_missile_owners = []
        evasion_candidates = []
        bite_owners = []
        track_loss_owners = []
        motion_records = []
        for m in self.missiles:
            if not m.alive:
                continue
            c = self.aircraft_settings[m.owner]
            m.age += 1
            target = self.aircraft[m.target]
            if not target.alive or m.age > c.missile_lifetime:
                m.alive = False
                m.phase = "expired"
                self.events.append({"type": "expired", "missile": m.id,
                                    "reason": "target_lost" if not target.alive else "lifetime"})
                missed_missile_owners.append(m.owner)
                if target.alive:
                    evasion_candidates.append((m.target, m.id in self._missile_bitten, m.id))
                continue
            previous = m.phase
            seeker_tracking = self._visible(m, target, c.seeker_range, c.seeker_fov)
            supported = False
            self._missile_supporters[m.id] = ()
            if m.phase != "active":
                owner = self.aircraft[m.owner]
                # Any living teammate may provide midcourse radar updates when
                # track sharing is enabled. Without sharing, only the launcher
                # can support its missile.
                observers = (ally for ally in self.aircraft.values()
                             if ally.alive and ally.team == owner.team
                             and (c.share_tracks or ally.id == owner.id))
                supporters = tuple(sorted(ally.id for ally in observers if self.radar_visible(ally, target)))
                self._missile_supporters[m.id] = supporters
                supported = bool(supporters)
                if supported:
                    m.aim_x, m.aim_y, m.aim_z = target.x, target.y, target.z
                m.phase = "midcourse" if supported else "coast"
                if m.age >= c.seeker_delay and seeker_tracking:
                    m.phase = "active"
            if m.phase == 'active':
                self._missile_supporters[m.id] = ()
            sensor_tracking = seeker_tracking if m.phase == 'active' else supported
            was_tracking = self._missile_tracking.get(m.id, previous == 'midcourse')
            if sensor_tracking:
                confirmation = (c.missile_reacquisition_confirm_ticks if was_tracking else
                                self._missile_reacquisition_ticks.get(m.id, 0) + 1)
            else:
                confirmation = 0
            self._missile_reacquisition_ticks[m.id] = confirmation
            tracking = (sensor_tracking and
                        (was_tracking or confirmation >= c.missile_reacquisition_confirm_ticks))
            if m.phase == 'active' and tracking and m.id not in self._missile_bitten:
                self._missile_bitten.add(m.id)
                bite_owners.append(m.owner)
                self.events.append({'type': 'bite', 'missile': m.id, 'aircraft': m.owner, 'target': m.target})
            if (was_tracking and not tracking
                    and m.id not in self._missile_track_loss_penalized):
                self._missile_track_loss_penalized.add(m.id)
                track_loss_owners.append(m.owner)
                self.events.append({'type': 'missile_track_loss', 'missile': m.id,
                                    'aircraft': m.owner, 'target': m.target,
                                    'tracking_source': 'seeker' if m.phase == 'active' else 'radar',
                                    'reason': ('reacquisition_pending' if sensor_tracking else
                                               'seeker_contact_lost' if m.phase == 'active' else 'radar_support_lost')})
            if not tracking:
                guidance_strength = 0.0
            elif was_tracking:
                guidance_strength = min(1.0, self._missile_guidance_strength.get(m.id, 1.0)
                                        + c.aero_tick_seconds/c.missile_reacquisition_ramp_seconds)
            else:
                guidance_strength = min(1.0,
                                        c.aero_tick_seconds/c.missile_reacquisition_ramp_seconds)
            self._missile_tracking[m.id] = tracking
            self._missile_guidance_strength[m.id] = guidance_strength
            if m.phase == "active" and tracking:
                m.aim_x, m.aim_y, m.aim_z = target.x, target.y, target.z
            if previous != m.phase:
                self.events.append({"type": "phase", "missile": m.id, "phase": m.phase})
            old_x, old_y, old_z = m.x, m.y, m.z
            waypoint = Waypoint(m.aim_x, m.aim_y, m.aim_z,
                                guidance_enabled=tracking,
                                guidance_strength=guidance_strength)
            if tracking:
                # Supply a synchronized target segment only while a sensor can
                # update the track. Lost tracks coast without steering toward
                # a stale point, including active seekers outside their FOV.
                before = previous_positions[target.id]
                velocity = tuple((now-old)/c.aero_tick_seconds for now,old in zip(
                    (target.x, target.y, target.z), before))
                waypoint = Waypoint(*before, velocity=velocity,
                                    guidance_strength=guidance_strength)
            motion_records.append((m, c, target, (old_x, old_y, old_z, m.elapsed), waypoint))

        # Missile dynamics are independent within a tick. Group identical
        # configurations into array kernels, then resolve all swept impacts.
        poses = [None] * len(motion_records)
        groups = {}
        for index, (m, c, target, old, waypoint) in enumerate(motion_records):
            if not self._custom_missile_motion:
                groups.setdefault(id(c), (c, []) )[1].append(index)
            else:
                poses[index] = self.missile_mover(c).advance(Pose.from_body(m), waypoint, c)
        for c, indices in groups.values():
            mover = self.missile_mover(c)
            batch = mover.advance_many(
                [Pose.from_body(motion_records[i][0]) for i in indices],
                [motion_records[i][4] for i in indices], c)
            for index, pose in zip(indices, batch):
                poses[index] = pose

        for pose, (m, c, target, old, waypoint) in zip(poses, motion_records):
            old_x, old_y, old_z, old_elapsed = old
            pose.apply_to(m)
            target_before = previous_positions[target.id]
            # Ground contact may end a missile segment before the tick ends.
            # Sweep both bodies over that same interval: using the target's
            # full tick can create hits after impact and miss hits before it.
            fraction = 1.0
            if m.z <= 0:
                fraction = max(0.0, min(1.0, (m.elapsed-old_elapsed)/c.aero_tick_seconds))
            sx, sy, sz = (m.x-old_x-fraction*(target.x-target_before[0]),
                          m.y-old_y-fraction*(target.y-target_before[1]),
                          m.z-old_z-fraction*(target.z-target_before[2]))
            squared = sx*sx+sy*sy+sz*sz
            u = max(0, min(1, ((target_before[0]-old_x)*sx+(target_before[1]-old_y)*sy+(target_before[2]-old_z)*sz)/squared)) if squared else 0
            if math.sqrt((target_before[0]-(old_x+u*sx))**2 + (target_before[1]-(old_y+u*sy))**2 + (target_before[2]-(old_z+u*sz))**2) <= c.hit_radius:
                hits.add(target.id)
                m.alive = False
                m.phase = "hit"
                self.events.append({"type": "hit", "missile": m.id, "target": target.id})
            elif m.z <= 0:
                m.alive = False
                m.phase = 'expired'
                self.events.append({'type': 'expired', 'missile': m.id, 'reason': 'ground'})
                missed_missile_owners.append(m.owner)
                evasion_candidates.append((m.target, m.id in self._missile_bitten, m.id))
        for ident in hits:
            self.aircraft[ident].alive = False
        evaded_missiles = []
        for target_id, bitten, missile_id in evasion_candidates:
            if self.aircraft[target_id].alive:
                evaded_missiles.append((target_id, bitten))
                self.events.append({'type': 'missile_evaded', 'missile': missile_id,
                                    'aircraft': target_id, 'bitten': bitten})
        teams = {a.team for a in self.aircraft.values() if a.alive}
        # A time-limit draw is a terminal game result, not an interruption.
        timed_out = len(teams) >= 2 and self.tick >= self.config.max_steps
        terminated = len(teams) < 2 or timed_out
        truncated = False
        self.done = terminated or truncated
        self.winner = next(iter(teams)) if terminated and len(teams) == 1 else None
        rewards = self.reward_system.calculate(RewardContext(
            self.config, self.aircraft, previous_positions, self.done, self.winner,
            shot_down=tuple(sorted(hits)),
            missed_missile_owners=tuple(missed_missile_owners),
            evaded_missiles=tuple(evaded_missiles), bite_owners=tuple(bite_owners),
            track_loss_owners=tuple(track_loss_owners),
            radar_detection_fractions=radar_detection_fractions,
            previous_poses=previous_poses, aircraft_settings=self.aircraft_settings,
            crashed=tuple(e['aircraft'] for e in self.events if e['type']=='crash'),
            events=tuple(self.events), radar_detected_targets=radar_detected_targets,
            tick=self.tick, max_steps=self.config.max_steps, timed_out=timed_out,
            missiles=tuple(self.missiles), missile_telemetry={
                m.id: {
                    'tracking': bool(m.alive and self._missile_tracking.get(m.id, False)),
                    'tracking_source': ('seeker' if m.phase == 'active' else 'radar')
                        if m.alive and self._missile_tracking.get(m.id, False) else 'none',
                    'supporters': self._missile_supporters.get(m.id, ()) if m.alive else (),
                    'ever_seeker_acquired': m.id in self._missile_bitten,
                } for m in self.missiles}))
        for ident, reward in rewards.items():
            self.episode_returns[ident] += reward
        return self.observations(), rewards, terminated, truncated, self.info()

    @staticmethod
    def _missile_available_lateral_acceleration(missile, config):
        """Current aerodynamic steering authority in SI acceleration units."""
        altitude = missile.z*config.aero_length_scale
        world_speed = config.aircraft_speed if missile.speed is None else missile.speed
        physical_speed = world_speed*config.aero_length_scale/config.aero_tick_seconds
        authority = min(1.0, density(altitude)/1.225
                        * (physical_speed/config.aero_missile_control_speed)**2)
        return config.aero_missile_max_lateral_acceleration*authority

    def snapshot(self, *, include_rewards=True):
        """Privileged state for rendering/debugging only; never a policy observation."""
        aircraft = []
        live = [a for a in self.aircraft.values() if a.alive]
        flying = [m for m in self.missiles if m.alive]
        radar = {a.id: {target.id for target in live
                        if target.team != a.team and self.radar_visible(a, target)} for a in live}
        missile_radar = {a.id: {m.id for m in flying
                                if m.team != a.team and self.radar_visible(a, m)} for a in live}
        for a in self.aircraft.values():
            own = self.aircraft_settings[a.id]
            contacts, missile_contacts = [], []
            if a.alive:
                contacts = [{'id': ally.id, 'source': 'ally'} for ally in live
                            if ally.team == a.team and ally.id != a.id]
                contacts.extend({'id': target.id, 'source': 'radar' if target.id in radar[a.id] else 'datalink'}
                                for target in self.tracks(a, radar_visible=lambda observer, target: target.id in radar[observer.id]))
                observers = [o for o in live if o.team == a.team and (self.config.share_tracks or o.id == a.id)]
                for m in flying:
                    if m.team == a.team:
                        missile_contacts.append({'id': m.id, 'source': 'own' if m.owner == a.id else 'ally'})
                    elif any(m.id in missile_radar[o.id] for o in observers):
                        missile_contacts.append({'id': m.id, 'source': 'radar' if m.id in missile_radar[a.id] else 'datalink'})
            aircraft.append({**asdict(a), "radar_range": own.radar_range, "radar_fov": own.radar_fov,
                             "hud_contacts": contacts, "hud_missiles": missile_contacts,
                             "control": self.control_diagnostics.get(a.id)})
        missiles = []
        for m in self.missiles:
            if m.alive:
                own = self.aircraft_settings[m.owner]
                available_acceleration = self._missile_available_lateral_acceleration(m, own)
                guidance_strength = self._missile_guidance_strength.get(m.id, 0.0)
                missiles.append({**asdict(m), "tracking": bool(self._missile_tracking.get(m.id, False)),
                                 "guidance_strength": guidance_strength,
                                 "available_lateral_acceleration": available_acceleration,
                                 "guidance_acceleration_limit": guidance_strength*available_acceleration,
                                 "reacquisition_ticks": self._missile_reacquisition_ticks.get(m.id, 0),
                                 "seeker_range": own.seeker_range, "seeker_fov": own.seeker_fov})
        frame = {"tick": self.tick, "aircraft": aircraft, "missiles": missiles, "events": list(self.events)}
        if include_rewards:
            components = self.reward_system.last_components
            frame["rewards"] = {
                "components": {name: dict(values) for name, values in components.items()
                               if name != 'HookReward'},
                "clipping": {reason: {ident: dict(entry) for ident, entry in rows.items()}
                             for reason, rows in self.reward_system.last_clipping.items()},
                "episode_totals": dict(self.episode_returns),
            }
        return frame
