"""Replaceable movement only; detection, lifecycle and rewards belong to ToyEnv."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np
from typing import TYPE_CHECKING, Protocol

from .geometry import wrap
from .atmosphere import density, thrust_fraction
from . import numba_aircraft
from . import numba_missile

if TYPE_CHECKING:
    from .env import Config


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    heading: float
    pitch: float
    speed: float | None = None
    elapsed: float = 0.0

    @classmethod
    def from_body(cls, body):
        return cls(body.x, body.y, body.z, body.heading, body.pitch,
                   getattr(body, 'speed', None), getattr(body, 'elapsed', 0.0))

    def apply_to(self, body):
        body.x, body.y, body.z = self.x, self.y, self.z
        body.heading, body.pitch = self.heading, self.pitch
        if self.speed is not None and hasattr(body, 'speed'):
            body.speed = self.speed
        if hasattr(body, 'elapsed'):
            body.elapsed = self.elapsed


@dataclass(frozen=True)
class FlightControl:
    # Normalized angle targets: turn*45 deg; vertical maps -1/+1 to -45/+15 deg.
    turn: float = 0
    vertical: float = 0
    target_heading: float | None = None
    rear_turn_direction: int = 0

    def stabilized(self, heading, previous=None):
        """Keep a turn's side when adjacent rear targets straddle the angle seam.

        Only rear targets (>=150 deg error) within 45 deg of the previous
        target participate. New distant targets and forward targets respond
        immediately. State is carried by the owning aircraft's last control.
        """
        if self.target_heading is None:
            return self
        error = wrap(self.target_heading-heading)
        direction = 0
        if abs(error) >= math.radians(150):
            direction = 1 if error > 0 else -1
            if (previous is not None and previous.target_heading is not None
                    and abs(wrap(self.target_heading-previous.target_heading)) <= math.pi/4
                    and previous.rear_turn_direction):
                direction = previous.rear_turn_direction
        return replace(self, rear_turn_direction=direction)

    @property
    def target_pitch(self):
        return math.radians(self.vertical*(45 if self.vertical < 0 else 15))

    def heading_command(self, heading):
        target = self.target_heading if self.target_heading is not None else heading + self.turn*math.pi/4
        error = wrap(target-heading)
        if (self.rear_turn_direction*error < 0 and abs(error) >= math.radians(120)):
            error += self.rear_turn_direction*2*math.pi
        return max(-1.0, min(1.0, error/(math.pi/4)))


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    z: float
    # Optional tracked velocity in game distance units per second. Position is
    # at the beginning of the integration tick. Unknown tracks remain fixed.
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Without a live track, coast under thrust, drag and gravity rather than
    # circle a stale point. A later sensor update can re-enable steering.
    guidance_enabled: bool = True
    # Steering authority after a real track interruption.  A continuous
    # midcourse-to-seeker handoff remains at full authority.
    guidance_strength: float = 1.0


class AircraftMotion(Protocol):
    def advance(self, pose: Pose, control: FlightControl, config: Config) -> Pose:
        """Return the next pose for one game tick. Do not mutate inputs."""
        ...


class MissileMotion(Protocol):
    def advance(self, pose: Pose, waypoint: Waypoint, config: Config) -> Pose:
        """Move toward the supplied game waypoint without accessing world state."""
        ...


class AerodynamicAircraftMotion:
    """Still-air point mass with an ideal lift-vector autopilot, integrated in SI.

    Heading/pitch describe velocity, not body attitude. Lift is perpendicular
    to velocity; gravity always acts down. No artificial minimum/maximum speed.
    """

    density = staticmethod(density)

    def acceleration(self, position, velocity, control, c):
        vx, vy, vz = velocity
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        horizontal = math.hypot(vx, vy)
        heading = math.atan2(vy, vx)
        pitch = math.atan2(vz, horizontal)
        cp, sp = math.cos(pitch), math.sin(pitch)
        ch, sh = math.cos(heading), math.sin(heading)
        forward = (cp*ch, cp*sh, sp)
        up = (-sp*ch, -sp*sh, cp)
        right = (-sh, ch, 0.0)
        # Proportional target tracking: discrete target angles produce distinct
        # initial rates instead of every nonzero target immediately saturating.
        desired_pitch = max(-c.max_pitch, min(c.max_pitch, control.target_pitch))
        pitch_error = (desired_pitch-pitch)/c.pitch_error_scale
        pitch_rate = c.pitch_rate*max(-1.0, min(1.0, pitch_error))/c.aero_tick_seconds
        normal = max(0.0, c.aero_gravity*cp + speed*pitch_rate)
        lateral = horizontal*control.heading_command(heading)*c.turn_rate/c.aero_tick_seconds
        # Restore turn-priority allocation while retaining at least 25% of
        # the requested unloading from trim. Lift/load caps apply below.
        bank_tangent = math.tan(c.aero_max_bank)
        trim = max(0.0, c.aero_gravity*cp)
        if normal < trim:
            normal_limit = trim-.25*(trim-normal)
            normal = max(normal, min(normal_limit, abs(lateral)/bank_tangent))
        lateral = math.copysign(min(abs(lateral), normal*bank_tangent), lateral)
        demand = math.hypot(normal, lateral)
        air_density = self.density(position[2])
        q_area = .5*air_density*speed*speed*c.aero_wing_area
        lift = min(c.aero_mass*demand, q_area*c.aero_cl_max,
                   c.aero_mass*c.aero_max_load*9.80665)
        cl = lift/q_area if q_area > 0 else 0.0
        drag = q_area*(c.aero_cd0 + c.aero_induced_drag*cl*cl)
        scale = lift/(c.aero_mass*demand) if demand > 0 else 0.0
        thrust = c.aero_thrust*thrust_fraction(air_density, c.aero_thrust_density_exponent)
        along = (thrust-drag)/c.aero_mass
        return (along*forward[0] + scale*(normal*up[0]+lateral*right[0]),
                along*forward[1] + scale*(normal*up[1]+lateral*right[1]),
                along*forward[2] + scale*(normal*up[2]+lateral*right[2]) - c.aero_gravity)

    def advance(self, pose, control, config):
        c = config
        control = replace(control, target_heading=wrap(pose.heading + control.turn*math.pi/4)) if control.target_heading is None else control
        # Subclasses may override acceleration for instrumentation or custom
        # dynamics, so only the stock implementation uses the compiled path.
        if numba_aircraft.AVAILABLE and type(self) is AerodynamicAircraftMotion:
            values = numba_aircraft.advance(
                pose.x, pose.y, pose.z, pose.heading, pose.pitch,
                math.nan if pose.speed is None else pose.speed,
                control.turn, control.target_pitch,
                math.nan if control.target_heading is None else control.target_heading,
                control.rear_turn_direction, c.aircraft_speed, c.max_pitch,
                c.pitch_error_scale, c.pitch_rate, c.aero_tick_seconds,
                c.aero_gravity, c.turn_rate, c.aero_max_bank,
                c.aero_wing_area, c.aero_mass, c.aero_cl_max,
                c.aero_max_load, c.aero_cd0, c.aero_induced_drag,
                c.aero_thrust, c.aero_length_scale, c.aero_substeps,
                c.aero_thrust_density_exponent)
            return Pose(*values)
        units = c.aero_length_scale
        speed_scale = units/c.aero_tick_seconds
        speed = (c.aircraft_speed if pose.speed is None else pose.speed)*speed_scale
        velocity = (speed*math.cos(pose.pitch)*math.cos(pose.heading),
                    speed*math.cos(pose.pitch)*math.sin(pose.heading), speed*math.sin(pose.pitch))
        position = tuple(v*units for v in (pose.x, pose.y, pose.z))
        dt = c.aero_tick_seconds/c.aero_substeps
        for _ in range(c.aero_substeps):
            acceleration = self.acceleration(position, velocity, control, c)
            mid_v = (velocity[0]+.5*dt*acceleration[0],
                     velocity[1]+.5*dt*acceleration[1],
                     velocity[2]+.5*dt*acceleration[2])
            mid_p = (position[0]+.5*dt*velocity[0],
                     position[1]+.5*dt*velocity[1],
                     position[2]+.5*dt*velocity[2])
            mid_a = self.acceleration(mid_p, mid_v, control, c)
            next_p = (position[0]+dt*mid_v[0], position[1]+dt*mid_v[1], position[2]+dt*mid_v[2])
            next_v = (velocity[0]+dt*mid_a[0], velocity[1]+dt*mid_a[1], velocity[2]+dt*mid_a[2])
            if next_p[2] <= 0:
                fraction = position[2]/(position[2]-next_p[2]) if position[2] > 0 else 0
                position = tuple(position[i]+fraction*(next_p[i]-position[i]) for i in range(3))
                return Pose(position[0]/units, position[1]/units, 0.0,
                            pose.heading, pose.pitch, 0.0)
            position, velocity = next_p, next_v
        vx, vy, vz = velocity
        return Pose(position[0]/units, position[1]/units, position[2]/units,
                    math.atan2(vy, vx), math.atan2(vz, math.hypot(vx, vy)),
                    math.sqrt(vx*vx+vy*vy+vz*vz)/speed_scale)


class AerodynamicMissileMotion:
    """SI point-mass missile motion mapped onto the game coordinate system.

    The waypoint still uses a capped guidance turn, but forces are evaluated
    in SI: thrust, quadratic aerodynamic drag and gravity act on a mass point.
    Tracked velocity feeds proportional navigation; no attitude/fin model is included.
    All per-projectile state lives in Pose, so reset/parallel envs are isolated.
    """

    def advance(self, pose, waypoint, config):
        c = config
        if numba_missile.AVAILABLE and type(self) is AerodynamicMissileMotion:
            values = numba_missile.advance_one(
                __import__('numpy').asarray((pose.x,pose.y,pose.z,pose.heading,pose.pitch,
                    math.nan if pose.speed is None else pose.speed,pose.elapsed),dtype='float64'),
                __import__('numpy').asarray((waypoint.x,waypoint.y,waypoint.z,*waypoint.velocity,
                    float(waypoint.guidance_enabled),waypoint.guidance_strength),dtype='float64'),
                numba_missile.config_array(c))
            return Pose(*values)
        length_scale = c.aero_length_scale
        tick = c.aero_tick_seconds
        x, y, z = (value*length_scale for value in (pose.x, pose.y, pose.z))
        speed_scale = length_scale/tick
        speed = (c.aircraft_speed if pose.speed is None else pose.speed)*speed_scale
        velocity = (speed*math.cos(pose.pitch)*math.cos(pose.heading),
                speed*math.cos(pose.pitch)*math.sin(pose.heading),
                speed*math.sin(pose.pitch))
        elapsed = pose.elapsed
        dt = tick/c.aero_substeps
        target_velocity = tuple(v*length_scale for v in waypoint.velocity)
        for _ in range(c.aero_substeps):
            # Split exactly at burnout, even when it falls between substeps.
            remaining = dt
            while remaining > 1e-12:
                boost_end = c.aero_missile_boost_seconds
                sustain_end = boost_end + c.aero_missile_sustain_seconds
                staged = boost_end > 0 or c.aero_missile_sustain_seconds > 0
                if staged and elapsed < boost_end:
                    thrust = c.aero_missile_boost_thrust
                    burn_end = boost_end
                elif staged and elapsed < sustain_end:
                    thrust = c.aero_missile_sustain_thrust
                    burn_end = sustain_end
                elif staged:
                    thrust = 0.0
                    burn_end = None
                else:
                    thrust = c.aero_missile_thrust if elapsed < c.aero_missile_burn_seconds else 0.0
                    burn_end = c.aero_missile_burn_seconds if thrust else None
                burning = thrust > 0
                # Even a zero-thrust stage has an end time: do not skip the
                # beginning of the next stage when it falls within a substep.
                h = min(remaining, burn_end-elapsed) if burn_end is not None else remaining
                velocity_speed = math.sqrt(sum(value*value for value in velocity))
                forward = tuple(value/velocity_speed for value in velocity) if velocity_speed > 1e-12 else (
                    math.cos(pose.pitch)*math.cos(pose.heading),
                    math.cos(pose.pitch)*math.sin(pose.heading), math.sin(pose.pitch))
                delta = tuple(target*length_scale + target_velocity[i]*(elapsed-pose.elapsed)-position
                              for i, (target, position) in enumerate(zip(
                                  (waypoint.x, waypoint.y, waypoint.z), (x, y, z))))
                rlen = math.sqrt(sum(v*v for v in delta))
                angle = 0.0
                if waypoint.guidance_enabled and rlen > 1e-12 and speed > 1e-12:
                    rel = tuple(target_velocity[i]-velocity[i] for i in range(3))
                    closing = max(0.0, -sum(delta[i]*rel[i] for i in range(3))/max(rlen, 1e-9))
                    los = tuple(v/rlen for v in delta) if rlen > 1e-9 else (0.0, 0.0, 0.0)
                    los_rate = (los[1]*rel[2]-los[2]*rel[1], los[2]*rel[0]-los[0]*rel[2], los[0]*rel[1]-los[1]*rel[0])
                    # LOS angular velocity has units 1/s; omega x LOS points
                    # toward the transverse relative velocity, not away from it.
                    los_rate = tuple(v/max(rlen, 1e-9) for v in los_rate)
                    pn = (los_rate[1]*los[2]-los_rate[2]*los[1], los_rate[2]*los[0]-los_rate[0]*los[2], los_rate[0]*los[1]-los_rate[1]*los[0])
                    # Proportional navigation already accounts for target
                    # velocity. The centering term only restores the current
                    # line of sight; aiming it at a constant-velocity lead
                    # point can extrapolate a climbing target far above its
                    # real altitude after reacquisition.
                    projection = sum(los[i]*forward[i] for i in range(3))
                    acquisition = tuple(velocity_speed/c.aero_missile_acquisition_time *
                                        (los[i]-projection*forward[i]) for i in range(3))
                    commanded = tuple(velocity[i] + (c.missile_navigation_constant*closing*pn[i]
                                      + acquisition[i])*h for i in range(3))
                    commanded_len = math.sqrt(sum(v*v for v in commanded))
                    desired = tuple(v/commanded_len for v in commanded) if commanded_len > 1e-9 else forward
                    dot = max(-1.0, min(1.0, sum(a*b for a, b in zip(forward, desired))))
                    error = math.acos(dot)
                    authority = min(1.0, AerodynamicAircraftMotion.density(z)/1.225 *
                                    (velocity_speed/c.aero_missile_control_speed)**2)
                    acceleration_limit = c.aero_missile_max_lateral_acceleration*authority
                    rate_limit = (waypoint.guidance_strength *
                                  min(c.aero_missile_turn_rate,
                                      acceleration_limit/max(velocity_speed, 1e-9)))
                    angle = min(error, rate_limit*h)
                    tangent = tuple(desired[i]-dot*forward[i] for i in range(3))
                    norm = math.sqrt(sum(v*v for v in tangent))
                    if norm < 1e-10:
                        # Deterministic perpendicular for a waypoint directly behind.
                        axis = min(range(3), key=lambda i: abs(forward[i]))
                        tangent = tuple((1.0 if i == axis else 0.0)-forward[axis]*forward[i]
                                        for i in range(3))
                        norm = math.sqrt(sum(v*v for v in tangent))
                    forward = tuple(math.cos(angle)*forward[i] +
                                    math.sin(angle)*tangent[i]/norm for i in range(3))
                if angle:
                    velocity = tuple(velocity_speed*math.exp(-c.aero_missile_turn_drag*angle)*value
                                     for value in forward)
                speed = math.sqrt(sum(value*value for value in velocity))
                thrust_acceleration = thrust/c.aero_missile_mass if burning else 0.0
                drag_factor = (.5*AerodynamicAircraftMotion.density(z)
                               *c.aero_missile_drag*c.aero_missile_area/c.aero_missile_mass)
                drag_acceleration = drag_factor*speed*speed
                acceleration = tuple((thrust_acceleration-drag_acceleration)*forward[i]
                                     - (c.aero_missile_gravity if i == 2 else 0.0)
                                     for i in range(3))
                mid_velocity = tuple(velocity[i]+.5*h*acceleration[i] for i in range(3))
                mid_speed = math.sqrt(sum(value*value for value in mid_velocity))
                mid_drag = (.5*AerodynamicAircraftMotion.density(z+.5*h*velocity[2])
                            *c.aero_missile_drag*c.aero_missile_area/c.aero_missile_mass*mid_speed*mid_speed)
                mid_acceleration = tuple((thrust_acceleration-mid_drag)*forward[i]
                                         - (c.aero_missile_gravity if i == 2 else 0.0)
                                         for i in range(3))
                nx, ny, nz = (position+h*mid_velocity[i] for i, position in enumerate((x, y, z)))
                if nz <= 0:
                    # Terminate at contact; do not continue a path below ground.
                    contact = h
                    if z <= 0:
                        contact = 0.0
                    else:
                        # Solve the same quadratic used by this substep's
                        # midpoint position, including thrust/drag vertically.
                        root = math.sqrt(max(0.0, velocity[2]**2-2*acceleration[2]*z))
                        denominator = root-velocity[2]
                        if denominator > 0:
                            contact = 2*z/denominator
                    contact = max(0.0, min(h, contact))
                    contact_position = tuple((x, y, z)[i]+contact*velocity[i]
                                             +.5*contact*contact*acceleration[i] for i in range(3))
                    return Pose(contact_position[0]/length_scale, contact_position[1]/length_scale, 0.0,
                                math.atan2(mid_velocity[1], mid_velocity[0]),
                                math.atan2(mid_velocity[2], math.hypot(mid_velocity[0], mid_velocity[1])),
                                0.0, elapsed+contact)
                x, y, z = nx, ny, nz
                velocity = tuple(velocity[i]+h*mid_acceleration[i] for i in range(3))
                speed = math.sqrt(sum(value*value for value in velocity))
                elapsed += h
                remaining -= h
        heading = math.atan2(velocity[1], velocity[0])
        pitch = math.atan2(velocity[2], math.hypot(velocity[0], velocity[1]))
        return Pose(x/length_scale, y/length_scale, z/length_scale, heading, pitch,
                    speed/speed_scale, pose.elapsed+tick)

    def advance_many(self, poses, waypoints, config):
        """Advance independent missiles in one compiled array kernel."""
        if not numba_missile.AVAILABLE:
            return [self.advance(pose, waypoint, config)
                    for pose, waypoint in zip(poses, waypoints)]
        pose_array = np.asarray([
            (p.x,p.y,p.z,p.heading,p.pitch,math.nan if p.speed is None else p.speed,p.elapsed)
            for p in poses], dtype=np.float64)
        waypoint_array = np.asarray([
            (w.x,w.y,w.z,*w.velocity,float(w.guidance_enabled),w.guidance_strength)
            for w in waypoints], dtype=np.float64)
        values = numba_missile.advance_batch(
            pose_array, waypoint_array, numba_missile.config_array(config))
        return [Pose(*row) for row in values]
