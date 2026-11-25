"""Optional Numba kernel for aerodynamic aircraft integration."""
from __future__ import annotations

import math

from .atmosphere import density as _density, thrust_fraction

try:
    from numba import njit
except ImportError:  # pragma: no cover - the Python motion remains the fallback
    njit = None


AVAILABLE = njit is not None


if AVAILABLE:
    @njit(cache=True)
    def _wrap(value):
        return (value + math.pi) % (2.0 * math.pi) - math.pi


    @njit(cache=True)
    def _acceleration(px, py, pz, vx, vy, vz, target_pitch, target_heading,
                      rear_direction, max_pitch, pitch_error_scale, pitch_rate,
                      tick, gravity, turn_rate, max_bank, wing_area, mass,
                      cl_max, max_load, cd0, induced_drag, thrust, thrust_density_exponent):
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        horizontal = math.hypot(vx, vy)
        heading = math.atan2(vy, vx)
        pitch = math.atan2(vz, horizontal)
        cp, sp = math.cos(pitch), math.sin(pitch)
        ch, sh = math.cos(heading), math.sin(heading)
        desired_pitch = max(-max_pitch, min(max_pitch, target_pitch))
        pitch_error = (desired_pitch-pitch)/pitch_error_scale
        requested_pitch_rate = pitch_rate*max(-1.0, min(1.0, pitch_error))/tick
        normal = max(0.0, gravity*cp + speed*requested_pitch_rate)
        error = _wrap(target_heading-heading)
        if rear_direction*error < 0.0 and abs(error) >= math.radians(120.0):
            error += rear_direction*2.0*math.pi
        command = max(-1.0, min(1.0, error/(math.pi/4.0)))
        lateral = horizontal*command*turn_rate/tick
        bank_tangent = math.tan(max_bank)
        trim = max(0.0, gravity*cp)
        if normal < trim:
            normal_limit = trim-.25*(trim-normal)
            normal = max(normal, min(normal_limit, abs(lateral)/bank_tangent))
        lateral = math.copysign(min(abs(lateral), normal*bank_tangent), lateral)
        demand = math.hypot(normal, lateral)
        air_density = _density(pz)
        q_area = .5*air_density*speed*speed*wing_area
        lift = min(mass*demand, q_area*cl_max, mass*max_load*9.80665)
        cl = lift/q_area if q_area > 0.0 else 0.0
        drag = q_area*(cd0 + induced_drag*cl*cl)
        scale = lift/(mass*demand) if demand > 0.0 else 0.0
        along = (thrust*thrust_fraction(air_density, thrust_density_exponent)-drag)/mass
        fx, fy, fz = cp*ch, cp*sh, sp
        ux, uy, uz = -sp*ch, -sp*sh, cp
        rx, ry = -sh, ch
        return (along*fx + scale*(normal*ux+lateral*rx),
                along*fy + scale*(normal*uy+lateral*ry),
                along*fz + scale*normal*uz - gravity)


    @njit(cache=True)
    def advance(x, y, z, heading, pitch, pose_speed, turn, target_pitch,
                target_heading, rear_direction, aircraft_speed, max_pitch,
                pitch_error_scale, pitch_rate, tick, gravity, turn_rate,
                max_bank, wing_area, mass, cl_max, max_load, cd0, induced_drag,
                thrust, length_scale, substeps, thrust_density_exponent=0.0):
        if math.isnan(target_heading):
            target_heading = _wrap(heading + turn*math.pi/4.0)
        speed_scale = length_scale/tick
        speed = aircraft_speed if math.isnan(pose_speed) else pose_speed
        speed *= speed_scale
        vx = speed*math.cos(pitch)*math.cos(heading)
        vy = speed*math.cos(pitch)*math.sin(heading)
        vz = speed*math.sin(pitch)
        px, py, pz = x*length_scale, y*length_scale, z*length_scale
        dt = tick/substeps
        for _ in range(substeps):
            args = (target_pitch, target_heading, rear_direction, max_pitch,
                    pitch_error_scale, pitch_rate, tick, gravity, turn_rate,
                    max_bank, wing_area, mass, cl_max, max_load, cd0,
                    induced_drag, thrust, thrust_density_exponent)
            ax, ay, az = _acceleration(px, py, pz, vx, vy, vz, *args)
            mvx, mvy, mvz = vx+.5*dt*ax, vy+.5*dt*ay, vz+.5*dt*az
            mpx, mpy, mpz = px+.5*dt*vx, py+.5*dt*vy, pz+.5*dt*vz
            max_, may, maz = _acceleration(mpx, mpy, mpz, mvx, mvy, mvz, *args)
            npx, npy, npz = px+dt*mvx, py+dt*mvy, pz+dt*mvz
            nvx, nvy, nvz = vx+dt*max_, vy+dt*may, vz+dt*maz
            if npz <= 0.0:
                fraction = pz/(pz-npz) if pz > 0.0 else 0.0
                px += fraction*(npx-px)
                py += fraction*(npy-py)
                return px/length_scale, py/length_scale, 0.0, heading, pitch, 0.0
            px, py, pz, vx, vy, vz = npx, npy, npz, nvx, nvy, nvz
        return (px/length_scale, py/length_scale, pz/length_scale,
                math.atan2(vy, vx), math.atan2(vz, math.hypot(vx, vy)),
                math.sqrt(vx*vx+vy*vy+vz*vz)/speed_scale)
else:
    def advance(*args):  # pragma: no cover
        raise RuntimeError("Numba is not installed")
