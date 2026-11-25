"""Numba kernels for aerodynamic missile integration."""
from __future__ import annotations

import math

from .atmosphere import density as _density
import numpy as np
from numba import njit

AVAILABLE = True


@njit(cache=True)
def advance_one(pose, waypoint, c):
    # Config layout is produced by config_array below.
    length_scale, tick, aircraft_speed, substeps = c[0], c[1], c[2], int(c[3])
    x, y, z = pose[0]*length_scale, pose[1]*length_scale, pose[2]*length_scale
    heading, pitch = pose[3], pose[4]
    speed_scale = length_scale/tick
    speed = (aircraft_speed if math.isnan(pose[5]) else pose[5])*speed_scale
    vx = speed*math.cos(pitch)*math.cos(heading)
    vy = speed*math.cos(pitch)*math.sin(heading)
    vz = speed*math.sin(pitch)
    initial_elapsed = pose[6]
    elapsed = initial_elapsed
    dt = tick/substeps
    tvx, tvy, tvz = waypoint[3]*length_scale, waypoint[4]*length_scale, waypoint[5]*length_scale
    for _ in range(substeps):
        remaining = dt
        while remaining > 1e-12:
            boost_end, sustain_duration = c[4], c[5]
            sustain_end = boost_end+sustain_duration
            staged = boost_end > 0.0 or sustain_duration > 0.0
            if staged and elapsed < boost_end:
                thrust, burn_end = c[6], boost_end
            elif staged and elapsed < sustain_end:
                thrust, burn_end = c[7], sustain_end
            elif staged:
                thrust, burn_end = 0.0, -1.0
            else:
                thrust = c[8] if elapsed < c[9] else 0.0
                burn_end = c[9] if thrust else -1.0
            burning = thrust > 0.0
            h = min(remaining, burn_end-elapsed) if burn_end >= 0.0 else remaining
            velocity_speed = math.sqrt(vx*vx+vy*vy+vz*vz)
            if velocity_speed > 1e-12:
                fx, fy, fz = vx/velocity_speed, vy/velocity_speed, vz/velocity_speed
            else:
                fx = math.cos(pitch)*math.cos(heading)
                fy = math.cos(pitch)*math.sin(heading)
                fz = math.sin(pitch)
            drift = elapsed-initial_elapsed
            dx = waypoint[0]*length_scale+tvx*drift-x
            dy = waypoint[1]*length_scale+tvy*drift-y
            dz = waypoint[2]*length_scale+tvz*drift-z
            rlen = math.sqrt(dx*dx+dy*dy+dz*dz)
            angle = 0.0
            if waypoint[6] != 0.0 and rlen > 1e-12 and speed > 1e-12:
                rvx, rvy, rvz = tvx-vx, tvy-vy, tvz-vz
                closing = max(0.0, -(dx*rvx+dy*rvy+dz*rvz)/max(rlen,1e-9))
                if rlen > 1e-9:
                    losx, losy, losz = dx/rlen, dy/rlen, dz/rlen
                else:
                    losx=losy=losz=0.0
                lrx = losy*rvz-losz*rvy
                lry = losz*rvx-losx*rvz
                lrz = losx*rvy-losy*rvx
                inverse_range = 1.0/max(rlen,1e-9)
                lrx,lry,lrz = lrx*inverse_range,lry*inverse_range,lrz*inverse_range
                pnx = lry*losz-lrz*losy
                pny = lrz*losx-lrx*losz
                pnz = lrx*losy-lry*losx
                projection = losx*fx+losy*fy+losz*fz
                acquisition = velocity_speed/c[20]
                cmdx = vx+(c[11]*closing*pnx+acquisition*(losx-projection*fx))*h
                cmdy = vy+(c[11]*closing*pny+acquisition*(losy-projection*fy))*h
                cmdz = vz+(c[11]*closing*pnz+acquisition*(losz-projection*fz))*h
                cmdlen = math.sqrt(cmdx*cmdx+cmdy*cmdy+cmdz*cmdz)
                if cmdlen > 1e-9: desx,desy,desz = cmdx/cmdlen,cmdy/cmdlen,cmdz/cmdlen
                else: desx,desy,desz = fx,fy,fz
                dot = max(-1.0,min(1.0,fx*desx+fy*desy+fz*desz))
                error = math.acos(dot)
                authority = min(1.0,_density(z)/1.225*(velocity_speed/c[19])**2)
                rate_limit = waypoint[7]*min(c[12],c[18]*authority/max(velocity_speed,1e-9))
                angle = min(error,rate_limit*h)
                tx,ty,tz = desx-dot*fx,desy-dot*fy,desz-dot*fz
                norm = math.sqrt(tx*tx+ty*ty+tz*tz)
                if norm < 1e-10:
                    axis = 0
                    if abs(fy) < abs(fx): axis=1
                    if abs(fz) < (abs(fy) if axis==1 else abs(fx)): axis=2
                    component = fx if axis==0 else fy if axis==1 else fz
                    tx = (1.0 if axis==0 else 0.0)-component*fx
                    ty = (1.0 if axis==1 else 0.0)-component*fy
                    tz = (1.0 if axis==2 else 0.0)-component*fz
                    norm = math.sqrt(tx*tx+ty*ty+tz*tz)
                ca,sa = math.cos(angle),math.sin(angle)
                fx,fy,fz = ca*fx+sa*tx/norm,ca*fy+sa*ty/norm,ca*fz+sa*tz/norm
            if angle:
                retained = velocity_speed*math.exp(-c[13]*angle)
                vx,vy,vz = retained*fx,retained*fy,retained*fz
            speed = math.sqrt(vx*vx+vy*vy+vz*vz)
            thrust_acceleration = thrust/c[14] if burning else 0.0
            drag_factor = .5*_density(z)*c[15]*c[16]/c[14]
            drag_acceleration = drag_factor*speed*speed
            ax = (thrust_acceleration-drag_acceleration)*fx
            ay = (thrust_acceleration-drag_acceleration)*fy
            az = (thrust_acceleration-drag_acceleration)*fz-c[17]
            mvx,mvy,mvz = vx+.5*h*ax,vy+.5*h*ay,vz+.5*h*az
            mid_speed = math.sqrt(mvx*mvx+mvy*mvy+mvz*mvz)
            mid_drag = .5*_density(z+.5*h*vz)*c[15]*c[16]/c[14]*mid_speed*mid_speed
            max_=(thrust_acceleration-mid_drag)*fx
            may=(thrust_acceleration-mid_drag)*fy
            maz=(thrust_acceleration-mid_drag)*fz-c[17]
            nx,ny,nz = x+h*mvx,y+h*mvy,z+h*mvz
            if nz <= 0.0:
                contact = h
                if z <= 0.0: contact=0.0
                else:
                    root=math.sqrt(max(0.0,vz*vz-2.0*az*z)); denominator=root-vz
                    if denominator>0.0: contact=2.0*z/denominator
                contact=max(0.0,min(h,contact))
                cx=x+contact*vx+.5*contact*contact*ax
                cy=y+contact*vy+.5*contact*contact*ay
                return np.array((cx/length_scale,cy/length_scale,0.0,
                    math.atan2(mvy,mvx),math.atan2(mvz,math.hypot(mvx,mvy)),0.0,elapsed+contact))
            x,y,z=nx,ny,nz
            vx,vy,vz=vx+h*max_,vy+h*may,vz+h*maz
            speed=math.sqrt(vx*vx+vy*vy+vz*vz)
            elapsed+=h; remaining-=h
    return np.array((x/length_scale,y/length_scale,z/length_scale,
        math.atan2(vy,vx),math.atan2(vz,math.hypot(vx,vy)),speed/speed_scale,initial_elapsed+tick))


@njit(cache=True)
def advance_batch(poses, waypoints, c):
    result=np.empty_like(poses)
    for i in range(poses.shape[0]): result[i]=advance_one(poses[i],waypoints[i],c)
    return result


def config_array(c):
    return np.asarray((c.aero_length_scale,c.aero_tick_seconds,c.aircraft_speed,c.aero_substeps,
        c.aero_missile_boost_seconds,c.aero_missile_sustain_seconds,c.aero_missile_boost_thrust,
        c.aero_missile_sustain_thrust,c.aero_missile_thrust,c.aero_missile_burn_seconds,
        c.missile_lifetime,c.missile_navigation_constant,c.aero_missile_turn_rate,
        c.aero_missile_turn_drag,c.aero_missile_mass,c.aero_missile_drag,
        c.aero_missile_area,c.aero_missile_gravity, c.aero_missile_max_lateral_acceleration,
        c.aero_missile_control_speed,c.aero_missile_acquisition_time),dtype=np.float64)
