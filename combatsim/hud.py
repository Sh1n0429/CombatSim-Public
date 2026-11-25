"""Spectator-only missile reach estimates; never used by policy observations."""
import math

import numpy as np
from numba import njit

from .numba_missile import advance_one, config_array


@njit(cache=True)
def _intercepts(ranges, launch, target, azimuth, c, hit_radius):
    hits = np.zeros(len(ranges), dtype=np.bool_)
    tick = c[1]
    velocity = np.array((math.cos(target[4])*math.cos(target[3]),
                         math.cos(target[4])*math.sin(target[3]),
                         math.sin(target[4]))) * target[5]/tick
    for index in range(len(ranges)):
        pose = launch.copy()
        point = np.array((math.cos(azimuth)*ranges[index],
                          math.sin(azimuth)*ranges[index], target[2]))
        for _ in range(int(c[10])):
            end = point + velocity*tick
            if end[2] <= 0:
                break
            waypoint = np.array((point[0], point[1], point[2],
                                 velocity[0], velocity[1], velocity[2], 1., 1.))
            moved = advance_one(pose, waypoint, c)
            fraction = min(1., max(0., (moved[6]-pose[6])/tick))
            relative = moved[:3]-pose[:3]-velocity*tick*fraction
            offset = point-pose[:3]
            squared = np.dot(relative, relative)
            along = min(1., max(0., np.dot(offset, relative)/squared)) if squared else 0.
            closest = offset-along*relative
            if np.dot(closest, closest) <= hit_radius*hit_radius:
                hits[index] = True
                break
            if moved[2] <= 0:
                break
            pose, point = moved, end
    return hits


def estimate_rmax(observer, target, config):
    """Slant range in km, with constant target velocity and maintained guidance."""
    c = config
    unit, tick = c.aero_length_scale, c.aero_tick_seconds
    own_speed = observer.get('speed')
    own_speed = c.aircraft_speed if own_speed is None else own_speed
    target_speed = target.get('speed')
    target_speed = c.aircraft_speed if target_speed is None else target_speed
    launch = np.array((0., 0., observer['z'], observer['heading'],
                       observer['pitch'], own_speed, 0.))
    other = np.array((0., 0., target['z'], target['heading'],
                      target['pitch'], target_speed, 0.))
    dx, dy = target['x']-observer['x'], target['y']-observer['y']
    distance = math.hypot(dx, dy)
    azimuth = math.atan2(dy, dx) if distance else observer['heading']
    life = c.missile_lifetime*tick
    if c.aero_missile_boost_seconds or c.aero_missile_sustain_seconds:
        boost = min(life, c.aero_missile_boost_seconds)
        sustain = min(max(0., life-boost), c.aero_missile_sustain_seconds)
        impulse = boost*c.aero_missile_boost_thrust+sustain*c.aero_missile_sustain_thrust
    else:
        impulse = min(life, c.aero_missile_burn_seconds)*c.aero_missile_thrust
    # A drag-free speed bound brackets the search, including gravity and target motion.
    bound = ((own_speed+target_speed)*unit/tick+impulse/c.aero_missile_mass
             + c.aero_missile_gravity*life)*life/unit+c.hit_radius
    low = min(100/unit, bound)
    ranges = np.unique(np.r_[0., np.geomspace(low, bound, 24), min(distance, bound)])
    parameters = config_array(c)
    hits = _intercepts(ranges, launch, other, azimuth, parameters, c.hit_radius)
    reached = np.flatnonzero(hits)
    if not len(reached):
        return 0.
    index = reached[-1]
    lo, hi = ranges[index], ranges[min(index+1, len(ranges)-1)]
    for _ in range(8):
        middle = (lo+hi)/2
        if _intercepts(np.array([middle]), launch, other, azimuth, parameters, c.hit_radius)[0]:
            lo = middle
        else:
            hi = middle
    return math.hypot(lo, target['z']-observer['z'])*unit/1000


class HudRanges:
    """Keep live/replay range estimates out of the learning and physics paths."""

    def __init__(self, aircraft_settings):
        self.settings = aircraft_settings
        self.cache = {}

    def annotate(self, frame):
        aircraft = {a['id']: a for a in frame['aircraft']}
        tick = frame['tick']
        for observer in aircraft.values():
            if not observer['alive']:
                continue
            config = self.settings[observer['id']]
            interval = max(1, round(5/config.aero_tick_seconds))
            for contact in observer.get('hud_contacts', []):
                if contact['source'] not in ('radar', 'datalink'):
                    continue
                target = aircraft.get(contact['id'])
                if not target or not target['alive']:
                    continue
                key = observer['id'], target['id']
                old_tick, value = self.cache.get(key, (-interval, None))
                if value is None or tick < old_tick or tick-old_tick >= interval:
                    old_tick, value = tick, round(estimate_rmax(observer, target, config), 1)
                    self.cache[key] = old_tick, value
                contact.update(rmax_km=value, rmax_tick=old_tick)
        return frame
