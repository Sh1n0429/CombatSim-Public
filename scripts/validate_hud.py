"""Range forecasts and observer-authorized missile telemetry."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from combatsim.env import Config, ToyEnv, Missile
from combatsim.factory import load_settings
from combatsim.hud import HudRanges, _intercepts, estimate_rmax
from combatsim.numba_missile import config_array
import numpy as np


class LinearMotion:
    def advance(self, pose, control, config):
        speed = pose.speed
        return replace(pose, x=pose.x+speed*math.cos(pose.pitch)*math.cos(pose.heading),
                       y=pose.y+speed*math.cos(pose.pitch)*math.sin(pose.heading),
                       z=pose.z+speed*math.sin(pose.pitch))


class HudChecks(unittest.TestCase):
    def setUp(self):
        self.config = Config(**load_settings('configs/scenario-default.json'))
        self.own = dict(id='blue', alive=True, x=0., y=0., z=24., heading=0., pitch=0., speed=1.)
        self.target = dict(id='red', alive=True, x=160., y=0., z=24., heading=math.pi, pitch=0., speed=1.)

    def test_aspect_speed_altitude_and_initial_turn_change_range(self):
        before = deepcopy((self.own, self.target, self.config))
        approaching = estimate_rmax(self.own, self.target, self.config)
        leaving = estimate_rmax(self.own, {**self.target, 'heading': 0.}, self.config)
        crossing = estimate_rmax(self.own, {**self.target, 'heading': math.pi/2}, self.config)
        self.assertGreater(approaching, crossing)
        self.assertGreater(crossing, leaving)
        self.assertGreater(estimate_rmax({**self.own, 'speed': 1.5}, self.target, self.config), approaching)
        self.assertLess(estimate_rmax(self.own, {**self.target, 'heading': 0., 'speed': 1.5}, self.config), leaving)
        self.assertNotAlmostEqual(estimate_rmax(self.own, {**self.target, 'z': 40.}, self.config), approaching, places=1)
        self.assertLess(estimate_rmax({**self.own, 'heading': math.pi/2}, self.target, self.config), approaching)
        self.assertEqual((self.own, self.target, self.config), before)

    def test_reported_boundary_brackets_a_real_intercept(self):
        maximum = estimate_rmax(self.own, self.target, self.config)*1000/self.config.aero_length_scale
        hits = _intercepts(np.array([maximum*.99, maximum*1.01]),
            np.array([0., 0., 24., 0., 0., 1., 0.]),
            np.array([0., 0., 24., math.pi, 0., 1., 0.]), 0.,
            config_array(self.config), self.config.hit_radius)
        self.assertEqual(hits.tolist(), [True, False])

    def test_forecast_agrees_with_real_missile_flight_and_swept_hits(self):
        c = replace(self.config, aircraft=None, blue=1, red=1, radar_range=10000.,
                    radar_fov=2*math.pi, seeker_range=10000., seeker_fov=2*math.pi)
        for distance, heading in [(160., math.pi), (340., math.pi), (80., 0.), (160., 0.)]:
            with self.subTest(distance=distance, heading=heading):
                world = ToyEnv(c, aircraft_motion=LinearMotion())
                a, b = world.aircraft['blue_0'], world.aircraft['red_0']
                a.x, a.y, a.z, a.heading, a.pitch, a.speed = 0., 0., 24., 0., 0., 1.
                b.x, b.y, b.z, b.heading, b.pitch, b.speed = distance, 0., 24., heading, 0., 1.
                missile = Missile('test', a.id, a.team, b.id, 0., 0., 0., b.x, b.y,
                                  z=24., aim_z=24., speed=1., phase='active')
                world.missiles.append(missile)
                predicted = bool(_intercepts(np.array([distance]), np.array([0., 0., 24., 0., 0., 1., 0.]),
                                 np.array([0., 0., 24., heading, 0., 1., 0.]), 0., config_array(c), c.hit_radius)[0])
                for _ in range(c.missile_lifetime):
                    if not missile.alive or world.done:
                        break
                    world.step({})
                self.assertEqual(predicted, missile.phase == 'hit')

    def test_cached_ranges_only_annotate_observed_enemies_and_keep_physics_unchanged(self):
        frame = {'tick': 0, 'aircraft': [{**self.own, 'hud_contacts': [
            {'id': 'red', 'source': 'datalink'}, {'id': 'friend', 'source': 'ally'}]},
            self.target, {**self.own, 'id': 'friend'}]}
        ranges = HudRanges({'blue': self.config, 'red': self.config, 'friend': self.config})
        with patch('combatsim.hud.estimate_rmax', return_value=42.3) as compute:
            first = ranges.annotate(deepcopy(frame))
            self.assertEqual(first['aircraft'][0]['hud_contacts'][0]['rmax_km'], 42.3)
            self.assertNotIn('rmax_km', first['aircraft'][0]['hud_contacts'][1])
            ranges.annotate({**deepcopy(frame), 'tick': 4})
            self.assertEqual(compute.call_count, 1)
            updated = ranges.annotate({**deepcopy(frame), 'tick': 5})
            self.assertEqual(compute.call_count, 2)
            self.assertEqual(updated['aircraft'][0]['hud_contacts'][0]['rmax_tick'], 5)
            ranges.annotate(deepcopy(frame))
            self.assertEqual(compute.call_count, 3, 'a backward timeline cannot reuse future estimates')
        self.assertNotIn('rmax_km', frame['aircraft'][0]['hud_contacts'][0])

    def test_snapshot_missiles_match_radar_datalink_and_friendly_telemetry(self):
        world = ToyEnv(Config(blue=2, red=1, radar_range=50., radar_fov=.5, share_tracks=True))
        for ident, x in [('blue_0', 100.), ('blue_1', 110.), ('red_0', 140.)]:
            a = world.aircraft[ident]
            a.x, a.y, a.z, a.heading, a.pitch = x, 100., 20., 0., 0.
        for ident, owner, team, x in [('own', 'blue_0', 'blue', 120.), ('ally', 'blue_1', 'blue', 120.),
                                     ('enemy', 'red_0', 'red', 140.), ('unseen', 'red_0', 'red', 300.)]:
            world.missiles.append(Missile(ident, owner, team, 'blue_0', x, 100., 0., 100., 100., z=20., aim_z=20.))
        def sources():
            tracks = deepcopy(world._sensor_tracks)
            values = next(a['hud_missiles'] for a in world.snapshot()['aircraft'] if a['id']=='blue_0')
            self.assertEqual(world._sensor_tracks, tracks)
            return {m['id']: m['source'] for m in values}
        self.assertEqual(sources(), {'own': 'own', 'ally': 'ally', 'enemy': 'radar'})
        world.aircraft['blue_0'].heading = math.pi
        self.assertEqual(sources(), {'own': 'own', 'ally': 'ally', 'enemy': 'datalink'})
        world.config.share_tracks = False
        self.assertEqual(sources(), {'own': 'own', 'ally': 'ally'})
        world.missiles[0].alive = False
        self.assertEqual(sources(), {'ally': 'ally'})
        world.aircraft['blue_0'].alive = False
        self.assertEqual(sources(), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
