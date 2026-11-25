"""Check tracking, integration, collision timing and missile motion backends.

Run with: python -m scripts.validate_missile_motion
"""
from dataclasses import astuple, replace
import math
import unittest

import numpy as np

from combatsim.env import Config, Missile, ToyEnv
from combatsim.motion import AerodynamicMissileMotion, Pose, Waypoint


class PythonMotion(AerodynamicMissileMotion):
    """Exercise the reference path rather than the stock compiled fast path."""


class FrozenAircraftMotion:
    def advance(self, pose, control, config):
        return pose


class RecordingMotion:
    def __init__(self):
        self.waypoints = []

    def advance(self, pose, waypoint, config):
        self.waypoints.append(waypoint)
        return pose


class MissileMotionChecks(unittest.TestCase):
    def test_zero_mass_is_rejected_before_integration(self):
        with self.assertRaisesRegex(ValueError, 'aero_missile_mass'):
            Config(aero_missile_mass=0)

    def test_zero_thrust_stage_still_splits_at_stage_boundary(self):
        config = Config(aero_substeps=1, aero_missile_boost_seconds=.25,
                        aero_missile_boost_thrust=0, aero_missile_sustain_seconds=1,
                        aero_missile_sustain_thrust=100, aero_missile_drag=0,
                        aero_missile_gravity=0)
        pose = Pose(50, 50, 40, 0, 0, 1)
        waypoint = Waypoint(0, 0, 0, guidance_enabled=False)
        for mover in (PythonMotion(), AerodynamicMissileMotion()):
            with self.subTest(backend=type(mover).__name__):
                result = mover.advance(pose, waypoint, config)
                self.assertAlmostEqual(result.speed, 1+.75/250, places=12)
                self.assertAlmostEqual(result.x, 51+.5*.75**2/250, places=12)

    def test_ground_contact_limits_collision_time(self):
        class MovingAircraft:
            def advance(self, pose, control, config):
                return replace(pose, x=pose.x+20)

        class GroundContact:
            def advance(self, pose, waypoint, config):
                return replace(pose, z=0, speed=0, elapsed=pose.elapsed+.1)

        for target_x, expected_hit in ((40, False), (49, True)):
            with self.subTest(target_x=target_x):
                env = ToyEnv(Config(missile_motion='aerodynamic', radar_range=1,
                                    seeker_range=1, hit_radius=.01),
                             aircraft_motion=MovingAircraft(), missile_motion=GroundContact())
                env.reset(seed=123)
                owner, target = env.aircraft['blue_0'], env.aircraft['red_0']
                owner.x, owner.y, owner.z = 10, 10, 20
                target.x, target.y, target.z = target_x, 50, .1
                missile = Missile('ground-regression', owner.id, owner.team, target.id,
                                  50, 50, 0, 80, 50, age=10, z=.2, speed=2, elapsed=2)
                env.missiles.append(missile)
                env.step({})
                self.assertEqual(not target.alive, expected_hit)
                self.assertEqual(missile.phase, 'hit' if expected_hit else 'expired')
                if not expected_hit:
                    self.assertTrue(any(e['type']=='expired' and e['reason']=='ground'
                                        for e in env.events))

    def test_untracked_missile_does_not_circle_stale_point(self):
        config = Config(missile_motion='aerodynamic')
        for mover in (PythonMotion(), AerodynamicMissileMotion()):
            with self.subTest(backend=type(mover).__name__):
                pose = Pose(50, 50, 40, 0, 0, 2.0, 20)
                waypoint = Waypoint(49, 50, 41, guidance_enabled=False)
                for _ in range(50):
                    following = mover.advance(pose, waypoint, config)
                    self.assertGreater(following.x, pose.x)
                    self.assertLessEqual(following.z, pose.z)
                    self.assertAlmostEqual(following.heading, 0.0)
                    pose = following
                    if pose.z == 0:
                        self.assertEqual(pose.speed, 0.0)
                        break

    def test_coasting_still_applies_thrust_drag_and_gravity(self):
        config = Config(aero_missile_gravity=0.0, aero_missile_drag=0.0)
        pose = Pose(50, 50, 40, 0, 0, 2.0)
        waypoint = Waypoint(0, 0, 0, guidance_enabled=False)
        for mover in (PythonMotion(), AerodynamicMissileMotion()):
            boosted = mover.advance(pose, waypoint, config)
            self.assertGreater(boosted.speed, pose.speed)
            drag = mover.advance(pose, waypoint, replace(config, aero_missile_thrust=0,
                                                       aero_missile_drag=0.3))
            self.assertLess(drag.speed, pose.speed)
            falling = mover.advance(pose, waypoint, replace(config, aero_missile_thrust=0,
                                                          aero_missile_gravity=9.80665))
            self.assertLess(falling.z, pose.z)

    def test_guidance_uses_transverse_velocity_per_unit_range(self):
        config = Config(aero_length_scale=1, aero_tick_seconds=0.1, aero_substeps=1,
                        aero_missile_thrust=0, aero_missile_drag=0,
                        aero_missile_gravity=0, aero_missile_turn_drag=0,
                        aero_missile_turn_rate=10, aero_missile_acquisition_time=1e15,
                        aero_missile_max_lateral_acceleration=1e9)
        pose = Pose(0, 0, 1000, 0, 0, 10)
        for distance in (100, 1000, 10000):
            delta = np.array([distance, 0, distance/10])
            los = delta / np.linalg.norm(delta)
            velocity = np.array([100., 0., 0.])
            relative = -velocity
            closing = -np.dot(los, relative)
            transverse = relative - np.dot(relative, los)*los
            acceleration = config.missile_navigation_constant*closing*transverse/np.linalg.norm(delta)
            commanded = velocity + config.aero_tick_seconds*acceleration
            desired = commanded/np.linalg.norm(commanded)
            expected = math.atan2(desired[2], desired[0])
            for mover in (PythonMotion(), AerodynamicMissileMotion()):
                with self.subTest(distance=distance, backend=type(mover).__name__):
                    result = mover.advance(pose, Waypoint(distance, 0, 1000+distance/10), config)
                    self.assertAlmostEqual(result.pitch, expected, places=12)

    def test_acquisition_centers_current_los_instead_of_extrapolated_climb(self):
        config = Config(aero_length_scale=1, aero_tick_seconds=.1, aero_substeps=1,
                        aero_missile_thrust=0, aero_missile_drag=0,
                        aero_missile_gravity=0, aero_missile_turn_drag=0,
                        missile_navigation_constant=1e-15,
                        aero_missile_turn_rate=10, aero_missile_acquisition_time=1,
                        aero_missile_max_lateral_acceleration=1e9)
        pose = Pose(0, 0, 1000, 0, 0, 100)
        stationary = Waypoint(1000, 0, 1200)
        climbing = replace(stationary, velocity=(0, 0, 100))
        for mover in (PythonMotion(), AerodynamicMissileMotion()):
            with self.subTest(backend=type(mover).__name__):
                base = mover.advance(pose, stationary, config)
                rising = mover.advance(pose, climbing, config)
                self.assertAlmostEqual(rising.pitch, base.pitch, places=12)

    def test_lateral_limit_and_low_speed_authority(self):
        c = Config(aero_length_scale=1, aero_tick_seconds=.01, aero_substeps=1,
                   aero_missile_thrust=0, aero_missile_drag=0,
                   aero_missile_gravity=0, aero_missile_turn_drag=0,
                   aero_missile_turn_rate=100, aero_missile_acquisition_time=.01)
        for mover in (PythonMotion(), AerodynamicMissileMotion()):
            rates=[]
            for speed in (30,300,1200):
                pose=Pose(0,0,100,0,0,speed*.01)
                result=mover.advance(pose,Waypoint(100,1000,100),c)
                rate=abs(result.heading)/.01
                self.assertLessEqual(rate*speed,c.aero_missile_max_lateral_acceleration+1e-8)
                rates.append(rate)
            self.assertLess(rates[0],rates[1])
            self.assertLess(rates[2],rates[1])

    def test_stationary_target_intercept(self):
        c=Config(aero_length_scale=1,aero_tick_seconds=.1,
                 aero_missile_thrust=0,aero_missile_drag=0,
                 aero_missile_gravity=0,aero_missile_turn_drag=0)
        target=Waypoint(1000,100,1000)
        for mover in (PythonMotion(),AerodynamicMissileMotion()):
            pose=Pose(0,0,1000,0,0,30)
            minimum=1e9
            for _ in range(50):
                pose=mover.advance(pose,target,c)
                minimum=min(minimum,math.dist((pose.x,pose.y,pose.z),(target.x,target.y,target.z)))
            self.assertLess(minimum,20)

    def test_python_single_and_batch_agree(self):
        poses = [Pose(50, 50, 40, 0.1, 0.05, 2, elapsed)
                 for elapsed in (0, 2.98, 14.98, 25)]
        for substeps in (10, 20, 40):
            config = Config(aero_substeps=substeps, aero_missile_boost_seconds=3,
                            aero_missile_sustain_seconds=12,
                            aero_missile_boost_thrust=27000,
                            aero_missile_sustain_thrust=3200)
            for enabled in (False, True):
                waypoints = [Waypoint(100, 70, 44, (0.5, 0.1, 0.02), enabled)]*len(poses)
                batch = AerodynamicMissileMotion().advance_many(poses, waypoints, config)
                for pose, waypoint, actual in zip(poses, waypoints, batch):
                    expected = PythonMotion().advance(pose, waypoint, config)
                    single = AerodynamicMissileMotion().advance(pose, waypoint, config)
                    np.testing.assert_allclose(astuple(actual), astuple(expected), rtol=1e-11, atol=1e-11)
                    np.testing.assert_allclose(astuple(single), astuple(expected), rtol=1e-11, atol=1e-11)

    def test_lost_track_and_reacquisition_reach_motion_backend(self):
        for phase in ('midcourse', 'active'):
            with self.subTest(phase=phase):
                mover = RecordingMotion()
                config = Config(missile_motion='aerodynamic', radar_range=1,
                                seeker_range=10, seeker_fov=math.pi/2, hit_radius=.01,
                                missile_reacquisition_ramp_seconds=8.0)
                env = ToyEnv(config, aircraft_motion=FrozenAircraftMotion(), missile_motion=mover)
                env.reset(seed=123)
                owner, target = env.aircraft['blue_0'], env.aircraft['red_0']
                owner.x, owner.y, owner.z = 10, 10, 20
                target.x, target.y, target.z = 20, 50, 20
                missile = Missile('regression', owner.id, owner.team, target.id,
                                  50, 50, 0, 20, 50, age=10, phase=phase, z=20, speed=2)
                env.missiles.append(missile)
                env.step({})
                self.assertFalse(mover.waypoints[-1].guidance_enabled)
                target.x = 55
                env.step({})
                self.assertFalse(mover.waypoints[-1].guidance_enabled)
                self.assertEqual(env._missile_reacquisition_ticks[missile.id], 1)
                env.step({})
                self.assertTrue(mover.waypoints[-1].guidance_enabled)
                self.assertAlmostEqual(mover.waypoints[-1].guidance_strength, .125)
                env.step({})
                self.assertAlmostEqual(mover.waypoints[-1].guidance_strength, .25)
                env.step({})
                self.assertAlmostEqual(mover.waypoints[-1].guidance_strength, .375)
                env.step({})
                self.assertAlmostEqual(mover.waypoints[-1].guidance_strength, .5)
                for _ in range(4):
                    env.step({})
                self.assertAlmostEqual(mover.waypoints[-1].guidance_strength, 1.0)
                self.assertEqual(missile.phase, 'active')

    def test_fixed_recovery_duration_is_independent_of_altitude_and_speed(self):
        for altitude, speed, tick in ((1, 2, 1.), (40, .6, 1.), (20, 1, .5)):
            with self.subTest(altitude=altitude, speed=speed, tick=tick):
                mover = RecordingMotion()
                config = Config(missile_motion='aerodynamic', radar_range=1,
                                seeker_range=10, seeker_fov=math.pi/2, hit_radius=.01,
                                aero_tick_seconds=tick,
                                missile_reacquisition_ramp_seconds=20.,
                                missile_reacquisition_confirm_ticks=5)
                env = ToyEnv(config, aircraft_motion=FrozenAircraftMotion(), missile_motion=mover)
                env.reset(seed=123)
                owner, target = env.aircraft['blue_0'], env.aircraft['red_0']
                owner.x, owner.y, owner.z = 10, 10, altitude
                target.x, target.y, target.z = 20, 50, altitude
                missile = Missile('recovery', owner.id, owner.team, target.id,
                                  50, 50, 0, 20, 50, age=1, phase='active',
                                  z=altitude, speed=speed)
                env.missiles.append(missile)
                env.step({})
                target.x = 55
                for _ in range(4):
                    env.step({})
                    self.assertFalse(mover.waypoints[-1].guidance_enabled)
                for step in range(1, int(20/tick)+2):
                    env.step({})
                    self.assertTrue(mover.waypoints[-1].guidance_enabled)
                    self.assertAlmostEqual(mover.waypoints[-1].guidance_strength,
                                           min(1., step*tick/20))
                target.x = 20
                env.step({})
                self.assertEqual(mover.waypoints[-1].guidance_strength, 0.)
                target.x = 55
                for _ in range(5):
                    env.step({})
                self.assertAlmostEqual(mover.waypoints[-1].guidance_strength, tick/20)

    def test_single_tick_reacquisition_does_not_restore_guidance(self):
        mover = RecordingMotion()
        config = Config(missile_motion='aerodynamic', radar_range=1,
                        seeker_range=10, seeker_fov=math.pi/2, hit_radius=.01,
                        missile_reacquisition_confirm_ticks=2)
        env = ToyEnv(config, aircraft_motion=FrozenAircraftMotion(), missile_motion=mover)
        env.reset(seed=123)
        owner, target = env.aircraft['blue_0'], env.aircraft['red_0']
        owner.x, owner.y, owner.z = 10, 10, 20
        target.x, target.y, target.z = 20, 50, 20
        missile = Missile('flicker', owner.id, owner.team, target.id,
                          50, 50, 0, 20, 50, age=10, phase='active', z=20, speed=2)
        env.missiles.append(missile)
        env.step({})
        target.x = 55
        env.step({})
        self.assertFalse(mover.waypoints[-1].guidance_enabled)
        target.x = 20
        env.step({})
        self.assertFalse(mover.waypoints[-1].guidance_enabled)
        self.assertEqual(env._missile_reacquisition_ticks[missile.id], 0)

    def test_wingman_shared_radar_track_supports_midcourse_missile(self):
        for share_tracks, expected in ((True, True), (False, False)):
            with self.subTest(share_tracks=share_tracks):
                mover = RecordingMotion()
                config = Config(blue=2, red=1, missile_motion='aerodynamic',
                                share_tracks=share_tracks, radar_range=20,
                                radar_fov=math.pi, seeker_range=1,
                                seeker_fov=math.pi/2, hit_radius=.01)
                env = ToyEnv(config, aircraft_motion=FrozenAircraftMotion(), missile_motion=mover)
                env.reset(seed=123)
                owner = env.aircraft['blue_0']
                wingman = env.aircraft['blue_1']
                target = env.aircraft['red_0']
                owner.x, owner.y, owner.z, owner.heading = 10, 10, 20, 0
                wingman.x, wingman.y, wingman.z, wingman.heading = 90, 50, 20, 0
                target.x, target.y, target.z = 100, 50, 20
                missile = Missile('wingman-support', owner.id, owner.team, target.id,
                                  50, 50, 0, 100, 50, age=1, phase='coast', z=20, speed=2)
                env.missiles.append(missile)
                env.step({})
                self.assertFalse(mover.waypoints[-1].guidance_enabled)
                env.step({})
                self.assertEqual(mover.waypoints[-1].guidance_enabled, expected)
                self.assertEqual(missile.phase, 'midcourse' if expected else 'coast')

    def test_reacquisition_strength_limits_three_dimensional_turn(self):
        config = Config(aero_length_scale=1, aero_tick_seconds=.1, aero_substeps=1,
                        aero_missile_thrust=0, aero_missile_drag=0,
                        aero_missile_gravity=0, aero_missile_turn_drag=0,
                        aero_missile_turn_rate=10,
                        aero_missile_max_lateral_acceleration=1e9)
        pose = Pose(0, 0, 100, 0, 0, 100)
        full = Waypoint(100, 0, 200, guidance_strength=1.0)
        half = replace(full, guidance_strength=.5)
        for mover in (PythonMotion(), AerodynamicMissileMotion()):
            full_result = mover.advance(pose, full, config)
            half_result = mover.advance(pose, half, config)
            self.assertAlmostEqual(half_result.pitch, full_result.pitch/2, places=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
