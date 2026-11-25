"""Check observation reuse against fresh generation and mutable world state."""
from collections import Counter
import copy
import math
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

from combatsim.env import Config, ToyEnv
from combatsim.factory import load_settings
from combatsim.rllib_env import CombatSimFlightEnv
from scripts.validate_rewards import ROOT


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


class ObservationChecks(TestCase):
    def test_reuse_matches_fresh_adapter_across_control_modes(self):
        for interval in (1, 3):
            for side in (None, 'blue', 'red'):
                for shared in (False, True):
                    with self.subTest(interval=interval, rule_side=side, share_tracks=shared):
                        settings = load_settings(ROOT/'configs'/'scenario-default.json')
                        settings.update(max_steps=75, decision_interval_steps=interval, share_tracks=shared)
                        config = {'settings': settings, 'rule_ratio': 0}
                        reused, fresh = CombatSimFlightEnv(copy.deepcopy(config)), CombatSimFlightEnv(copy.deepcopy(config))
                        a = reused.reset(seed=2001, options={'rule_side': side})
                        b = fresh.reset(seed=2001, options={'rule_side': side})
                        self.assertEqual(plain(a), plain(b))
                        fresh_step = fresh.env.step
                        with patch.object(fresh.env, 'step', side_effect=lambda actions, **kw: fresh_step(actions)):
                            while not reused.env.done:
                                actions = {agent: (5, 5, 1) for agent in a[0]}
                                a, b = reused.step(actions), fresh.step(actions)
                                self.assertEqual(plain(a), plain(b))
                                self.assertEqual(reused.env.snapshot(), fresh.env.snapshot())
                                self.assertEqual(reused.env._sensor_tracks, fresh.env._sensor_tracks)
                        # Reset must discard observations and track associations from the last match.
                        self.assertEqual(plain(reused.reset(seed=2002)), plain(fresh.reset(seed=2002)))

    def test_only_one_observation_generation_per_physics_tick(self):
        for side in (None, 'red'):
            with self.subTest(rule_side=side):
                settings = load_settings(ROOT/'configs'/'scenario-default.json')
                settings.update(max_steps=8, decision_interval_steps=3)
                env = CombatSimFlightEnv({'settings': settings, 'rule_ratio': 0})
                observations, _ = env.reset(seed=2001, options={'rule_side': side})
                with patch.object(env.env, 'observations', wraps=env.env.observations) as generate:
                    while not env.env.done:
                        before = env.env.tick
                        result = env.step({agent: (5, 5, 1) for agent in observations})
                        self.assertEqual(generate.call_count, env.env.tick-before)
                        generate.reset_mock()
                        observations = result[0]

    def test_local_caches_do_not_survive_calls_or_hide_mutations(self):
        env = ToyEnv(Config(blue=2, red=2, radar_range=1000, radar_fov=2*math.pi))
        env.reset(seed=3)
        original_radar, original_velocity = env.radar_visible, env._velocity
        radar_calls, velocity_calls = Counter(), Counter()

        def radar(observer, target):
            radar_calls[id(observer), id(target)] += 1
            return original_radar(observer, target)

        def velocity(body):
            velocity_calls[id(body)] += 1
            return original_velocity(body)

        with patch.object(env, 'radar_visible', side_effect=radar), patch.object(env, '_velocity', side_effect=velocity):
            old = env.observations()
            self.assertTrue(radar_calls)
            self.assertEqual(max(radar_calls.values()), 1)
            self.assertEqual(max(velocity_calls.values()), 1)
            env.aircraft['blue_0'].speed = 2
            env.aircraft['blue_0'].heading = math.pi/2
            env.config.radar_range = .01
            updated = env.observations()
            self.assertNotEqual(old['blue_0']['self']['velocity'], updated['blue_0']['self']['velocity'])
            self.assertEqual(updated['blue_0']['contacts'], [])
            self.assertEqual(max(velocity_calls.values()), 2)

    def test_public_step_generates_fresh_decision_observations(self):
        env = ToyEnv(Config())
        env.reset(seed=3)
        with patch.object(env, 'observations', wraps=env.observations) as generate:
            env.aircraft['blue_0'].speed = 2
            env.step({'blue_0': (5, 5, 0), 'red_0': (5, 5, 0)})
            self.assertEqual(generate.call_count, 2)

    def test_hud_contacts_match_sensors_without_changing_policy_tracks(self):
        env = ToyEnv(Config(blue=2, red=2, radar_range=50, radar_fov=.5, share_tracks=True))
        env.reset(seed=3)
        for ident, x in [('blue_0', 100), ('blue_1', 110), ('red_0', 140), ('red_1', 300)]:
            a = env.aircraft[ident]
            a.x, a.y, a.z, a.heading, a.pitch = x, 100, 20, 0, 0

        def contacts():
            raw = env.observations()['blue_0']
            tracks, numbers = copy.deepcopy(env._sensor_tracks), dict(env._next_sensor_track)
            with patch.object(env, 'observations', side_effect=AssertionError('snapshot must not generate policy tracks')):
                snapshot = env.snapshot()
            self.assertEqual(env._sensor_tracks, tracks)
            self.assertEqual(env._next_sensor_track, numbers)
            hud = {a['id']: a['hud_contacts'] for a in snapshot['aircraft']}['blue_0']
            self.assertEqual(sum(c['source'] != 'ally' for c in hud), len(raw['contacts']))
            for source, detected in [('radar', True), ('datalink', False)]:
                self.assertEqual(sum(c['source'] == source for c in hud),
                                 sum(c['detected_by_self'] == detected for c in raw['contacts']))
            return hud

        self.assertEqual(contacts(), [{'id': 'blue_1', 'source': 'ally'}, {'id': 'red_0', 'source': 'radar'}])
        env.aircraft['blue_0'].heading = math.pi
        self.assertEqual(contacts(), [{'id': 'blue_1', 'source': 'ally'}, {'id': 'red_0', 'source': 'datalink'}])
        env.config.share_tracks = False
        self.assertEqual(contacts(), [{'id': 'blue_1', 'source': 'ally'}])
        env.config.share_tracks = True
        env.aircraft['blue_1'].alive = False
        self.assertEqual(contacts(), [])
        env.aircraft['blue_0'].heading = 0
        env.aircraft['red_0'].alive = False
        self.assertEqual(contacts(), [])
        env.aircraft['blue_0'].alive = False
        self.assertEqual(next(a for a in env.snapshot()['aircraft'] if a['id'] == 'blue_0')['hud_contacts'], [])


if __name__ == '__main__':
    main(verbosity=2)
