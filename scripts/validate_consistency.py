"""Game/learning contract checks, including finite horizons and team symmetry."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from combatsim.decision import DecisionScheduler
from combatsim.env import Config, Missile, ToyEnv
from combatsim.factory import load_settings
from combatsim.inference import select_actions
from combatsim.observation_schema import SELF_INDEX
from combatsim.policies import RulePolicy
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.rllib_rating import RatingCoordinator
from combatsim.set_transformer import entity_features, SetTransformerActorCritic

ROOT = Path(__file__).resolve().parents[1]


class ConsistencyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_time_limit_is_terminal_draw(self):
        e = CombatSimFlightEnv({'settings': {'max_steps': 2, 'ammunition': 0,
                          'rewards': [{'type': 'victory', 'draw': -.1}]}})
        obs, _ = e.reset(seed=10)
        for _ in range(2):
            obs, rewards, terminated, truncated, _ = e.step({a: (5, 5, 0) for a in obs})
        self.assertTrue(all(terminated.values()))
        self.assertFalse(any(truncated.values()))
        self.assertIsNone(e.env.winner)
        self.assertTrue(all(v == -.1 for v in rewards.values()))

    def test_dead_aircraft_retains_time_and_only_noop(self):
        e = CombatSimFlightEnv({'settings': {'blue': 2, 'red': 1, 'max_steps': 6,
                          'decision_interval_steps': 2, 'ammunition': 0,
                          'rewards': [{'type': 'victory', 'draw': 1.}]}})
        obs, _ = e.reset(seed=10)
        e.env.aircraft['blue_0'].alive = False
        n = SetTransformerActorCritic().eval()
        collected = []
        for _ in range(3):
            obs, rewards, terminated, _, _ = e.step({a: (5, 5, 0) for a in obs})
            collected.append(rewards['blue_0'])
            dead = obs['blue_0']
            self.assertEqual(dead['self'][0, SELF_INDEX['alive']], 0)
            self.assertAlmostEqual(float(dead['common'][0, 0]), e.env.tick/6)
            for kind in ('allies', 'contacts', 'missiles', 'friendly_missiles'):
                self.assertEqual(dead[kind+'_mask'].sum(), 0)
            np.testing.assert_array_equal(select_actions(n, obs, ['blue_0']), [[0, 0, 0]])
            self.assertEqual(terminated['blue_0'], e.env.tick == 6)
        self.assertEqual(collected, [0., 0., 1.])

    def test_rule_and_neural_decide_at_same_frequency(self):
        e = CombatSimFlightEnv({'settings': {'decision_interval_steps': 3, 'max_steps': 10}})
        obs, _ = e.reset(seed=9, options={'rule_side': 'red'})
        calls = []
        original = e.env.step
        def capture(actions, **kwargs):
            calls.append(dict(actions))
            return original(actions, **kwargs)
        with patch.object(e.rule_policy, 'act', return_value=(5, 5, 1)) as decide, \
                patch.object(e.env, 'step', side_effect=capture):
            e.step({'blue_0': (5, 5, 1)})
        self.assertEqual(decide.call_count, 1)
        self.assertEqual([row['red_0'][2] for row in calls], [1, 0, 0])
        scheduler = DecisionScheduler(3)
        with patch.object(e.rule_policy, 'act', return_value=(5, 5, 1)) as decide:
            raw = e.env.observations()['red_0']
            for tick in range(6):
                scheduler.policy_action(e.rule_policy, raw, tick)
        self.assertEqual(decide.call_count, 2)

    def test_paired_spawns_and_rule_actions_are_symmetric(self):
        settings = load_settings(ROOT/'configs/scenario-default.json')
        settings['paired_spawn'] = True
        settings['ammunition'] = 0
        for spec in settings['aircraft']:
            spec.setdefault('performance', {})['ammunition'] = 0
        for seed in (3, 17, 123):
            e = ToyEnv(Config(**settings)); obs, _ = e.reset(seed=seed)
            policy = RulePolicy(settings['rule_policy'], seed=seed)
            for _ in range(30):
                actions = {a: policy.act(raw) for a, raw in obs.items()}
                for i in range(2):
                    b, r = f'blue_wing{i}', f'red_wing{i}'
                    self.assertEqual(actions[b], actions[r])
                    left, right = entity_features(obs[b]), entity_features(obs[r])
                    for key in left:
                        np.testing.assert_allclose(left[key], right[key], atol=1e-5)
                obs, *_ = e.step(actions)

    def test_friendly_telemetry_relates_only_visible_contacts(self):
        e = ToyEnv(Config(radar_range=1000., radar_fov=2*math.pi))
        e.reset(seed=4)
        b, r = e.aircraft['blue_0'], e.aircraft['red_0']
        m = Missile('m-test', b.id, b.team, r.id, b.x, b.y, b.heading, r.x, r.y,
                    z=b.z, speed=2., phase='active')
        e.missiles.append(m); e._missile_tracking[m.id] = False
        raw = e.observations()[b.id]; features = entity_features(raw)
        self.assertEqual(raw['friendly_missiles'][0]['phase'], 'active')
        self.assertFalse(raw['friendly_missiles'][0]['tracking'])
        self.assertEqual(features['missile_targets'].sum(), 1.)
        e.config.radar_range = .001
        raw = e.observations()[b.id]; features = entity_features(raw)
        self.assertIsNone(raw['friendly_missiles'][0]['target_track'])
        self.assertEqual(features['missile_targets'].sum(), 0.)

    def test_full_match_rotates_with_team_swap_and_common_random_numbers(self):
        settings = load_settings(ROOT/'configs/scenario-default.json')
        settings['paired_spawn'] = True
        settings['max_steps'] = 180
        n = SetTransformerActorCritic().eval()
        envs = [CombatSimFlightEnv({'settings': deepcopy(settings)}) for _ in range(2)]
        observations = [envs[0].reset(seed=31, options={'rule_side': 'red', 'rule_profile': 'fix'})[0],
                        envs[1].reset(seed=31, options={'rule_side': 'blue', 'rule_profile': 'fix'})[0]]
        generators = [torch.Generator().manual_seed(7) for _ in envs]
        states = [{}, {}]
        while not envs[0].env.done:
            for i, e in enumerate(envs):
                ids = sorted(observations[i])
                actions = select_actions(n, observations[i], ids, 'sample', generators[i], states[i])
                observations[i], *_ = e.step(dict(zip(ids, actions)))
            self.assertEqual(envs[0].env.done, envs[1].env.done)
            for side in ('blue', 'red'):
                opposite = 'red' if side == 'blue' else 'blue'
                for slot in range(2):
                    a = envs[0].env.aircraft[f'{side}_wing{slot}']
                    b = envs[1].env.aircraft[f'{opposite}_wing{slot}']
                    self.assertEqual(a.alive, b.alive)
                    np.testing.assert_allclose([a.x, a.y, a.z],
                        [settings['arena_x']-b.x, settings['arena_y']-b.y, b.z], atol=1e-6)

    def test_override_observation_is_interval_fraction(self):
        e = CombatSimFlightEnv({'settings': {'decision_interval_steps': 3,
                                       'boundary_override': {'enabled': True}}})
        e.reset(seed=1)
        e.env.aircraft['blue_0'].z = e.config.ceiling-.01
        obs, *_ = e.step({'blue_0': (5, 8, 0), 'red_0': (5, 5, 0)})
        counts = e.episode_control_counts['blue_0']
        self.assertEqual(counts['ticks'], 3)
        self.assertGreater(counts['overrides'], 0)
        self.assertAlmostEqual(float(obs['blue_0']['self'][0, SELF_INDEX['boundary_override_fraction']]),
                               counts['overrides']/counts['ticks'])

    def test_explicit_seed_resets_all_policy_randomness(self):
        e = CombatSimFlightEnv({'settings': {'rule_policy_profiles': {'fix': {}, 'random': {}}}, 'rule_ratio': 1., 'random_rate': .5})
        first = e.reset(seed=81)[0]; side, profile = e.rule_side, e.rule_profile
        second = e.reset(seed=81)[0]
        self.assertEqual((side, profile), (e.rule_side, e.rule_profile))
        for agent in first:
            for key in first[agent]:
                np.testing.assert_array_equal(first[agent][key], second[agent][key])

    def test_rule_ratings_and_returns_remain_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            c = RatingCoordinator.__ray_metadata__.modified_class(str(Path(folder)/'log'), 24, 0, 1, 10)
            for profile, winner in (('fix', 'blue'), ('random', 'red')):
                c.report({'episode_id': profile, 'winner': winner, 'rule_side': 'red',
                          'rule_profile': profile, 'snapshot_side': None, 'snapshot_policy_id': None,
                          'simulation_steps': 3, 'learning_returns_by_side': {'blue': 1 if winner == 'blue' else -1}})
            self.assertGreater(c.rule_profile_ratings['fix'], 1000)
            self.assertLess(c.rule_profile_ratings['random'], 1000)
            self.assertEqual(c.state()['matchup_statistics']['rule:fix:blue']['mean_return'], 1)
            restored = RatingCoordinator.__ray_metadata__.modified_class(
                str(Path(folder)/'resume'), 24, 0, 1, 10, c.export_state())
            self.assertEqual(restored.state(), c.state())

    def test_recurrent_rollouts_accept_odd_and_different_team_sizes(self):
        from combatsim.tournament import settings_for_team_sizes
        base = load_settings(ROOT/'configs/scenario-default.json')
        n = SetTransformerActorCritic().eval()
        for blue, red in ((1, 1), (3, 3), (4, 4), (3, 2)):
            settings = settings_for_team_sizes(base, blue, red)
            settings['max_steps'] = 12
            e = CombatSimFlightEnv({'settings': settings})
            obs, _ = e.reset(seed=9); states = {}
            while not e.env.done:
                ids = sorted(obs)
                actions = select_actions(n, obs, ids, states=states)
                obs, *_ = e.step(dict(zip(ids, actions)))
                self.assertEqual(len(obs), blue+red)
                self.assertTrue(all(torch.isfinite(h).all() for pair in states.values() for h in pair))


if __name__ == '__main__':
    unittest.main(verbosity=2)
