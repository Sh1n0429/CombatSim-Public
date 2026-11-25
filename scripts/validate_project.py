"""Regression checks for the project-wide training/evaluation audit."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from ray.rllib.env.env_context import EnvContext

from combatsim.env import Config, ToyEnv
from combatsim.factory import load_settings
from combatsim.policies import RulePolicy
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.rating import update_rating
from combatsim.rllib_rating import RatingCoordinator, RLLibRatingCallbacks
from combatsim.rllib_logging import iteration_record, format_iteration
from combatsim.config_defaults import APPLICATION_DEFAULTS
from combatsim.rllib_evaluation import evaluate_network
from combatsim.tournament import MODEL_FORMAT, file_sha256, load_model, team_assignments
from combatsim.observation_schema import OBSERVATION_VERSION
from combatsim.motion import AerodynamicMissileMotion, Pose, Waypoint
from combatsim.set_transformer import SetTransformerActorCritic
from scripts.train import parse_args

ROOT=Path(__file__).resolve().parents[1]

class ProjectChecks(unittest.TestCase):
    def test_resume_resolves_newer_numbered_checkpoint(self):
        from combatsim.checkpoints import resolve_resume_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            numbered = root/'iteration-000002'
            numbered.mkdir()
            (root/'training-state.json').touch()
            (numbered/'training-state.json').touch()
            pointer = root/'latest-checkpoint.txt'
            pointer.write_text(numbered.name, encoding='utf-8')
            self.assertEqual(resolve_resume_checkpoint(root), numbered)
            self.assertEqual(resolve_resume_checkpoint(pointer), numbered)
            self.assertEqual(resolve_resume_checkpoint(numbered), numbered)

    def test_boundary_penalty_caps_per_aircraft_and_matches_hooks(self):
        from dataclasses import replace
        from combatsim.rewards import RewardContext, RewardSystem
        from combatsim.reward_hooks import ORIGINAL_REWARD_REASONS
        from scripts.validate_rewards import sample_state
        from dataclasses import asdict
        specs = json.loads((ROOT/'configs/rewards.json').read_text(encoding='utf-8'))['rewards']
        boundary = next((spec for spec in specs if spec['type'] == 'boundary_override'), None)
        if boundary is None:
            boundary = {'type': 'boundary_override'}
            specs.append(boundary)
        boundary.update(penalty=-.03, max_total=.05)
        normal = RewardSystem(specs)
        with patch('combatsim.config_defaults.read_defaults', return_value={'rewards': specs}):
            hooked = RewardSystem([{'type': 'hooks', 'profile': 'original'}])
        aircraft = {a.id: SimpleNamespace(**asdict(a)) for a in sample_state().aircraft}
        positions = {k: (a.x, a.y, a.z) for k, a in aircraft.items()}
        horizontal = {'type': 'boundary_override', 'aircraft': 'blue_0', 'horizontal': True, 'lower': True}
        vertical = {'type': 'boundary_override', 'aircraft': 'blue_1', 'lower': True}
        context = RewardContext(Config(), aircraft, positions, False, None,
                                events=(horizontal,), tick=1)
        for tick, expected in enumerate((-.03, -.02, 0., 0.), start=1):
            context = replace(context, tick=tick, events=(horizontal, vertical) if tick == 3 else (horizontal,))
            self.assertEqual(normal.calculate(context), hooked.calculate(context))
            values = hooked.last_components.get('HookReward:boundary_override', {})
            self.assertAlmostEqual(values.get('blue_0', 0), expected)
            if tick == 3:
                self.assertEqual(values['blue_1'], -.03)
            else:
                self.assertNotIn('blue_1', values)
        context = replace(context, events=({'type': 'boundary_override',
                                            'aircraft': 'blue_1', 'horizontal': True},))
        self.assertEqual(normal.calculate(context), hooked.calculate(context))
        self.assertAlmostEqual(hooked.last_components['HookReward:boundary_override']['blue_1'], -.02)
        normal.reset(); hooked.reset()
        self.assertEqual(normal.calculate(context), hooked.calculate(context))
        self.assertEqual(hooked.last_components['HookReward:boundary_override']['blue_1'], -.03)
        self.assertIn('boundary_override', ORIGINAL_REWARD_REASONS)
        for invalid in ({'penalty': .01}, {'max_total': 0}, {'penalty': float('nan')}):
            with self.assertRaises(ValueError):
                RewardSystem([{'type': 'boundary_override', **invalid}])

    def test_outside_penalty_matches_hooks_and_ignores_assistance(self):
        from dataclasses import replace, asdict
        from combatsim.rewards import RewardContext, RewardSystem
        from scripts.validate_rewards import sample_state
        specs = json.loads((ROOT/'configs/rewards.json').read_text(encoding='utf-8'))['rewards']
        spec = next(x for x in specs if x['type'] == 'out_of_bounds')
        # Isolate outside costs even when the baseline enables intervention costs.
        specs = [x for x in specs if x['type'] != 'boundary_override']
        spec.update(penalty=-.03, max_total=.05)
        for mode in ('per_step', 'on_exit'):
            spec['mode'] = mode
            normal = RewardSystem(specs)
            with patch('combatsim.config_defaults.read_defaults', return_value={'rewards': specs}):
                hooked = RewardSystem([{'type': 'hooks', 'profile': 'original'}])
            aircraft = {a.id: SimpleNamespace(**asdict(a)) for a in sample_state().aircraft}
            c = Config()
            for a in aircraft.values():
                a.x, a.y, a.z = c.arena_x/2, c.arena_y/2, c.ceiling/2
            positions = {k: (a.x, a.y, a.z) for k, a in aircraft.items()}
            ctx = RewardContext(c, aircraft, positions, False, None,
                events=({'type': 'boundary_override', 'aircraft': 'blue_0'},), tick=1)
            self.assertEqual(normal.calculate(ctx), hooked.calculate(ctx))
            self.assertNotIn('HookReward:boundary_override', hooked.last_components)
            self.assertNotIn('HookReward:out_of_bounds', hooked.last_components)
            aircraft['blue_0'].x = -1
            aircraft['blue_0'].z = c.ceiling+1
            aircraft['blue_1'].z = c.ceiling+1
            self.assertEqual(normal.calculate(ctx), hooked.calculate(ctx))
            self.assertEqual(hooked.last_components['HookReward:out_of_bounds'], {'blue_0': -.03, 'blue_1': -.03})
            outside = {k: (a.x, a.y, a.z) for k, a in aircraft.items()}
            ctx = replace(ctx, previous_positions=outside, tick=2)
            self.assertEqual(normal.calculate(ctx), hooked.calculate(ctx))
            expected = -.02 if mode == 'per_step' else 0
            self.assertAlmostEqual(hooked.last_components.get('HookReward:out_of_bounds', {}).get('blue_0', 0), expected)
            normal.reset(); hooked.reset()
            ctx = replace(ctx, previous_positions=positions, tick=3)
            self.assertEqual(normal.calculate(ctx), hooked.calculate(ctx))
            self.assertEqual(hooked.last_components['HookReward:out_of_bounds']['blue_0'], -.03)

    def test_episode_budget_crash_and_unlimited_outside_parity(self):
        from dataclasses import asdict, replace
        from combatsim.rewards import RewardContext, RewardSystem
        from scripts.validate_rewards import sample_state
        specs = json.loads((ROOT/'configs/rewards.json').read_text(encoding='utf-8'))['rewards']
        budget = next(x['max_total'] for x in specs if x['type'] == 'time_penalty')
        for horizon in (10, 1200):
            normal = RewardSystem(specs)
            hooked = RewardSystem([{'type': 'hooks', 'profile': 'original'}])
            aircraft = {a.id: SimpleNamespace(**asdict(a)) for a in sample_state().aircraft}
            cfg = Config(max_steps=horizon)
            for a in aircraft.values():
                a.x, a.y, a.z = cfg.arena_x/2, cfg.arena_y/2, cfg.ceiling/2
            aircraft['blue_0'].x = -1
            total = 0
            for tick in range(1, horizon+1):
                positions = {k: (a.x,a.y,a.z) for k,a in aircraft.items()}
                ctx = RewardContext(cfg, aircraft, positions, False, None, tick=tick, max_steps=horizon)
                left, right = normal.calculate(ctx), hooked.calculate(ctx)
                for agent in left:
                    self.assertAlmostEqual(left[agent], right[agent])
                total += hooked.last_components['HookReward:time_penalty']['blue_0']
                self.assertAlmostEqual(hooked.last_components['HookReward:out_of_bounds']['blue_0'], -.003)
                if tick < horizon:
                    self.assertGreater(total, -budget)
            self.assertAlmostEqual(total, -budget)
            normal.reset(); hooked.reset()
            aircraft['blue_0'].alive = False
            ctx = replace(ctx, tick=1, crashed=('blue_0',))
            self.assertEqual(normal.calculate(ctx), hooked.calculate(ctx))
            self.assertEqual(hooked.last_components['HookReward:friendly_loss'], {'blue_0': -.3, 'blue_1': -.3})
            self.assertNotIn('HookReward:enemy_kill', hooked.last_components)
            self.assertNotIn('blue_0', hooked.last_components['HookReward:time_penalty'])
        for kwargs in ({'mode':'invalid'}, {'mode':'episode_budget','max_total':None}):
            with self.assertRaises(ValueError):
                RewardSystem([{'type':'time_penalty', **kwargs}])

    def test_recent_win_rate_excludes_selfplay_and_survives_resume(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(
                APPLICATION_DEFAULTS['training'], {'win_rate_window': 3}):
            coordinator_class = RatingCoordinator.__ray_metadata__.modified_class
            c = coordinator_class(str(Path(folder)/'matches.jsonl'), 24, 0, 1, 100)
            def report(winner, matchup='rule'):
                c.report({'winner': winner, 'rule_side': 'red' if matchup == 'rule' else None,
                          'snapshot_side': 'red' if matchup == 'snapshot' else None,
                          'snapshot_policy_id': 'snapshot_0' if matchup == 'snapshot' else None,
                          'simulation_steps': 1})
            report('blue')
            report('blue', 'selfplay')
            report('red', 'snapshot')
            report(None)
            report('blue')
            record = iteration_record({'training_iteration': 1}, c.state(), 'test')
            self.assertEqual(record['opponent_results']['games'], 4)
            self.assertEqual(record['opponent_results']['win_rate'], .5)
            self.assertEqual(record['recent_opponent_results'], {
                'games': 3, 'wins': 1, 'draws': 1, 'losses': 1, 'win_rate': 1/3, 'window': 3})
            self.assertIn('WIN RATE 50.0%', format_iteration(record))
            self.assertIn('WIN RATE 33.3%', format_iteration(record))
            restored = coordinator_class(str(Path(folder)/'resume.jsonl'), 24, 0, 1, 100,
                                         json.loads(json.dumps(c.export_state())))
            self.assertEqual(c.state(), restored.state())
            legacy = c.export_state().copy()
            legacy.pop('recent_reference_results')
            old = coordinator_class(str(Path(folder)/'legacy.jsonl'), 24, 0, 1, 100, legacy)
            self.assertEqual(old.state()['recent_reference_results'], [])
            self.assertEqual(old.state()['reference_results'], c.state()['reference_results'])

    def test_win_rate_empty_partial_and_tensorboard(self):
        from scripts.train import write_tensorboard_iteration
        from unittest.mock import Mock
        state = {'completed': 0, 'main_rating': 1000}
        record = iteration_record({'training_iteration': 1}, state, 'test')
        self.assertIsNone(record['recent_opponent_results']['win_rate'])
        self.assertIn('WIN RATE -', format_iteration(record))
        json.dumps(record, allow_nan=False)
        state.update(recent_reference_results=['draw', 'win'], reference_results={'draw': 1, 'win': 1})
        record = iteration_record({'training_iteration': 2}, state, 'test')
        self.assertEqual(record['recent_opponent_results']['games'], 2)
        self.assertEqual(record['recent_opponent_results']['win_rate'], .5)
        writer = Mock()
        write_tensorboard_iteration(writer, record)
        writer.add_scalar.assert_any_call('training_opponents/main/recent/win_rate', .5, 2)
        writer.add_scalar.assert_any_call('training_opponents/main/all/win_rate', .5, 2)

    def test_explicit_fix_without_profile_does_not_sample_curriculum(self):
        settings = {'blue': 1, 'red': 1,
                    'rule_policy_profiles': {'random': {'fire_range': 12.}}}
        environment = CombatSimFlightEnv({'settings': settings, 'rule_ratio': 1, 'random_rate': 1.})
        with patch('combatsim.rllib_env.RulePolicy', wraps=RulePolicy) as policy:
            environment.reset(seed=17, options={'rule_side': 'red', 'rule_profile': 'fix'})
            self.assertEqual(environment.rule_profile, 'fix')
            self.assertEqual(policy.call_args.args[0], environment.config.rule_policy)
            environment.reset(seed=17, options={'rule_side': 'red'})
            self.assertEqual(environment.rule_profile, 'random')
            self.assertEqual(policy.call_args.args[0]['fire_range'], 12.)

    def test_spawn_spacing_is_reproducible_and_bounded(self):
        import math
        settings = load_settings(ROOT/'configs/scenario-default.json')
        env = ToyEnv(Config(**settings))
        for seed in range(100):
            env.reset(seed=seed)
            first = [(a.x, a.y, a.z) for a in env.aircraft.values()]
            for team in ('blue', 'red'):
                allies = [a for a in env.aircraft.values() if a.team == team]
                self.assertGreaterEqual(math.hypot(allies[0].x-allies[1].x,
                                                  allies[0].y-allies[1].y), 30.)
            env.reset(seed=seed)
            self.assertEqual(first, [(a.x, a.y, a.z) for a in env.aircraft.values()])
        # Impossible spacing must finish and respect the configured spawn box.
        cramped = ToyEnv(Config(blue=4, spawn_min_horizontal_distance=1000.,
                                blue_spawn=((.2, .2), (.4, .4), (.5, .5))))
        self.assertTrue(all((a.x, a.y, a.z) == (20., 40., 20.)
                            for a in cramped.aircraft.values() if a.team == 'blue'))

    def test_metrics_exclude_frozen_opponent_rewards(self):
        environment=CombatSimFlightEnv({'settings':{'blue':1,'red':1}})
        environment.reset(seed=2001)
        environment.env.episode_returns={'blue_0':2.,'red_0':-2.}
        environment.episode_reward_reasons={'win':{'blue_0':1.,'red_0':-1.}}
        environment.episode_reward_clipping_by_agent['red_0']['events']=5
        episode=SimpleNamespace(episode_id='test',custom_metrics={},
                                user_data={'snapshot_side':'red','snapshot_policy_id':'snapshot_0'})
        base=SimpleNamespace(get_sub_environments=lambda:[environment])
        actor=SimpleNamespace(report=SimpleNamespace(remote=lambda _:None))
        callback=RLLibRatingCallbacks()
        with patch.object(callback,'_actor',return_value=actor), \
                patch('combatsim.rllib_rating.ray.get',return_value={'rating':1000}), \
                patch('combatsim.rllib_rating.format_match',return_value=''):
            callback.on_episode_end(episode=episode,base_env=base,env_index=0)
        self.assertEqual(episode.custom_metrics['learning_return'],2.)
        self.assertEqual(episode.custom_metrics['reward/original'],1.)
        for reason in ('boundary_penalty', 'friendly_loss', 'loss', 'time_penalty'):
            self.assertEqual(episode.custom_metrics[f'reward_reason/{reason}'], 0.0)
        self.assertNotIn('reward_reason/boundary_override', episode.custom_metrics)
        self.assertNotIn('reward_reason/out_of_bounds', episode.custom_metrics)
        self.assertEqual(episode.custom_metrics['reward_clipping/events'],0)
        record=iteration_record({'training_iteration':1,'env_runners':{'custom_metrics':
            {'learning_return_mean':2.},'episode_return_mean':0.}},
            {'completed':1,'main_rating':1000},'test')
        self.assertEqual(record['mean_return'],2.)
    def test_elo_draw_moves_toward_opponent_rating(self):
        self.assertEqual(update_rating(1000,.5,1000),1000)
        self.assertGreater(update_rating(900,.5,1100),900)
        self.assertLess(update_rating(1100,.5,900),1100)
        self.assertAlmostEqual(update_rating(1000,1,1000),1012)
    def test_snapshot_updates_wait_for_inflight_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            coordinator=RatingCoordinator.__ray_metadata__.modified_class(
                str(Path(temporary)/'matches.jsonl'),24,1,1,1)
            choice=coordinator.choose_snapshot('episode-1')
            self.assertEqual(choice['policy_id'],'snapshot_0')
            coordinator.pending=1
            self.assertIsNone(coordinator.commit_snapshot())
            self.assertIsNone(coordinator.choose_snapshot('episode-2'))
            self.assertEqual(coordinator.pending,1)
            coordinator.report({'episode_id':'episode-1','winner':None,'rule_side':None,
                'snapshot_side':choice['side'],'snapshot_policy_id':choice['policy_id'],'simulation_steps':1})
            self.assertIsNotNone(coordinator.commit_snapshot())
            self.assertEqual(coordinator.pending,1)  # This completed match queued the next capture.
    def test_training_seeds_vary_and_reproduce_across_workers(self):
        def environment(worker=1,vector=0):
            return CombatSimFlightEnv(EnvContext({'settings':{'blue':1,'red':1},'seed':123},
                                             worker_index=worker,vector_index=vector))
        def positions(env,seed=None):
            env.reset(seed=seed)
            return [(a.x,a.y,a.z) for a in env.env.aircraft.values()]
        left,right=environment(),environment()
        sequence=[positions(left) for _ in range(3)]
        self.assertEqual(sequence,[positions(right) for _ in range(3)])
        self.assertEqual(len({str(item) for item in sequence}),3)
        self.assertNotEqual(sequence[0],positions(environment(worker=2)))
        self.assertNotEqual(sequence[0],positions(environment(vector=1)))
        self.assertEqual(positions(left,2001),positions(environment(worker=3),2001))
    def test_cli_paths_are_fixed_before_ray_changes_directory(self):
        args=parse_args(['--experiment',str(ROOT/'experiments/smoke.yaml'),
                         '--checkpoint-dir','artifacts/project-audit/checkpoint',
                         '--metrics-log','artifacts/project-audit/training.jsonl'])
        self.assertEqual(args.checkpoint_dir,str((Path.cwd()/'artifacts/project-audit/checkpoint').resolve()))
        self.assertTrue(Path(args.metrics_log).is_absolute())
        self.assertTrue(Path(args.eval_log).is_absolute())
    def test_model_rejects_nonfinite_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            weights=SetTransformerActorCritic().state_dict()
            first=next(iter(weights));weights[first].view(-1)[0]=float('nan')
            torch.save(weights,folder/'model.pth')
            (folder/'metadata.json').write_text(json.dumps({'model_id':'bad', 'format':MODEL_FORMAT,
                'observation_version':OBSERVATION_VERSION,'sha256':file_sha256(folder/'model.pth')}))
            with self.assertRaisesRegex(ValueError,'NaN'):load_model(folder)
    def test_evaluation_rejects_empty_and_duplicate_seeds(self):
        network=SetTransformerActorCritic()
        for seeds in ([],[1,1],[-1],[True]):
            with self.assertRaises(ValueError):evaluate_network(network,{},seeds)
    def test_ballistic_trajectory_matches_si_freefall(self):
        mover=AerodynamicMissileMotion()
        endpoints=[]
        for tick in (.5,1.):
            config=Config(aero_tick_seconds=tick,aero_length_scale=100.,
                          aero_missile_thrust=0.,aero_missile_drag=0.)
            pose=Pose(0.,0.,10.,0.,0.,speed=4*tick)
            for _ in range(round(5/tick)):
                pose=mover.advance(pose,Waypoint(100.,0.,10.,guidance_enabled=False),config)
            self.assertAlmostEqual(pose.x,20.,places=9)
            self.assertAlmostEqual(pose.z,10-.5*config.aero_missile_gravity*25/100,places=9)
            endpoints.append((pose.x,pose.y,pose.z))
        np.testing.assert_allclose(*endpoints,atol=1e-10)
    def test_model_team_follows_scenario_roster(self):
        for name in ('scenario-default.json','scenario-tournament.json'):
            settings=load_settings(ROOT/'configs'/name)
            assignments=team_assignments('A','B',settings['aircraft'])
            self.assertEqual(set(assignments),{a['id'] for a in settings['aircraft']})
            self.assertTrue(all(assignments[a['id']]==('A' if a['team']=='blue' else 'B') for a in settings['aircraft']))
    def test_rule_flights_remain_finite_and_terminal(self):
        # Exercise complete multi-missile trajectories, including guidance loss,
        # turns, ground contact, boundary recovery and simultaneous impacts.
        for seed in range(2001,2007):
            settings=load_settings(ROOT/'configs/scenario-default.json')
            env=ToyEnv(Config(**settings));observations,_=env.reset(seed=seed)
            policy=RulePolicy(settings['rule_policy'],seed=seed)
            while not env.done:
                actions={agent:policy.act(obs) for agent,obs in observations.items() if env.aircraft[agent].alive}
                observations,rewards,_,_,_=env.step(actions,decision_observations=observations)
                self.assertTrue(all(np.isfinite(value) for value in rewards.values()))
                for body in [*env.aircraft.values(),*env.missiles]:
                    self.assertTrue(np.isfinite([body.x,body.y,body.z,body.heading,body.pitch,body.speed]).all())
                    self.assertGreaterEqual(body.z,0)
                    self.assertGreaterEqual(body.speed,0)
                if env.tick%30==0:
                    for obs in observations.values():
                        from combatsim.set_transformer import entity_features
                        self.assertTrue(all(np.isfinite(value).all() for value in entity_features(obs).values()))
            self.assertLessEqual(env.tick,env.config.max_steps)
            self.assertTrue(env.missiles,'must exercise missile trajectories')

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
