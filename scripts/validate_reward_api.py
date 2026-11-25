"""利用者向けフックの発火条件とミサイル消滅理由を確認する。"""
from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest

from combatsim.env import Config, Missile
from combatsim.rewards import HookReward, RewardContext
from combatsim.reward_hooks import EmptyRewardHooks, CustomRewardHooks, RewardSignal
from scripts.validate_rewards import sample_state


class RewardAPIChecks(unittest.TestCase):
    def context(self, events=()):
        state = sample_state()
        aircraft = {a.id: SimpleNamespace(**asdict(a)) for a in state.aircraft}
        return RewardContext(
            Config(), aircraft, {a.id: (a.x, a.y, a.z) for a in state.aircraft},
            False, None, events=events,
            missiles=(Missile('m0', 'red_0', 'red', 'blue_0', 0, 0, 0, 0, 0),))

    def test_expiration_reason_reaches_model_hook(self):
        class ExpirationPenalty(EmptyRewardHooks):
            def on_missile_expired(self, event, state):
                self.event = event
                if event.reason == 'lifetime':
                    return RewardSignal(-.01, (event.actor,), 'expired_lifetime')
                return ()

        for raw, expected in [('lifetime', 'lifetime'), ('ground', 'ground'),
                              ('target_lost', 'target_destroyed')]:
            reward = HookReward(profile='empty')
            reward.hooks = ExpirationPenalty()
            actual = reward.calculate(self.context((
                {'type': 'expired', 'missile': 'm0', 'reason': raw},)))
            event = reward.hooks.event
            self.assertEqual((event.actor, event.target, event.missile, event.reason),
                             ('red_0', 'blue_0', 'm0', expected))
            self.assertEqual(actual, {'red_0': -.01} if raw == 'lifetime' else {})

    def test_concrete_tracking_events_dispatch(self):
        class TrackingRewards(EmptyRewardHooks):
            def on_seeker_acquired(self, event, state):
                return RewardSignal(.01, (event.actor,), 'seeker_acquired')

            def on_missile_track_lost(self, event, state):
                return RewardSignal(-.02, (event.actor,), 'track_lost')

        reward = HookReward(profile='empty')
        reward.hooks = TrackingRewards()
        actual = reward.calculate(self.context(tuple(
            {'type': kind, 'missile': 'm0'} for kind in ('bite', 'missile_track_loss'))))
        self.assertEqual(actual, {'red_0': -.01})
        self.assertEqual(set(reward.last_reasons), {'seeker_acquired', 'track_lost'})

    def test_removed_hooks_are_not_dispatched_or_rewarded(self):
        removed = ('on_support', 'on_team_separation', 'on_focus_fire',
                   'on_numerical_advantage', 'on_aircraft_destroyed', 'on_missile_evaded',
                   'on_boundary_warning')
        reward = HookReward()
        context = self.context(({'type': 'missile_evaded', 'missile': 'm0',
                                 'aircraft': 'blue_0'},))
        # 生存数が偏り、同一標的へのミサイルが複数ある場合も旧フックは生成しない。
        context.aircraft['blue_1'].alive = False
        context.aircraft['blue_0'].x = 0  # 境界上でも削除したwarningフックは生成しない。
        context = replace(context, missiles=(*context.missiles, replace(context.missiles[0], id='m1')),
                          crashed=('blue_1',))
        names = {name for name, _ in reward._events(context, reward._state(context))}
        for name in removed:
            self.assertNotIn(name, names)
            self.assertFalse(hasattr(CustomRewardHooks(), name))
        reward.calculate(context)
        self.assertNotIn('captured_missile_evaded', reward.last_reasons)


class RewardUsabilityChecks(RewardAPIChecks):
    def test_queries_report_local_detection_and_live_missiles(self):
        context = self.context()
        context = replace(context, radar_detected_targets={'blue_0': ('red_0',)},
                          missiles=(*context.missiles, replace(context.missiles[0], id='dead', alive=False)),
                          missile_telemetry={'m0': {'supporters': ('red_1',)},
                                             'dead': {'supporters': ('red_1',)}})
        state = HookReward._state(context)
        self.assertEqual(state.detected_targets('blue_0'), ('red_0',))
        self.assertEqual(state.detected_targets('blue_1'), ())
        self.assertEqual(tuple(m.id for m in state.targeting_missiles('blue_0')), ('m0',))
        self.assertFalse(state.targeting_missiles('blue_0')[0].tracking)
        self.assertEqual(tuple(m.id for m in state.supported_missiles('red_1')), ('m0',))
        self.assertEqual(state.supported_missiles('red_0'), ())
        self.assertEqual(state.targeting_missiles('missing'), ())

    def test_replay_records_actual_custom_hook_and_clipping(self):
        from combatsim.factory import make_env
        class Example(CustomRewardHooks):
            def on_step(self, event, state):
                return (*super().on_step(event, state),
                        RewardSignal(.15, state.alive_team_members('blue'), 'example'))
        env = make_env(Config(rewards=[{'type': 'hooks', 'max_step_total': .1}]))
        env.reward_system.components[0].hooks = Example()
        obs, _ = env.reset(seed=27)
        initial = env.snapshot()['rewards']
        _, rewards, _, _, info = env.step({ident: (5,5,0) for ident in obs})
        recorded = env.snapshot()['rewards']
        self.assertEqual(initial['components'], {})
        self.assertNotIn('HookReward', recorded['components'])
        for ident, actual in rewards.items():
            self.assertAlmostEqual(sum(part.get(ident, 0) for part in recorded['components'].values()), actual)
            self.assertAlmostEqual(recorded['episode_totals'][ident], actual)
        self.assertTrue(recorded['components']['HookReward:example'])
        self.assertEqual(recorded['clipping'], info['reward_clipping'])
        self.assertTrue(all(abs(row['discarded_abs']-.05)<1e-12
                            for row in recorded['clipping']['example'].values()))

    def test_absolute_additional_budget_preserves_base_rewards(self):
        class Mixed(CustomRewardHooks):
            def on_step(self, event, state):
                return [RewardSignal(.6, ('blue_0',), 'plus'),
                        RewardSignal(-.6, ('blue_0',), 'minus')]
        reward = HookReward(max_step_total=1, max_episode_total=1)
        reward.hooks = Mixed()
        context = replace(self.context(), done=True, winner='blue')
        values = reward.calculate(context)
        self.assertAlmostEqual(values['blue_0'], 1.2)
        self.assertAlmostEqual(reward.episode_totals['blue_0'], 1)
        self.assertAlmostEqual(reward.last_clipping['minus']['blue_0']['discarded_abs'], .2)
        reward.calculate(replace(context, done=False))
        self.assertEqual(reward.last_reasons['plus']['blue_0'], 0)
        reward.reset()
        reward.calculate(replace(context, done=False))
        self.assertAlmostEqual(reward.last_reasons['plus']['blue_0'], .6)

    def test_super_kill_example_keeps_individual_allocation(self):
        class Extra(CustomRewardHooks):
            def on_kill(self, event, state):
                signals = self._signals(super().on_kill(event, state))
                signals.append(RewardSignal(.01, (event.shooter,), 'extra'))
                return signals
        context = self.context(({'type': 'hit', 'missile': 'm0', 'target': 'blue_0'},))
        context.aircraft['blue_0'].alive = False
        context = replace(context, shot_down=('blue_0',))
        reward = HookReward(); reward.hooks = Extra()
        values = reward.calculate(context)
        self.assertAlmostEqual(values['red_0'], .31-.0005)
        self.assertLess(values.get('red_1', 0), 0)  # only the baseline time cost
        self.assertEqual(reward.last_reasons['enemy_kill'], {'red_0': .3})

    def test_start_has_initial_state_previous_state_and_end_runs_last(self):
        from combatsim.factory import make_env
        class Record(EmptyRewardHooks):
            def reset(self):
                super().reset(); self.calls = []
            def on_episode_start(self, event, state):
                self.calls.append(('start', state))
                return RewardSignal(.01, state.alive_team_members('blue'), 'start_bonus')
            def on_step(self, event, state): self.calls.append(('step', state))
            def on_draw(self, event, state): self.calls.append(('draw', state))
            def on_timeout(self, event, state): self.calls.append(('timeout', state))
            def on_episode_end(self, event, state): self.calls.append(('end', state))
        env = make_env(Config(max_steps=1, rewards=[{'type': 'hooks', 'profile': 'empty'}]))
        hooks = Record();env.reward_system.components[0].hooks = hooks
        observations, _ = env.reset(seed=11)
        initial = hooks.calls[0][1]
        self.assertEqual(initial.tick, 0)
        self.assertEqual(initial.previous_aircraft, ())
        _, rewards, _, _, _ = env.step({i: (5,5,0) for i in observations})
        names = [name for name, _ in hooks.calls]
        self.assertEqual(names[0:2], ['start', 'step'])
        self.assertEqual(names[-2:], ['timeout', 'end'])
        state = hooks.calls[1][1]
        self.assertEqual(state.previous_aircraft, initial.aircraft)
        self.assertTrue(all(rewards[i] == .01 for i in initial.alive_team_members('blue')))
        env.reset(seed=11)
        self.assertEqual(len(hooks.calls), 1)

    def test_first_detection_and_loss_causes(self):
        reward = HookReward(profile='empty')
        context = replace(self.context(), radar_detected_targets={'blue_0': ('red_0',)})
        def events(c):
            return reward._events(c, reward._state(c))
        first = next(e for n,e in events(context) if n == 'on_target_detected')
        self.assertTrue(first.first_occurrence)
        self.assertEqual(first.observer, 'blue_0')
        lost = next(e for n,e in events(replace(context, radar_detected_targets={})) if n == 'on_target_lost')
        self.assertEqual(lost.reason, 'radar_contact_lost')
        again = next(e for n,e in events(context) if n == 'on_target_detected')
        self.assertFalse(again.first_occurrence)
        context.aircraft['red_0'].alive = False
        lost = next(e for n,e in events(replace(context, radar_detected_targets={})) if n == 'on_target_lost')
        self.assertEqual(lost.reason, 'target_destroyed')

    def test_loss_aliases_and_crash_reason(self):
        context = replace(self.context(({'type': 'hit', 'missile': 'm0', 'target': 'blue_0'},)),
                          shot_down=('blue_0',), crashed=('blue_1',))
        reward = HookReward(profile='empty')
        losses = [e for n,e in reward._events(context,reward._state(context)) if n == 'on_friendly_loss']
        shot = next(e for e in losses if e.victim == 'blue_0')
        crash = next(e for e in losses if e.victim == 'blue_1')
        self.assertEqual((shot.shooter,shot.reason), ('red_0','shot_down'))
        self.assertEqual((crash.shooter,crash.reason), (None,'crash'))

    def test_once_and_duplicate_recipients(self):
        hooks = EmptyRewardHooks()
        signal = RewardSignal(.01, ('blue_0','blue_1'), 'once')
        self.assertEqual(hooks.once(('detect','red_0'),signal), (signal,))
        self.assertEqual(hooks.once(('detect','red_0'),signal), ())
        self.assertEqual(hooks.once(('detect','red_1'),signal), (signal,))
        hooks.reset()
        self.assertEqual(hooks.once(('detect','red_0'),signal), (signal,))
        with self.assertRaises(ValueError): RewardSignal(.01, ('blue_0','blue_0'), 'bad')

    def test_missile_telemetry_and_tracking_cause(self):
        reward = HookReward(profile='empty')
        context = replace(self.context(), missile_telemetry={'m0': {
            'tracking': True, 'tracking_source': 'radar', 'supporters': ('red_1',),
            'ever_seeker_acquired': False}})
        reward.calculate(context)
        context = replace(context, missile_telemetry={'m0': {
            'tracking': False, 'tracking_source': 'none', 'supporters': (),
            'ever_seeker_acquired': True}},events=({'type':'missile_track_loss','missile':'m0',
                'reason':'seeker_contact_lost','tracking_source':'seeker'},))
        class Record(EmptyRewardHooks):
            def on_missile_track_lost(self, event, state): self.event, self.state = event, state
        record=Record();reward.hooks=record;reward.calculate(context)
        self.assertEqual(record.event.previous_supporters, ('red_1',))
        self.assertEqual(record.event.tracking_source, 'seeker')
        self.assertEqual(record.event.reason, 'seeker_contact_lost')
        self.assertTrue(record.event.ever_seeker_acquired)
        self.assertEqual(record.state.missile_by_id('m0').x, context.missiles[0].x)
        self.assertEqual(record.state.previous_missiles[0].supporters, ('red_1',))

    def test_real_rollout_exposes_actual_supporters_without_changing_physics(self):
        from combatsim.factory import make_env
        from combatsim.policies import RulePolicy
        env=make_env();observations,_=env.reset(seed=27);policy=RulePolicy();seen=False
        reference=make_env(replace(env.config,rewards=[]));reference.reset(seed=27)
        for _ in range(250):
            actions={i:policy.act(o) for i,o in observations.items()}
            observations,_,done,truncated,_=env.step(actions)
            reference.step(actions)
            self.assertEqual(env.snapshot(include_rewards=False), reference.snapshot(include_rewards=False))
            state=env.reward_system.components[0].previous_state
            for m in state.missiles:
                if not m.alive: continue
                seen=True
                self.assertEqual(m.tracking,env._missile_tracking.get(m.id,False))
                for ident in m.supporters:
                    self.assertTrue(env.aircraft[ident].alive)
                    self.assertEqual(env.aircraft[ident].team,m.team)
                if m.phase=='active':self.assertEqual(m.supporters,())
            if done or truncated:break
        self.assertTrue(seen)


if __name__ == '__main__':
    unittest.main()
