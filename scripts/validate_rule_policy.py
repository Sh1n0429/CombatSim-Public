"""Regression checks for shared rule/learner shot legality."""
import math
import unittest
from combatsim.policies import RulePolicy
from combatsim.entities import shot_choices

class RuleActionChecks(unittest.TestCase):
    def observation(self):
        return {'self': {'id':'blue_0', 'team':'blue', 'x':.5, 'y':.5,
                        'z':.5, 'heading':0., 'pitch':0., 'ammo':6,
                        'shoot_cooldown':0},
                'airspace':{'x':1000.,'y':1000.},
                'performance':{'radar_range':280.},
                'contacts':[{'id':'local', 'distance':.1, 'bearing':.5,
                             'elevation':.3, 'detected_by_self':True}]}

    def policy(self):
        return RulePolicy({'fire_range':240.})

    def test_aspect_range_continues_to_decrease_past_beam(self):
        for distance, expected in ((260, (1, 0, 0)), (240, (1, 1, 0)), (218, (1, 1, 1))):
            for angle, shot in zip((math.pi, math.pi/2, 0), expected):
                with self.subTest(distance=distance, angle=angle):
                    obs = self.observation()
                    contact = obs['contacts'][0]
                    contact.update(distance=distance/1000, heading=contact['bearing']+angle)
                    policy = RulePolicy({'fire_range':265, 'nonclosing_fire_range':220})
                    self.assertEqual(policy.act(obs)[2], shot)

    def test_legal_off_axis_contact_can_be_selected(self):
        obs=self.observation()
        action=self.policy().act(obs)
        self.assertEqual(action[2],1)
        self.assertIn(action[2],shot_choices(obs))

    def test_shared_only_nearest_does_not_hide_local_contact(self):
        obs=self.observation()
        obs['contacts'].insert(0,dict(obs['contacts'][0],id='shared',
                                     distance=.08,detected_by_self=False))
        self.assertEqual(self.policy().act(obs)[2],2)
        obs['self']['shoot_cooldown']=1
        policy=self.policy()
        self.assertEqual(policy.act(obs)[2],0)
        self.assertEqual(policy.memory['blue_0']['target_id'],'local')

    def test_visibility_ammunition_and_cooldown_are_required(self):
        for change in ('visibility','ammo','cooldown'):
            obs=self.observation()
            if change=='visibility':obs['contacts'][0]['detected_by_self']=False
            elif change=='ammo':obs['self']['ammo']=0
            else:obs['self']['shoot_cooldown']=1
            self.assertEqual(self.policy().act(obs)[2],0,change)

    def test_profile_range_and_warning_maneuver_remain(self):
        obs=self.observation()
        self.assertEqual(RulePolicy({'fire_range':80.}).act(obs)[2],0)
        obs['self'].update(mws_active=True,mws_direction=(1.,0.,0.))
        policy=self.policy()
        self.assertEqual(policy.act(obs)[2],1)
        self.assertEqual(policy.mode('blue_0'),'evade')

    def test_additional_shot_preserves_existing_support(self):
        obs=self.observation()
        policy=self.policy()
        policy.act(obs)
        obs['friendly_missiles']=[{'id':'first','is_own':True,'age':1,'phase':'midcourse'}]
        action=policy.act(obs)
        self.assertEqual(action[2],1)
        self.assertEqual(policy.mode('blue_0'),'support')
        self.assertEqual(policy.memory['blue_0']['support_missile_id'],'first')
        self.assertFalse(policy.memory['blue_0']['pending_launch'])

    def test_fire_pacing_uses_ticks_without_freezing_maneuvers(self):
        obs=self.observation()
        policy=RulePolicy({'fire_range':240., 'fire_interval_ticks':20})
        obs['tick']=100
        first=policy.act(obs)
        self.assertEqual(first[2],1)
        obs['tick']=119
        obs['contacts'][0]['bearing']=-.5
        second=policy.act(obs)
        self.assertEqual(second[2],0)
        self.assertNotEqual(first[0],second[0])
        obs['tick']=120
        self.assertEqual(policy.act(obs)[2],1)
        # Re-reading the same physical tick cannot consume the delay.
        self.assertEqual(policy.act(obs)[2],0)

    def test_fire_pacing_validation(self):
        for value in (-1, .5, True):
            with self.assertRaises(ValueError):
                RulePolicy({'fire_interval_ticks':value})

    def test_own_missile_limit_prevents_salvo_spam(self):
        obs=self.observation()
        policy=RulePolicy({'fire_range':240., 'max_simultaneous_own_missiles':1})
        obs['tick']=1
        obs['friendly_missiles']=[{'id':'wingman-shot','is_own':False,'age':1,'phase':'midcourse'}]
        self.assertEqual(policy.act(obs)[2],1)
        obs['tick']=2
        obs['friendly_missiles'].append(
            {'id':'own-shot','is_own':True,'age':1,'phase':'midcourse'})
        self.assertEqual(policy.act(obs)[2],0)
        obs['tick']=3
        obs['friendly_missiles']=[]
        self.assertEqual(policy.act(obs)[2],1)

    def test_warning_delay_uses_physical_ticks(self):
        for delay in (0, 4, 8):
            policy=RulePolicy({'warning_response_delay_ticks':delay})
            obs=self.observation()
            obs['self'].update(mws_active=True,mws_direction=(1.,0.,0.))
            for tick in range(100, 110):
                obs['tick']=tick
                policy.act(obs)
                self.assertEqual(policy.mode('blue_0')=='evade', tick>=100+delay)
                policy.act(obs)  # Same tick must not advance the timer.
                self.assertEqual(policy.mode('blue_0')=='evade', tick>=100+delay)

    def test_warning_delay_resets_on_clear_and_episode_reset(self):
        policy=RulePolicy({'warning_response_delay_ticks':4})
        obs=self.observation()
        obs['self'].update(mws_active=True,mws_direction=(1.,0.,0.))
        for tick, active in ((10,True),(13,True),(14,False),(20,True),(23,True)):
            obs['tick']=tick
            obs['self']['mws_active']=active
            policy.act(obs)
            self.assertNotEqual(policy.mode('blue_0'),'evade')
        obs['tick']=24
        policy.act(obs)
        self.assertEqual(policy.mode('blue_0'),'evade')
        policy.reset()
        policy.act(obs)
        self.assertNotEqual(policy.mode('blue_0'),'evade')

    def test_warning_delay_does_not_freeze_flight_or_fire(self):
        policy=RulePolicy({'warning_response_delay_ticks':8,'fire_range':240.})
        obs=self.observation()
        obs['self'].update(mws_active=True,mws_direction=(1.,0.,0.))
        obs['tick']=10
        a=policy.act(obs)
        obs['tick']=11
        obs['contacts'][0]['bearing']=-.5
        b=policy.act(obs)
        self.assertEqual((a[2],b[2]),(1,1))
        self.assertNotEqual(a[0],b[0])
        obs['self']['x']=.001
        policy.act(obs)
        self.assertEqual(policy.mode('blue_0'),'boundary')

    def test_radar_warning_uses_same_delay(self):
        policy=RulePolicy({'warning_response_delay_ticks':4})
        obs=self.observation()
        obs['missiles']=[{'distance':.01,'bearing':0.,'elevation':.1}]
        for tick in (30,33,34):
            obs['tick']=tick
            policy.act(obs)
            self.assertEqual(policy.mode('blue_0')=='evade',tick==34)

    def test_warning_delay_validation(self):
        for value in (-1,.5,True):
            with self.assertRaises(ValueError):
                RulePolicy({'warning_response_delay_ticks':value})

if __name__=='__main__':unittest.main()
