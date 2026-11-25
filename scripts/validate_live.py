"""End-to-end checks for schedules, live sessions, persistence and HTTP controls."""
import json
import pickle
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import torch
from combatsim.evaluation import EvaluationSession, build_schedule
from combatsim.factory import load_settings
from combatsim.live import make_server
from combatsim.set_transformer import SetTransformerActorCritic
from combatsim.tournament import MODEL_FORMAT, file_sha256
from combatsim.tournament import settings_for_team_sizes
from combatsim.observation_schema import OBSERVATION_VERSION

ROOT=Path(__file__).resolve().parents[1]

class LiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        model=self.folder/'model';model.mkdir()
        torch.manual_seed(12)
        torch.save(SetTransformerActorCritic().state_dict(),model/'model.pth')
        (model/'metadata.json').write_text(json.dumps({'format':MODEL_FORMAT,'observation_version':OBSERVATION_VERSION,'model_id':'test','sha256':file_sha256(model/'model.pth')}))
        self.session=EvaluationSession(ROOT,self.folder/'output')
        self.model=self.session.models.add('model',str(model))['id']
        settings=load_settings(ROOT/'configs/scenario-default.json');settings['max_steps']=8
        scenario=self.folder/'scenario.json';scenario.write_text(json.dumps(settings))
        self.session.scenarios['simulator']=scenario
        self.request={'blue_model':self.model,'red_model':self.model,'seed_count':2,'pace':0}
    def tearDown(self):
        self.session.stop()
        if self.session.worker:self.session.worker.join(10)
        self.temp.cleanup()
    def test_schedule(self):
        catalog=self.session.models.entries
        plan=build_schedule({**self.request,'blue_count':1,'red_count':3},catalog)
        self.assertEqual(len(plan),4)
        self.assertEqual((plan[1]['blue_count'],plan[1]['red_count']),(3,1))
        mixed=build_schedule({**self.request,'mode':'mixed','blue_lineup':[self.model,self.model],'red_lineup':[self.model]},catalog)
        self.assertEqual(len(mixed[0]['assignments']),3)
        rr=build_schedule({**self.request,'mode':'round_robin','models':[self.model,'rule:fix']},catalog)
        self.assertEqual(len(rr),4)
        for bad in ({'seeds':[1,1]},{'blue_model':'missing'},{'seed_count':True}):
            with self.assertRaises(ValueError):build_schedule({**self.request,**bad},catalog)
    def test_rule_only_single_match_finishes_and_tick_stays_fixed(self):
        request={**self.request,'blue_model':'rule:fix','red_model':'rule:fix',
                 'single_match':True,'seed_count':5,'swap_sides':True,'pace':0}
        self.assertEqual(self.session.preview(request)['total'],1)
        self.session.start(request)
        self.session.worker.join(10)
        self.assertEqual(self.session.job['status'],'completed',self.session.job)
        self.assertEqual(self.session.job['completed'],1)
        first=self.session.watch()
        self.assertTrue(first['done'])
        time.sleep(.1)
        self.assertEqual(self.session.watch()['frame']['tick'],first['frame']['tick'])
        self.assertEqual(first['frame']['tick'],8)
        self.assertTrue(self.session.matches[0]['personal'])
        self.assertEqual(self.session.matches[0]['action_mode'], 'sample')

    def test_random_rule_is_selectable_reproducible_and_logged(self):
        request={**self.request,'red_model':'rule:random','seed_count':1,
                 'swap_sides':False,'save_replays':False,'pace':0}
        self.session.start(request)
        self.session.worker.join(15)
        self.assertEqual(self.session.job['status'],'completed',self.session.job)
        first=self.session.matches[0]
        self.assertEqual(first['rule_profiles_by_side'], {'red':'random'})
        parameters=first['rule_parameters_by_side']['red']
        settings = load_settings(self.session.scenarios['simulator'])
        ranges = settings['rule_policy_profiles']['random']['randomize']
        for field in ('fire_range', 'fire_interval_ticks'):
            lower, upper = ranges[field]
            self.assertTrue(lower <= parameters[field] <= upper)
        self.session.start(first['retry_request'])
        self.session.worker.join(15)
        self.assertEqual(self.session.matches[0]['rule_parameters_by_side']['red'],parameters)
    def test_hud_range_and_missile_metadata_survive_live_and_saved_replay(self):
        path=self.session.scenarios['simulator']
        settings=json.loads(path.read_text())
        settings.update(blue_spawn=[[.10,.10],[.5,.5],[.4,.4]],
                        red_spawn=[[.15,.15],[.5,.5],[.4,.4]])
        path.write_text(json.dumps(settings))
        self.session.start({**self.request,'blue_model':'rule:fix','red_model':'rule:fix',
                            'single_match':True,'blue_count':1,'red_count':1})
        self.session.worker.join(15)
        self.assertEqual(self.session.job['status'],'completed',self.session.job)
        frames=self.session.watch()['new_frames']
        self.assertTrue(all(any('rmax_km' in t for t in a['hud_contacts'])
                            for a in frames[0]['aircraft']))
        flying=[f for f in frames if f['missiles']]
        self.assertTrue(flying)
        self.assertTrue(any(t['source']=='own' for f in flying for a in f['aircraft'] for t in a['hud_missiles']))
        saved=json.loads(next((Path(self.session.job['output'])/'replays').glob('*.json')).read_text())
        self.assertEqual(saved['frames'],frames)

    def test_batch_logs_replay_and_selfplay(self):
        self.session.start(self.request);self.session.worker.join(15)
        state=self.session.state();self.assertEqual(state['job']['status'],'completed',state['job'])
        self.assertEqual(state['job']['completed'],4)
        self.assertEqual(len(state['models']),2)
        self.assertTrue(all(row['played']==4 for row in state['models']))
        output=Path(state['job']['output'])
        self.assertEqual(len((output/'matches.jsonl').read_text(encoding='utf-8').splitlines()),4)
        for name in ['manifest.json','summary.json','team-standings.csv','model-standings.csv']:self.assertTrue((output/name).exists())
        self.assertEqual(len(list((output/'replays').glob('*.html'))),4)
        watch=self.session.watch();self.assertTrue(watch['done']);self.assertEqual(watch['frame']['tick'],8)
        model_name = self.session.models.entries[self.model]['name']
        for side in ('blue', 'red'):
            aircraft = [a for a in watch['frame']['aircraft'] if a['team'] == side]
            self.assertEqual({a['label'] for a in aircraft},
                             {model_name if len(aircraft) == 1 else f'{model_name} #{i}'
                              for i in range(1, len(aircraft)+1)})
        self.assertTrue(all('hud_contacts' in a for a in watch['frame']['aircraft']))
        replay = json.loads(next((output/'replays').glob('*.json')).read_text(encoding='utf-8'))
        self.assertTrue(all('label' in a for f in replay['frames'] for a in f['aircraft']))
        self.assertTrue(all('hud_contacts' in a for f in replay['frames'] for a in f['aircraft']))
        self.assertEqual([f['tick'] for f in replay['frames']], list(range(9)))
        cumulative = {}
        for frame in replay['frames']:
            for values in frame['rewards']['components'].values():
                for ident, value in values.items():
                    cumulative[ident] = cumulative.get(ident, 0) + value
            for ident, total in frame['rewards']['episode_totals'].items():
                self.assertAlmostEqual(cumulative.get(ident, 0), total)
        self.assertEqual(self.session.watch(watch['match_id'],8)['new_frames'],[])

    def test_retry_preserves_swapped_match_and_previous_results(self):
        self.session.start({**self.request, 'red_model': 'rule:fix',
                            'blue_count': 1, 'red_count': 2, 'seed_count': 1})
        self.session.worker.join(15)
        previous = Path(self.session.job['output'])
        match = self.session.matches[1]
        self.assertEqual(match['reference_side'], 'red')
        plan = build_schedule(match['retry_request'], self.session.models.entries)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['assignments'], match['assignments'])
        self.assertEqual(plan[0]['seed'], match['seed'])
        self.assertEqual(plan[0]['team_names'], match['team_names'])
        self.session.start(match['retry_request'])
        self.session.worker.join(15)
        self.assertEqual(self.session.job['completed'], 1)
        self.assertNotEqual(Path(self.session.job['output']), previous)
        self.assertTrue((previous/'summary.json').is_file())
        self.assertEqual(self.session.matches[0]['winner'], match['winner'])
    def test_rule_and_pause_stop(self):
        self.session.start({**self.request,'red_model':'rule:fix','pace':1})
        deadline=time.monotonic()+5
        while not self.session.current and time.monotonic()<deadline:time.sleep(.01)
        self.session.pause(True);self.assertEqual(self.session.state()['job']['status'],'paused')
        self.session.set_pace(120)
        self.assertEqual(self.session.state()['job']['pace'],120)
        self.session.set_pace('realtime')
        self.assertEqual(self.session.watch()['pace'],'realtime')
        time.sleep(.1) # Allow an already-running step to finish.
        tick=self.session.watch()['frame']['tick']
        time.sleep(.15)
        self.assertEqual(self.session.watch()['frame']['tick'],tick)
        for invalid in (-1,1001,True,'60'):
            with self.assertRaises(ValueError):self.session.set_pace(invalid)
        with self.assertRaises(ValueError):self.session.start(self.request)
        self.session.pause(False);self.session.stop();self.session.worker.join(5)
        self.assertEqual(self.session.state()['job']['status'],'stopped')
        self.assertEqual(self.session.state()['job']['completed'],0)
        self.session.start({**self.request,'red_model':'rule:fix','seed_count':1,'swap_sides':False})
        self.session.worker.join(10)
        self.assertEqual(self.session.state()['job']['status'],'completed',self.session.state()['job'])
    def test_checkpoint_mixed_and_round_robin(self):
        folder=self.folder/'checkpoint';policy=folder/'policies/main';policy.mkdir(parents=True)
        weights=self.session.models.networks[self.model].state_dict()
        (policy/'policy_state.pkl').write_bytes(pickle.dumps({'weights':{'network.'+k:v.numpy() for k,v in weights.items()}}))
        torch.save(weights,folder/'run-initial-weights.pth')
        (folder/'training-state.json').write_text(json.dumps(
            {'observation_version':OBSERVATION_VERSION}))
        self.session.models.checkpoint(folder)
        checkpoint=next(m['id'] for m in self.session.models.catalog() if m.get('policy_id')=='main')
        self.assertEqual(next(self.session.models.networks[checkpoint].parameters()).device.type,'cpu')
        self.session.start({**self.request,'mode':'mixed','blue_lineup':[self.model,checkpoint],
                            'red_lineup':[checkpoint],'seed_count':1,'swap_sides':False})
        self.session.worker.join(10)
        self.assertEqual(self.session.job['status'],'completed',self.session.job)
        self.assertEqual(len(self.session.state()['models']),3)
        aircraft = self.session.watch()['frame']['aircraft']
        blue = [a for a in aircraft if a['team']=='blue']
        red = [a for a in aircraft if a['team']=='red']
        self.assertEqual({a['model_color'] for a in blue}, {0, 1})
        self.assertEqual({a['model_color'] for a in red}, {0})
        self.session.start({**self.request,'mode':'round_robin','models':[self.model,checkpoint,'rule:fix'],
                            'seed_count':1,'swap_sides':False})
        self.session.worker.join(10)
        self.assertEqual(self.session.job['completed'],3,self.session.job)
        settings=load_settings(ROOT/'configs/scenario-default.json')
        settings['aircraft'][0]['performance']={'aircraft_speed':.7}
        resized=settings_for_team_sizes(settings,3,1)
        self.assertEqual(resized['aircraft'][2]['performance']['aircraft_speed'],.7)
    def test_http(self):
        server=make_server(self.session,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        def fetch(path,data=None,token=True,extra=None):
            headers=extra or {}
            if token:headers['X-CombatSim-Token']=server.session_token
            return urlopen(Request(base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers))
        try:
            for route in ['/','/evaluation.js','/viewer','/arena','/arena.js','/arena-scene.js','/arena-webgl.js','/arena-audio.js',
                          '/arena-assets/f16.glb','/arena-assets/three.module.js',
                          '/arena-assets/three.core.js','/arena-assets/addons/loaders/GLTFLoader.js',
                          '/arena-assets/addons/utils/BufferGeometryUtils.js',
                          '/arena-assets/addons/utils/SkeletonUtils.js',
                          '/arena-assets/addons/objects/Sky.js','/arena-assets/addons/objects/Water.js',
                          '/arena-assets/addons/libs/meshopt_decoder.module.js','/arena-assets/NOTICE.txt',
                          '/api/state','/api/watch','/api/history']:
                with fetch(route) as response:self.assertEqual(response.status,200)
            with fetch('/arena-assets/three.module.js') as response:
                self.assertIn('javascript', response.headers['Content-Type'])
            with fetch('/arena-assets/f16.glb') as response:
                self.assertEqual(response.read(4), b'glTF')
            for name in ['score','engine','launch','impact']:
                with fetch(f'/arena-assets/arena-{name}.ogg') as response:
                    self.assertIn('audio', response.headers['Content-Type'])
                    self.assertEqual(response.read(4), b'OggS')
            with self.assertRaises(HTTPError):fetch('/arena-assets/%2e%2e/live.py')
            with self.assertRaises(HTTPError):fetch('/arena-assets/%2e%2e/arena.js')
            with self.assertRaises(HTTPError):fetch('/arena-assets/%2e%2e/%2e%2e/live.py')
            with fetch('/viewer') as response:
                page = response.read().decode('utf-8')
                self.assertNotIn('__BOOTSTRAP__', page)
                self.assertNotIn('__SCRIPT__', page)
                self.assertIn('function sampleRecordedFrame(', page)
            self.session.frames[0]['aircraft'][0]['label'] = '</script><img src=x>'
            with fetch('/arena') as response:
                page = response.read().decode('utf-8')
                self.assertIn('CombatSim ARENA', page)
                self.assertNotIn('</script><img src=x>', page)
                self.assertIn(r'\u003c/script>', page)
                self.assertNotIn('__BOOTSTRAP__', page)
            with self.assertRaises(HTTPError) as error:fetch('/api/start',self.request,False)
            self.assertEqual(error.exception.code,403)
            with self.assertRaises(HTTPError) as error:fetch('/api/state',extra={'Origin':'https://example.com'})
            self.assertEqual(error.exception.code,403)
            with self.assertRaises(HTTPError):fetch('/files/../catalog.json')
            with fetch('/api/start',self.request) as response:self.assertEqual(response.status,200)
            self.session.worker.join(15)
            with fetch('/files/'+self.session.job['id']+'/matches.jsonl') as response:self.assertEqual(len(response.read().splitlines()),4)
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
