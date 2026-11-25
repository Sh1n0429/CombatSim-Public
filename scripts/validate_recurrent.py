"""Temporal-policy regression checks for decision history and recurrent PPO."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from combatsim.checkpoints import validate_checkpoint_schema
from combatsim.env import ACTION_SHAPE, decode_action
from combatsim.inference import select_actions
from combatsim.observation_schema import SELF_INDEX
from combatsim.observation_schema import CONTACT_INDEX
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.rllib_model import CombatSimSetTransformerModel
from combatsim.set_transformer import SetTransformerActorCritic, EMBED_SIZE

class RecurrentChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
 def fixture(self):
  env=CombatSimFlightEnv({'settings':{'blue':1,'red':1,'decision_interval_steps':3,
                                'boundary_override':{'enabled':True}}})
  obs,_=env.reset(seed=17)
  return env,obs
 def test_decision_survives_hold_and_override_and_resets(self):
  env,obs=self.fixture();a=env.env.aircraft['blue_0'];a.z=env.config.ceiling-.01;a.pitch=.1
  env._raw_observations=env.env.observations()
  actions={'blue_0':(10,8,7),'red_0':(0,0,0)}
  obs,*_=env.step(actions)
  row=obs['blue_0']['self'][0];decoded=decode_action(actions['blue_0'])
  self.assertAlmostEqual(float(row[SELF_INDEX['previous_turn']]),decoded[0],places=6)
  self.assertEqual(float(row[SELF_INDEX['previous_vertical']]),decoded[1])
  self.assertAlmostEqual(float(row[SELF_INDEX['previous_fire']]),7/31,places=6)
  self.assertNotEqual(env.env.previous_actions['blue_0'][1],decoded[1])
  self.assertEqual(env.env.previous_actions['blue_0'][2],False)
  obs,_=env.reset(seed=17)
  for k in ['previous_turn','previous_vertical','previous_fire']:self.assertEqual(obs['blue_0']['self'][0,SELF_INDEX[k]],0.)
 def test_sequence_matches_stream_and_padding_ignores_tail(self):
  env,obs=self.fixture();sample=obs['blue_0'];n=SetTransformerActorCritic().eval()
  tokens={k:torch.as_tensor(np.stack([v]*6)) for k,v in sample.items()}
  tokens['common'][:,0,0]=torch.arange(6)/10
  lengths=torch.tensor([3,2]);seq,value,hidden=n.forward_sequence(tokens,None,lengths)
  for b,length in enumerate(lengths.tolist()):
   state=None
   for t in range(length):
    step={k:v[b*3+t:b*3+t+1] for k,v in tokens.items()}
    logits,val,state=n.forward_recurrent(step,state)
    for x,y in zip(seq,logits):torch.testing.assert_close(x[b*3+t],y[0],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(value[b*3+t],val[0],atol=2e-6,rtol=2e-5)
   for x,y in zip(hidden,state):torch.testing.assert_close(x[b],y[0,0],atol=2e-6,rtol=2e-5)
 def test_gradient_reaches_previous_observation(self):
  _,obs=self.fixture();sample=obs['blue_0'];n=SetTransformerActorCritic()
  tokens={k:torch.as_tensor(np.stack([v]*3)) for k,v in sample.items()};tokens['self'].requires_grad_()
  _,value,_=n.forward_sequence(tokens,None,torch.tensor([3]));value[-1].backward()
  self.assertGreater(tokens['self'].grad[0].abs().sum().item(),0.)
 def test_agent_memory_is_independent_and_new_match_resets(self):
  _,obs=self.fixture();n=SetTransformerActorCritic().eval();agents=sorted(obs);states={}
  select_actions(n,obs,agents,states=states);first={a:tuple(v.clone() for v in states[a]) for a in agents}
  select_actions(n,obs,list(reversed(agents)),states=states)
  for a in agents:
   solo={a:tuple(v.clone() for v in first[a])};select_actions(n,obs,[a],states=solo)
   for x,y in zip(states[a],solo[a]):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
  fresh={};select_actions(n,obs,agents,states=fresh)
  for a in agents:
   for x,y in zip(first[a],fresh[a]):torch.testing.assert_close(x,y)
 def test_adapter_returns_sequence_state_and_values(self):
  env,obs=self.fixture();model=CombatSimSetTransformerModel(env.get_observation_space(),env.get_action_space(),sum(ACTION_SHAPE),{},'test')
  tokens={k:torch.as_tensor(np.stack([v]*6)) for k,v in obs['blue_0'].items()}
  logits,state=model.forward({'obs':tokens},[torch.zeros(2,EMBED_SIZE),torch.zeros(2,EMBED_SIZE)],torch.tensor([3,2]))
  self.assertEqual(logits.shape,(6,sum(ACTION_SHAPE)));self.assertEqual(model.value_function().shape,(6,));self.assertEqual(state[0].shape,(2,EMBED_SIZE))
 def test_old_semantics_cannot_silently_resume(self):
  with tempfile.TemporaryDirectory() as folder:
   Path(folder,'training-state.json').write_text(json.dumps({'observation_version':1}))
   with self.assertRaisesRegex(ValueError,'incompatible'):validate_checkpoint_schema(folder)

 def test_selected_contact_is_previous_decision_not_last_physics_tick(self):
  env,_=self.fixture()
  blue,red=env.env.aircraft['blue_0'],env.env.aircraft['red_0']
  blue.x,blue.y,blue.z,blue.heading=30.,50.,20.,0.
  red.x,red.y,red.z,red.heading=70.,50.,20.,float(np.pi)
  env._raw_observations=env.env.observations()
  obs,*_=env.step({'blue_0':(5,5,1),'red_0':(5,5,0)})
  self.assertGreater(obs['blue_0']['contacts_mask'].sum(),0)
  self.assertEqual(obs['blue_0']['contacts'][0,CONTACT_INDEX['selected_last_step']],1.)
  obs,*_=env.step({'blue_0':(5,5,0),'red_0':(5,5,0)})
  self.assertEqual(obs['blue_0']['contacts'][0,CONTACT_INDEX['selected_last_step']],0.)

 def test_odd_sequence_batch_has_independent_states(self):
  _,obs=self.fixture();sample=obs['blue_0'];n=SetTransformerActorCritic().eval()
  tokens={k:torch.as_tensor(np.stack([v]*9)) for k,v in sample.items()}
  lengths=torch.tensor([3,1,2]);_,_,hidden=n.forward_sequence(tokens,None,lengths)
  for b,length in enumerate(lengths.tolist()):
   single={k:v[b*3:b*3+length] for k,v in tokens.items()}
   _,_,own=n.forward_sequence(single,None,torch.tensor([length]))
   for x,y in zip(hidden,own):torch.testing.assert_close(x[b],y[0],atol=2e-6,rtol=2e-5)

 def test_resume_hyperparameters_match_restored_optimizer(self):
  from types import SimpleNamespace
  from scripts.train import restore_training_parameters
  saved={'training_parameters':{'lr':4e-5,'batch_size':512}}
  args=SimpleNamespace(lr=2e-5,batch_size=2048)
  restore_training_parameters(args,saved,[])
  self.assertEqual(args.lr,4e-5);self.assertEqual(args.batch_size,512)
  with self.assertRaisesRegex(ValueError,'restored PPO'):
   restore_training_parameters(SimpleNamespace(lr=1e-4,batch_size=512),saved,['--lr=0.0001'])

if __name__=='__main__':unittest.main(verbosity=2)
