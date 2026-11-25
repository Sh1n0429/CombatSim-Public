"""Output-head gradients, architecture persistence and legacy inference."""
from pathlib import Path
import json,pickle,tempfile,unittest
import torch
from torch import nn
from combatsim.config_defaults import APPLICATION_DEFAULTS
from combatsim.checkpoints import load_checkpoint_network
from combatsim.observation_schema import OBSERVATION_VERSION
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.set_transformer import SetTransformerActorCritic,head_hidden_sizes_from_weights,observation_tensor
from combatsim.tournament import load_model,MODEL_FORMAT,file_sha256
from scripts.train import parse_args,restore_training_parameters

class OutputHeadChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def fixture(self):
  env=CombatSimFlightEnv({'settings':{'blue':1,'red':1}})
  obs,_=env.reset(seed=12)
  return observation_tensor(obs['blue_0'])
 def test_independent_hidden_heads_receive_gradients(self):
  net=SetTransformerActorCritic(head_hidden_sizes=[32])
  heads,value=net(self.fixture())
  for module in (net.turn_actor,net.vertical_actor,net.fire_actor,net.critic):
   self.assertIsInstance(module,nn.Sequential)
   self.assertEqual(sum(isinstance(m,nn.Linear) for m in module),2)
  net.zero_grad();(-heads[0].log_softmax(-1)[0,0]).backward()
  self.assertGreater(float(net.turn_actor[0].weight.grad.norm()),0)
  self.assertIsNone(net.critic[0].weight.grad)
  self.assertIsNone(net.vertical_actor[0].weight.grad)
  net.zero_grad();heads,value=net(self.fixture());value.square().sum().backward()
  self.assertGreater(float(net.critic[0].weight.grad.norm()),0)
  self.assertIsNone(net.turn_actor[0].weight.grad)
 def test_checkpoint_and_model_preserve_old_and_new_shapes(self):
  observation=self.fixture()
  for sizes in ([],[32],[32,16]):
   with self.subTest(sizes=sizes),tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp);net=SetTransformerActorCritic(head_hidden_sizes=sizes).eval()
    weights=net.state_dict();self.assertEqual(head_hidden_sizes_from_weights(weights),sizes)
    (path/'policies/main').mkdir(parents=True)
    (path/'training-state.json').write_text(json.dumps({'observation_version':OBSERVATION_VERSION}))
    (path/'policies/main/policy_state.pkl').write_bytes(pickle.dumps({'weights':{'network.'+k:v.numpy() for k,v in weights.items()}}))
    loaded,_,_=load_checkpoint_network(path)
    torch.save(weights,path/'model.pth')
    (path/'metadata.json').write_text(json.dumps({'format':MODEL_FORMAT,'observation_version':OBSERVATION_VERSION,'model_id':'heads','sha256':file_sha256(path/'model.pth')}))
    _,submitted,_=load_model(path)
    with torch.inference_mode():
     expected=net(observation)
     for candidate in (loaded,submitted):
      actual=candidate(observation)
      for left,right in zip(expected[0],actual[0]):torch.testing.assert_close(left,right,rtol=0,atol=0)
      torch.testing.assert_close(expected[1],actual[1],rtol=0,atol=0)
 def test_configuration_and_resume_architecture(self):
  self.assertEqual(parse_args([]).head_hidden_sizes,APPLICATION_DEFAULTS['training']['head_hidden_sizes'])
  args=parse_args(['--algorithm','ppo','--head-hidden-sizes','32','16'])
  self.assertEqual(args.head_hidden_sizes,[32,16])
  restore_training_parameters(args,{'training_parameters':{'head_hidden_sizes':[]}},[])
  self.assertEqual(args.head_hidden_sizes,[])
  with self.assertRaisesRegex(ValueError,'preserves saved'):
   restore_training_parameters(parse_args(['--algorithm','ppo','--head-hidden-sizes','32']),{'training_parameters':{'head_hidden_sizes':[]}},['--head-hidden-sizes','32'])
  for sizes in ([0],[-1],[True]):
   with self.subTest(sizes=sizes),self.assertRaises(ValueError):SetTransformerActorCritic(head_hidden_sizes=sizes)

if __name__=='__main__':unittest.main()
