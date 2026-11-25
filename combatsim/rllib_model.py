"""RLlib adapter for CombatSim's masked, variable-entity Set Transformer."""
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import explained_variance

from .set_transformer import SetTransformerActorCritic, EMBED_SIZE, OBSERVATION_SHAPE
from .observation_schema import SELF_INDEX, CONTACT_INDEX
from .central_critic import CentralSetCritic


class CombatSimSetTransformerModel(TorchModelV2, nn.Module):
    """Keep typed entity tensors and masks intact through RLlib's policy model."""

    def __init__(self, observation_space, action_space, num_outputs, model_config, name,
                 compact_attention=None, head_hidden_sizes=None, **kwargs):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        if compact_attention is None:
            compact_attention = model_config.get('custom_model_config', {}).get('compact_attention', True)
        if head_hidden_sizes is None:
            head_hidden_sizes = model_config.get('custom_model_config', {}).get('head_hidden_sizes')
        self.network = SetTransformerActorCritic(
            *map(int, action_space.nvec),
            compact_attention=compact_attention,
            head_hidden_sizes=head_hidden_sizes,
        )
        self.algorithm = model_config.get('custom_model_config', {}).get('algorithm', 'ppo')
        if self.algorithm not in ('ppo', 'mappo'):
            raise ValueError('algorithm must be ppo or mappo')
        if self.algorithm == 'mappo':
            self.central_critic = CentralSetCritic()
        self._value = None
        self._metrics = {}

    def forward(self, input_dict, state, seq_lens):
        observation = {
            key: value.float() for key, value in input_dict['obs'].items()
        }
        if seq_lens is None:
            seq_lens = torch.ones(observation['self'].shape[0], dtype=torch.int32)
        expected_states = 4 if self.algorithm == 'mappo' else 2
        if state and len(state) != expected_states:
            raise ValueError(f'{self.algorithm} requires {expected_states} recurrent state tensors')
        policy, self._value, actor_state = self.network.forward_sequence(
            {key: observation[key] for key in OBSERVATION_SHAPE},
            tuple(state[:2]) if state else None, seq_lens)
        if self.algorithm == 'mappo':
            self._value, critic_state = self.central_critic.forward_sequence(
                observation, tuple(state[2:]) if state else None, seq_lens)
            updated_state = (*actor_state, *critic_state)
        else:
            updated_state = actor_state
        with torch.no_grad():
            lengths = torch.as_tensor(seq_lens, device=self._value.device)
            time = self._value.numel() // len(lengths)
            valid = (torch.arange(time, device=lengths.device)[None] < lengths[:, None]).reshape(-1)
            alive = valid & (observation['self'][:, 0, SELF_INDEX['alive']] > .5)
            head_entropy = [torch.distributions.Categorical(logits=head).entropy() for head in policy]
            entropy = sum(head_entropy)
            ready = alive & (observation['self'][:, 0, SELF_INDEX['ammo']] > 0)
            ready &= observation['self'][:, 0, SELF_INDEX['shoot_cooldown']] == 0
            ready &= observation['self'][:, 0, SELF_INDEX['missile_capacity_available']] > .5
            ready &= ((observation['contacts_mask'] > .5) &
                      (observation['contacts'][..., CONTACT_INDEX['detected_by_self']] > .5)).any(-1)
            firing = 1-policy[2].softmax(-1)[:, 0]
            self._metrics = {'alive_entropy': (entropy*alive).sum()/alive.sum().clamp_min(1),
                             'alive_sample_fraction': alive.sum()/valid.sum().clamp_min(1),
                             'turn_entropy_alive': (head_entropy[0]*alive).sum()/alive.sum().clamp_min(1),
                             'pitch_entropy_alive': (head_entropy[1]*alive).sum()/alive.sum().clamp_min(1),
                             'fire_probability_ready': (firing*ready).sum()/ready.sum().clamp_min(1),
                             'fire_gate_above_half_ready': ((firing > .5)*ready).sum()/ready.sum().clamp_min(1),
                             'fire_ready_fraction': ready.sum()/alive.sum().clamp_min(1)}
        # RLlib's MultiDiscrete distribution splits this concatenated vector
        # back into turn, vertical, and target-selection logits.
        return torch.cat(policy, dim=-1), list(updated_state)

    def get_initial_state(self):
        count = 4 if self.algorithm == 'mappo' else 2
        return [next(self.parameters()).new_zeros(EMBED_SIZE) for _ in range(count)]

    def metrics(self):
        return self._metrics

    def custom_loss(self, policy_loss, loss_inputs):
        """Correct reporting only: RLlib's PPO explained variance includes padding.

        PPO's loss already excludes padded rows. Keep its loss and gradients
        unchanged, but report value metrics over the same real rows.
        """
        if 'value_targets' not in loss_inputs or self._value is None:
            return policy_loss
        with torch.no_grad():
            values = self._value.reshape(-1)
            targets = loss_inputs['value_targets'].reshape(-1).to(values)
            lengths = loss_inputs.get('seq_lens')
            if lengths is None:
                valid = torch.ones_like(values, dtype=torch.bool)
            else:
                lengths = torch.as_tensor(lengths, device=values.device)
                time = values.numel()//len(lengths)
                valid = (torch.arange(time, device=values.device)[None] < lengths[:, None]).reshape(-1)
            if int(valid.sum()) > 1:
                raw = explained_variance(targets, values)
                corrected = explained_variance(targets[valid], values[valid])
                self.tower_stats['vf_explained_var'] = corrected
                self._metrics.update({
                    'vf_explained_var_with_padding': raw,
                    'vf_explained_var_valid': corrected,
                    'padding_fraction': (~valid).float().mean(),
                    'value_target_variance_valid': targets[valid].var(unbiased=False),
                    'value_error_variance_valid': (targets[valid]-values[valid]).var(unbiased=False),
                })
        return policy_loss

    def value_function(self):
        if self._value is None:
            raise RuntimeError('forward() must run before value_function()')
        return self._value
