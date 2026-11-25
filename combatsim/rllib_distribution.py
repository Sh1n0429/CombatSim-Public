"""Hierarchical firing semantics with exact categorical PPO logp/entropy/KL."""
import torch
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical
from .inference import greedy_fire_action


class GatedMultiCategorical(TorchMultiCategorical):
    def __init__(self, inputs, model, **kwargs):
        super().__init__(inputs, model, tuple(map(int, model.action_space.nvec)), model.action_space)

    def deterministic_sample(self):
        self.last_sample = torch.stack((self.cats[0].probs.argmax(-1), self.cats[1].probs.argmax(-1),
                                       greedy_fire_action(self.cats[2].logits)), -1)
        return self.last_sample
