"""Shared observation batching, action selection and rule-match random seeds."""
import numpy as np
import torch


def greedy_fire_action(logits):
    """Threshold the learned gate at 0.5, then select its conditional target."""
    fire = torch.logsumexp(logits[:, 1:], -1) > logits[:, 0]
    target = logits[:, 1:].argmax(-1) + 1
    return torch.where(fire, target, torch.zeros_like(target))


def rule_match_generator(seed, learning_side, device='cpu'):
    """Use the same action stream for evaluation and its matching replay."""
    if learning_side not in ('blue', 'red'):
        raise ValueError('learning_side must be blue or red')
    return torch.Generator(device=device).manual_seed(seed*2 + (learning_side == 'red'))


@torch.inference_mode()
def select_actions(network, observations, agents, mode='greedy', generator=None, states=None):
    """Return one action row per agent, preserving the caller's agent order."""
    if mode not in ('greedy', 'sample'):
        raise ValueError('action mode must be greedy or sample')
    device = next(network.parameters()).device
    features = {
        key: torch.as_tensor(np.stack([observations[agent][key] for agent in agents]), device=device)
        for key in observations[agents[0]]
    }
    if states is None:
        logits, _ = network(features)
    else:
        hidden = [states[agent] if agent in states else network.init_hidden(1) for agent in agents]
        batched = tuple(torch.cat([h[i].reshape(1, -1) for h in hidden], dim=0)
                        for i in range(2))
        logits, _, updated = network.forward_recurrent(features, batched)
        for index, agent in enumerate(agents):
            states[agent] = tuple(h[index:index+1].detach().clone() for h in updated)
    if mode == 'greedy':
        actions = torch.stack((logits[0].argmax(-1), logits[1].argmax(-1),
                               greedy_fire_action(logits[2])), -1)
    else:
        actions = torch.stack([
            torch.multinomial(head.softmax(-1), 1, generator=generator).squeeze(-1)
            for head in logits
        ], -1)
    return actions.cpu().numpy()
