"""Typed entity observations and per-type MLPs followed by attention and LSTM."""

import numpy as np
import torch
from torch import nn

from .env import ACTION_SHAPE
from .config_defaults import APPLICATION_DEFAULTS
from .entities import TOKEN_COUNT, selected_entities

EMBED_SIZE = 128
PMA_SEEDS = 4
from .observation_schema import (OBSERVATION_VERSION, PERFORMANCE_KEYS, PERFORMANCE_SCALES,
                                 ENTITY_DIMS, FEATURE_NAMES, SELF_INDEX, CONTACT_INDEX)
ENTITY_CAPACITIES = {key: 1 if key in ('common', 'self') else TOKEN_COUNT-1 for key in ENTITY_DIMS}
SET_TOKEN_COUNT = TOKEN_COUNT + 1  # Common context plus the existing entity budget.
OBSERVATION_SHAPE = {key: (ENTITY_CAPACITIES[key], dim) for key, dim in ENTITY_DIMS.items()}
OBSERVATION_SHAPE.update({key+'_mask': (capacity,) for key, capacity in ENTITY_CAPACITIES.items()})
OBSERVATION_SHAPE['contact_observers'] = (ENTITY_CAPACITIES['contacts'], ENTITY_CAPACITIES['allies'])
OBSERVATION_SHAPE['missile_targets'] = (ENTITY_CAPACITIES['friendly_missiles'], ENTITY_CAPACITIES['contacts'])


def squash(value):
    return value/(1+abs(value))


def observation_tensor(observation):
    """Add a single batch dimension to every typed feature/mask array."""
    return {key: torch.from_numpy(value).unsqueeze(0) for key, value in observation.items()}


def performance_features(performance):
    return [squash(performance[key]/PERFORMANCE_SCALES.get(key,1.0)) for key in PERFORMANCE_KEYS]


def _entity_values(entity, kind):
    values = [*entity['team_position'], *map(squash, entity['velocity'])]
    if kind == 'allies':
        values.extend([squash(entity['ammo']), squash(entity['shoot_cooldown']),
                       *performance_features(entity['performance'])])
    elif kind == 'contacts':
        values.extend([float(entity['detected_by_self']), squash(entity['detector_count']),
                       float(entity.get('selected_last_step',False))])
    elif kind == 'friendly_missiles':
        values.extend([squash(entity['elapsed']), float(entity['is_own']),
                       float(entity['phase'] == 'midcourse'), float(entity['phase'] == 'active'),
                       float(entity.get('tracking', False))])
    return values


def _observer_relations(result, groups):
    for i, missile in enumerate(groups['friendly_missiles']):
        for j, contact in enumerate(groups['contacts']):
            if result['friendly_missiles_mask'][i] and result['contacts_mask'][j]:
                result['missile_targets'][i, j] = float(missile.get('target_track') == contact['id'])
    for i, contact in enumerate(groups['contacts'][:ENTITY_CAPACITIES['contacts']]):
        if not result['contacts_mask'][i]:
            continue
        detectors = set(contact['detected_by'])
        for j, ally in enumerate(groups['allies'][:ENTITY_CAPACITIES['allies']]):
            if result['allies_mask'][j] and ally['id'] in detectors:
                result['contact_observers'][i, j] = 1
    return result


def entity_features(obs):
    """Separate natural-width matrices; masks pad entity counts, never features.

    Feature widths are defined by observation_schema.FEATURE_NAMES.
    One common token plus at most 32 entities, selected by category and distance.
    """
    result = {key: np.zeros(shape, dtype=np.float32) for key, shape in OBSERVATION_SHAPE.items()}
    result['common'][0] = [obs['time']]
    result['common_mask'][0] = 1
    own = obs['self']
    action, performance = obs['previous_action'], obs['performance']
    values = [*own['team_position'], *map(squash, own['velocity']),
              squash(own['ammo']), squash(own['shoot_cooldown']),
              *[float(action[k]) for k in ('turn', 'vertical', 'fire')],
              *performance_features(performance),
              *[squash(wall['distance']) for wall in obs['boundaries']],
              float(own.get('alive', True)), float(obs.get('boundary_override_fraction', 0.)),
              float(own.get('mws_active', False)), *own.get('mws_direction', (0., 0., 0.)),
              squash(own.get('own_missiles_in_flight', 0)), float(own.get('missile_capacity_available', True))]
    result['self'][0] = values
    result['self_mask'][0] = 1
    groups = selected_entities(obs)
    remaining = TOKEN_COUNT-1
    for index in range(min(TOKEN_COUNT-1, max(map(len, groups.values()), default=0))):
        for key, group in groups.items():
            if index < len(group):
                result[key][index] = _entity_values(group[index], key)
                result[key+'_mask'][index] = 1
                remaining -= 1
                if remaining == 0:
                    return _observer_relations(result, groups)
    return _observer_relations(result, groups)


class PoolingByMultiheadAttention(nn.Module):
    """Four learned queries pool a set into four distinct 128-dimensional slots."""

    def __init__(self):
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, PMA_SEEDS, EMBED_SIZE))
        # Give queries distinct directions at a scale comparable to normalized
        # tokens. Small near-zero queries can be overwhelmed by the same pooled
        # context. This changes initialization only, without assigning roles.
        nn.init.orthogonal_(self.seed[0], gain=EMBED_SIZE**.5)
        self.input_ff = nn.Sequential(nn.Linear(EMBED_SIZE, EMBED_SIZE), nn.ReLU(),
                                      nn.Linear(EMBED_SIZE, EMBED_SIZE))
        self.attention = nn.MultiheadAttention(EMBED_SIZE, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(EMBED_SIZE)
        self.output_ff = nn.Sequential(nn.Linear(EMBED_SIZE, EMBED_SIZE), nn.ReLU(),
                                       nn.Linear(EMBED_SIZE, EMBED_SIZE))
        self.norm2 = nn.LayerNorm(EMBED_SIZE)

    def forward(self, entities, padding_mask):
        empty = padding_mask.all(-1)
        safe_mask = padding_mask.clone()
        safe_mask[empty, 0] = False
        values = self.input_ff(entities.masked_fill(padding_mask.unsqueeze(-1), 0))
        values = values.masked_fill(padding_mask.unsqueeze(-1), 0)
        seed = self.seed.expand(entities.size(0), -1, -1)
        attended, _ = self.attention(seed, values, values, key_padding_mask=safe_mask,
                                     need_weights=False)
        pooled = self.norm1(seed + attended)
        pooled = self.norm2(pooled + self.output_ff(pooled)).flatten(1)
        return pooled.masked_fill(empty.unsqueeze(-1), 0)


def validate_head_hidden_sizes(sizes):
    if not isinstance(sizes, (list, tuple)) or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError('head_hidden_sizes must be a list of positive integers')
    return tuple(sizes)


def output_head(output_size, hidden_sizes):
    if not hidden_sizes:
        return nn.Linear(EMBED_SIZE, output_size)
    layers = []
    width = EMBED_SIZE
    for size in hidden_sizes:
        layers.extend((nn.Linear(width, size), nn.ReLU()))
        width = size
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)


def gated_fire_logits(gate, target_logits, valid):
    """P(hold)=1-g; P(target i)=g*P(i|fire), retaining the 32-choice API."""
    if target_logits.shape[-1] == 0:
        return torch.zeros_like(gate).unsqueeze(-1)
    available = valid.any(-1)
    conditional = torch.log_softmax(target_logits.masked_fill(~valid, -1e9), -1)
    shots = (torch.nn.functional.logsigmoid(gate).unsqueeze(-1) + conditional).masked_fill(~valid, -1e9)
    hold = torch.where(available, torch.nn.functional.logsigmoid(-gate), torch.zeros_like(gate))
    return torch.cat((hold.unsqueeze(-1), shots), -1)


def head_hidden_sizes_from_weights(weights):
    """Read saved head shapes so old linear heads keep their exact semantics."""
    import re
    if 'turn_actor.weight' in weights:
        return []
    indices = sorted(int(match.group(1)) for name in weights
                     if (match := re.fullmatch(r'turn_actor\.(\d+)\.weight', name)))
    if not indices:
        raise ValueError('Checkpoint has no recognizable actor head')
    return [int(weights[f'turn_actor.{index}.weight'].shape[0]) for index in indices[:-1]]


class SetTransformerActorCritic(nn.Module):
    def __init__(self, turn_actions=ACTION_SHAPE[0], vertical_actions=ACTION_SHAPE[1], fire_actions=ACTION_SHAPE[2], compact_attention=True, head_hidden_sizes=None):
        super().__init__()
        self.compact_attention = compact_attention
        self.compact_entity_mlp = True
        self.attention_need_weights = False
        self.head_hidden_sizes = validate_head_hidden_sizes(
            APPLICATION_DEFAULTS['training']['head_hidden_sizes'] if head_hidden_sizes is None else head_hidden_sizes)
        self.entity_mlps = nn.ModuleDict({
            key: nn.Sequential(nn.Linear(dim, EMBED_SIZE), nn.ReLU(),
                               nn.Linear(EMBED_SIZE, EMBED_SIZE), nn.ReLU())
            for key, dim in ENTITY_DIMS.items()
        })
        self.observer_projection = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.missile_target_projection = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(EMBED_SIZE, 4, batch_first=True),
            nn.MultiheadAttention(EMBED_SIZE, 4, batch_first=True),
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(EMBED_SIZE), nn.LayerNorm(EMBED_SIZE)])
        self.pma = PoolingByMultiheadAttention()
        self.lstm = nn.LSTM(PMA_SEEDS * EMBED_SIZE, EMBED_SIZE, batch_first=True)
        self.turn_actor = output_head(turn_actions, self.head_hidden_sizes)
        self.vertical_actor = output_head(vertical_actions, self.head_hidden_sizes)
        self.fire_gate_token = nn.Parameter(torch.randn(1, 1, EMBED_SIZE) * .02)
        self.target_attention = nn.MultiheadAttention(EMBED_SIZE, 4, batch_first=True)
        self.target_norm = nn.LayerNorm(EMBED_SIZE)
        self.fire_actor = output_head(1, self.head_hidden_sizes)
        self.fire_gate = output_head(1, self.head_hidden_sizes)
        self.critic = output_head(1, self.head_hidden_sizes)

    def _encode(self, observation):
        embeddings, masks = [], []
        for key, mlp in self.entity_mlps.items():
            mask = observation[key+'_mask'] <= .5
            # Ignore placeholder contents before the MLP, including its biases.
            values = observation[key].masked_fill(mask.unsqueeze(-1), 0)
            if self.compact_attention and self.compact_entity_mlp:
                columns = (~mask).any(0).nonzero(as_tuple=True)[0]
                encoded = mlp(values.index_select(1, columns))
                # Keep original entity rows for target/action associations.
                # Even empty selections keep zero gradients (rather than None)
                # so Adam momentum semantics match the unoptimized path.
                embeddings.append(encoded.new_zeros(values.shape[0], values.shape[1], EMBED_SIZE)
                                  .index_copy(1, columns, encoded))
            else:
                embeddings.append(mlp(values))
            masks.append(mask)
        contact_embeddings = embeddings[list(ENTITY_DIMS).index('contacts')]
        # Associate each enemy with the actual retained allied entity embeddings.
        # IDs are used only to build edges, never as ordinal numeric features.
        ally_index, contact_index = list(ENTITY_DIMS).index('allies'), list(ENTITY_DIMS).index('contacts')
        edges = observation['contact_observers'].masked_fill(masks[contact_index].unsqueeze(-1), 0)
        edges = edges.masked_fill(masks[ally_index].unsqueeze(1), 0)
        observers = edges @ embeddings[ally_index]
        observers = observers / edges.sum(-1, keepdim=True).clamp_min(1)
        embeddings[contact_index] = embeddings[contact_index] + self.observer_projection(observers)
        missile_index = list(ENTITY_DIMS).index('friendly_missiles')
        target_edges = observation['missile_targets'].masked_fill(masks[missile_index].unsqueeze(-1), 0)
        target_edges = target_edges.masked_fill(masks[contact_index].unsqueeze(1), 0)
        embeddings[missile_index] = embeddings[missile_index] + self.missile_target_projection(
            target_edges @ embeddings[contact_index])
        hidden = torch.cat(embeddings, dim=1)
        mask = torch.cat(masks, dim=1)
        # Compact padding after projection, retaining common and entity tokens.
        # On the current CPU learner, omit columns that are padding for every
        # sample. No valid entity is removed, and checkpoint tensor shapes stay fixed.
        token_count = (max(1, min(SET_TOKEN_COUNT, int((~mask).sum(-1).max().item())))
                       if getattr(self, 'compact_attention', True) else SET_TOKEN_COUNT)
        indices = mask.to(torch.int64).argsort(dim=1, stable=True)[:, :token_count]
        hidden = hidden.gather(1, indices.unsqueeze(-1).expand(-1, -1, EMBED_SIZE))
        mask = mask.gather(1, indices)
        safe_mask = mask.clone()
        safe_mask[mask.all(-1), 0] = False
        for attention, norm in zip(self.attention, self.norms):
            attended, _ = attention(hidden, hidden, hidden, key_padding_mask=safe_mask,
                                    need_weights=self.attention_need_weights)
            hidden = norm(hidden + attended)
        return self.pma(hidden, mask), contact_embeddings

    def forward_sequence(self, tokens, hidden, seq_lens):
        """Flattened batch-major [B*T,...] observations with padded sequences."""
        pooled, contacts = self._encode(tokens)
        batch = len(seq_lens)
        time = pooled.shape[0] // batch
        pooled = pooled.reshape(batch, time, -1)
        packed = nn.utils.rnn.pack_padded_sequence(
            pooled, torch.as_tensor(seq_lens).detach().cpu().clamp_min(1),
            batch_first=True, enforce_sorted=False)
        if hidden is not None:
            hidden = tuple(value.unsqueeze(0) if value.dim() == 2 else value for value in hidden)
        recurrent, hidden = self.lstm(packed, hidden)
        recurrent, _ = nn.utils.rnn.pad_packed_sequence(
            recurrent, batch_first=True, total_length=time)
        logits, value = self._heads(tokens, recurrent.reshape(batch*time, -1), contacts)
        return logits, value, tuple(value.squeeze(0) for value in hidden)

    def forward_recurrent(self, tokens, hidden=None):
        pooled, contacts = self._encode(tokens)
        pooled = pooled.unsqueeze(1)
        squeeze_hidden = hidden is not None and hidden[0].dim() == 2
        if squeeze_hidden:
            hidden = tuple(value.unsqueeze(0) for value in hidden)
        recurrent, hidden = self.lstm(pooled, hidden)
        if squeeze_hidden:
            hidden = tuple(value.squeeze(0) for value in hidden)
        recurrent = recurrent[:, -1]
        policy, value = self._heads(tokens, recurrent, contacts)
        return policy, value, hidden

    def _heads(self, tokens, recurrent, contacts):
        valid = (tokens['contacts_mask']>.5) & (tokens['contacts'][...,CONTACT_INDEX['detected_by_self']]>.5)
        ready = (tokens['self'][:,0,SELF_INDEX['ammo']]>0) & (tokens['self'][:,0,SELF_INDEX['shoot_cooldown']]==0)
        ready = ready & (tokens['self'][:,0,SELF_INDEX['missile_capacity_available']] > .5)
        valid = valid & ready.unsqueeze(-1)
        fire_count = valid.size(1) + 1
        columns = torch.arange(valid.size(1), device=valid.device).expand_as(valid)
        contact_count = (int(columns.masked_fill(~valid, -1).max().item()) + 1
                         if getattr(self, 'compact_attention', True) else valid.size(1))
        contacts, valid = contacts[:, :contact_count], valid[:, :contact_count]
        # Recurrent context, gate query, then target candidates.
        candidates = torch.cat((recurrent.unsqueeze(1),
                                self.fire_gate_token.expand(recurrent.size(0), -1, -1),
                                contacts), dim=1)
        candidate_mask = torch.cat((valid.new_zeros(valid.size(0), 2), ~valid), dim=1)
        attended, _ = self.target_attention(candidates, candidates, candidates,
                                            key_padding_mask=candidate_mask,
                                            need_weights=self.attention_need_weights)
        candidates = self.target_norm(candidates + attended)
        targets = self.fire_actor(candidates[:, 2:]).squeeze(-1)
        gate = self.fire_gate(candidates[:, 1]).squeeze(-1)
        shot_logits = gated_fire_logits(gate, targets, valid)
        shot_logits = torch.nn.functional.pad(shot_logits, (0, fire_count-shot_logits.size(1)), value=-1e9)
        heads = (self.turn_actor(recurrent), self.vertical_actor(recurrent), shot_logits)
        # Destroyed aircraft retain the team's timeline but have exactly one
        # no-op. Their policy entropy/gradient is zero; value learning continues.
        alive = tokens['self'][:, 0, SELF_INDEX['alive']] > .5
        heads = tuple(torch.where(alive[:, None], head,
                      torch.where(torch.arange(head.shape[-1], device=head.device)[None] == 0,
                                  torch.zeros_like(head), torch.full_like(head, -1e9)))
                      for head in heads)
        return heads, self.critic(recurrent).squeeze(-1)

    def forward(self, tokens):
        policy, value, _ = self.forward_recurrent(tokens)
        return policy, value

    def init_hidden(self, batch_size=None):
        device = next(self.parameters()).device
        if batch_size is None or batch_size == []:
            shape = (1, EMBED_SIZE)
        elif isinstance(batch_size, (list, tuple)):
            shape = (*batch_size, EMBED_SIZE)
        else:
            shape = (batch_size, EMBED_SIZE)
        return (torch.zeros(*shape, device=device), torch.zeros(*shape, device=device))
