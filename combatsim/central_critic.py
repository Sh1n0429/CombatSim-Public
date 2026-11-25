"""Independent set-based MAPPO critic; privileged inputs never enter the actor."""
import torch
from torch import nn

from .central_state import AIRCRAFT_STATE_FIELDS, MISSILE_STATE_FIELDS
from .set_transformer import EMBED_SIZE, PMA_SEEDS, PoolingByMultiheadAttention


def embedding(width):
    return nn.Sequential(nn.Linear(width, EMBED_SIZE), nn.ReLU(),
                         nn.Linear(EMBED_SIZE, EMBED_SIZE), nn.ReLU())


class CentralSetCritic(nn.Module):
    """128-dim type embeddings, two attention layers, four-slot PMA, value MLP.

    All aircraft (including dead ones) and all live missiles are represented.
    Owner/target edges use aircraft embeddings rather than ordinal identity IDs.
    Privileged memory is separate from the actor's recurrent hidden state.
    """
    def __init__(self):
        super().__init__()
        self.common_mlp = embedding(1)
        self.aircraft_mlp = embedding(len(AIRCRAFT_STATE_FIELDS))
        self.missile_mlp = embedding(len(MISSILE_STATE_FIELDS))
        self.owner_projection = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.target_projection = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(EMBED_SIZE, 4, batch_first=True) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(EMBED_SIZE) for _ in range(2)])
        self.pma = PoolingByMultiheadAttention()
        self.lstm = nn.LSTM(PMA_SEEDS*EMBED_SIZE, EMBED_SIZE, batch_first=True)
        self.value_head = nn.Sequential(nn.Linear(EMBED_SIZE, EMBED_SIZE),
                                        nn.ReLU(), nn.Linear(EMBED_SIZE, 1))

    def _encode(self, observation):
        state = observation['central_state']
        common = self.common_mlp(state[:, :1]).unsqueeze(1)
        aircraft = self.aircraft_mlp(state[:, 1:].reshape(
            state.shape[0], -1, len(AIRCRAFT_STATE_FIELDS)))
        invalid = observation['central_missiles_mask'] <= .5
        # Drop columns empty across the batch; never drop a live projectile.
        columns = (~invalid).any(0).nonzero(as_tuple=True)[0]
        invalid = invalid[:, columns]
        values = observation['central_missiles'][:, columns].masked_fill(invalid.unsqueeze(-1), 0)
        missiles = self.missile_mlp(values)
        owners = observation['central_missile_owners'][:, columns].masked_fill(invalid.unsqueeze(-1), 0)
        targets = observation['central_missile_targets'][:, columns].masked_fill(invalid.unsqueeze(-1), 0)
        missiles = missiles + self.owner_projection(owners @ aircraft) + self.target_projection(targets @ aircraft)
        hidden = torch.cat((common, aircraft, missiles), dim=1)
        mask = torch.cat((invalid.new_zeros(state.shape[0], 1+aircraft.shape[1]), invalid), dim=1)
        for attention, norm in zip(self.attention, self.norms):
            attended, _ = attention(hidden, hidden, hidden, key_padding_mask=mask, need_weights=False)
            hidden = norm(hidden + attended)
        return self.pma(hidden, mask)

    def forward_sequence(self, observation, hidden, seq_lens):
        pooled = self._encode(observation)
        batch = len(seq_lens)
        time = pooled.shape[0] // batch
        packed = nn.utils.rnn.pack_padded_sequence(
            pooled.reshape(batch, time, -1), torch.as_tensor(seq_lens).detach().cpu().clamp_min(1),
            batch_first=True, enforce_sorted=False)
        if hidden is not None:
            hidden = tuple(h.unsqueeze(0) if h.dim() == 2 else h for h in hidden)
        recurrent, updated = self.lstm(packed, hidden)
        recurrent, _ = nn.utils.rnn.pad_packed_sequence(recurrent, batch_first=True, total_length=time)
        values = self.value_head(recurrent.reshape(batch*time, -1)).squeeze(-1)
        return values, tuple(h.squeeze(0) for h in updated)

    def forward(self, observation):
        # One-step convenience for diagnostics; RLlib uses forward_sequence.
        lengths = torch.ones(observation['central_state'].shape[0], dtype=torch.int32)
        return self.forward_sequence(observation, None, lengths)[0]
