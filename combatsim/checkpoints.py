"""Atomic tensor saves and shared loading of locally produced RLlib checkpoints."""
from pathlib import Path
import hashlib
import io
import json
import os
import pickle
import tempfile
import torch


def resolve_resume_checkpoint(path):
    """Accept an exact checkpoint or its training output directory."""
    path = Path(path).resolve()
    pointer = path if path.name == 'latest-checkpoint.txt' else path/'latest-checkpoint.txt'
    if pointer.is_file():
        target = pointer.read_text(encoding='utf-8').strip()
        if target:
            candidate = (pointer.parent / target).resolve()
            if (candidate/'training-state.json').is_file():
                return candidate
        raise ValueError(f'No completed checkpoint in {pointer}')
    if (path/'training-state.json').is_file():
        return path
    raise ValueError(f'No training checkpoint or latest-checkpoint.txt in {path}')


def validate_checkpoint_schema(checkpoint):
    from .observation_schema import OBSERVATION_VERSION
    path = Path(checkpoint)/'training-state.json'
    metadata = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if metadata.get('observation_version') != OBSERVATION_VERSION:
        raise ValueError('Checkpoint observation/recurrent semantics are incompatible; '
                         f'preserve the old checkpoint and start a new version-{OBSERVATION_VERSION} training run.')
    return metadata


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_torch(path, value):
    stream = io.BytesIO()
    torch.save(value,stream)
    data = stream.getvalue()
    atomic_bytes(path,data)
    return data


def read_policy_checkpoint(checkpoint, policy_id='main'):
    """Return CPU weights, RLlib policy metadata and the source SHA-256.

    Native RLlib pickle files must come from a trusted local training run.
    Exchanged models use tournament.load_model and weights_only=True.
    """
    checkpoint = Path(checkpoint)
    if policy_id == 'initial':
        data = (checkpoint/'run-initial-weights.pth').read_bytes()
        weights = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        state = {}
    else:
        data = (checkpoint/'policies'/policy_id/'policy_state.pkl').read_bytes()
        state = pickle.loads(data)
        weights = {key.removeprefix('network.'): torch.as_tensor(value).detach().cpu()
                   for key, value in state['weights'].items() if key.startswith('network.')}
    return weights, state, hashlib.sha256(data).hexdigest()


def load_checkpoint_network(checkpoint, policy_id='main'):
    from .set_transformer import SetTransformerActorCritic, head_hidden_sizes_from_weights

    validate_checkpoint_schema(checkpoint)
    weights, state, digest = read_policy_checkpoint(checkpoint, policy_id)
    network = SetTransformerActorCritic(head_hidden_sizes=head_hidden_sizes_from_weights(weights))
    network.load_state_dict(weights, strict=True)
    network.eval()
    return network, state, digest
