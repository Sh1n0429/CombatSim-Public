"""Single environment construction point for the CLI (run/train/evaluate/live)."""
import json
import math
from pathlib import Path

from .env import Config, ToyEnv
from .config_defaults import CONFIG_DIR


def load_settings(path, _parents=()):
    """Resolve environment, airspace and reward files relative to their parent."""
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError('Circular environment_config reference')
    settings = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Config must be a JSON object")
    if 'environment_config' in settings:
        reference = settings.pop('environment_config')
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError('environment_config must be a non-empty file path')
        base = load_settings(path.parent/reference, (*_parents, path))
    else:
        base = {}
    if "airspace_config" in settings:
        reference = settings.pop("airspace_config")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("airspace_config must be a non-empty file path")
        airspace = json.loads((path.parent/reference).read_text(encoding="utf-8"))
        if not isinstance(airspace, dict) or set(airspace) - {"arena_x", "arena_y", "ceiling", "blue_spawn", "red_spawn"}:
            raise ValueError("Airspace config supports only arena_x, arena_y, ceiling, blue_spawn and red_spawn")
        if set(settings) & set(airspace):
            raise ValueError("Airspace settings must not be duplicated in the main config")
        settings.update(airspace)
    if 'reward_config' in settings:
        reference = settings.pop('reward_config')
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError('reward_config must be a non-empty file path')
        if 'rewards' in settings:
            raise ValueError('Specify either reward_config or inline rewards, not both')
        reward_settings = json.loads((path.parent/reference).read_text(encoding='utf-8'))
        if (not isinstance(reward_settings, dict) or set(reward_settings) != {'rewards'}
                or not isinstance(reward_settings['rewards'], list)):
            raise ValueError('Reward config must contain only a rewards list')
        settings['rewards'] = reward_settings['rewards']
    merged = {**base, **settings}
    # Optional human-readable terminal/time reward overrides.  Keep the
    # component list as the canonical runtime representation so older reward
    # files and checkpoints remain valid.
    draw_reward = merged.pop('draw_reward', None)
    time_reward = merged.pop('time_reward_at_limit', None)
    if draw_reward is not None or time_reward is not None:
        rewards = [dict(spec) for spec in merged.get('rewards', [])]
        if draw_reward is not None:
            if type(draw_reward) not in (int, float) or not math.isfinite(draw_reward):
                raise ValueError('draw_reward must be a finite number')
            victory = next((spec for spec in rewards if spec.get('type') == 'victory'), None)
            if victory is None:
                rewards.insert(0, {'type': 'victory', 'draw': draw_reward})
            else:
                victory['draw'] = draw_reward
        if time_reward is not None:
            if (type(time_reward) not in (int, float) or not math.isfinite(time_reward)
                    or time_reward > 0):
                raise ValueError('time_reward_at_limit must be a finite nonpositive number')
            max_steps = merged.get('max_steps', Config.max_steps)
            if type(max_steps) is not int or max_steps < 1:
                raise ValueError('max_steps must be positive for time_reward_at_limit')
            rewards = [spec for spec in rewards if spec.get('type') != 'time_penalty']
            if time_reward:
                rewards.append({'type': 'time_penalty',
                                'penalty': time_reward/max_steps,
                                'max_total': abs(time_reward)})
        merged['rewards'] = rewards
    return merged


def make_env(config: Config | None = None) -> ToyEnv:
    if config is None:
        config = Config(**load_settings(CONFIG_DIR / 'scenario-default.json'))
    return ToyEnv(config)
