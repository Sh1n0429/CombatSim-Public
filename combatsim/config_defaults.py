"""Configuration defaults shared by entry points and low-level components."""
from copy import deepcopy
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs'


def read_defaults(name):
    with (CONFIG_DIR / name).open(encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f'{name} must contain an object')
    return data


ENVIRONMENT_DEFAULTS = read_defaults('environment-defaults.json')
RULE_DEFAULTS = read_defaults('rule-defaults.json')
APPLICATION_DEFAULTS = read_defaults('application.json')


def environment_default(name):
    if name in ('blue_spawn', 'red_spawn'):
        return tuple(tuple(bounds) for bounds in ENVIRONMENT_DEFAULTS[name])
    return deepcopy(ENVIRONMENT_DEFAULTS[name])

REWARD_DEFAULTS = read_defaults('reward-defaults.json')
