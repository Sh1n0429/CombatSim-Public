"""Fast installation and reward-hook smoke test; performs no training."""
from pathlib import Path

from combatsim.env import Config
from combatsim.factory import load_settings, make_env


def main():
    root = Path(__file__).resolve().parents[1]
    settings = load_settings(root / 'configs' / 'scenario-default.json')
    env = make_env(Config(**settings))
    observations, _ = env.reset(seed=123)
    steps = 0
    while not env.done and steps < 5:
        actions = {agent: (5, 5, 0) for agent in observations}
        observations, rewards, _, _, info = env.step(actions)
        assert all(isinstance(value, float) for value in rewards.values())
        steps += 1
    component = env.reward_system.components[0]
    assert component.profile == 'custom'
    print(f'OK: Python package, scenario and CustomRewardHooks ({steps} steps)')
    print(f'Reward breakdown keys: {sorted(info["reward_components"])}')


if __name__ == '__main__':
    main()
