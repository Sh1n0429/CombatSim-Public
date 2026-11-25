"""Run a local simulated match and export a replay of the configured rewards."""
import argparse
from dataclasses import asdict
from pathlib import Path

from combatsim.env import Config
from combatsim.factory import load_settings
from combatsim.tournament import MatchRunner
from combatsim.rule_profiles import normalize_rule_settings
from combatsim.replay import write_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', default='configs/scenario-default.json')
    parser.add_argument('--blue-profile', choices=('fix', 'random'), default='fix')
    parser.add_argument('--red-profile', choices=('fix', 'random'), default='fix')
    parser.add_argument('--seed', type=int, default=27)
    parser.add_argument('--output', default='artifacts/reward-preview/replay.html')
    args = parser.parse_args()
    settings = load_settings(args.scenario)
    config = Config(**normalize_rule_settings(settings))
    profiles = {'blue': args.blue_profile, 'red': args.red_profile}
    roster = config.aircraft or [
        {'id': f'{side}_{i}', 'team': side}
        for side, count in (('blue', config.blue), ('red', config.red))
        for i in range(count)]
    assignments = {a['id']: f"rule:{profiles[a['team']]}" for a in roster}
    runner = MatchRunner({}, assignments, settings, args.seed,
                         rule_profiles={f'rule:{profile}': profile for profile in profiles.values()})
    frames = [runner.snapshot()]
    while not runner.done:
        runner.step(record_tick=frames.append)
    write_replay(args.output, frames, {**asdict(runner.env.config),
                 'models': {side: {'policy': f'ルール / {profile}'}
                            for side, profile in profiles.items()}})
    print(Path(args.output).resolve())


if __name__ == '__main__':
    main()
