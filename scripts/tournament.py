"""Run model-versus-model team matches or mixed-team exchange matches."""
from combatsim.config_defaults import APPLICATION_DEFAULTS

import argparse
from itertools import combinations
import json
from pathlib import Path

import torch

from combatsim.env import Config
from combatsim.tournament import (load_model,
                              load_tournament_settings, run_match,
                              settings_for_team_sizes, standings_from_matches,
                              model_team_assignments, team_assignments,
                              team_standings_from_matches, write_results)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=str(root / "models"))
    parser.add_argument("--scenario", default=str(root / "configs" / "scenario-default.json"))
    parser.add_argument("--mode", choices=("team", "exchange"), default="team")
    parser.add_argument("--teams",
                        help="JSON file defining the two fixed teams; required for exchange mode")
    # Each seed is played twice with Blue/Red swapped: 50 seeds = 100 matches per pair.
    # Use --seeds only when an explicit seed list is needed.
    parser.add_argument("--seed-start", type=int, default=APPLICATION_DEFAULTS['tournament']['seed_start'])
    parser.add_argument("--num-seeds", type=int, default=APPLICATION_DEFAULTS['tournament']['num_seeds'])
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Explicit seed list; overrides --seed-start/--num-seeds")
    parser.add_argument("--action-mode", choices=("greedy", "sample"), default=APPLICATION_DEFAULTS['tournament']['action_mode'])
    parser.add_argument("--output", default=str(root / "artifacts" / "tournament"))
    parser.add_argument("--replays", action="store_true",
                        help="Save every match as HTML; this uses more disk space")
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("--num-seeds must be positive")
    if args.seeds is None:
        args.seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    if any(not 0 <= seed < 2**32 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be distinct integers between 0 and 4294967295')
    torch.set_num_threads(1)
    folders = sorted(path for path in Path(args.models).iterdir() if path.is_dir())
    loaded = [load_model(path) for path in folders]
    networks = {model_id: network for model_id, network, _ in loaded}
    if len(networks) != len(loaded):
        raise ValueError("Every model must have a unique model_id")
    if len(networks) < 2:
        raise ValueError("At least two valid models are required")
    settings = load_tournament_settings(args.scenario)
    scenario = Config(**settings)
    roster = scenario.aircraft or [
        {'id': f'{side}_{index}', 'team': side}
        for side, count in (('blue', scenario.blue), ('red', scenario.red))
        for index in range(count)]
    output = Path(args.output) / args.mode
    matches = []
    if args.mode == "team":
        schedule = []
        for left, right in combinations(sorted(networks), 2):
            for seed in args.seeds:
                schedule.extend([(seed, team_assignments(left, right, roster)),
                                 (seed, team_assignments(right, left, roster))])
    else:
        if not args.teams:
            raise ValueError("Exchange mode requires --teams examples/teams.json")
        team_config = json.loads(Path(args.teams).read_text(encoding="utf-8"))
        if not isinstance(team_config, dict) or len(team_config) != 2:
            raise ValueError("The teams file must contain exactly two teams")
        team_names = list(team_config)
        def parse_team(name):
            value = team_config[name]
            if isinstance(value, list) and value:
                return list(map(str, value)), None
            if (isinstance(value, dict) and set(value) == {"models", "lineup"}
                    and isinstance(value["models"], list) and value["models"]
                    and isinstance(value["lineup"], list) and value["lineup"]):
                return list(map(str, value["models"])), list(map(str, value["lineup"]))
            raise ValueError(f"Invalid team definition: {name}")
        (first, first_lineup), (second, second_lineup) = (
            parse_team(name) for name in team_names)
        if (first_lineup is None) != (second_lineup is None):
            raise ValueError("Use explicit lineup for both teams or for neither team")
        configured = first + second
        if len(set(configured)) != len(configured):
            raise ValueError("Every model_id must belong to exactly one team")
        if set(configured) != set(networks):
            missing = sorted(set(networks) - set(configured))
            unknown = sorted(set(configured) - set(networks))
            raise ValueError(f"Team/model mismatch; unassigned={missing}, unknown={unknown}")
        schedule = []
        for index, seed in enumerate(args.seeds):
            # Every seed produces a paired match. This separates team strength
            # from any blue/red advantage in spawn geometry or learned behavior.
            first_blue = model_team_assignments(
                first, second, index, seed, first_lineup, second_lineup)
            second_blue = model_team_assignments(
                second, first, index, seed, second_lineup, first_lineup)
            schedule.extend([
                (seed, first_blue, {"blue": team_names[0], "red": team_names[1]}),
                (seed, second_blue, {"blue": team_names[1], "red": team_names[0]}),
            ])
    if args.mode == "team":
        schedule = [(seed, assignments, None) for seed, assignments in schedule]
    for number, (seed, assignments, match_team_names) in enumerate(schedule, 1):
        replay = output / "replays" / f"match-{number:04d}-seed-{seed}.html" if args.replays else None
        match_settings = settings
        if match_team_names:
            blue_count = sum(agent.startswith("blue_") for agent in assignments)
            red_count = sum(agent.startswith("red_") for agent in assignments)
            match_settings = settings_for_team_sizes(settings, blue_count, red_count)
        result = run_match(networks, assignments, match_settings, seed, args.action_mode, replay)
        result.update(match=number, mode=args.mode, action_mode=args.action_mode)
        if match_team_names:
            result["team_names"] = match_team_names
        matches.append(result)
        print(f"[{number:04d}/{len(schedule):04d}] seed={seed} winner={result['winner'] or 'draw'}")
    standings = standings_from_matches(matches)
    team_standings = team_standings_from_matches(matches) if args.mode == "exchange" else None
    write_results(output, matches, standings, team_standings)
    if team_standings is not None:
        print("Team standings:")
        print(json.dumps(team_standings, ensure_ascii=False, indent=2))
    print("Model statistics:")
    print(json.dumps(standings, ensure_ascii=False, indent=2))
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
