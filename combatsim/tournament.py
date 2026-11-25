"""Safe portable models and deterministic model-versus-model matches."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import random

import torch

from .factory import load_settings
from .inference import select_actions
from .observation_schema import OBSERVATION_VERSION
from .replay import write_replay
from .rllib_env import CombatSimFlightEnv
from .set_transformer import SetTransformerActorCritic, head_hidden_sizes_from_weights

MODEL_FORMAT = "combatsim-model-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(directory):
    """Load tensors only. Student pickle/code is never executed."""
    directory = Path(directory).resolve()
    metadata_path, model_path = directory / "metadata.json", directory / "model.pth"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != MODEL_FORMAT:
        raise ValueError(f"Unsupported model format in {metadata_path}")
    if metadata.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError(f"Observation version mismatch in {metadata_path}")
    if metadata.get("sha256") != file_sha256(model_path):
        raise ValueError(f"Model checksum mismatch in {directory}")
    model_id = metadata.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"model_id is missing in {metadata_path}")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError(f"model.pth must contain only a tensor state_dict: {directory}")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError(f"Model contains NaN or infinite weights: {directory}")
    network = SetTransformerActorCritic(head_hidden_sizes=head_hidden_sizes_from_weights(state))
    network.load_state_dict(state, strict=True)
    network.eval()
    return model_id, network, metadata


class MatchRunner:
    """Incremental match engine shared by command-line and live evaluation."""

    def __init__(self, networks, assignments, settings, seed, action_mode="greedy", rule_profiles=None):
        if action_mode not in ("greedy", "sample"):
            raise ValueError("action_mode must be greedy or sample")
        self.networks, self.assignments = networks, dict(assignments)
        self.seed, self.action_mode = seed, action_mode
        self.env = CombatSimFlightEnv({"settings": deepcopy(settings), "rule_ratio": 0})
        if set(assignments) != set(self.env.possible_agents):
            raise ValueError("Assignments must name every aircraft exactly once")
        rule_profiles = rule_profiles or {}
        unknown = set(assignments.values()) - set(networks) - set(rule_profiles)
        if unknown:
            raise ValueError(f"Unknown models in assignments: {sorted(unknown)}")
        rule_sides = {}
        for side in ('blue', 'red'):
            controllers = {assignments[a.id] for a in self.env.env.aircraft.values() if a.team == side}
            rules = controllers & set(rule_profiles)
            if rules:
                if len(controllers) != 1:
                    raise ValueError("ルール相手はチーム全体に指定してください。混成チームには学習モデルを指定します。")
                rule_sides[side] = rule_profiles[next(iter(rules))]
        self.rule_profiles_by_side = dict(rule_sides)
        side = next(iter(rule_sides), None) if len(rule_sides) == 1 else None
        options = {'rule_side': side}
        if side is not None:
            options['rule_profile'] = rule_sides[side]
        if len(rule_sides) == 2:
            options['rule_profiles_by_side'] = rule_sides
        self.observations, _ = self.env.reset(seed=seed, options=options)
        self.generators = {name: torch.Generator().manual_seed(seed*1009+index)
                           for index, name in enumerate(sorted(networks))}
        self.recurrent_states = {name: {} for name in networks}

    @property
    def done(self):
        return self.env.env.done

    def snapshot(self):
        return self.env.env.snapshot()

    @torch.inference_mode()
    def step(self, *, record_tick=None):
        if self.done:
            return self.snapshot()
        actions, grouped = {}, defaultdict(list)
        for agent in sorted(self.observations):
            if self.assignments[agent] in self.networks:
                grouped[self.assignments[agent]].append(agent)
        for model_id, agents in grouped.items():
            batch = select_actions(self.networks[model_id], self.observations, agents,
                                   self.action_mode, self.generators[model_id],
                                   states=self.recurrent_states[model_id])
            actions.update(zip(agents, batch))
        def on_tick():
            record_tick(self.snapshot())

        self.observations, _, _, _, _ = self.env.step(
            actions, on_tick=on_tick if record_tick is not None else None)
        return self.snapshot()

    def result(self, replay=None):
        if not self.done:
            raise ValueError("対戦が完了していません。")
        world = self.env.env
        missiles = world.missiles
        personal = {}
        for agent, model_id in self.assignments.items():
            aircraft = world.aircraft[agent]
            personal[agent] = {
                "model_id": model_id, "team": aircraft.team,
                "return": world.episode_returns[agent], "survived": bool(aircraft.alive),
                "boundary_override_fraction": self.env.episode_control_counts.get(agent, {}).get('overrides', 0) /
                    max(1, self.env.episode_control_counts.get(agent, {}).get('ticks', 0)),
                "launches": sum(m.owner == agent for m in missiles),
                "kills": len({m.target for m in missiles if m.owner == agent and m.phase == "hit"}),
            }
        rule_parameters = {}
        for side in self.rule_profiles_by_side:
            policy = (self.env.rule_policy if self.env.rule_side == side else
                      self.env.rule_policies_by_side[side])
            rule_parameters[side] = dict(policy.parameters)
        return {"seed": self.seed, "winner": world.winner, "draw": world.winner is None,
                "ticks": world.tick, "assignments": dict(self.assignments), "personal": personal,
                "rule_profiles_by_side": dict(self.rule_profiles_by_side),
                "rule_parameters_by_side": rule_parameters,
                "replay": str(Path(replay).resolve()) if replay else None}


def run_match(networks, assignments, settings, seed, action_mode="greedy", replay=None):
    """Run one configured match; assignments maps every aircraft ID to a model ID."""
    runner = MatchRunner(networks, assignments, settings, seed, action_mode)
    frames = [runner.snapshot()] if replay else None
    while not runner.done:
        frame = runner.step()
        if replay:
            frames.append(frame)
    if replay:
        write_replay(replay, frames, asdict(runner.env.config))
    return runner.result(replay)


def team_assignments(blue, red, roster=None):
    if roster is None:  # Preserve the original four-aircraft helper interface.
        roster = [{'id': f'{side}_wing{i}', 'team': side}
                  for side in ('blue', 'red') for i in range(4)]
    return {aircraft['id']: blue if aircraft['team'] == 'blue' else red
            for aircraft in roster}


def model_team_assignments(blue_models, red_models, round_index, seed=None,
                             blue_lineup=None, red_lineup=None):
    """Give everyone one aircraft and randomly fill the smaller team's shortage."""
    blue_models, red_models = list(blue_models), list(red_models)
    if not blue_models or not red_models:
        raise ValueError("Both exchange teams require at least one model")
    if len(set(blue_models)) != len(blue_models) or len(set(red_models)) != len(red_models):
        raise ValueError("A model may appear only once in a team roster")
    if set(blue_models) & set(red_models):
        raise ValueError("A model cannot belong to both exchange teams")
    if (blue_lineup is None) != (red_lineup is None):
        raise ValueError("Specify a lineup for both teams or for neither team")
    if blue_lineup is not None:
        blue_lineup, red_lineup = list(blue_lineup), list(red_lineup)
        if not blue_lineup or len(blue_lineup) != len(red_lineup):
            raise ValueError("Explicit lineups must be nonempty and have the same aircraft count")
        if set(blue_lineup) - set(blue_models) or set(red_lineup) - set(red_models):
            raise ValueError("A lineup may contain only models from its own team")
        if set(blue_models) - set(blue_lineup) or set(red_models) - set(red_lineup):
            raise ValueError("Every model must appear at least once in an explicit lineup")
        return {
            **{f"blue_wing{index}": model for index, model in enumerate(blue_lineup)},
            **{f"red_wing{index}": model for index, model in enumerate(red_lineup)},
        }
    # Rotate aircraft numbers so the same model is not always the lead aircraft.
    blue_shift = round_index % len(blue_models)
    red_shift = round_index % len(red_models)
    blue_models = blue_models[blue_shift:] + blue_models[:blue_shift]
    red_models = red_models[red_shift:] + red_models[:red_shift]
    target_size = max(len(blue_models), len(red_models))
    rng = random.Random(f"combatsim-team-fill:{seed}:{round_index}")

    def fill(models):
        seats = list(models)
        while len(seats) < target_size:
            seats.append(rng.choice(models))
        return seats

    blue_seats, red_seats = fill(blue_models), fill(red_models)
    return {
        **{f"blue_wing{index}": model for index, model in enumerate(blue_seats)},
        **{f"red_wing{index}": model for index, model in enumerate(red_seats)},
    }


def settings_for_team_sizes(settings, blue_count, red_count):
    """Create an N-vs-M roster while retaining the official aircraft performance."""
    if blue_count < 1 or red_count < 1:
        raise ValueError("Both sides require at least one aircraft")
    # The SetTransformer selects at most 32 entities. Keep the tournament within
    # that documented observation budget, including the observing aircraft.
    if blue_count + red_count > 32:
        raise ValueError("At most 32 models can participate in one match")
    result = deepcopy(settings)
    roster = []
    for side, count in (("blue", blue_count), ("red", red_count)):
        templates = [a for a in (settings.get("aircraft") or []) if a["team"] == side]
        for index in range(count):
            template = deepcopy(templates[index % len(templates)]) if templates else {}
            roster.append({**template, "id": f"{side}_wing{index}", "team": side})
    result["aircraft"] = roster
    return result


def team_standings_from_matches(matches):
    """Aggregate results for the two named simulator teams."""
    table = defaultdict(lambda: {"played": 0, "wins": 0, "draws": 0, "losses": 0,
                                 "points": 0, "launches": 0, "kills": 0, "deaths": 0,
                                 "return": 0.0, "survivals": 0})
    for match in matches:
        side_names = match.get("team_names")
        if not side_names:
            continue
        for side in ("blue", "red"):
            name, row = side_names[side], table[side_names[side]]
            own = [result for result in match["personal"].values() if result["team"] == side]
            row["played"] += 1
            won = match["winner"] == side
            lost = match["winner"] is not None and not won
            row["wins"] += int(won)
            row["draws"] += int(match["draw"])
            row["losses"] += int(lost)
            row["points"] += 3 * int(won) + int(match["draw"])
            row["launches"] += sum(result["launches"] for result in own)
            row["kills"] += sum(result["kills"] for result in own)
            row["deaths"] += sum(not result["survived"] for result in own)
            row["survivals"] += sum(result["survived"] for result in own)
            row["return"] += sum(result["return"] for result in own)
    rows = [{"team": name, **values,
             "kill_difference": values["kills"] - values["deaths"],
             "mean_launches": values["launches"] / max(1, values["played"]),
             "mean_return": values["return"] / max(1, values["played"])}
            for name, values in table.items()]
    rows.sort(key=lambda row: (-row["points"], -row["kill_difference"],
                               -row["mean_return"], row["team"]))
    return [{"rank": rank, **row} for rank, row in enumerate(rows, 1)]


def standings_from_matches(matches):
    table = defaultdict(lambda: {"played": 0, "wins": 0, "draws": 0, "losses": 0,
                                 "points": 0, "launches": 0, "kills": 0, "deaths": 0,
                                 "return": 0.0, "survivals": 0})
    for match in matches:
        by_model = defaultdict(list)
        for result in match["personal"].values():
            by_model[result["model_id"]].append(result)
        for model_id, aircraft_results in by_model.items():
            row = table[model_id]
            row["played"] += 1
            teams = {item["team"] for item in aircraft_results}
            won = match["winner"] is not None and match["winner"] in teams
            lost = match["winner"] is not None and not won
            row["wins"] += int(won)
            row["draws"] += int(match["draw"])
            row["losses"] += int(lost)
            row["points"] += 3 * int(won) + int(match["draw"])
            row["launches"] += sum(item["launches"] for item in aircraft_results)
            row["kills"] += sum(item["kills"] for item in aircraft_results)
            row["deaths"] += sum(not item["survived"] for item in aircraft_results)
            row["survivals"] += sum(item["survived"] for item in aircraft_results)
            row["return"] += sum(item["return"] for item in aircraft_results)
    rows = [{"model_id": name, **values,
             "kill_difference": values["kills"] - values["deaths"],
             "mean_launches": values["launches"] / max(1, values["played"]),
             "mean_return": values["return"] / max(1, values["played"])}
            for name, values in table.items()]
    rows = sorted(rows, key=lambda row: (-row["points"], -row["kill_difference"],
                                         -row["mean_return"], row["model_id"]))
    return [{"rank": rank, **row} for rank, row in enumerate(rows, 1)]


def _write_table(output_dir, stem, rows):
    (output_dir / f"{stem}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (output_dir / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_results(output_dir, matches, standings, team_standings=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "matches.jsonl").open("w", encoding="utf-8") as stream:
        for match in matches:
            stream.write(json.dumps(match, ensure_ascii=False) + "\n")
    _write_table(output_dir, "standings", standings)
    if team_standings is not None:
        _write_table(output_dir, "team-standings", team_standings)


def load_tournament_settings(path):
    return load_settings(path)
