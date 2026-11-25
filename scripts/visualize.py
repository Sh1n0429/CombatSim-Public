"""Create an HTML replay of a learned checkpoint against a rule opponent."""
from combatsim.config_defaults import APPLICATION_DEFAULTS

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from combatsim.factory import load_settings
from combatsim.checkpoints import load_checkpoint_network, validate_checkpoint_schema
from combatsim.inference import rule_match_generator, select_actions
from combatsim.replay import write_replay
from combatsim.rllib_env import CombatSimFlightEnv


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(root/'artifacts'/'checkpoint'))
    parser.add_argument('--scenario', help='Override the scenario saved in the checkpoint')
    parser.add_argument('--seed', type=int, default=APPLICATION_DEFAULTS['replay']['seed'])
    parser.add_argument('--learned-side', choices=('blue','red'), default=APPLICATION_DEFAULTS['replay']['learned_side'])
    parser.add_argument('--rule-profile', choices=('fix', 'random'), default=APPLICATION_DEFAULTS['replay']['rule_profile'])
    parser.add_argument('--action-mode', choices=('sample','greedy'), default=APPLICATION_DEFAULTS['replay']['action_mode'])
    parser.add_argument('--output', default=str(root/'artifacts'/'replay.html'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    network, _, _ = load_checkpoint_network(args.checkpoint)
    settings = (load_settings(args.scenario) if args.scenario else
                validate_checkpoint_schema(args.checkpoint)['scenario_settings'])
    env = CombatSimFlightEnv({'settings': settings, 'rule_ratio': 1})
    rule_side = 'red' if args.learned_side == 'blue' else 'blue'
    observations, _ = env.reset(seed=args.seed,
                                options={'rule_side':rule_side,
                                         'rule_profile':args.rule_profile})
    frames = [env.env.snapshot()]
    generator = rule_match_generator(args.seed, args.learned_side)
    recurrent_states = {}
    with torch.inference_mode():
        while True:
            agents = sorted(observations)
            actions = select_actions(network, observations, agents, args.action_mode, generator,
                                     states=recurrent_states)
            observations, _, terminated, truncated, _ = env.step(
                dict(zip(agents, actions)))
            frames.append(env.env.snapshot())
            if terminated['__all__'] or truncated['__all__']:
                break
    write_replay(args.output, frames, asdict(env.config))
    summary = {'winner':env.env.winner, 'ticks':env.env.tick, 'seed':args.seed,
               'learned_side':args.learned_side, 'rule_profile':args.rule_profile,
               'output':str(Path(args.output).resolve())}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
