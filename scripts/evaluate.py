"""Evaluate saved policy weights using the experiment YAML; no training or Ray cluster."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import uuid

import torch

from combatsim.rllib_evaluation import evaluate_network, format_opponent_results
from combatsim.rllib_logging import append_record
from combatsim.checkpoints import load_checkpoint_network
if __package__:
    from .train import parse_args
else:
    from train import parse_args


def evaluate_policy(checkpoint, policy_id, seeds, action_mode, rule_profile='fix', settings=None):
    torch.set_num_threads(1)
    checkpoint = Path(checkpoint)
    management = json.loads((checkpoint / 'training-state.json').read_text(encoding='utf-8'))
    network, state, digest = load_checkpoint_network(checkpoint, policy_id)
    result = evaluate_network(network, settings or management['scenario_settings'], seeds,
                              action_mode, rule_profile)
    result.update(type='heldout_evaluation', policy_id=policy_id, seeds=seeds,
                  action_mode=action_mode, checkpoint=str(checkpoint.resolve()),
                  rule_profile=rule_profile,
                  checkpoint_sha256=digest,
                  initial_iteration=management.get('initial_iteration'),
                  gradient_updates=state.get('num_grad_updates'),
                  snapshot_generation=management['generations'].get(policy_id))
    return result


def main():
    args = parse_args()
    checkpoint = Path(args.resume or args.checkpoint_dir).resolve()
    if args.eval_num_workers < 1:
        raise ValueError('eval_num_workers must be positive')
    run_id = uuid.uuid4().hex[:12]
    print(f'[HELDOUT] policies={args.eval_policy_ids} seeds={len(args.eval_seeds)} '
          f'x both sides | checkpoint={checkpoint}', flush=True)
    with ProcessPoolExecutor(max_workers=args.eval_num_workers) as pool:
        futures = [pool.submit(evaluate_policy, str(checkpoint), policy_id,
                               args.eval_seeds, args.eval_action_mode,
                               args.eval_rule_profile)
                   for policy_id in args.eval_policy_ids]
        for future in as_completed(futures):
            result = future.result()
            result['run_id'] = run_id
            append_record(args.eval_log, result)
            print(f"[HELDOUT {result['policy_id']}] {format_opponent_results(result)} | W/D/L="
                  f"{result['wins']}/{result['draws']}/{result['losses']} "
                  f"score={result['score']:.3f} return={result['mean_team_return']:+.3f} "
                  f"launches={result['mean_launches']:.2f} kills={result['mean_kills']:.2f} "
                  f"outside={result['outside_decision_fraction']:.1%}", flush=True)


if __name__ == '__main__':
    main()
