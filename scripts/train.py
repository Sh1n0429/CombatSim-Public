"""Train a shared PPO or MAPPO policy through self-play and rule matches."""
import argparse
import copy
import cProfile
import json
import os
import secrets
import signal
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from combatsim.ray_runtime import configure_ray_runtime

configure_ray_runtime()  # Ray reads this setting at import time.

import ray
import torch
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.train._internal.storage import StorageContext
from torch.utils.tensorboard import SummaryWriter

from combatsim.checkpoints import (save_torch, validate_checkpoint_schema, read_policy_checkpoint,
                                   resolve_resume_checkpoint)
from combatsim.tournament import load_model
from combatsim.set_transformer import head_hidden_sizes_from_weights, validate_head_hidden_sizes
from combatsim.observation_schema import OBSERVATION_VERSION
from combatsim.config_defaults import APPLICATION_DEFAULTS, ENVIRONMENT_DEFAULTS
from combatsim.factory import load_settings
from combatsim.rule_profiles import normalize_rule_settings
from combatsim.rllib_env import CombatSimFlightEnv
from combatsim.rllib_evaluation import evaluate_algorithm
from combatsim.rllib_experiment import PATH_FIELDS, load_experiment
from combatsim.rllib_logging import append_record, iteration_record, format_iteration
from combatsim.rllib_model import CombatSimSetTransformerModel
from combatsim.rllib_distribution import GatedMultiCategorical
from combatsim.rllib_rating import RatingCoordinator, RLLibRatingCallbacks


TRAINING_PARAMETER_FIELDS = (
    'algorithm',
    'head_hidden_sizes', 'max_seq_len', 'batch_size', 'minibatch_size', 'num_epochs', 'lr', 'gamma',
    'compact_attention', 'gae_lambda', 'clip_param', 'entropy_coeff',
    'vf_loss_coeff', 'vf_clip_param', 'grad_clip', 'kl_coeff', 'kl_target',
)


def restore_training_parameters(args, saved_state, argv=None):
    """A resume restores PPO/optimizer state, so keep matching hyperparameters."""
    explicit = {token.split('=')[0].removeprefix('--').replace('-', '_').removeprefix('no_')
                for token in (sys.argv[1:] if argv is None else argv)
                if token.startswith('--')}
    parameters = {'algorithm': 'ppo', **saved_state.get('training_parameters', {})}
    if getattr(args, 'algorithm', 'ppo') != parameters['algorithm']:
        raise ValueError('Cannot resume with a different algorithm; start a new training run')
    for key, value in parameters.items():
        if key not in TRAINING_PARAMETER_FIELDS:
            continue
        if key in explicit and getattr(args, key) != value:
            raise ValueError(f'--resume preserves saved {key}={value}; '
                             'a different value would disagree with restored PPO/optimizer state')
        setattr(args, key, value)


def restore_opponent_parameters(args, saved_state):
    """Keep saved opponent sampling unless the run explicitly overrides it."""
    args.snapshot_pool_size = saved_state['pool_size']
    if args.resume_override_opponents:
        saved_state = dict(saved_state)
        saved_state.update(k=args.rating_k, snapshot_ratio=args.snapshot_ratio,
                           interval=args.snapshot_interval_matches,
                           rule_ratio=args.rule_ratio, random_rate=args.random_rate)
    else:
        args.rating_k = saved_state['k']
        args.snapshot_ratio = saved_state['snapshot_ratio']
        args.snapshot_interval_matches = saved_state['interval']
        args.rule_ratio = saved_state.get('rule_ratio', args.rule_ratio)
        args.random_rate = saved_state.get('random_rate', args.random_rate)
    return saved_state


def policy_mapping(agent_id, episode, **kwargs):
    del kwargs
    if not agent_id.startswith(('blue_', 'red_')):
        raise ValueError(f'Unknown aircraft agent ID: {agent_id}')
    side = 'blue' if agent_id.startswith('blue_') else 'red'
    return (
        episode.user_data['snapshot_policy_id']
        if episode.user_data.get('snapshot_side') == side else 'main'
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_experiment = Path(__file__).resolve().parents[1] / 'experiments' / 'default.yaml'
    parser.add_argument('--experiment', default=str(default_experiment),
                        help='Experiment YAML (default: experiments/default.yaml)')
    parser.add_argument('--eval-policy-ids', nargs='+')
    parser.add_argument('--eval-num-workers', type=int)
    parser.add_argument('--eval-enabled', action=argparse.BooleanOptionalAction)
    parser.add_argument('--eval-interval', type=int)
    parser.add_argument('--eval-seeds', type=int, nargs='+')
    parser.add_argument('--eval-action-mode', choices=('sample', 'greedy'))
    parser.add_argument('--eval-rule-profile', choices=('fix', 'random', 'both'))
    parser.add_argument('--eval-stop-win-rate', type=float,
                        help='Stop after consecutive evaluations and independent confirmation reach this win rate')
    parser.add_argument('--eval-stop-consecutive', type=int)
    parser.add_argument('--eval-confirm-seeds', type=int, nargs='+')
    parser.add_argument('--eval-log')
    parser.add_argument('--config')
    parser.add_argument('--algorithm', choices=('ppo', 'mappo'),
                        help='PPO with local critic, or MAPPO with privileged aircraft critic')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--checkpoint-interval', type=int,
                        help='Save a numbered checkpoint every N updates; 0 saves only at end')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-epochs', type=int, help='PPO passes per batch; lower values change learning behavior')
    parser.add_argument('--compact-attention', action=argparse.BooleanOptionalAction)
    parser.add_argument('--minibatch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--gae-lambda', type=float)
    parser.add_argument('--clip-param', type=float)
    parser.add_argument('--entropy-coeff', type=float)
    parser.add_argument('--vf-loss-coeff', type=float)
    parser.add_argument('--vf-clip-param', type=float)
    parser.add_argument('--grad-clip', type=float)
    parser.add_argument('--kl-coeff', type=float)
    parser.add_argument('--kl-target', type=float)
    parser.add_argument('--profile', help='Write driver cProfile data and final RLlib timing metrics')
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--resume', help='Checkpoint directory or training output directory with latest-checkpoint.txt')
    parser.add_argument('--init-model', type=Path, help='Start a new run from a model.pth and metadata.json folder')
    parser.add_argument('--resume-override-opponents', action=argparse.BooleanOptionalAction,
                        help='Use opponent ratios from this run when resuming')
    parser.add_argument('--rating-log')
    parser.add_argument('--metrics-log')
    parser.add_argument('--tensorboard-dir',
                        help='TensorBoard root; each run gets its own subdirectory')
    parser.add_argument('--rating-k', type=float)
    parser.add_argument('--rule-ratio', type=float)
    parser.add_argument('--random-rate', type=float,
                        help='Fraction of rule matches using randomized fix parameters; '
                             '0 uses fixed fix, 1 always randomized, omitted uses experiment YAML')
    parser.add_argument('--snapshot-ratio', type=float)
    parser.add_argument('--snapshot-pool-size', type=int)
    parser.add_argument('--snapshot-interval-matches', type=int)
    parser.add_argument('--num-env-runners', type=int)
    parser.add_argument('--num-envs-per-runner', type=int)
    parser.add_argument('--cpus-per-env-runner', type=float)
    parser.add_argument('--learner-threads', type=int,
                        help='CPU threads for the learner; 0 keeps the PyTorch default')
    parser.add_argument('--ray-address')
    parser.add_argument('--ray-temp-dir',
                        help='Writable local Ray session/log directory')
    parser.add_argument('--max-seq-len', type=int)
    parser.add_argument('--head-hidden-sizes', type=int, nargs='*', help='Hidden widths in each output head; empty keeps linear heads')
    parser.add_argument('--seed', type=int)
    selected, _ = parser.parse_known_args(argv)
    try:
        defaults = dict(APPLICATION_DEFAULTS['training'])
        defaults['learner_threads'] = 0
        defaults['algorithm'] = 'ppo'
        defaults.update(load_experiment(default_experiment))
        defaults.update(load_experiment(selected.experiment))
        parser.set_defaults(**defaults)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    args = parser.parse_args(argv)
    if args.init_model is not None:
        if args.resume:
            parser.error('--init-model and --resume cannot be used together')
        args.init_model = str(args.init_model.resolve())
    if args.learner_threads < 0:
        parser.error('learner-threads must be nonnegative')
    # Ray can change the process working directory after construction. Resolve
    # CLI overrides before that, just as the YAML loader resolves its paths.
    for key in PATH_FIELDS:
        value = getattr(args, key, None)
        if value is not None:
            setattr(args, key, str(Path(value).resolve()))
    try:
        validate_head_hidden_sizes(args.head_hidden_sizes)
    except ValueError as error:
        parser.error(str(error))
    if args.max_seq_len < 1:
        parser.error('max-seq-len must be positive')
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error('seed must be between 0 and 4294967295')
    if type(args.random_rate) not in (int, float) or not 0 <= args.random_rate <= 1:
        parser.error('random-rate must be between 0 and 1')
    if args.eval_stop_win_rate is not None:
        if not 0 < args.eval_stop_win_rate <= 1:
            parser.error('eval-stop-win-rate must be in (0, 1]; it applies only when evaluation is enabled')
        if args.eval_stop_consecutive < 1:
            parser.error('eval-stop-consecutive must be positive')
        seeds = args.eval_confirm_seeds
        if (not seeds or len(set(seeds)) != len(seeds)
                or any(not 0 <= seed < 2**32 for seed in seeds)
                or set(seeds) & set(args.eval_seeds)):
            parser.error('eval-confirm-seeds must be unique valid seeds disjoint from eval-seeds')
    return args


def find_metric(result, name):
    """Find a metric across old/new RLlib result nesting."""
    if isinstance(result, dict):
        if name in result:
            return result[name]
        for value in result.values():
            found = find_metric(value, name)
            if found is not None:
                return found
    return None


def write_tensorboard_iteration(writer, record):
    """Write the human-facing metrics with stable, readable tag names."""
    step = record['iteration']
    scalars = {
        'performance/rating': record.get('main_rating'),
        'performance/mean_return': record.get('mean_return'),
        'time/iteration_seconds': record.get('iteration_seconds'),
        'time/sample_seconds': record.get('sample_seconds'),
        'time/learn_seconds': record.get('learn_seconds'),
        'behavior/boundary_override_fraction': record.get('boundary_override_fraction'),
    }
    breakdown = record.get('reward_breakdown', {})
    scalars['reward/original'] = breakdown.get('original')
    scalars['reward/additional'] = breakdown.get('additional')
    scalars['reward_clipping/events'] = breakdown.get('clipping_events')
    scalars['reward_clipping/discarded_abs'] = breakdown.get('clipped_value_abs')
    for reason, value in breakdown.get('reasons', {}).items():
        scalars[f'reward_reason/{reason}'] = value
    for name, value in record.get('learner', {}).items():
        scalars[f'learner/{name}'] = value
    for name, value in record.get('model', {}).items():
        scalars[f'model/{name}'] = value
    for tag, value in scalars.items():
        if value is not None:
            writer.add_scalar(tag, value, step)
    for scope, key in (('all', 'opponent_results'), ('recent', 'recent_opponent_results')):
        for name, value in record.get(key, {}).items():
            if value is not None:
                writer.add_scalar(f'training_opponents/main/{scope}/{name}', value, step)
    for name, values in record.get('rule_opponent_results', {}).items():
        for key, value in values.items():
            if value is not None:
                writer.add_scalar(f'training_opponents/{name}/{key}', value, step)
    for name, value in record.get('rule_profile_ratings', {}).items():
        writer.add_scalar(f'performance/rule_index/{name}', value, step)
    for name, value in record.get('matchup_statistics', {}).items():
        writer.add_scalar(f'matchup/{name}/mean_return', value['mean_return'], step)
        writer.add_scalar(f'matchup/{name}/win_rate', value['wins']/value['games'], step)
    writer.flush()


def write_tensorboard_evaluation(writer, result):
    step = result['iteration']
    for tag, key in (
            ('evaluation/win_rate', 'win_rate'),
            ('evaluation/score', 'score'),
            ('evaluation/mean_team_return', 'mean_team_return'),
            ('evaluation/mean_kills', 'mean_kills'),
            ('evaluation/mean_survivors', 'mean_survivors'),
            ('evaluation/mean_launches', 'mean_launches'),
            ('evaluation/boundary_override_fraction', 'override_decision_fraction')):
        writer.add_scalar(tag, result[key], step)
    for name, values in result.get('by_opponent', {}).items():
        for key in ('win_rate', 'wins', 'draws', 'losses', 'games', 'score', 'mean_team_return'):
            writer.add_scalar(f'evaluation/{name}/{key}', values[key], step)
        for side, side_values in values['by_side'].items():
            writer.add_scalar(f'evaluation/{name}/{side}/win_rate', side_values['win_rate'], step)
    for side, values in result.get('by_side', {}).items():
        writer.add_scalar(f'evaluation/{side}/win_rate', values['win_rate'], step)
    writer.flush()


def save_training_checkpoint(algorithm, coordinator, path, initial_weights,
                             initial_iteration, scenario_settings, args):
    """Save native PPO and management state at a synchronous update boundary."""
    checkpoint = Path(path).resolve()
    checkpoint.mkdir(parents=True, exist_ok=True)
    # Synchronous PPO sampling has returned before this boundary. Invalidate
    # old metadata first so an interrupted overwrite cannot be resumed.
    state_file = checkpoint / 'training-state.json'
    state_file.unlink(missing_ok=True)
    algorithm.save_checkpoint(str(checkpoint))
    save_torch(checkpoint / 'run-initial-weights.pth', initial_weights)
    management_state = ray.get(coordinator.export_state.remote())
    management_state.update(observation_version=OBSERVATION_VERSION,
                            training_iteration=algorithm.iteration, initial_iteration=initial_iteration, scenario_settings=scenario_settings,
                            rule_ratio=args.rule_ratio, random_rate=args.random_rate, seed=args.seed,
                            training_parameters={key: getattr(args, key) for key in TRAINING_PARAMETER_FIELDS})
    temporary = checkpoint / 'training-state.json.tmp'
    temporary.write_text(json.dumps(management_state, indent=2), encoding='utf-8')
    temporary.replace(state_file)
    latest = Path(args.checkpoint_dir).resolve() / 'latest-checkpoint.txt'
    latest.parent.mkdir(parents=True, exist_ok=True)
    pending_latest = latest.with_suffix('.tmp')
    pending_latest.write_text(str(checkpoint), encoding='utf-8')
    pending_latest.replace(latest)
    print(f'[CHECKPOINT] iteration={algorithm.iteration} path={checkpoint}', flush=True)
    return checkpoint


def initialize_new_policies(algorithm, snapshot_policies, network_weights=None):
    """Initialize inference weights without restoring optimizer or learner state."""
    if network_weights is not None:
        algorithm.get_policy('main').model.network.load_state_dict(network_weights, strict=True)
    main_weights = algorithm.get_weights(['main'])['main']
    if snapshot_policies:
        algorithm.set_weights({policy_id: main_weights for policy_id in snapshot_policies})
    policies = list(snapshot_policies)
    if network_weights is not None:
        policies.append('main')
    if policies:
        algorithm.env_runner_group.sync_weights(policies=policies)


def main():
    args = parse_args()
    if args.learner_threads:
        torch.set_num_threads(args.learner_threads)
    saved_state = None
    initial_network_weights = None
    if args.init_model:
        _, network, metadata = load_model(args.init_model)
        initial_network_weights = network.state_dict()
        args.head_hidden_sizes = head_hidden_sizes_from_weights(initial_network_weights)
        print(f"[INIT MODEL] {args.init_model} sha256={metadata['sha256']}", flush=True)
    if args.resume:
        resume_path = resolve_resume_checkpoint(args.resume)
        args.resume = str(resume_path)
        saved_state = validate_checkpoint_schema(resume_path)
        if saved_state.get('version') != 1:
            raise ValueError('Unsupported training state version')
        weights, _, _ = read_policy_checkpoint(resume_path)
        saved_state.setdefault('training_parameters', {}).setdefault(
            'head_hidden_sizes', head_hidden_sizes_from_weights(weights))
        restore_training_parameters(args, saved_state)
        saved_state = restore_opponent_parameters(args, saved_state)
        args.seed = saved_state.get('seed', args.seed)
    scenario_settings = (saved_state['scenario_settings']
                         if saved_state and 'scenario_settings' in saved_state
                         else load_settings(Path(args.config).resolve()))
    scenario_settings = normalize_rule_settings(scenario_settings)
    if args.eval_interval < 1 or not args.eval_seeds:
        raise ValueError('evaluation interval and seeds must be positive/nonempty')
    if args.num_epochs < 1 or args.batch_size < 1 or args.minibatch_size < 1:
        raise ValueError('--num-epochs, --batch-size and --minibatch-size must be positive')
    if args.checkpoint_interval < 0:
        raise ValueError('--checkpoint-interval must be non-negative')
    if args.iterations < 1:
        raise ValueError('--iterations must be at least 1')
    if not 0.0 <= args.rule_ratio <= 1.0:
        raise ValueError('--rule-ratio must be between 0 and 1')
    if not 0.0 <= args.snapshot_ratio <= 1.0:
        raise ValueError('--snapshot-ratio must be between 0 and 1')
    if args.snapshot_pool_size < 1:
        raise ValueError('--snapshot-pool-size must be at least 1')
    if args.snapshot_interval_matches < 1:
        raise ValueError('--snapshot-interval-matches must be at least 1')
    if args.num_env_runners < 0 or args.num_envs_per_runner < 1:
        raise ValueError('environment runner counts must be non-negative/positive')

    ray_options = {'address': args.ray_address or None, 'include_dashboard': False}
    if not args.ray_address:
        ray_temp_dir = Path(args.ray_temp_dir).resolve()
        ray_temp_dir.mkdir(parents=True, exist_ok=True)
        ray_options['_temp_dir'] = str(ray_temp_dir)
        # Recent Ray versions enable token authentication for local clusters
        # and otherwise create ~/.ray/auth_token.  A per-run token avoids any
        # write to HOME while keeping authentication enabled. Workers inherit
        # the environment and therefore use the same token.
        os.environ.setdefault('RAY_AUTH_TOKEN', secrets.token_urlsafe(32))
    ray.init(**ray_options)
    ModelCatalog.register_custom_model(
        'combatsim_set_transformer', CombatSimSetTransformerModel
    )
    ModelCatalog.register_custom_action_dist('combatsim_gated', GatedMultiCategorical)
    run_id = uuid.uuid4().hex[:12]
    actor_name = f'combatsim-rating-{ray.get_runtime_context().get_job_id()}'
    coordinator = RatingCoordinator.options(name=actor_name).remote(
        args.rating_log, args.rating_k, args.snapshot_ratio,
        args.snapshot_pool_size, args.snapshot_interval_matches, saved_state, run_id,
    )
    RLLibRatingCallbacks.configure(actor_name)
    snapshot_policies = {
        f'snapshot_{index}' for index in range(args.snapshot_pool_size)
    }
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            CombatSimFlightEnv,
            env_config={
                'settings': scenario_settings, 'seed': args.seed,
                'algorithm': args.algorithm,
                'rule_ratio': args.rule_ratio,
                'random_rate': args.random_rate,
                'rating_actor_name': actor_name,
            },
        )
        .framework('torch')
        .debugging(seed=args.seed)
        .callbacks(RLLibRatingCallbacks)
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_runner,
            num_cpus_per_env_runner=args.cpus_per_env_runner,
            batch_mode='complete_episodes',
        )
        .learners(num_learners=0, num_gpus_per_learner=0)
        .training(
            train_batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            minibatch_size=min(args.minibatch_size, args.batch_size),
            lr=args.lr, gamma=args.gamma, lambda_=args.gae_lambda,
            clip_param=args.clip_param, entropy_coeff=args.entropy_coeff,
            vf_loss_coeff=args.vf_loss_coeff, vf_clip_param=args.vf_clip_param,
            grad_clip=args.grad_clip, kl_coeff=args.kl_coeff, kl_target=args.kl_target,
            model={'custom_model': 'combatsim_set_transformer',
                   'custom_action_dist': 'combatsim_gated',
                   'max_seq_len': args.max_seq_len,
                   'custom_model_config': {'compact_attention': args.compact_attention,
                                           'algorithm': args.algorithm,
                                           'head_hidden_sizes': args.head_hidden_sizes}},
        )
        .multi_agent(
            policies={'main', *snapshot_policies},
            policies_to_train=['main'],
            policy_mapping_fn=policy_mapping,
            count_steps_by='env_steps',
        )
    )

    algorithm = None
    tensorboard = SummaryWriter(log_dir=str(Path(args.tensorboard_dir).resolve() / run_id))
    manifest = {'observation_version': OBSERVATION_VERSION, 'run_id': run_id,
                'parameters': vars(args), 'scenario_settings': asdict(CombatSimFlightEnv(
                    {'settings': scenario_settings}).config),
                'semantics': {'timeout': 'terminal_draw', 'dead_agents': 'absorbing_noop',
                              'rule_decision_interval': 'same_as_neural'}}
    Path(tensorboard.log_dir, 'resolved-experiment.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    previous_sigint = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        if not stop_requested:
            stop_requested = True
            print("[STOP REQUESTED] Finishing the current operation, then saving. Please wait.", flush=True)

    try:
        # AlgorithmConfig.build_algo() does not expose Trainable's storage
        # argument and therefore falls back to ~/ray_results.  Provide the
        # storage context explicitly so read-only HOME environments work.
        storage = StorageContext(
            storage_path=str((Path(args.ray_temp_dir).resolve() / 'results')),
            experiment_dir_name='combatsim-simulator',
            trial_dir_name=run_id,
        )
        algorithm = config.algo_class(config=copy.deepcopy(config), storage=storage)
        if args.resume:
            algorithm.load_checkpoint(str(resume_path))
        else:
            initialize_new_policies(algorithm, snapshot_policies, initial_network_weights)
        roster = scenario_settings.get('aircraft', [])
        blue = sum(plane['team'] == 'blue' for plane in roster) if roster else scenario_settings.get('blue', ENVIRONMENT_DEFAULTS['blue'])
        red = sum(plane['team'] == 'red' for plane in roster) if roster else scenario_settings.get('red', ENVIRONMENT_DEFAULTS['red'])
        append_record(args.metrics_log, {'type': 'resume' if args.resume else 'start',
            'run_id': run_id, 'blue': blue, 'red': red, 'parameters': vars(args),
            'scenario_source': 'checkpoint' if args.resume else args.config})
        print(f'[EXPERIMENT] {blue} vs {red} | run={run_id} | '
              f'{args.algorithm.upper()} epochs={args.num_epochs} batch={args.batch_size} | '
              f'workers={args.num_env_runners} x {args.num_envs_per_runner}', flush=True)
        print(f'[LOGS] matches={args.rating_log} | metrics={args.metrics_log}', flush=True)
        initial_iteration = algorithm.iteration
        initial_weights = {key: value.detach().cpu().clone()
                           for key, value in algorithm.get_policy('main').model.network.state_dict().items()}
        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        if args.eval_enabled:
            evaluation = evaluate_algorithm(
                algorithm, scenario_settings, args.eval_seeds, args.eval_action_mode,
                args.eval_log, run_id, initial_weights, args.eval_rule_profile)
            write_tensorboard_evaluation(tensorboard, evaluation)
        profiler = cProfile.Profile() if args.profile else None
        success_streak = 0
        for iteration in range(1, args.iterations + 1):
            if stop_requested:
                break
            if profiler:
                profiler.enable()
            try:
                result = algorithm.train()
            finally:
                if profiler:
                    profiler.disable()
                    profile_path = Path(args.profile).resolve()
                    profile_path.parent.mkdir(parents=True, exist_ok=True)
                    profiler.dump_stats(str(profile_path))
            if profiler:
                timing = {key: value for key, value in result.items()
                          if key in ('timers', 'time_this_iter_s', 'time_total_s', 'training_iteration')}
                profile_path.with_suffix('.timing.json').write_text(
                    json.dumps(timing, indent=2, default=str), encoding='utf-8')
            state = ray.get(coordinator.state.remote())
            while state['pending_captures']:
                snapshot = ray.get(coordinator.commit_snapshot.remote())
                if snapshot is None:
                    # Vector environments can retain unfinished games even
                    # though this synchronous sampling call has returned.
                    break
                slot = snapshot['policy_id']
                algorithm.set_weights({slot: algorithm.get_weights(['main'])['main']})
                algorithm.env_runner_group.sync_weights(policies=[slot])
                print(
                    f"[snapshot] match={snapshot['completed']} policy={slot} "
                    f"generation={snapshot['generation']} "
                    f"rating={snapshot['snapshot_rating']:.1f}", flush=True,
                )
                state = ray.get(coordinator.state.remote())
            record = iteration_record(result, state, run_id)
            append_record(args.metrics_log, record)
            write_tensorboard_iteration(tensorboard, record)
            print(format_iteration(record), flush=True)
            if stop_requested:
                break
            if args.checkpoint_interval and iteration % args.checkpoint_interval == 0:
                numbered = Path(args.checkpoint_dir) / f'iteration-{algorithm.iteration:06d}'
                save_training_checkpoint(algorithm, coordinator, numbered, initial_weights,
                                         initial_iteration, scenario_settings, args)
            if stop_requested:
                break
            if args.eval_enabled and (iteration % args.eval_interval == 0 or iteration == args.iterations):
                evaluation = evaluate_algorithm(
                    algorithm, scenario_settings, args.eval_seeds, args.eval_action_mode,
                    args.eval_log, run_id, initial_weights, args.eval_rule_profile)
                write_tensorboard_evaluation(tensorboard, evaluation)

                if stop_requested:
                    break
                if args.eval_stop_win_rate is not None:
                    reached = evaluation['wins'] / evaluation['games'] >= args.eval_stop_win_rate
                    success_streak = success_streak + 1 if reached else 0
                    if success_streak >= args.eval_stop_consecutive:
                        confirmation_log = Path(args.eval_log).with_name(
                            Path(args.eval_log).stem + '-confirmation.jsonl')
                        confirmation = evaluate_algorithm(
                            algorithm, scenario_settings, args.eval_confirm_seeds,
                            args.eval_action_mode, confirmation_log, run_id,
                            initial_weights, args.eval_rule_profile)
                        confirmed = confirmation['wins'] / confirmation['games'] >= args.eval_stop_win_rate
                        append_record(args.metrics_log, {
                            'type': 'target_check', 'run_id': run_id,
                            'iteration': algorithm.iteration, 'confirmed': confirmed,
                            'threshold': args.eval_stop_win_rate,
                            'success_streak': success_streak,
                            'confirmation_wins': confirmation['wins'],
                            'confirmation_games': confirmation['games'],
                        })
                        if confirmed:
                            print(f'[TARGET REACHED] iteration={algorithm.iteration}', flush=True)
                            break
                        success_streak = 0

        final_path = (Path(args.checkpoint_dir) / f'iteration-{algorithm.iteration:06d}'
                      if stop_requested else args.checkpoint_dir)
        save_training_checkpoint(algorithm, coordinator, final_path,
                                 initial_weights, initial_iteration, scenario_settings, args)
    finally:
        tensorboard.close()
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)


if __name__ == '__main__':
    main()
