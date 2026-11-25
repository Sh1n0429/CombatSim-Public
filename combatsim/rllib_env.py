"""RLlib multi-agent adapter for the CombatSim combat task.

The adapter exposes one agent per aircraft, including the fire decision.
"""
from pathlib import Path
import random

import numpy as np

try:
    from gymnasium import spaces
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError as error:  # pragma: no cover
    raise ImportError('Install the project dependencies with: pip install -e .') from error

from .config_defaults import CONFIG_DIR, APPLICATION_DEFAULTS
from .rule_profiles import normalize_rule_settings
from .decision import held_action
from .env import Config, ACTION_SHAPE, decode_action
from .factory import load_settings, make_env
from .entities import selected_entities
from .policies import RulePolicy
from .set_transformer import OBSERVATION_SHAPE, entity_features
from .observation_schema import SELF_INDEX
from .central_state import central_state, central_state_size, central_missiles, missile_capacity, MISSILE_STATE_FIELDS


class CombatSimFlightEnv(MultiAgentEnv):
    """Simultaneous multi-aircraft flight control environment for RLlib."""

    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}
        settings = env_config.get('settings')
        if settings is None:
            config_path = Path(env_config.get('config', CONFIG_DIR / 'scenario-default.json')).resolve()
            settings = load_settings(config_path)
        self.config = Config(**normalize_rule_settings(settings))
        self.env = make_env(self.config)
        self.possible_agents = list(self.env.aircraft)
        self.agents = list(self.possible_agents)
        self.algorithm = env_config.get('algorithm', 'ppo')
        if self.algorithm not in ('ppo', 'mappo'):
            raise ValueError('algorithm must be ppo or mappo')
        # Preserve the complete action layout, including all shot tokens.
        self._action_space = spaces.MultiDiscrete(list(ACTION_SHAPE))
        self._observation_space = spaces.Dict({
            key: spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
            for key, shape in OBSERVATION_SHAPE.items()
        })
        self._seed = env_config.get('seed')
        if self.algorithm == 'mappo':
            self._missile_capacity = missile_capacity(self.config)
            n = len(self.possible_agents)
            self._observation_space = spaces.Dict({
                **self._observation_space.spaces,
                'central_state': spaces.Box(-np.inf, np.inf,
                    (central_state_size(len(self.possible_agents)),), dtype=np.float32),
                **{key: spaces.Box(-np.inf, np.inf, shape, dtype=np.float32) for key, shape in {
                    'central_missiles': (self._missile_capacity, len(MISSILE_STATE_FIELDS)),
                    'central_missiles_mask': (self._missile_capacity,),
                    'central_missile_owners': (self._missile_capacity, n),
                    'central_missile_targets': (self._missile_capacity, n),
                }.items()},
            })
        self._worker_index = getattr(env_config, 'worker_index', 0)
        self._vector_index = getattr(env_config, 'vector_index', 0)
        self._rule_ratio = float(env_config.get('rule_ratio', 0.0))
        self._random_rate = env_config.get('random_rate', APPLICATION_DEFAULTS['environment_adapter']['random_rate'])
        if (type(self._random_rate) not in (int, float)
                or not 0 <= self._random_rate <= 1):
            raise ValueError('random_rate must be between 0 and 1')
        if self._random_rate > 0 and 'random' not in self.config.rule_policy_profiles:
            raise ValueError('random_rate > 0 requires a random profile in the scenario')
        self.rating_actor_name = env_config.get('rating_actor_name')
        if not 0.0 <= self._rule_ratio <= 1.0:
            raise ValueError('rule_ratio must be between 0 and 1')
        self._episode_index = 0
        self.rule_side = None
        self.rule_profile = None
        self.rule_policy = None

    def _rule_settings(self, profile):
        # Inherit the fix baseline, then apply the random profile's explicit
        # overrides and episode sampling ranges.
        baseline = (self.config.rule_policy_profiles.get('fix', {})
                    if profile == 'random' else {})
        return {**self.config.rule_policy, **baseline,
                **self.config.rule_policy_profiles.get(profile, {})}

    def _obs(self, aircraft_id):
        return self._features(aircraft_id, self.env.observations()[aircraft_id])

    def _features(self, agent, raw):
        # Chosen neural decisions, not the last held/overridden physics control.
        raw = dict(raw)
        raw['previous_action'] = self._previous_decisions.get(
            agent, {'turn': 0., 'vertical': 0., 'fire': 0.})
        raw['boundary_override_fraction'] = self._decision_overrides.get(agent, 0.)
        target = self._decision_targets.get(agent)
        raw['contacts'] = [{**contact, 'selected_last_step': contact['id'] == target}
                           for contact in raw['contacts']]
        return self._with_central_state(agent, entity_features(raw))

    def _with_central_state(self, agent, features):
        if self.algorithm == 'mappo':
            features['central_state'] = central_state(self.env, agent)
            features.update(central_missiles(self.env, self._missile_capacity))
        return features

    def _absorbing_observation(self):
        # No privileged sensors after death. Keep time and a valid self token;
        # alive=0 makes every policy head a deterministic no-op.
        features = {key: np.zeros(shape, dtype=np.float32) for key, shape in OBSERVATION_SHAPE.items()}
        features['self_mask'][0] = 1.
        features['common_mask'][0] = 1.
        features['common'][0, 0] = self.env.tick/self.config.max_steps
        return features

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None and self._seed is not None:
            # A run seed defines a reproducible sequence, not one repeated map.
            # Keep independent streams for parallel workers and vector slots.
            seed = int(np.random.SeedSequence([
                self._seed, self._worker_index, self._vector_index, self._episode_index,
            ]).generate_state(1)[0])
        elif seed is None:
            seed = int(self.np_random.integers(0, 2**32, dtype=np.uint64))
        self.episode_seed = seed
        raw_observations, _ = self.env.reset(seed=seed)
        self._raw_observations = raw_observations
        matchup_rng = random.Random(f'combatsim-rllib:{seed}')
        self._episode_index += 1
        self.rule_side = (
            matchup_rng.choice(('blue', 'red'))
            if matchup_rng.random() < self._rule_ratio else None
        )
        if options and 'rule_side' in options:
            if options['rule_side'] not in (None, 'blue', 'red'):
                raise ValueError('rule_side must be blue, red, or None')
            self.rule_side = options['rule_side']
        requested_profile = options.get('rule_profile') if options else None
        if requested_profile is not None and requested_profile not in self.config.rule_policy_profiles:
            # "fix" means the base rule policy, including scenarios that do
            # not define named curriculum profiles.
            if requested_profile != 'fix':
                raise ValueError('unknown rule_profile')
            # Keep the explicit name: .get('fix', {}) below selects the
            # base policy instead of sampling a training opponent.
        self.rule_profile = None
        if self.rule_side:
            if requested_profile is not None:
                self.rule_profile = requested_profile
            else:
                # Keep this RNG separate: changing random_rate must not change
                # spawn, side selection or the non-rule matchup stream.
                sample = random.Random(f'combatsim-rule-variant:{seed}').random()
                self.rule_profile = 'random' if sample < self._random_rate else 'fix'
        policy_settings = self._rule_settings(self.rule_profile)
        self.rule_policy = (
            RulePolicy(policy_settings, seed=f'combatsim-rule:{seed}')
            if self.rule_side else None
        )
        self.rule_policies_by_side = {}
        for side, profile in (options or {}).get('rule_profiles_by_side', {}).items():
            if (side not in ('blue', 'red')
                    or (profile != 'fix' and profile not in self.config.rule_policy_profiles)):
                raise ValueError('Invalid rule side/profile')
            settings = self._rule_settings(profile)
            self.rule_policies_by_side[side] = RulePolicy(settings, seed=f'combatsim-rule:{seed}')
        self.agents = [
            agent for agent in self.possible_agents
            if self.env.aircraft[agent].team != self.rule_side
        ]
        self._previous_decisions = {}
        self._decision_targets = {}
        self._decision_overrides = {}
        observations = {agent: self._features(agent, raw_observations[agent]) for agent in self.agents
                        if self.env.aircraft[agent].alive}
        self._episode_agents = set(observations)
        self.episode_control_counts = {agent: {'ticks': 0, 'overrides': 0} for agent in self._episode_agents}
        # Accumulate hook reasons across the whole episode.  ToyEnv.info()
        # contains only the current physics tick, while an RLlib iteration
        # summary needs the complete episode.
        self.episode_reward_reasons = {}
        self.episode_reward_clipping = {'events': 0, 'discarded_abs': 0.0}
        self.episode_reward_clipping_by_agent = {
            agent: {'events': 0, 'discarded_abs': 0.0} for agent in self._episode_agents}
        return observations, {agent: {} for agent in observations}

    def step(self, action_dict, *, on_tick=None):
        mapped = {}
        for agent, action in action_dict.items():
            if agent not in self.env.aircraft or not self.env.aircraft[agent].alive:
                continue
            turn, vertical, fire = map(int, action)
            mapped[agent] = (turn, vertical, fire)
            decoded = decode_action(mapped[agent])
            self._previous_decisions[agent] = {
                'turn': float(decoded[0]), 'vertical': float(decoded[1]),
                'fire': fire / (ACTION_SHAPE[2]-1),
            }
            contacts = selected_entities(self._raw_observations[agent])['contacts']
            self._decision_targets[agent] = (contacts[fire-1]['id']
                                             if 0 < fire <= len(contacts) else None)

        rewards = {agent: 0.0 for agent in self._episode_agents}
        terminated = truncated = False
        info = {}
        # One RLlib action is a neural decision. Advance the lower-level physics
        # for the configured interval, holding flight controls and pulsing fire
        # only on the first tick.
        raw_observations = self._raw_observations
        rule_decisions = ({agent: self.rule_policy.act(raw_observations[agent])
                           for agent, aircraft in self.env.aircraft.items()
                           if aircraft.alive and aircraft.team == self.rule_side}
                          if self.rule_side else {})
        for agent, aircraft in self.env.aircraft.items():
            policy = self.rule_policies_by_side.get(aircraft.team)
            if policy is not None and aircraft.alive:
                rule_decisions[agent] = policy.act(raw_observations[agent])
        decision_counts = {agent: [0, 0] for agent in self._episode_agents}
        for offset in range(self.config.decision_interval_steps):
            learning_actions = mapped if offset == 0 else {
                agent: held_action(action) for agent, action in mapped.items()
                if self.env.aircraft[agent].alive
            }
            rule_actions = {agent: action if offset == 0 else held_action(action)
                            for agent, action in rule_decisions.items()
                            if self.env.aircraft[agent].alive}
            tick_actions = {**learning_actions, **rule_actions}
            raw_observations, tick_rewards, terminated, truncated, info = self.env.step(
                tick_actions, decision_observations=raw_observations)
            if on_tick is not None:
                on_tick()
            for agent, diagnostic in info.get('controls', {}).items():
                if agent in decision_counts:
                    count = int(diagnostic.get('boundary_override', False))
                    decision_counts[agent][0] += count
                    decision_counts[agent][1] += 1
                    self.episode_control_counts[agent]['overrides'] += count
                    self.episode_control_counts[agent]['ticks'] += 1
            for name, values in info.get('reward_components', {}).items():
                if not name.startswith('HookReward:'):
                    continue
                reason = name.split(':', 1)[1]
                totals = self.episode_reward_reasons.setdefault(reason, {})
                for agent, value in values.items():
                    if agent in self._episode_agents:
                        totals[agent] = totals.get(agent, 0.0) + float(value)
            for agents in info.get('reward_clipping', {}).values():
                for agent, values in agents.items():
                    if agent in self._episode_agents:
                        self.episode_reward_clipping['events'] += int(values['events'])
                        self.episode_reward_clipping['discarded_abs'] += float(values['discarded_abs'])
                        own = self.episode_reward_clipping_by_agent[agent]
                        own['events'] += int(values['events'])
                        own['discarded_abs'] += float(values['discarded_abs'])
            for agent, reward in tick_rewards.items():
                if agent in rewards:
                    rewards[agent] += float(reward)
            if terminated or truncated:
                break

        self._raw_observations = raw_observations
        self._decision_overrides = {agent: counts[0]/max(1, counts[1])
                                    for agent, counts in decision_counts.items()}
        alive = {
            agent for agent, aircraft in self.env.aircraft.items()
            if aircraft.alive and aircraft.team != self.rule_side
        }
        observations = {agent: self._features(agent, raw_observations[agent]) if agent in alive
                        else self._with_central_state(agent, self._absorbing_observation())
                        for agent in self._episode_agents}
        ended = terminated or truncated
        terminateds = {
            agent: bool(terminated)
            for agent in self._episode_agents
        }
        truncateds = {
            agent: bool(truncated)
            for agent in self._episode_agents
        }
        terminateds['__all__'] = bool(terminated)
        truncateds['__all__'] = bool(truncated)
        self.agents = [] if ended else list(self._episode_agents)
        infos = {agent: info for agent in observations}
        return observations, rewards, terminateds, truncateds, infos

    def get_observation_space(self, agent_id=None):
        return self._observation_space

    def get_action_space(self, agent_id=None):
        return self._action_space
