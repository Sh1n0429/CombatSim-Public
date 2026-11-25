"""報酬設計のために編集できるフック。

イベントの検出と報酬の配布は環境側で行う。
``CustomRewardHooks``を編集し、``RewardSignal``を返すことで報酬を設定する。
フックに渡される状態は変更できないスナップショットで、シミュレーションには影響しない。
狙い別の記入例と各hookの注意点は docs/REWARD_API.md を参照。
独自RewardSignalは追加報酬枠を使い、基本報酬の枠とは分離される。
"""
from dataclasses import dataclass
import math


# OriginalRewardHooksが生成する報酬理由。
# 学習ログでは、この一覧を使って元の報酬と追加した報酬を区別する。
# 報酬を追加するときは、"launch_cost"など内容が分かる新しい理由名を使う。
ORIGINAL_REWARD_REASONS = frozenset({
    'time_penalty', 'win', 'loss', 'draw', 'enemy_kill', 'friendly_loss',
    'boundary_override', 'out_of_bounds',
})


@dataclass(frozen=True)
class AircraftSnapshot:
    id: str
    team: str
    x: float
    y: float
    z: float
    heading: float
    pitch: float
    speed: float
    ammo: int
    alive: bool


@dataclass(frozen=True)
class MissileSnapshot:
    id: str
    owner: str
    team: str
    target: str
    phase: str
    age: int
    alive: bool
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0
    pitch: float = 0.0
    speed: float | None = None
    tracking: bool = False
    tracking_source: str = 'none'
    supporters: tuple[str, ...] = ()
    ever_seeker_acquired: bool = False


@dataclass(frozen=True)
class RewardState:
    tick: int
    max_steps: int
    done: bool
    winner: str | None
    timed_out: bool
    aircraft: tuple[AircraftSnapshot, ...]
    missiles: tuple[MissileSnapshot, ...]
    previous_aircraft: tuple[AircraftSnapshot, ...] = ()
    previous_missiles: tuple[MissileSnapshot, ...] = ()
    arena_x: float = 0.0
    arena_y: float = 0.0
    ceiling: float = 0.0
    tick_seconds: float = 1.0

    radar_contacts: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def detected_targets(self, aircraft_id: str) -> tuple[str, ...]:
        """自機のレーダー探知リスト。共有情報は含まない。"""
        return next((targets for observer, targets in self.radar_contacts
                     if observer == aircraft_id), ())

    def targeting_missiles(self, aircraft_id: str) -> tuple[MissileSnapshot, ...]:
        """指定機体を標的にする飛翔中の弾。追尾の有無は問わない。"""
        return tuple(m for m in self.missiles if m.alive and m.target == aircraft_id)

    def supported_missiles(self, aircraft_id: str) -> tuple[MissileSnapshot, ...]:
        """指定機体がレーダー更新を提供できる飛翔中の弾。"""
        return tuple(m for m in self.missiles if m.alive and aircraft_id in m.supporters)

    def previous_aircraft_by_id(self, ident):
        return next((a for a in self.previous_aircraft if a.id == ident), None)

    def missile_by_id(self, ident):
        return next((m for m in self.missiles if m.id == ident), None)

    def aircraft_by_id(self, ident):
        return next((a for a in self.aircraft if a.id == ident), None)

    def alive_team_members(self, team):
        return tuple(a.id for a in self.aircraft if a.team == team and a.alive)

    def team_members(self, team):
        return tuple(a.id for a in self.aircraft if a.team == team)


@dataclass(frozen=True)
class RewardEvent:
    type: str
    actor: str | None = None
    target: str | None = None
    team: str | None = None
    missile: str | None = None
    reason: str | None = None
    outside_horizontal: bool = False
    outside_vertical: bool = False
    previous_outside_horizontal: bool = False
    previous_outside_vertical: bool = False
    shooter: str | None = None
    victim: str | None = None
    observer: str | None = None
    first_occurrence: bool = False
    tracking_source: str | None = None
    supporters: tuple[str, ...] = ()
    previous_supporters: tuple[str, ...] = ()
    target_alive: bool | None = None
    ever_seeker_acquired: bool = False


@dataclass(frozen=True)
class RewardSignal:
    value: float
    recipients: tuple[str, ...]
    reason: str
    budget: str = 'additional'

    def __post_init__(self):
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError('RewardSignal.value must be a finite number')
        if not isinstance(self.recipients, tuple) or any(
                not isinstance(ident, str) or not ident for ident in self.recipients):
            raise ValueError('RewardSignal.recipients must be a tuple of aircraft IDs')
        if self.budget not in ('base', 'additional'):
            raise ValueError('RewardSignal.budget must be base or additional')
        if len(set(self.recipients)) != len(self.recipients):
            raise ValueError('RewardSignal.recipients must not contain duplicates')
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError('RewardSignal.reason must be a nonempty string')


class EmptyRewardHooks:
    """報酬を返さないフック一式。個別の報酬を試すときに使う。"""

    def reset(self):
        self._reward_once = set()

    def once(self, key, signal):
        """各機体・キーについて試合中一度だけ配る。resetで履歴を消す。"""
        if not hasattr(self, '_reward_once'):
            self._reward_once = set()
        recipients = tuple(i for i in signal.recipients if (key, i) not in self._reward_once)
        self._reward_once.update((key, i) for i in recipients)
        return (RewardSignal(signal.value, recipients, signal.reason, signal.budget),) if recipients else ()

    def on_episode_start(self, event, state): return ()
    def on_episode_end(self, event, state): return ()
    def on_win(self, event, state): return ()
    def on_loss(self, event, state): return ()
    def on_draw(self, event, state): return ()
    def on_timeout(self, event, state): return ()
    def on_step(self, event, state): return ()
    def on_out_of_bounds(self, event, state): return ()
    def on_boundary_override(self, event, state): return ()
    def on_target_detected(self, event, state): return ()
    def on_target_lost(self, event, state): return ()
    def on_seeker_acquired(self, event, state): return ()
    def on_missile_track_lost(self, event, state): return ()
    def on_fire(self, event, state): return ()
    def on_missile_expired(self, event, state): return ()
    def on_missile_hit(self, event, state): return ()
    def on_kill(self, event, state): return ()
    def on_friendly_loss(self, event, state): return ()


class OriginalRewardHooks(EmptyRewardHooks):
    """通常シナリオの報酬をフックで表現したもの。

    基本の報酬値はconfigs/rewards.jsonから読み込む。
    """

    def __init__(self):
        from combatsim.config_defaults import read_defaults
        self._baseline = {item['type']: item for item in read_defaults('rewards.json')['rewards']}
        from combatsim.config_defaults import REWARD_DEFAULTS
        configured = self._baseline.get('boundary_override')
        self._boundary_spec = (None if configured is None else
                               {**REWARD_DEFAULTS['BoundaryOverridePenalty'], **configured})
        if self._boundary_spec is not None:
            penalty, cap = self._boundary_spec['penalty'], self._boundary_spec['max_total']
            if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty > 0:
                raise ValueError('boundary_override penalty must be finite and nonpositive')
            if cap is not None and (type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0):
                raise ValueError('boundary_override max_total must be positive and finite')
        configured = self._baseline.get('out_of_bounds')
        self._outside_spec = (None if configured is None else
                              {**REWARD_DEFAULTS['OutOfBoundsPenalty'], **configured})
        if self._outside_spec is not None:
            spec = self._outside_spec
            if (type(spec['penalty']) not in (int, float) or not math.isfinite(spec['penalty'])
                    or spec['penalty'] > 0 or spec['mode'] not in ('per_step', 'on_exit')
                    or type(spec['include_altitude']) is not bool):
                raise ValueError('Invalid out_of_bounds reward settings')
            cap = spec['max_total']
            if cap is not None and (type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0):
                raise ValueError('out_of_bounds max_total must be positive and finite')
        from combatsim.tactical_rewards import TimePenalty, validate_tactical_reward
        self._time_reward = TimePenalty(**{k: v for k, v in self._baseline['time_penalty'].items() if k != 'type'})
        validate_tactical_reward(self._time_reward)
        self.reset()

    def reset(self):
        super().reset()
        self._time_spent = {}
        self._boundary_spent = {}
        self._outside_spent = {}

    def on_step(self, event, state):
        rewards = []
        for aircraft in state.aircraft:
            if not aircraft.alive:
                continue
            spent = self._time_spent.get(aircraft.id, 0.0)
            cost = -self._time_reward.per_tick(state.max_steps)
            if self._time_reward.max_total is not None:
                cost = min(cost, max(0.0, self._time_reward.max_total-spent))
            self._time_spent[aircraft.id] = spent+cost
            if cost:
                rewards.append(RewardSignal(-cost, (aircraft.id,), 'time_penalty', budget='base'))
        return tuple(rewards)

    def on_out_of_bounds(self, event, state):
        spec = self._outside_spec
        if spec is None or event.actor is None:
            return ()
        outside = event.outside_horizontal or (spec['include_altitude'] and event.outside_vertical)
        previous = event.previous_outside_horizontal or (spec['include_altitude'] and event.previous_outside_vertical)
        if not outside or (spec['mode'] == 'on_exit' and previous):
            return ()
        spent = self._outside_spent.get(event.actor, 0.0)
        cost = -spec['penalty']
        if spec['max_total'] is not None:
            cost = min(cost, max(0.0, spec['max_total']-spent))
        self._outside_spent[event.actor] = spent+cost
        return (RewardSignal(-cost, (event.actor,), 'out_of_bounds', budget='base'),) if cost else ()

    def on_boundary_override(self, event, state):
        if event.actor is None or self._boundary_spec is None:
            return ()
        spent = self._boundary_spent.get(event.actor, 0.0)
        cost = -self._boundary_spec['penalty']
        cap = self._boundary_spec['max_total']
        if cap is not None:
            cost = min(cost, max(0.0, cap-spent))
        self._boundary_spent[event.actor] = spent+cost
        return (RewardSignal(-cost, (event.actor,), 'boundary_override', budget='base'),) if cost else ()

    def on_win(self, event, state):
        return RewardSignal(self._baseline['victory']['win'], (event.actor,), 'win', budget='base')

    def on_loss(self, event, state):
        return RewardSignal(self._baseline['victory']['loss'], (event.actor,), 'loss', budget='base')

    def on_draw(self, event, state):
        return RewardSignal(self._baseline['victory']['draw'], (event.actor,), 'draw', budget='base')

    def on_kill(self, event, state):
        return RewardSignal(self._baseline['shootdown']['kill'], state.team_members(event.team), 'enemy_kill', budget='base')

    def on_friendly_loss(self, event, state):
        return RewardSignal(self._baseline['shootdown']['loss'], state.team_members(event.team), 'friendly_loss', budget='base')


class IndividualRewardHooks(OriginalRewardHooks):
    """シミュレーター用の基本報酬。撃墜・喪失は本人に配布する。"""

    def on_kill(self, event, state):
        # 撃墜報酬は射手だけに与える。勝利報酬はチーム全体への配布を維持する。
        # 親クラスのチーム配布に加算せず、この機体への配布に置き換える。
        if event.actor is None:
            return ()
        return RewardSignal(self._baseline['shootdown']['kill'],
                            (event.actor,), 'enemy_kill', budget='base')

    def on_friendly_loss(self, event, state):
        # 機体喪失の罰は失った機体だけに与える。敗北の罰はチーム全体に与える。
        # 親クラスと同様に、墜落による喪失も対象にする。
        if event.actor is None:
            return ()
        return RewardSignal(self._baseline['shootdown']['loss'],
                            (event.actor,), 'friendly_loss', budget='base')



class CustomRewardHooks(IndividualRewardHooks):
    """報酬の編集場所。実験で必要なメソッドを編集する。

    多くのメソッドは``super()``の報酬をそのまま返す。
    現在、撃墜報酬は射手だけ、被撃墜の罰は失った機体だけに与える。
    元の報酬を残して加算する場合は、``super()``の結果に追加する::

        rewards = self._signals(super().on_kill(event, state))
        rewards.append(RewardSignal(.1, (event.actor,), 'shooter_bonus'))
        return rewards

    元のフックが単一の報酬や報酬のタプルを返す場合は、
    ``_signals``でリストに揃えて扱う。
    """

    @staticmethod
    def _signals(value):
        if value is None:
            return []
        if isinstance(value, RewardSignal):
            return [value]
        return list(value)

    # 試合の開始・終了 --------------------------------------------------
    def on_episode_start(self, event, state):
        rewards = self._signals(super().on_episode_start(event, state))
        # 既存の報酬なし。reset時の初期状態で1回呼ばれる。ここで返す報酬は最初のstepで配布する。
        # rewards.append(RewardSignal(...))
        return rewards

    def on_episode_end(self, event, state):
        rewards = self._signals(super().on_episode_end(event, state))
        # 既存の報酬なし。勝敗・時間切れのフックを呼んだ後、最後に1回呼ばれる。
        return rewards

    def on_win(self, event, state):
        rewards = self._signals(super().on_win(event, state))
        # 各機体ごとに呼ばれる。全員への配布を書くと重複する。
        # 設定された勝利報酬をevent.actor（死亡済みも含む勝利チームの各機体）に与える。
        # 追加例: rewards.append(RewardSignal(.1, (event.actor,), 'win_bonus'))
        return rewards

    def on_loss(self, event, state):
        rewards = self._signals(super().on_loss(event, state))
        # 設定された敗北の罰をevent.actor（敗北チームの各機体）に与える。
        return rewards

    def on_draw(self, event, state):
        rewards = self._signals(super().on_draw(event, state))
        # 設定された引き分けの報酬をevent.actor（全機体）に与える。
        return rewards

    def on_timeout(self, event, state):
        rewards = self._signals(super().on_timeout(event, state))
        # ここでは報酬なし。時間経過の罰はon_stepで少しずつ与えるため、
        # タイムアウト時に二重には加算しない。
        return rewards

    # 時間経過と境界 ----------------------------------------------------
    def on_step(self, event, state):
        rewards = self._signals(super().on_step(event, state))
        # 生存機体ごとに毎ステップ、設定された時間経過の罰を与える。
        # 味方同士の距離・生存数・標的の重複など、継続状態の報酬もここに書く。
        # state.previous_aircraft / previous_missilesで前tickとも比較できる。
        # 生存条件・累積量を確認し、条件と配布先を自分で決める。
        return rewards

    def on_boundary_override(self, event, state):
        rewards = self._signals(super().on_boundary_override(event, state))
        # boundary_overrideが設定されている場合だけ、境界介入の罰を与える。
        # 継続介入でも呼ばれる。実際の場外とは別。
        # 水平・高度方向に同時に介入しても、イベントは1回として扱う。
        return rewards

    def on_out_of_bounds(self, event, state):
        # 実際に水平・高度の範囲外にいる機体に、設定された場外の罰を与える。
        # 滞在中も呼ぶ。退出瞬間だけならprevious_outside_*と比較する。
        # outside_horizontal / outside_verticalで、どちらの範囲外かを判別できる。
        return self._signals(super().on_out_of_bounds(event, state))

    # 探知と追尾 --------------------------------------------------------
    def on_target_detected(self, event, state):
        rewards = self._signals(super().on_target_detected(event, state))
        # event.observerが探知した機体。first_occurrenceで再探知の重複加点を防げる。
        # 既存の報酬なし。event.targetがevent.actorのレーダー探知リストに加わったとき。
        return rewards

    def on_target_lost(self, event, state):
        rewards = self._signals(super().on_target_lost(event, state))
        # 既存の報酬なし。event.targetがevent.actorのレーダー探知リストから消えたとき。
        # reasonでobserver_destroyed / target_destroyed / radar_contact_lostを区別する。
        # 探知範囲外への移動だけでなく、死亡などによる消失も含む。
        return rewards

    def on_seeker_acquired(self, event, state):
        rewards = self._signals(super().on_seeker_acquired(event, state))
        # 既存の報酬なし。ミサイルのシーカーが標的を初めて追尾したとき、弾ごとに1回。
        # event.shooterは射手、event.targetは標的、event.missileはミサイルID。
        # previous_supportersは前tickの支援者。シーカー捕捉は命中とは別。
        return rewards

    def on_missile_track_lost(self, event, state):
        rewards = self._signals(super().on_missile_track_lost(event, state))
        # 既存の報酬なし。ミサイルが追尾あり→なしに変わった最初の1回。
        # reason・tracking_source・previous_supportersを確認し、射手の責任と決めつけない。
        # レーダー支援による追尾とシーカー追尾の両方を含む。event.shooterは射手。
        return rewards

    # 射撃とミサイル ----------------------------------------------------
    def on_fire(self, event, state):
        rewards = self._signals(super().on_fire(event, state))
        # 既存の報酬なし。event.actorがevent.targetに発射したときに呼ばれる。
        # 追加例: rewards.append(RewardSignal(-.01, (event.actor,), 'launch_cost'))
        return rewards

    def on_missile_expired(self, event, state):
        rewards = self._signals(super().on_missile_expired(event, state))
        # 既存の報酬なし。命中せずミサイルが消滅したとき。event.actorは射手。
        # event.reasonはlifetime（寿命切れ）、ground（地面衝突）、
        # target_destroyed（標的が既に死亡）。すべてを射手の失敗とは扱わない。
        return rewards

    def on_missile_hit(self, event, state):
        rewards = self._signals(super().on_missile_hit(event, state))
        # ここでは報酬なし。撃墜報酬はon_killで与える。
        # event.shooterは射手、event.victimは被弾機。on_killとの二重加点に注意。
        return rewards

    # 撃墜と機体の喪失 --------------------------------------------------
    def on_kill(self, event, state):
        # 元の個人報酬を維持。追加する場合はrewards.append(...)。
        return self._signals(super().on_kill(event, state))

    def on_friendly_loss(self, event, state):
        return self._signals(super().on_friendly_loss(event, state))
