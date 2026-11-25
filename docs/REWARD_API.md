# 報酬フックの使い方

編集する場所は `combatsim/reward_hooks.py` の `CustomRewardHooks` です。
「何を覚えさせたいか」「誰の行動か」「何回まで配るか」を決めてから、対応するメソッドを編集します。
値を変えただけで強くなるとは限りません。固定した対戦条件で勝敗・発射数・飛び方も比較してください。

## 何を調べたい？

| 知りたいこと | 使うもの | 返り値・値の例 | 設計するときの問い |
|---|---|---|---|
| 今、新しく誰が誰を見つけた？ | `on_target_detected` | `event.observer` と `event.target` は機体IDの文字列。 `first_occurrence` は初回ならTrue | 初回と再探知をどう区別する？ |
| 今、自機で誰を探知している？ | `state.detected_targets("blue_0")` | 機体IDのタプル。例：`("red_1",)`。共有情報を含まない | 探知した瞬間と見えている間を区別する？ |
| 今、自機を標的にしている弾は？ | `state.targeting_missiles("blue_0")` | 飛翔中の `MissileSnapshot` のタプル。各弾の `.tracking` で追尾の有無を読める | 標的になったことと追尾されていることを区別する？ |
| 今、自機がレーダー更新を提供できる弾は？ | `state.supported_missiles("blue_0")` | 飛翔中の `MissileSnapshot` のタプル。各弾の `.owner` は発射した機体のID | 更新を提供できることと結果をどう区別する？ |
| 誰が誰に発射した？ | `on_fire` | `event.shooter` と `event.target` は機体ID | 発射そのものを数える？ 結果を見る？ |
| 誰が誰を撃墜した？ | `on_kill` | `event.shooter` と `event.victim` は機体ID | 既存の個人報酬を変える必要はある？ |
| どの機体が、なぜ失われた？ | `on_friendly_loss` | `event.victim` は機体ID。 `event.reason` は `shot_down` または `crash` | 原因によって区別する？ |

問い合わせで該当がなければ `()` が返ります。hookの `return ()` は「そのhookから報酬を返さない」という意味です。

```python
def on_target_detected(self, event, state):
    observer = event.observer             # 今、新しく見つけた機体のID
    target = event.target                 # 今、新しく見つかった相手のID
    visible = state.detected_targets(observer)  # そのtickに自機で見えた相手の一覧
    return ()                             # 報酬なし
```

相手を探知し続けてもhookは呼ばれ直しません。その間の一覧を調べるなら `on_step` から問い合わせます。
探知一覧はそのtickのレーダー判定結果です。後続の命中処理で同じtickに相手が失われる場合もあります。

## 最初に試す例

まず、発射時に情報を読むだけならこう書けます。

```python
def on_fire(self, event, state):
    shooter = event.shooter  # 例: "blue_0"
    target = event.target    # 例: "red_1"
    return ()               # このhookからは報酬なし
```

### 無駄撃ちを減らしたい

`on_fire` に追加します。発射した本人にコストを付けます。発射を減らす仮説であり、
命中率を直接評価する条件ではありません。強すぎると撃たなくなるため、撃墜数も確認します。
以下の数値は記入例で、推奨値や現行設定ではありません。例をコピーしただけでは有効になりません。

```python
def on_fire(self, event, state):
    rewards = self._signals(super().on_fire(event, state))
    rewards.append(RewardSignal(-0.01, (event.shooter,), 'launch_cost'))
    return rewards
```

### 最初の探知を促したい

同じ敵を見失って再発見するたびに得をしないよう、初回だけにします。
敵ごと・探知した機体ごとの初回です。チーム全体の初回とは異なります。

```python
def on_target_detected(self, event, state):
    if not event.first_occurrence:
        return ()
    return RewardSignal(0.01, (event.observer,), 'first_detection')
```

### 撃墜の個人報酬を残して追加したい

`CustomRewardHooks` の親は個人配布の `IndividualRewardHooks` です。
この `super()` は射手への基本報酬を返し、味方全員へ配りません。
基準報酬を残すメソッドでは `super()` を消さないでください。

```python
def on_kill(self, event, state):
    rewards = self._signals(super().on_kill(event, state))
    rewards.append(RewardSignal(0.01, (event.shooter,), 'extra_kill_bonus'))
    return rewards
```

## 戻り値と配布先

`RewardSignal(value, recipients, reason)`、そのリスト・タプル、`None`を返せます。
`value` は対象者一人あたりの値、`recipients` は機体IDのタプル、`reason` は内訳ログの名前です。
存在しない機体ID・重複した配布先・有限でない値はエラーになります。
別のSignalに同じ機体を含めれば加算されます。

- 本人に配る：`(event.actor,)`。ただしactorの意味は下表で確認します。
- 射手に配る：`(event.shooter,)`。ミサイル・撃墜hookで使います。
- 喪失した機体に配る：`(event.victim,)`。
- 生存している味方に配る：`state.alive_team_members(team)`。
- 死亡済みも含む味方に配る：`state.team_members(team)`。

勝敗hookは各機体ごとに呼ばれます。そこで全員へ配ると重複します。
死亡した機体にも最終勝敗の報酬を生成します。学習への取り込みは学習側の軌跡処理に従います。

## 基本報酬と追加報酬の上限

`budget` の既定値は `additional` です。追加報酬には `configs/reward-hooks.json` の
`max_step_total` と `max_episode_total` が機体ごとに適用されます。
消費量は配布した値の絶対値の累積です。加点と減点を相殺しても枠は復活しません。
上限を超えた部分は呼び出し順に切り捨てられ、`info['reward_clipping']` に記録されます。

基本報酬は `budget='base'` で、追加報酬の枠とは独立しています。
追加報酬を使い切っても勝敗・撃墜・機体喪失などの基本報酬は削られません。
基本報酬自身の時間・境界・場外の上限は従来どおりです。
独自報酬で `base` を指定すると追加報酬上限を回避するため、通常は指定しません。

試合中一度だけ配りたいときは、`self.once(key, signal)` が使えます。
キーと配布先ごとに重複を防ぎ、resetで履歴を消します。
上限で切られた場合もその試行を一度として数えます。

```python
def on_seeker_acquired(self, event, state):
    return self.once(('seeker', event.missile),
                     RewardSignal(0.01, (event.shooter,), 'seeker_bonus'))
```

独自の履歴を保存する場合は `reset()` で `super().reset()` を呼んでから初期化します。

## 全hookの契約と注意点

| hook | いつ・誰に関係するか | 設計時の注意 |
|---|---|---|
| `on_episode_start` | reset時に初期状態で呼ぶ。actorなし | 初期配置の保存に使える。返したSignalは最初のstepに繰り越して配布する。resetの戻り値には報酬がないため。行動と無関係な固定加点は学習の目標になりにくい。 |
| `on_step` | 各シミュレーションtickの処理後に呼ぶ。actorなし | 全機を自分で選ぶ。生存条件と累積量を確認する。モデルの行動選択間隔とは異なる。 |
| `on_boundary_override` | 自動境界補助が介入した機体がactor | 場外ではない。継続介入でも呼ぶ。場外罰との併用は意図を決める。 |
| `on_out_of_bounds` | 現在場外にいる機体がactor | 滞在中も呼ぶ。退出瞬間だけなら現在と前回のoutsideフラグを比較する。境界上は場内。 |
| `on_target_detected` | 探知した機体がactor・observer、敵がtarget | `first_occurrence` で出入りによる加点稼ぎを防ぐ。探知だけで有利な射撃とはいえない。 |
| `on_target_lost` | 探知していた機体がactor・observer | reasonは `observer_destroyed`・`target_destroyed`・`radar_contact_lost`。範囲と姿勢のどちらが原因かまでは断定しない。 |
| `on_fire` | 発射に成功した機体がactor・shooter、標的がtarget | 発射要求の失敗は対象外。加点は乱射、減点は撃ち渋りの可能性がある。 |
| `on_seeker_acquired` | ミサイルが初めてシーカー追尾を確立。actor・shooterは射手 | 弾ごとの初回のみ。命中ではない。前stepの支援者は `previous_supporters` で参照できる。新規発射と同tickなら空の場合がある。 |
| `on_missile_track_lost` | ミサイルの追尾あり→なしの最初の遷移。actor・shooterは射手 | reasonは `radar_support_lost`・`seeker_contact_lost`・`reacquisition_pending`。後者はセンサー情報があっても追尾確立待ち。再捕捉後の再喪失では再発火しない。射手の責任とは限らない。 |
| `on_missile_expired` | 命中せず消滅。actor・shooterは射手、標的がtarget | reasonは `lifetime`・`ground`・`target_destroyed`。弾が消えて標的が生存していても、回避操作が原因とは限らない。 |
| `on_missile_hit` | 命中。actor・shooterは射手、target・victimは被弾機 | 撃墜hookとの重複を意図して設計する。 |
| `on_kill` | 敵機撃墜。actor・shooterは射手、target・victimは被撃墜機 | 標準は射手だけに配る。支援や囮の貢献量は自動採点しない。 |
| `on_friendly_loss` | 被撃墜・墜落。actor・victimは失った機体、shooterは射手 | reasonは `shot_down` または `crash`。墜落ではshooterとtargetはNone。失った本人だけが標準の配布先。 |
| `on_win` | 勝利チームの各機体がactor | 死亡済みも含む。actorだけに配るとチーム各機に同じ勝利報酬が届く。 |
| `on_loss` | 敗北チームの各機体がactor | 勝利hookと同じ配布単位。機体喪失の減点とは別。 |
| `on_draw` | 引分時の各機体がactor | `winner is None` だけで試合中を引分扱いしない。独自条件ではdoneも確認する。 |
| `on_timeout` | 時間切れの終了時。actorなし | 勝敗hookの後。時間罰・引分罰との二重計上を確認する。 |
| `on_episode_end` | 終了時、勝敗・時間切れhookの後に最後に呼ぶ。actorなし | 集計・履歴整理向け。ここでも勝敗加点すると重複する。 |

通常の環境では開始→各tickのon_step・イベント→終了tickの勝敗→時間切れ→終了の順です。
同じtickのイベントはすべて処理後のstateを受け取ります。
レーダーの探知リストはミサイルの命中処理より前にサンプリングされるので、
そのtickの死亡による探知喪失は次tickになる場合があります。
`target_lost` は射撃対象の選択変更ではありません。

## 使える状態

stateは読み取り専用のスナップショットです。報酬側から環境を変更できません。
全機体・全ミサイルを含む特権情報なので、観測できない情報を使った報酬かどうかも説明してください。

| 名前 | 内容 |
|---|---|
| `tick`, `max_steps`, `tick_seconds` | 現在tick・試合上限・tickの物理時間 |
| `done`, `winner`, `timed_out` | 終了・勝利チーム・時間切れ |
| `arena_x`, `arena_y`, `ceiling` | 場の寸法。座標と同じ単位 |
| `aircraft`, `missiles` | 今stepの機体・ミサイル |
| `previous_aircraft`, `previous_missiles` | 前のreward評価時の状態。通常は前tick。開始時は空 |
| `aircraft_by_id(id)`, `previous_aircraft_by_id(id)`, `missile_by_id(id)` | ID検索。該当しなければNone |
| `team_members(team)`, `alive_team_members(team)` | チーム全員・生存機のIDタプル |

機体には `id, team, x, y, z, heading, pitch, speed, ammo, alive` があります。
座標は世界座標、角度はラジアンです。
ミサイルには `id, owner, team, target, phase, age, alive` に加えて
`x, y, z, heading, pitch, speed, tracking, tracking_source, supporters, ever_seeker_acquired` があります。
`speed` が未設定ならNoneです。

- `tracking`：追尾確立済みか。`phase == 'active'` でも必ず追尾しているわけではありません。
- `tracking_source`：`radar`・`seeker`・`none`。死んだミサイルではnoneです。
- `supporters`：そのtickに実際に使われるレーダー更新を提供できた味方ID。シーカー誘導中は空。
  支援可能だったことを表し、命中への因果的な貢献度を表す値ではありません。
- `ever_seeker_acquired`：これまでにシーカー追尾を確立したか。

イベントには `type, actor, target, team, missile, reason` と
`shooter, victim, observer, first_occurrence, tracking_source, supporters, previous_supporters,
target_alive, ever_seeker_acquired` があります。使わないID・原因はNone、タプルは空です。
`first_occurrence` は `on_target_detected` 専用です。ほかのイベントでは既定値のFalseです。
追尾喪失の `tracking_source` は喪失した追尾方式です。
場外hookは `outside_horizontal, outside_vertical, previous_outside_horizontal,
previous_outside_vertical` も持ちます。

## 報酬を確認する

サンプルは初期状態では無効です。hookを編集したらLiveを再起動して対戦を開始し、
観戦画面または保存リプレイの **「報酬の内訳」** を開きます。

スライダーで見たいtickに移動すると、**誰に・何の理由で・いくら入ったか**を確認できます。
上限で削られた量と、その時点までの機体別累計（割引前）も表示されます。
同じ理由・機体の値はtickごとに合算されます。報酬記録のない古いリプレイは、対戦を記録し直してください。

学習モデルがなくても、Liveで両チームにルール機を選べば確認できます。配点を比較するときは、シードと対戦設定を揃えてください。

<details>
<summary>設定ファイル・ログ・テストの詳細</summary>

- `configs/rewards.json`：基本報酬の値。
- `configs/reward-hooks.json`：追加報酬の上限。
- `info['reward_components']`：`HookReward:理由名` ごとの適用値。
- `info['reward_clipping']`：上限で削られた回数と絶対量。
- `uv run python -m scripts.validate_rewards`：標準報酬・内訳・上限の事前チェック。
- `uv run python -m unittest scripts.validate_reward_api`：hookの発火条件や配布先のテスト。

</details>
