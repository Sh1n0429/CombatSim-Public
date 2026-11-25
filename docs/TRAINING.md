# 強化学習とモデルの書き出し

強化学習は追加機能です。Liveでルール機同士を対戦させるだけなら、この手順は不要です。
通常設定はMAPPOで、Actorは局所観測、中央Criticは学習時の全体情報を使います。

## 動作確認と学習

```powershell
uv run --frozen python -m scripts.validate_rewards
uv run --frozen python -m scripts.train --experiment experiments/smoke.yaml
uv run --frozen python -m scripts.train --experiment experiments/default.yaml
```

スモーク設定は導入確認用です。通常学習を始める前に、ワーカー数と保存先をPCに合わせて確認してください。
アルゴリズム等は `training`、並列数は `workers`、相手構成は `opponents`、保存先は `output` にあります。
実験を比較するときはコピーしたYAMLで保存先を分けてください。

## 報酬と再開

`combatsim/reward_hooks.py` の `CustomRewardHooks` を編集できます。詳しくは[報酬API](REWARD_API.md)を参照してください。
学習を停止してhookを変更すると、次の起動から変更後のコードを使います。

```powershell
uv run --frozen python -m scripts.train --resume artifacts/checkpoint
```

再開は最新チェックポイントの重み・Critic・最適化状態と保存済みのシナリオを引き継ぎます。
設定JSONを書き換えても、保存されたシナリオは置き換わりません。

## モデルの書き出し

```powershell
uv run --frozen python -m scripts.export_model --checkpoint artifacts/checkpoint --model-id example --output models/example
uv run --frozen python -m scripts.validate_model models/example
```

モデルフォルダーをLiveに追加して観戦・評価できます。
モデルの重みだけで新規学習を始めるには、保存先を分けたYAMLと `--init-model models/example` を使います。
最適化状態や中央Criticは新規です。`--resume` とは併用できません。
