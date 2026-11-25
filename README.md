# CombatSim

複数機の空戦を実行し、LiveとARENAで観戦できるローカルシミュレーターです。
ルール機同士ならモデルの準備も学習も不要です。
機体・ミサイル・センサー・誘導・報酬を設定でき、強化学習も追加で試せます。

## 起動

Python環境は [uv](https://docs.astral.sh/uv/getting-started/installation/) で準備します。
このリポジトリのルートで実行してください。

```powershell
uv sync --frozen
uv run --frozen python -m scripts.check_setup
uv run --frozen python -m scripts.live
```

ブラウザに表示された対戦画面で、両チームを「ルール / fix」にして開始します。
ARENAへ切り替えると3D戦域、追尾カメラやHUD、音声付きで観戦できます。

- [Live・ARENA・リプレイの使い方](docs/MANUAL.md)
- [シナリオ・機体・センサー・報酬の設定](docs/REFERENCE.md)
- [追加で強化学習を試す](docs/TRAINING.md)
- [報酬フックAPI](docs/REWARD_API.md)
- [飛行モデルの根拠と簡略化](docs/FLIGHT-MODEL.md)
- [開発・検証・配布](docs/DEVELOPMENT.md)

本シミュレーターは簡略化したゲーム・研究用モデルで、実機性能の検証を目的としたものではありません。
Liveサーバーは起動したPCからのローカル接続用です。

## ライセンス

CombatSim本体は **MIT License** です。詳細は [LICENSE](LICENSE) を参照してください。
F-16モデルは **CC BY 4.0**、Three.js等は各ライセンスに従います。
同梱素材の出典と条件は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にまとめています。
