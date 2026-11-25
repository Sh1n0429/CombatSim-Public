# 開発・検証・配布

```powershell
uv sync --frozen
uv run --frozen python -m unittest discover -s scripts -p "validate_*.py"
node scripts/validate_viewer.cjs
node scripts/validate_arena.cjs
uv run --frozen python -m scripts.validate_rewards
uv run --frozen python -m scripts.build_distribution
```

配布ZIPは `artifacts/distribution/CombatSim.zip` です。ソース、設定、サンプル、文書、ライセンスとARENA素材を含みます。
作業用仮想環境、モデル、実行ログ、Git履歴は含めません。

主要ディレクトリは、環境・描画・推論の `combatsim/`、シナリオの `configs/`、
任意の学習設定の `experiments/`、編成例の `examples/`、CLIと検証の `scripts/` です。
新しい依存関係や外部素材を追加したら、出典・ライセンスと配布範囲も更新してください。
