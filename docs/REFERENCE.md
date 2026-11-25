# 設定の参照先

| 対象 | ファイル |
|---|---|
| 標準シナリオ・編成 | `configs/scenario-default.json` |
| 共通シナリオ | `configs/scenario.json` |
| 空域・初期配置 | `configs/airspace.json` |
| 機体・ミサイルの運動とセンサー | `configs/aerodynamic.json` |
| ルール機の設定・抽選範囲 | `configs/rule-defaults.json`・`configs/scenario-rule-curriculum.json` |
| 基本報酬 | `configs/rewards.json` |
| 追加報酬の上限 | `configs/reward-hooks.json` |
| Live・学習補助の初期値 | `configs/application.json` |

`environment_config` の参照先を先に読み、参照元の項目で上書きします。
`airspace_config`・`reward_config` は設定ファイルからの相対パスです。
機体ごとの `aircraft[].performance` は共通性能より優先されます。
設定を変更したら実行プロセスを再起動してください。

Liveでは標準シナリオを基に画面指定の機数を編成します。モデルに保存された学習シナリオとは別です。
対戦を比較するときはシナリオ・シード・機数・相手・行動選択を揃えます。

標準報酬では勝敗をチーム各機、撃墜を射手、機体喪失の罰を失った本人に配布します。
低水準の `Config()` は基礎値を使うため、通常は `make_env()` または明示したシナリオを使ってください。
