# Slither.io × RT-DLO 自動操作

Slither.io のヘビを **RT-DLO (Real-Time Deformable Linear Objects)** の考え方で扱い、骨格抽出・追跡・制御を行うプロジェクトです。

## アーキテクチャ（全体像）

```
Docker コンテナ (nvidia/cuda + Ubuntu 22.04)
├── Xvfb :99 (仮想ディスプレイ 1280×720)
├── x11vnc + noVNC (port 6080 で外部モニタリング)
├── Chromium (Selenium 経由で slither.io を自動操作)
└── Python メインループ
    ├── capture.py         ── mss で Xvfb をキャプチャ
    ├── snake_skeleton.py  ── 自機骨格抽出 (HSV マスク + 細線化)
    ├── dlo_instance.py    ── DLO データ構造 (全ヘビ統一表現)
    ├── dlo_tracker.py     ── フレーム間追跡・速度推定・変形予測
    ├── enemy_detection.py ── 敵・餌検出 + 敵骨格抽出 (DLO化)
    ├── mouse_control.py   ── pyautogui でマウス操作
    ├── game_env.py        ── Gymnasium 環境ラッパー (252次元観測)
    ├── agent_rl.py        ── PPO 学習エージェント (CUDA)
    ├── monitor.py         ── 認識モニタウィンドウ (2×2 DLO 表示)
    └── browser.py         ── Selenium ブラウザ制御
```

パイプライン:

1. **映像キャプチャ** — Xvfb 上の Chromium 画面をリアルタイム取得（mss）
2. **前処理** — 自機の色でマスク → 細線化 → 骨格の座標列に変換
3. **敵・餌検出 + DLO 化** — 背景 HSV マスク + 自機マスク除外 → 連結成分で分類 → 敵も骨格抽出して DLO インスタンスに
4. **DLO 追跡** — Hungarian 法でフレーム間マッチング → ID 維持 → 速度推定 → 1 ステップ先の変形予測
5. **意思決定** — PPO 強化学習エージェントが移動方向・加速を決定（予測衝突リスク含む 252 次元観測）
5. **入力操作** — pyautogui でマウスエミュレート

## セットアップ

### Docker（推奨・GPU 自動運転）

**前提**: Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) がインストール済み。

```bash
# ビルド & 起動
docker compose build
docker compose up

# ブラウザで認識モニタを確認
# http://localhost:6080
```

コンテナ内で自動的に Xvfb → VNC → Chromium → ゲーム開始 → RL 学習が始まります。

### ローカル（骨格可視化・HSV 調整用）

```bash
cd /path/to/slither_ai_dlo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 使い方

### 1. 自機の色を合わせる

Slither.io でゲームを開始し、**自機のヘビの色**を決めます。
`config.py` の `SNAKE_HSV_LOWER` と `SNAKE_HSV_UPPER` をその色に合わせてください（HSV 範囲）。

| 色の目安 | H (色相) |
|----------|----------|
| 赤       | 0–10 または 170–180 |
| 黄       | 20–35 |
| 緑       | 35–85 |
| 青       | 100–124 |
| 紫       | 125–155 |

デフォルトは緑系です。

### 2. 骨格の可視化

```bash
python main.py
```

- ゲーム画面をキャプチャし、自機の骨格を **緑（頭）→ 青（胴）→ 赤（尾）** でオーバーレイ表示します。
- 終了: 表示ウィンドウをアクティブにして **`q`** を押す。

### 3. マスクのデバッグ（HSV 調整用）

```bash
python main.py debug
```

- 左上: 元画像 / 右上: 色マスク
- 左下: 最大連結成分 / 右下: 細線化骨格
- 骨格がうまく出ない場合は、ここを見ながら `config.py` の HSV を調整してください。

### 4. 自動運転 Bot（Docker 内）

```bash
python main.py bot
```

- Chromium で slither.io を開き、自動でゲーム開始
- PPO 強化学習エージェントがリアルタイムで学習しながら操作
- 認識モニタウィンドウ（2×2 グリッド）で検出状態を確認可能
- ゲームオーバー時は自動リスタート
- モデルは `models/` に定期保存、次回起動時に継続学習

### 認識モニタ

`http://localhost:6080` の VNC 画面に 2×2 グリッドで表示されます。

| パネル | 内容 |
|--------|------|
| 左上 | 自機検出（緑マスク + 骨格線） |
| 右上 | 敵 DLO 検出（赤=骨格、橙=予測位置、ID ラベル、速度矢印、黄=餌） |
| 左下 | 統合 DLO オーバーレイ（自機 + 全敵骨格 + 予測 + 餌） |
| 右下 | RL 状態（報酬推移グラフ、行動、ステップ数） |

## キャプチャ領域

- `config.py` の `CAPTURE_MONITOR`:
  - `None` … 全画面
  - `1` … メインモニター
  - `{"left": 0, "top": 0, "width": 1920, "height": 1080}` … 指定領域（ゲームだけにすると軽くなります）

## 強化学習の設定

`config.py` で調整可能なパラメータ:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `RL_MODEL_DIR` | `models` | モデル保存先 |
| `RL_SAVE_INTERVAL` | `10000` | 保存間隔（ステップ） |
| `TOP_K_ENEMIES` | `8` | 観測に含む最寄り敵の数 |
| `TOP_M_FOOD` | `16` | 観測に含む最寄り餌の数 |
| `ENEMY_MIN_AREA` | `300` | 敵と判定する最小面積 |
| `FOOD_MAX_AREA` | `299` | 餌と判定する最大面積 |
| `ENEMY_SKELETON_POINTS` | `20` | 敵骨格のサンプル点数 |
| `DLO_MAX_LOST_FRAMES` | `5` | 追跡 ID 消失までの猶予フレーム数 |
| `DLO_VELOCITY_ALPHA` | `0.3` | 速度の指数移動平均係数 |
| `DLO_MATCH_MAX_DIST` | `200` | マッチング最大距離 (px) |

環境変数で上書き可能: `GAME_URL`, `NICKNAME`, `SCREEN_WIDTH`, `SCREEN_HEIGHT`, `RL_MODEL_DIR`, `RL_SAVE_INTERVAL`

## 技術スタック

| 要素 | 使用ツール |
|------|-----------|
| 言語 | Python 3.8+ |
| コンテナ | Docker + NVIDIA Container Toolkit |
| GPU | CUDA 12.2 (PyTorch) |
| 仮想ディスプレイ | Xvfb + x11vnc + noVNC |
| ブラウザ | Chromium + Selenium |
| 画面取得 | mss |
| 画像処理・骨格 | OpenCV, scikit-image, scipy |
| マウス操作 | pyautogui |
| 強化学習 | Gymnasium, Stable-Baselines3 (PPO) |

## DLO 統合アーキテクチャ

全てのヘビ（自機＋敵）を **DLO (Deformable Linear Object) インスタンス**として統一的に扱います。

### DLO パイプライン

1. **検出**: 背景除去 + 色マスクで前景の連結成分を分類（敵/餌）
2. **骨格抽出**: 各敵連結成分を細線化 → 骨格座標 (20 点) を抽出
3. **追跡**: Hungarian 法で前フレームとマッチング → 一意の ID を維持
4. **速度推定**: 重心差分 + 骨格点差分を指数移動平均で平滑化
5. **予測**: 現在の骨格 + 速度で 1 ステップ先を線形外挿 → 衝突リスク算出

### 観測空間 (252 次元)

| セグメント | 次元数 | 内容 |
|-----------|--------|------|
| 自機骨格 | 160 | 80 点 × (y, x) |
| 自機メタ | 4 | heading, length, vel_x, vel_y |
| 敵 DLO メタ | 48 | top-8 敵 × (center_dx, center_dy, heading, length, vel_dx, vel_dy) |
| 最寄り餌 | 32 | top-16 餌 × (dx, dy) |
| 衝突リスク | 8 | top-8 敵の予測骨格との最短距離 |

骨格座標は `snake_skeleton.skeleton_points_for_rt_dlo(points)` で **(x, y)** の float 配列に変換できます。

## 注意事項

- **座標系**: 内部は (y, x) numpy 順。OpenCV 描画は (x, y)。混同に注意。
- **処理速度**: 実時間性を保つため、負荷が高い場合はフレームスキップを検討してください。
- **GPU**: Bot モードは NVIDIA GPU (CUDA) を前提としています。CPU のみでも動作しますが学習速度は低下します。
- **共有メモリ**: Docker の `shm_size: 2g` は Chromium のクラッシュ防止に必要です。

## Git にアップロードする前の初期化手順

初めてこのリポジトリを Git で管理し、リモートにプッシュする場合の手順です。

### 1. リポジトリ内で Git を初期化

```bash
cd /path/to/slither_ai_dlo
git init
```

### 2. 除外設定の確認

プロジェクトルートに `.gitignore` があります。次のようなものがコミットされません。

- `__pycache__/`, `*.pyc`（Python キャッシュ）
- `.venv/`, `venv/`（仮想環境）
- `.env`, `.claude/`（ローカル設定・秘密）
- `weights/`, `*.pth`（モデル重みは必要に応じて別管理）

必要なら `.gitignore` を編集してから次へ進んでください。

### 3. 初回コミット

```bash
git add .
git status   # 追加されるファイルを確認（.venv 等が含まれていないこと）
git commit -m "Initial commit: Slither.io DLO pipeline + RL bot"
```

### 4. リモートの追加とプッシュ

GitHub / GitLab などで **空のリポジトリ** を作成したあと:

```bash
git remote add origin https://github.com/<user>/slither_ai_dlo.git
git branch -M main
git push -u origin main
```

SSH の場合は `git@github.com:<user>/slither_ai_dlo.git` に読み替えてください。

### チェックリスト

- [ ] `git init` 済み
- [ ] `.gitignore` で `.venv` や秘密ファイルが除外されていることを確認
- [ ] `git status` で意図しないファイルが add されていないことを確認
- [ ] リモートで空リポジトリを作成済み
- [ ] `git push` で初回アップロード完了

## ライセンス

MIT を想定（必要に応じて変更してください）。
