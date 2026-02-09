# Slither.io × RT-DLO 自動操作 AI

Slither.io のヘビを **RT-DLO (Real-Time Deformable Linear Objects)** の考え方で扱い、骨格抽出・追跡・強化学習による自律制御を行うプロジェクトです。

## アーキテクチャ

```
Docker コンテナ (Ubuntu 22.04 + NVIDIA/AMD/CPU 自動対応)
├── Xvfb :99 (仮想ディスプレイ 1280×720)
├── x11vnc + noVNC (port 6080 で外部モニタリング)
├── Chromium (Selenium 経由で slither.io を自動操作)
└── Python パイプライン
    ├── capture.py          ── mss で Xvfb をキャプチャ
    ├── snake_skeleton.py   ── 自機骨格抽出 (HSV マスク + 細線化)
    ├── color_detect.py     ── 自機カラー自動検出 (ROI 内色相ピーク)
    ├── dlo_instance.py     ── DLO データ構造 (全ヘビ統一表現)
    ├── dlo_tracker.py      ── フレーム間追跡・速度推定・変形予測
    ├── enemy_detection.py  ── 敵・餌検出 + 敵骨格抽出 (DLO 化)
    ├── mouse_control.py    ── pyautogui でマウス操作
    ├── game_env.py         ── Gymnasium 環境 (vector / hybrid 観測)
    ├── agent_rl.py         ── PPO 学習エージェント (MlpPolicy / MultiInputPolicy)
    ├── monitor.py          ── 認識モニタ (2×2 DLO 表示)
    ├── browser.py          ── Selenium ブラウザ制御
    └── config.py           ── 全設定パラメータ
```

### パイプライン

1. **映像キャプチャ** — Xvfb 上の Chromium 画面をリアルタイム取得（mss）
2. **自機カラー検出** — 画面中心 ROI の色相ピークから HSV 範囲を自動推定（初回 + 定期更新）
3. **前処理** — 自機の色でマスク → 細線化 → 骨格の座標列 (80 点) に変換
4. **敵・餌検出 + DLO 化** — 背景 HSV マスク + 自機マスク除外 → 連結成分で分類 → 敵も骨格抽出して DLO インスタンスに
5. **DLO 追跡** — Hungarian 法でフレーム間マッチング → ID 維持 → EMA 速度推定 → 1 ステップ先の変形予測
6. **意思決定** — PPO 強化学習エージェントが移動方向・加速を決定
7. **入力操作** — pyautogui でマウスエミュレート

## セットアップ

### Docker（推奨・GPU 自動運転）

GPU 環境に合わせて起動方法を選択してください。

#### NVIDIA GPU

**前提**: Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) がインストール済み。

```bash
GPU_TYPE=nvidia docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up --build
```

#### AMD GPU (Radeon / ROCm)

**前提**: Docker + [ROCm](https://rocm.docs.amd.com/) がインストール済み。`/dev/kfd` と `/dev/dri` へのアクセスが必要。

```bash
GPU_TYPE=amd docker compose -f docker-compose.yml -f docker-compose.amd.yml up --build
```

#### CPU のみ（Mac / Apple Silicon 含む）

```bash
docker compose up --build
```

Mac や Apple Silicon では同じコマンドで OK。イメージは arm64/amd64 に合わせてビルドされ、ブラウザは Chromium (.deb) を使用します。

#### 共通

```bash
# ブラウザで認識モニタを確認
# 同じマシン: http://localhost:6080/vnc.html
# リモート:   http://<サーバのIP>:6080
```

**「compose build requires buildx 0.17.0 or later」と出る場合**:

```bash
GPU_TYPE=nvidia docker build --build-arg GPU_TYPE=nvidia -t slither_ai_dlo-slither-bot .
docker compose up
```

**「RuntimeError: can't start new thread」で pip が落ちる場合**:

```bash
docker build --ulimit nproc=8192:8192 --build-arg GPU_TYPE=nvidia -t slither_ai_dlo-slither-bot .
```

コンテナ内で自動的に Xvfb → VNC → Chromium → ゲーム開始 → RL 学習が始まります。

### ローカル（どの環境でも同じ手順で立ち上がる）

**macOS / Linux:**

```bash
cd /path/to/slither_ai_dlo_2
./scripts/setup.sh          # 初回のみ。python3/python で .venv を作成し依存をインストール
./scripts/run.sh            # 骨格可視化
./scripts/run.sh debug      # HSV デバッグ
./scripts/run.sh bot        # 自動運転 + 強化学習
```

`run.sh` は仮想環境が無い／壊れている場合に自動で `setup.sh` を実行してから起動します。

**Windows (PowerShell):**

```powershell
cd \path\to\slither_ai_dlo_2
.\scripts\setup.ps1         # 初回のみ
.\scripts\run.ps1 bot       # 例: bot 起動
.\scripts\run.ps1 debug     # 例: デバッグ
```

**手動で venv を使う場合:**

```bash
cd /path/to/slither_ai_dlo_2
python3 -m venv .venv       # または python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py bot
```

## 使い方

### 1. 自機の色を合わせる

`AUTO_DETECT_COLOR = True`（デフォルト）であれば、ゲーム開始時に画面中心の色相分布から自機の色を自動検出します。手動設定する場合は `config.py` の `SNAKE_HSV_LOWER` / `SNAKE_HSV_UPPER` を変更してください。

| 色の目安 | H (色相) |
|----------|----------|
| 赤       | 0–10 または 170–180 |
| 黄       | 20–35 |
| 緑       | 35–85 |
| 青       | 100–124 |
| 紫       | 125–155 |

### 2. 骨格の可視化

```bash
./scripts/run.sh            # または source .venv/bin/activate && python main.py
```

- ゲーム画面をキャプチャし、自機の骨格を **緑（頭）→ 青（胴）→ 赤（尾）** でオーバーレイ表示
- 終了: 表示ウィンドウで **`q`** を押す

### 3. マスクのデバッグ（HSV 調整用）

```bash
./scripts/run.sh debug
```

- 左上: 元画像 / 右上: 色マスク / 左下: 最大連結成分 / 右下: 細線化骨格
- 骨格がうまく出ない場合はこの画面で HSV を調整

### 4. 自動運転 Bot

```bash
./scripts/run.sh bot        # ローカル（Slither.io が表示されている画面をキャプチャ）
# または Docker 内で python main.py bot が自動実行される
```

- Chromium で slither.io を開き、自動でゲーム開始
- PPO 強化学習エージェントがリアルタイムで学習しながら操作
- 認識モニタウィンドウ（2×2 グリッド）で検出状態を確認可能
- ゲームオーバー時は自動リスタート
- モデルは `models/` に定期保存、次回起動時に継続学習

## 観測モード

`config.py` の `RL_OBS_MODE` で切り替え（環境変数 `RL_OBS_MODE` でも上書き可能）。

### vector モード（`RL_OBS_MODE = "vector"`）

従来の固定長ベクトル観測。`MlpPolicy` で学習。

| セグメント | 次元数 | 内容 |
|-----------|--------|------|
| 自機骨格 | 160 | 80 点 × (y, x) |
| 自機メタ | 4 | heading, length, vel_x, vel_y |
| 敵 DLO メタ | 48 | top-8 敵 × (center_dx, center_dy, heading, length, vel_dx, vel_dy) |
| 最寄り餌 | 32 | top-16 餌 × (dx, dy) |
| 衝突リスク | 8 | top-8 敵の予測骨格との最短距離 |
| **合計** | **252** | |

### hybrid モード（`RL_OBS_MODE = "hybrid"`、デフォルト）

CNN + MLP のマルチ入力観測。`MultiInputPolicy`（SB3 の `CombinedExtractor`）で学習。

| キー | 形状 | 内容 |
|------|------|------|
| `image` | `(84, 84, 1)` uint8 | グレースケール画像 |
| `metadata` | `(60,)` float32 | self_meta(4) + enemy_dlo(48) + collision_risk(8) |

- `VecFrameStack(n_stack=4, channels_order="last")` により、実際のネットワーク入力は image `(84, 84, 4)` + metadata `(240,)` になる
- CNN 特徴量次元: `CNN_FEATURES_DIM = 256`
- バッチサイズ: `CNN_BATCH_SIZE = 256`

## 認識モニタ

noVNC (`http://localhost:6080/`) で 2×2 グリッドを表示。

| パネル | 内容 |
|--------|------|
| 左上 | 自機検出（緑マスク + 骨格線） |
| 右上 | 敵 DLO 検出（赤=骨格、橙=予測位置、ID ラベル、速度矢印、黄=餌） |
| 左下 | 統合 DLO オーバーレイ（自機 + 全敵骨格 + 予測 + 餌） |
| 右下 | RL 状態（報酬推移グラフ、行動、ステップ数） |

## DLO 統合アーキテクチャ

全てのヘビ（自機＋敵）を **DLO (Deformable Linear Object) インスタンス**として統一的に扱います。

### DLO パイプライン

1. **検出**: 背景除去 + 色マスクで前景の連結成分を分類（敵/餌）
2. **骨格抽出**: 各敵連結成分を細線化 → 骨格座標 (20 点) を抽出
3. **追跡**: Hungarian 法 (`scipy.linear_sum_assignment`) で前フレームとマッチング → 一意の ID を維持
4. **速度推定**: 重心差分 + 骨格点差分を指数移動平均 (EMA) で平滑化
5. **予測**: 現在の骨格 + 速度で 1 ステップ先を線形外挿 → 衝突リスク算出

## 報酬設計

| 報酬コンポーネント | 値 | 条件 |
|-------------------|-----|------|
| 生存報酬 | +0.1 / step | 常時 |
| 成長報酬 | 最大 +5.0 | 自機マスク面積が増加（餌獲得） |
| 餌接近報酬 | -0.3 ~ +0.5 | 最寄り餌への距離変化 |
| 敵近接ペナルティ | 最大 -0.5 | 敵が 100px 以内 |
| 予測衝突ペナルティ | 最大 -0.3 | DLO 予測骨格が 80px 以内 |
| 壁接近ペナルティ | 最大 -15.0 | 赤い境界が画面端に出現 |
| 通常死 | -10.0 | ゲームオーバー |
| 壁死 | -20.0 + 生存報酬取消 | 壁衝突によるゲームオーバー |

## 設定パラメータ

`config.py` で調整可能。一部は環境変数でも上書き可能。

### ゲーム・ブラウザ

| パラメータ | デフォルト | 環境変数 | 説明 |
|-----------|-----------|---------|------|
| `GAME_URL` | `http://slither.io` | `GAME_URL` | ゲーム URL |
| `NICKNAME` | `AI_Bot` | `NICKNAME` | ニックネーム |
| `SCREEN_WIDTH` | `1280` | `SCREEN_WIDTH` | 画面幅 |
| `SCREEN_HEIGHT` | `720` | `SCREEN_HEIGHT` | 画面高さ |

### カラー検出

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `AUTO_DETECT_COLOR` | `True` | 自機カラー自動検出の有効化 |
| `COLOR_DETECT_ROI_SIZE` | `200` | 中心 ROI の辺長 (px) |
| `COLOR_DETECT_HUE_MARGIN` | `15` | ピーク hue からの許容幅 |
| `COLOR_DETECT_MIN_FG_PIXELS` | `50` | 前景ピクセルの最低数 |

### 敵検出・DLO 追跡

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `ENEMY_MIN_AREA` | `300` | 敵と判定する最小面積 |
| `FOOD_MAX_AREA` | `299` | 餌と判定する最大面積 |
| `ENEMY_SKELETON_POINTS` | `20` | 敵骨格のサンプル点数 |
| `DLO_MAX_LOST_FRAMES` | `5` | ID 消失までの猶予フレーム数 |
| `DLO_VELOCITY_ALPHA` | `0.3` | 速度の EMA 係数 |
| `DLO_MATCH_MAX_DIST` | `200` | マッチング最大距離 (px) |

### 強化学習

| パラメータ | デフォルト | 環境変数 | 説明 |
|-----------|-----------|---------|------|
| `RL_OBS_MODE` | `hybrid` | `RL_OBS_MODE` | 観測モード (`vector` / `hybrid`) |
| `RL_MODEL_DIR` | `models` | `RL_MODEL_DIR` | モデル保存先 |
| `RL_SAVE_INTERVAL` | `10000` | `RL_SAVE_INTERVAL` | 保存間隔（ステップ） |
| `RL_DEVICE` | `auto` | `RL_DEVICE` | デバイス (`auto` / `cuda` / `cpu`) |
| `TOP_K_ENEMIES` | `8` | — | 観測に含む最寄り敵の数 |
| `TOP_M_FOOD` | `16` | — | 観測に含む最寄り餌の数 |

### CNN ハイブリッド観測

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `CNN_INPUT_SIZE` | `(84, 84)` | CNN 入力画像サイズ |
| `CNN_FRAME_STACK` | `4` | フレームスタック枚数 |
| `CNN_BATCH_SIZE` | `256` | PPO バッチサイズ（hybrid 時） |
| `CNN_FEATURES_DIM` | `256` | CombinedExtractor の CNN 出力次元 |

## 技術スタック

| 要素 | 使用ツール |
|------|-----------|
| 言語 | Python 3.8+ |
| コンテナ | Docker (+ NVIDIA Container Toolkit or ROCm) |
| GPU | CUDA (NVIDIA) / ROCm (AMD) / CPU 自動検出 |
| 仮想ディスプレイ | Xvfb + x11vnc + noVNC |
| ブラウザ | Chromium + Selenium |
| 画面取得 | mss |
| 画像処理・骨格 | OpenCV, scikit-image, scipy |
| マウス操作 | pyautogui |
| 強化学習 | Gymnasium, Stable-Baselines3 (PPO) |

## モジュール一覧

| ファイル | 役割 |
|---------|------|
| `main.py` | エントリポイント。vis / debug / bot の 3 モード |
| `config.py` | 全設定パラメータ（HSV, DLO, RL, CNN 等） |
| `capture.py` | mss による画面キャプチャ |
| `snake_skeleton.py` | 自機骨格抽出（HSV マスク → 細線化 → BFS → リサンプル） |
| `color_detect.py` | 画面中心 ROI の色相分布から自機 HSV 範囲を自動推定 |
| `dlo_instance.py` | `DLOInstance` / `DLOState` データクラス |
| `dlo_tracker.py` | Hungarian マッチング + EMA 速度推定 + 線形予測 |
| `enemy_detection.py` | 敵・餌検出 → `DLOState` 構築 |
| `game_env.py` | Gymnasium 環境。vector (252 次元) / hybrid (Dict) 観測 |
| `agent_rl.py` | PPO エージェント（MlpPolicy / MultiInputPolicy） |
| `monitor.py` | 2×2 認識モニタ（DLO 骨格 + 予測 + RL 状態） |
| `mouse_control.py` | pyautogui マウス操作（角度移動 + ブースト） |
| `browser.py` | Selenium ブラウザ制御（ゲーム開始・リスタート・状態取得） |

## 注意事項

- **座標系**: 内部は (y, x) numpy 順。OpenCV 描画は (x, y)。DLO の center/velocity は (x, y) float64。
- **GPU**: NVIDIA (CUDA) / AMD (ROCm) / CPU を自動検出。GPU があれば学習を高速化。CPU のみでも動作するが学習速度は低下。
- **共有メモリ**: Docker の `shm_size: 2g` は Chromium のクラッシュ防止に必要。

## Git にアップロードする前の初期化手順

```bash
cd /path/to/slither_ai_dlo
git init
git add .
git status   # .venv 等が含まれていないことを確認
git commit -m "Initial commit: Slither.io DLO pipeline + RL bot"
git remote add origin https://github.com/<user>/slither_ai_dlo.git
git branch -M main
git push -u origin main
```

`.gitignore` により `__pycache__/`, `.venv/`, `.env`, `.claude/`, `weights/`, `*.pth` 等は除外されます。

## ライセンス

MIT を想定（必要に応じて変更してください）。
