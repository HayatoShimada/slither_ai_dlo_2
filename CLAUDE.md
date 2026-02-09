# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slither.io automation AI. The system captures the game screen, extracts the player's snake skeleton via color masking + skeletonization, and controls the snake autonomously using reinforcement learning. Documentation (README.md, DESIGN.md) is written in Japanese.

**Current pipeline:** screen capture → color mask + skeletonization (all snakes as DLO instances) → frame-to-frame tracking (Hungarian matching) → velocity estimation + 1-step prediction → RL-based autonomous control.

**DLO integration:** All snakes (self + enemies) are treated as DLO (Deformable Linear Object) instances. Each instance has skeleton coordinates, heading, length, velocity, and predicted next-frame position. The tracker assigns persistent IDs across frames using center distance + shape similarity. This follows the RT-DLO philosophy (instance segmentation → centerline extraction → temporal tracking → deformation prediction) but implemented with OpenCV + scikit-image instead of a deep learning model.

## Commands

**どの環境でも立ち上がる（推奨）:**

```bash
# macOS / Linux: 初回セットアップ＋起動
./scripts/setup.sh           # 初回のみ。python3/python で .venv を作成・修復し依存をインストール
./scripts/run.sh             # 骨格可視化
./scripts/run.sh debug       # HSV デバッグ
./scripts/run.sh bot         # 自動運転 + 強化学習（venv が無い/壊れていれば setup を自動実行）
./scripts/run.sh record      # 人間プレイを記録（模倣学習用デモデータ収集）
./scripts/run.sh pretrain    # デモデータから Behavioral Cloning 事前訓練
```

```powershell
# Windows (PowerShell)
.\scripts\setup.ps1
.\scripts\run.ps1 bot
```

**手動で venv を使う場合:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py bot
```

**Docker (full autonomous setup):**

```bash
# NVIDIA GPU:
GPU_TYPE=nvidia docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up --build
# AMD GPU (Radeon/ROCm):
GPU_TYPE=amd docker compose -f docker-compose.yml -f docker-compose.amd.yml up --build
# CPU only:
docker compose up --build
# Access monitoring: http://localhost:6080 (noVNC)
```

**模倣学習ワークフロー:**

```bash
# Step 1: 人間プレイを記録（noVNC http://localhost:6080 でプレイ）
./scripts/run.sh record      # demos/ に .npz ファイルが保存される

# Step 2: Behavioral Cloning で事前訓練
./scripts/run.sh pretrain    # demos/ のデータで PPO ポリシーを教師あり学習

# Step 3: RL ファインチューニング（BC 事前訓練済みモデルから継続）
./scripts/run.sh bot         # models/slither_ppo.zip を読み込んで PPO 学習を継続
```

No test framework, linter, or build system is configured yet.

## Docker Setup

**Requirements:** Docker. GPU利用時は NVIDIA Container Toolkit (NVIDIA) または ROCm (AMD) が必要。CPU のみでも動作する。

The Docker container includes:
- **Xvfb** (virtual display 1280x720)
- **x11vnc + noVNC** (port 6080 for remote monitoring)
- **Chromium** (Selenium-driven, playing slither.io)
- **Python pipeline** (capture → detection → RL → control)
- **GPU** (CUDA/ROCm/CPU auto-detect for PyTorch PPO training)

`docker compose up` starts all services. Connect to `http://localhost:6080` to observe the game and recognition monitor via VNC.

## Architecture

**Current modules:**

- `capture.py` — Screen capture via `mss`. Returns BGR numpy arrays. Supports full-screen, single monitor, or region capture.
- `snake_skeleton.py` — Extraction pipeline: HSV color masking → morphological filtering → connected component analysis → skeletonization (scikit-image) → endpoint detection → BFS path tracing → uniform resampling. Returns `(N, 2)` array of ordered skeleton points.
- `main.py` — Entry point. `run_visualization()` for normal mode, `run_debug_mask()` for HSV tuning, `run_bot()` for autonomous RL mode, `run_record()` for imitation learning demo recording, `run_pretrain()` for Behavioral Cloning pre-training. Draws head (green), tail (red), body (blue) on the captured frame.
- `config.py` — All tunable parameters: HSV ranges, morphological kernel size, min area threshold, skeleton sample count, capture monitor selection, browser/game settings, enemy detection thresholds, DLO tracking settings, RL hyperparameters. Supports environment variable overrides.
- `browser.py` — Selenium-based Chromium control: driver creation, game start/restart, game state queries (playing, score).
- `mouse_control.py` — pyautogui mouse control: angle-based movement, absolute positioning, boost on/off.
- `dlo_instance.py` — DLO data structures: `DLOInstance` (single snake with skeleton, heading, length, velocity) and `DLOState` (all DLOs + food per frame). Helper functions for heading/length/center computation.
- `dlo_tracker.py` — Frame-to-frame DLO tracking: Hungarian matching (scipy `linear_sum_assignment`) using center distance + shape similarity cost. Exponential moving average velocity estimation. Linear extrapolation for 1-step prediction.
- `enemy_detection.py` — Enemy/food detection via background subtraction + self-mask exclusion + connected component analysis. `detect_all_objects()` returns `DLOState` with enemy skeletons extracted. `detect_enemies_and_food()` remains as backward-compatible wrapper returning `EnemyInfo`.
- `monitor.py` — Recognition monitoring window (2x2 grid): self-snake panel, enemy DLO panel (skeletons + IDs + velocity arrows + predicted positions), combined DLO overlay, RL status with reward graph.
- `game_env.py` — Gymnasium environment wrapper. Observation: 252-dim vector (skeleton 160 + self meta 4 + enemy DLO meta 48 + nearest food 32 + collision risk 8). Action: continuous angle + boost. Reward: survival + food + enemy proximity penalty + predicted collision risk penalty + death.
- `agent_rl.py` — PPO agent (Stable-Baselines3, CUDA/ROCm/CPU auto). Model creation, save/load, checkpoint callbacks, training loop.
- `imitation_data.py` — Demo data recording and dataset management for imitation learning. `DemoRecorder` captures human gameplay (observation, action pairs) via mouse position tracking. `DemoDataset` loads and batches saved `.npz` demo files for training.
- `imitation_learn.py` — Behavioral Cloning (BC) pre-training module. Trains PPO policy network with supervised learning on demo data. Angle uses sin/cos periodic loss, boost uses BCE loss. `pretrain_and_save()` provides end-to-end workflow.

**Infrastructure:**

- `Dockerfile` — Ubuntu 22.04 with Xvfb, x11vnc, noVNC, Chromium, Python. GPU_TYPE build arg for NVIDIA/AMD/CPU PyTorch.
- `docker-compose.yml` — Base config (no GPU). Use with override files for GPU.
- `docker-compose.nvidia.yml` — NVIDIA GPU overlay (nvidia driver, CUDA).
- `docker-compose.amd.yml` — AMD GPU overlay (ROCm, /dev/kfd + /dev/dri).
- `entrypoint.sh` — Startup orchestration: Xvfb → x11vnc → noVNC → python3 main.py bot.

## Key Conventions

**Coordinate systems** — This is the most important thing to get right:
- Internal/image space uses **(y, x)** (numpy indexing order)
- OpenCV drawing functions need **(x, y)**
- `skeleton_points_for_rt_dlo()` converts (y, x) → (x, y) float64 for外部連携用

**Data formats:**
- Images: `numpy.ndarray` BGR uint8
- Masks: binary uint8 (0/255)
- Skeleton points: `(N, 2)` numpy arrays
- DLO center/velocity: `(x, y)` float64 (OpenCV coordinate order)
- DLO skeleton_velocity: `(N, 2)` float64 in `(dy, dx)` per frame

**Code style:** Type hints throughout with `from __future__ import annotations`. Detailed docstrings with Parameters/Returns sections. Functions return `None` when snake is not detected.

## Dependencies

Core: `mss` (screen capture), `opencv-python` (image processing), `numpy`, `scikit-image` (skeletonization), `scipy` (Hungarian matching for DLO tracking), `pyautogui` (mouse control), `selenium` (browser automation), `gymnasium` (RL environment), `stable-baselines3` (PPO), `torch` (GPU/CPU training).

PyTorch is installed separately in the Dockerfile based on `GPU_TYPE` build arg: `nvidia` → CUDA wheels, `amd` → ROCm wheels, `cpu` (default) → CPU-only wheels.
