"""
Slither.io 自機検出用の設定。
ゲーム内で選んだヘビの色に合わせて HSV 範囲を調整してください。
"""

from __future__ import annotations

import os

# --- 画面キャプチャ ---
# キャプチャする領域 (None で全画面)。Slither.io のゲーム領域だけにすると高速化できる。
# 例: {"left": 0, "top": 0, "width": 1920, "height": 1080}
CAPTURE_MONITOR = None  # 1 でメイン画面、None で全画面

# --- 自機ヘビの色（HSV）---
# Slither.io ではヘビの色を選べる。その色を HSV で指定する。
# OpenCV の HSV: H∈[0,180], S∈[0,255], V∈[0,255]
# よく使う色の目安:
#   赤:   H 0-10 or 170-180
#   緑:   H 35-85
#   青:   H 100-124
#   黄:   H 20-35
#   紫:   H 125-155

SNAKE_HSV_LOWER = (35, 80, 80)   # 緑っぽい色の下限 (H, S, V) — デフォルト/フォールバック
SNAKE_HSV_UPPER = (85, 255, 255) # 緑の上限 — デフォルト/フォールバック

# 赤は H が 0 付近と 180 付近にまたがるので、必要なら2組の範囲を使う（snake_skeleton 側で対応可）

# --- 自機カラー自動検出 ---
AUTO_DETECT_COLOR = True           # True: ゲーム開始時に自機の色を自動検出する
COLOR_DETECT_ROI_SIZE = 200        # 画面中心から切り出す ROI の辺長 (px)
COLOR_DETECT_HUE_MARGIN = 20      # ピーク hue からの許容幅 (±)
COLOR_DETECT_MIN_FG_PIXELS = 50   # 前景ピクセルの最低数（これ未満は検出失敗）

# マスクのノイズ除去
MORPH_KERNEL_SIZE = (3, 3)  # クロージング・オープニングのカーネル
MIN_SNAKE_AREA = 500        # これより小さい連結成分は無視（ピクセル数）

# 骨格のサンプリング
SKELETON_SAMPLE_POINTS = 80  # 頭→尾まで何点でサンプルするか（RT-DLO や制御で使いやすい数）

# --- ブラウザ・ゲーム ---
GAME_URL = os.environ.get("GAME_URL", "http://slither.io")
NICKNAME = os.environ.get("NICKNAME", "AI_Bot")
SCREEN_WIDTH = int(os.environ.get("SCREEN_WIDTH", "1280"))
SCREEN_HEIGHT = int(os.environ.get("SCREEN_HEIGHT", "720"))

# ローカル実行時に headless モードでブラウザを起動するか自動判定する。
# Docker (Xvfb) 内では headed で VNC 表示、ローカルでは headless でフォーカス奪取を防止。
# HEADLESS_BROWSER=true/false で明示的に上書き可能。
_headless_env = os.environ.get("HEADLESS_BROWSER", "auto")
if _headless_env == "auto":
    HEADLESS_BROWSER = not os.path.exists("/.dockerenv")
else:
    HEADLESS_BROWSER = _headless_env.lower() in ("true", "1", "yes")

# --- 敵検出 ---
BG_HSV_LOWER = (0, 0, 0)       # 背景の HSV 下限
BG_HSV_UPPER = (180, 60, 80)   # 背景の HSV 上限 (低彩度・低明度)
ENEMY_MIN_AREA = 300            # これ以上 = 敵ヘビ（形状判定も併用）
FOOD_MAX_AREA = 299             # これ以下 = 餌（形状判定も併用）

# --- DLO 形状分類（敵 vs 餌） ---
# 面積だけでなく形状特徴で分類する。ヘビは細長く、餌は丸い。
SHAPE_CIRCULARITY_THRESH = 0.55  # 円形度: これ以上 = 餌っぽい（完全円=1.0）
SHAPE_ASPECT_RATIO_THRESH = 2.5  # アスペクト比: これ以上 = ヘビっぽい（細長い）
SHAPE_USE_FOR_CLASSIFICATION = True  # True: 形状特徴を使って敵/餌を再分類する

# --- DLO 追跡 ---
ENEMY_SKELETON_POINTS = 20   # 敵骨格のサンプル点数（自機80より少ない、計算コスト削減）
DLO_MAX_LOST_FRAMES = 5      # ID 消失までの猶予フレーム数
DLO_VELOCITY_ALPHA = 0.3     # 速度の指数移動平均係数 (0~1, 大きいほど最新値重視)
DLO_MATCH_MAX_DIST = 200     # マッチング最大距離 (px)

# --- 強化学習 ---
RL_MODEL_DIR = os.environ.get("RL_MODEL_DIR", "models")
RL_SAVE_INTERVAL = int(os.environ.get("RL_SAVE_INTERVAL", "10000"))


def _detect_device() -> str:
    """GPU デバイスを自動検出する。CUDA (NVIDIA) / ROCm (AMD) → 'cuda', それ以外 → 'cpu'。"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
            # ROCm は torch.cuda API 経由でアクセスされる（HIP バックエンド）
            backend = "ROCm" if hasattr(torch.version, "hip") and torch.version.hip else "CUDA"
            print(f"[GPU] {backend} device detected: {name}")
            return "cuda"
    except Exception:
        pass
    print("[GPU] No GPU detected, using CPU")
    return "cpu"


# "auto" の場合は実際にデバイスを検出してログ出力する
_rl_device_env = os.environ.get("RL_DEVICE", "auto")
RL_DEVICE = _detect_device() if _rl_device_env == "auto" else _rl_device_env
TOP_K_ENEMIES = 8
TOP_M_FOOD = 16

# --- CNN ハイブリッド観測 ---
RL_OBS_MODE = os.environ.get("RL_OBS_MODE", "hybrid")  # "vector" or "hybrid"
CNN_INPUT_SIZE = (84, 84)
CNN_FRAME_STACK = 4
CNN_BATCH_SIZE = 256
CNN_FEATURES_DIM = 256

# --- 報酬パラメータ（シンプル5コンポーネント設計） ---
# 成長報酬（JSスコア増加が主要報酬）
REWARD_GROWTH_SCORE_SCALE = 1.0 # JSスコア成長報酬のスケール
REWARD_GROWTH_SCORE_CAP = 1.0   # JSスコア成長報酬の上限 / step
REWARD_GROWTH_TOTAL_CAP = 1.0   # 成長報酬合計の上限 / step
# キル報酬（敵消滅検出: DLO ID消失 + 自機ヘッド近接）
REWARD_KILL = 2.0               # キル報酬（高価値イベント）
KILL_DETECT_RADIUS = 150        # キル検出半径 (px)
# 怠慢ペナルティ（長時間成長なしで発動→受動戦略を排除）
REWARD_IDLE_GRACE_STEPS = 100   # 怠慢ペナルティ猶予ステップ数
REWARD_IDLE_PENALTY = 0.02      # 怠慢ペナルティ / step
# 敵危険ペナルティ（旧: 敵近接 + 衝突予測を統合）
REWARD_ENEMY_DIST_THRESH = 60   # 敵危険ペナルティの距離閾値 (px)
REWARD_ENEMY_MAX_PENALTY = 0.5  # 敵危険ペナルティの最大値
# 壁ペナルティ（シンプル線形。旧: 線形+二次から縮小）
REWARD_WALL_THRESH = 0.7        # 壁ペナルティ発動閾値 (JS boundary_ratio: 0=中心, 1=端)
REWARD_WALL_MAX = 1.0           # 壁ペナルティの最大値
# 死亡ペナルティ（壁死/通常死を統一）
REWARD_DEATH = -5.0             # 死亡ペナルティ（統一）
REWARD_DEBUG_LOG = True          # True: ステップごとの報酬内訳をログ出力

# --- 模倣学習 (Imitation Learning) ---
IL_DEMO_DIR = os.environ.get("IL_DEMO_DIR", "demos")       # デモデータ保存ディレクトリ
IL_RECORD_INTERVAL = 0.05       # 記録間隔 (秒)。ゲームの step と同じ 50ms
IL_BC_EPOCHS = int(os.environ.get("IL_BC_EPOCHS", "50"))   # Behavioral Cloning のエポック数
IL_BC_BATCH_SIZE = int(os.environ.get("IL_BC_BATCH_SIZE", "64"))   # BC バッチサイズ
IL_BC_LR = float(os.environ.get("IL_BC_LR", "1e-3"))      # BC 学習率
IL_MIN_DEMO_STEPS = 50          # これ未満のエピソードは保存しない（短すぎるデモを除外）
