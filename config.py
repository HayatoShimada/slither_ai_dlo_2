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
COLOR_DETECT_HUE_MARGIN = 15      # ピーク hue からの許容幅 (±)
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

# --- 敵検出 ---
BG_HSV_LOWER = (0, 0, 0)       # 背景の HSV 下限
BG_HSV_UPPER = (180, 60, 80)   # 背景の HSV 上限 (低彩度・低明度)
ENEMY_MIN_AREA = 300            # これ以上 = 敵ヘビ
FOOD_MAX_AREA = 299             # これ以下 = 餌

# --- DLO 追跡 ---
ENEMY_SKELETON_POINTS = 20   # 敵骨格のサンプル点数（自機80より少ない、計算コスト削減）
DLO_MAX_LOST_FRAMES = 5      # ID 消失までの猶予フレーム数
DLO_VELOCITY_ALPHA = 0.3     # 速度の指数移動平均係数 (0~1, 大きいほど最新値重視)
DLO_MATCH_MAX_DIST = 200     # マッチング最大距離 (px)

# --- 強化学習 ---
RL_MODEL_DIR = os.environ.get("RL_MODEL_DIR", "models")
RL_SAVE_INTERVAL = int(os.environ.get("RL_SAVE_INTERVAL", "10000"))
# MlpPolicy は CPU の方が効くことが多い。GPU 警告を消すなら "cpu"、GPU を使うなら "cuda" / "auto"
RL_DEVICE = os.environ.get("RL_DEVICE", "auto")
TOP_K_ENEMIES = 8
TOP_M_FOOD = 16

# --- CNN ハイブリッド観測 ---
RL_OBS_MODE = os.environ.get("RL_OBS_MODE", "hybrid")  # "vector" or "hybrid"
CNN_INPUT_SIZE = (84, 84)
CNN_FRAME_STACK = 4
CNN_BATCH_SIZE = 256
CNN_FEATURES_DIM = 256
