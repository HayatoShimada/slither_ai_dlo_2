"""
Gymnasium 環境ラッパー。
slither.io のゲーム状態を観測空間・行動空間として定義し、
強化学習エージェントが学習できるインターフェースを提供する。

DLO 統合: 全ヘビを DLO インスタンスとして扱い、敵の骨格メタ情報・速度・
予測衝突リスクを観測空間に含める。
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    SKELETON_SAMPLE_POINTS,
    TOP_K_ENEMIES,
    TOP_M_FOOD,
    AUTO_DETECT_COLOR,
    RL_OBS_MODE,
    CNN_INPUT_SIZE,
)
from capture import capture_screen
from snake_skeleton import extract_snake_skeleton, mask_snake_bgr, largest_connected_component
from config import SNAKE_HSV_LOWER, SNAKE_HSV_UPPER, MIN_SNAKE_AREA
from enemy_detection import detect_all_objects, dlo_state_to_enemy_info, EnemyInfo
from dlo_instance import DLOInstance, DLOState
from dlo_tracker import DLOTracker
from mouse_control import move_to_angle, boost
from browser import restart_game
from color_detect import auto_detect_snake_color

import cv2


def detect_game_over(frame: np.ndarray) -> bool:
    """
    画像からゲームオーバーを検出する。

    ゲームオーバー時は画面が暗転し、中央に「Play Again」等のテキストが表示される。
    画面中央部の平均輝度が極端に低い + 自機マスクが消失 で判定。

    Returns
    -------
    bool
        ゲームオーバーなら True。
    """
    h, w = frame.shape[:2]
    # 画面中央 40% の領域
    y1, y2 = int(h * 0.3), int(h * 0.7)
    x1, x2 = int(w * 0.3), int(w * 0.7)
    center_region = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    # ゲームオーバー画面は暗い背景 + 白テキスト
    # 通常のゲーム画面は平均 40~100+、ゲームオーバーは 20~40
    return mean_brightness < 30


def detect_boundary_proximity(frame: np.ndarray) -> float:
    """
    画面内の赤い境界線を検出し、壁への近さを 0.0~1.0 で返す。

    slither.io のマップ境界は赤い帯として画面に表示される。
    画面の上下左右の端に赤いピクセルがどれだけあるかで判定する。

    Returns
    -------
    float
        0.0 = 境界なし（安全）, 1.0 = 画面の大部分が赤（危険）
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    # 赤色マスク（H が 0-10 または 170-180 の範囲）
    mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = mask1 | mask2

    # 画面の端 (上下左右の 15% 帯) の赤ピクセル比率を計算
    edge = int(min(h, w) * 0.15)
    regions = [
        red_mask[:edge, :],          # 上端
        red_mask[h - edge:, :],      # 下端
        red_mask[:, :edge],          # 左端
        red_mask[:, w - edge:],      # 右端
    ]
    total_pixels = 0
    red_pixels = 0
    for region in regions:
        total_pixels += region.size
        red_pixels += np.count_nonzero(region)

    if total_pixels == 0:
        return 0.0
    ratio = red_pixels / total_pixels
    # 赤が 1% 以上あれば壁が見えている。10% で最大危険度
    return min(ratio / 0.10, 1.0)


class SlitherEnv(gym.Env):
    """
    slither.io 用 Gymnasium 環境（DLO 統合版）。

    観測空間:
      vector モード (252 次元):
        自機骨格:       80 × 2 = 160
        自機メタ:       heading + length + velocity(2) = 4
        敵 DLO top-K=8: 各 (center_dx, center_dy, heading, length, vel_dx, vel_dy) = 6 × 8 = 48
        最寄り餌:       16 × 2 = 32
        予測衝突リスク: top-K 敵ごとの最短距離予測 = 8

      hybrid モード (Dict):
        image:    (84, 84, 1) uint8 グレースケール（VecFrameStack で 4 枚積み重ね）
        metadata: (60,) float32 = self_meta(4) + enemy_dlo(48) + collision_risk(8)

    行動空間:
        Box(2): [angle (0~360), boost (0~1)]
    """

    metadata = {"render_modes": ["human"]}

    # 観測空間の次元数
    SKELETON_DIM = SKELETON_SAMPLE_POINTS * 2  # 80 * 2 = 160
    SELF_META_DIM = 4                           # heading, length, vel_x, vel_y
    ENEMY_DLO_DIM = TOP_K_ENEMIES * 6          # 8 * 6 = 48
    FOOD_DIM = TOP_M_FOOD * 2                  # 16 * 2 = 32
    COLLISION_RISK_DIM = TOP_K_ENEMIES          # 8
    OBS_DIM = SKELETON_DIM + SELF_META_DIM + ENEMY_DLO_DIM + FOOD_DIM + COLLISION_RISK_DIM  # 252
    # hybrid モードのメタデータ次元: self_meta + enemy_dlo + collision_risk
    METADATA_DIM = SELF_META_DIM + ENEMY_DLO_DIM + COLLISION_RISK_DIM  # 60

    def __init__(
        self,
        driver,
        tracker: DLOTracker | None = None,
        hsv_lower: tuple[int, int, int] | None = None,
        hsv_upper: tuple[int, int, int] | None = None,
    ):
        """
        Parameters
        ----------
        driver : webdriver.Chrome
            Selenium WebDriver インスタンス。
        tracker : DLOTracker | None
            DLO 追跡器。None の場合は内部で生成する。
        hsv_lower : tuple | None
            自機の HSV 下限。None の場合は config デフォルト。
        hsv_upper : tuple | None
            自機の HSV 上限。None の場合は config デフォルト。
        """
        super().__init__()

        self.driver = driver
        self._tracker = tracker if tracker is not None else DLOTracker()
        self._hsv_lower = hsv_lower or SNAKE_HSV_LOWER
        self._hsv_upper = hsv_upper or SNAKE_HSV_UPPER
        self._obs_mode = RL_OBS_MODE  # "vector" or "hybrid"

        if self._obs_mode == "hybrid":
            self.observation_space = spaces.Dict({
                "image": spaces.Box(
                    low=0, high=255,
                    shape=(CNN_INPUT_SIZE[0], CNN_INPUT_SIZE[1], 1),
                    dtype=np.uint8,
                ),
                "metadata": spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(self.METADATA_DIM,),
                    dtype=np.float32,
                ),
            })
        else:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32,
            )

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([360.0, 1.0], dtype=np.float32),
        )

        self._prev_score = 0
        self._prev_length_px = 0.0
        self._prev_area_px = 0.0
        self._prev_min_food_dist = 0.0
        self._step_count = 0
        # 報酬コンポーネント累計（エピソードログ用）
        self._reward_survival = 0.0
        self._reward_growth = 0.0
        self._reward_food_approach = 0.0
        self._reward_enemy_penalty = 0.0
        self._reward_collision_penalty = 0.0
        self._reward_wall_penalty = 0.0
        self._prev_frame_small: np.ndarray | None = None
        self._stale_frame_count = 0
        self._last_enemies: EnemyInfo | None = None
        self._last_self_mask: np.ndarray | None = None
        self._last_skeleton: np.ndarray | None = None
        self._last_frame: np.ndarray | None = None
        self._last_dlo_state: DLOState | None = None
        self._last_predicted_dlos: list[DLOInstance] = []

    def reset(self, seed=None, options=None):
        """
        環境をリセットする。ゲームオーバー後の再スタートを行う。

        Returns
        -------
        tuple[np.ndarray, dict]
            (初期観測, info)
        """
        super().reset(seed=seed)

        # リセット前に報酬内訳を保存（SB3 が auto-reset するため）
        self._saved_reward_breakdown = {
            "survival": self._reward_survival,
            "growth": self._reward_growth,
            "food": self._reward_food_approach,
            "enemy": self._reward_enemy_penalty,
            "collision": self._reward_collision_penalty,
            "wall": self._reward_wall_penalty,
        }

        # ゲームオーバー判定（画像 + JS フォールバック）
        from browser import is_game_over
        frame = capture_screen()
        if detect_game_over(frame) or is_game_over(self.driver):
            restart_game(self.driver)
            time.sleep(0.5)
        self._empty_mask_count = 0

        # 自機カラー再検出（5エピソードに1回。毎回やると遅い）
        if AUTO_DETECT_COLOR and self._step_count == 0:
            if not hasattr(self, '_reset_count'):
                self._reset_count = 0
            self._reset_count += 1
            if self._reset_count <= 1 or self._reset_count % 5 == 0:
                self._hsv_lower, self._hsv_upper = auto_detect_snake_color(capture_screen)

        self._prev_score = 0
        self._prev_length_px = 0.0
        self._prev_area_px = 0.0
        self._prev_min_food_dist = 0.0
        self._step_count = 0
        self._reward_survival = 0.0
        self._reward_growth = 0.0
        self._reward_food_approach = 0.0
        self._reward_enemy_penalty = 0.0
        self._reward_collision_penalty = 0.0
        self._reward_wall_penalty = 0.0
        # 画面フリーズ検出をリセット
        self._prev_frame_small = None
        self._stale_frame_count = 0
        # 追跡器をリセット（新エピソードでは ID を引き継がない）
        self._tracker = DLOTracker()

        obs, info = self._get_observation()
        return obs, info

    def step(self, action):
        """
        1ステップ実行する。

        Parameters
        ----------
        action : np.ndarray
            [angle (0~360), boost (0~1)]

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict]
            (観測, 報酬, terminated, truncated, info)
        """
        angle = float(action[0])
        boost_val = float(action[1]) > 0.5

        # マウス操作
        move_to_angle(angle, distance=200)
        boost(boost_val)

        # ゲームの進行を待つ
        time.sleep(0.05)

        # 観測取得
        obs, info = self._get_observation()

        # 報酬計算
        reward = self._compute_reward(info)

        # 終了判定（JS 主体 + フレーム差分による画面フリーズ検出）
        from browser import is_game_over
        js_game_over = is_game_over(self.driver)

        # フレーム全体の画素差分で画面が動いているか判定
        # 通常プレイ中は背景・餌・他ヘビで常に変化する。
        # ゲームオーバー画面は完全に静止する。
        frame = info.get("frame")
        frame_diff = 999.0
        if frame is not None and self._prev_frame_small is not None:
            small = cv2.resize(frame, (64, 36))
            frame_diff = float(np.mean(np.abs(
                small.astype(np.float32) - self._prev_frame_small.astype(np.float32)
            )))
            if frame_diff < 1.0:  # ほぼ同一フレーム
                self._stale_frame_count += 1
            else:
                self._stale_frame_count = 0
        if frame is not None:
            self._prev_frame_small = cv2.resize(frame, (64, 36))
        stale_screen = self._stale_frame_count >= 8  # 8フレーム連続で静止 = フリーズ

        terminated = js_game_over or stale_screen

        # デバッグログ（10ステップごと）
        if self._step_count % 10 == 0 or terminated:
            area = info.get("area_px", 0)
            wall = info.get("boundary_ratio", 0)
            print(
                f"  [step {self._step_count}] area={area:.0f} wall={wall:.2f} "
                f"fdiff={frame_diff:.1f} js_over={js_game_over} "
                f"stale={self._stale_frame_count}"
                f"{'  >>> GAME OVER' if terminated else ''}",
                flush=True,
            )
        truncated = False

        if terminated:
            boundary_ratio = info.get("boundary_ratio", 0.0)
            if boundary_ratio > 0.01:
                # 壁死: 生存報酬を全取り消し + 重い死亡ペナルティ
                reward += -self._reward_survival  # 生存報酬ゼロ化
                reward += -20.0
                self._reward_wall_penalty += -self._reward_survival - 20.0
                self._reward_survival = 0.0
            else:
                # 通常死（敵衝突等）
                reward += -10.0
            boost(False)  # ブースト解除

        self._step_count += 1

        return obs, reward, terminated, truncated, info

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        BGR フレームを CNN 入力用に前処理する。

        BGR → グレースケール → 84x84 リサイズ → (84, 84, 1) uint8

        Parameters
        ----------
        frame : np.ndarray
            BGR フレーム。

        Returns
        -------
        np.ndarray
            (84, 84, 1) uint8 グレースケール画像。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, CNN_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]  # (84, 84, 1)

    def _get_observation(self) -> tuple[np.ndarray | dict, dict]:
        """
        現在のゲーム状態から観測を構築する。

        vector モード: 正規化された 252 次元ベクトル。
        hybrid モード: {"image": (84,84,1) uint8, "metadata": (60,) float32}。

        Returns
        -------
        tuple[np.ndarray | dict, dict]
            (観測, info dict)
        """
        frame = capture_screen()
        self._last_frame = frame

        # 自機骨格抽出
        skeleton_yx = extract_snake_skeleton(
            frame, hsv_lower=self._hsv_lower, hsv_upper=self._hsv_upper,
        )
        self._last_skeleton = skeleton_yx

        # 自機マスク
        mask = mask_snake_bgr(frame, self._hsv_lower, self._hsv_upper)
        self_mask = largest_connected_component(mask, MIN_SNAKE_AREA)
        self._last_self_mask = self_mask

        # DLO ベースの全オブジェクト検出
        dlo_state = detect_all_objects(frame, self_mask, skeleton_yx)
        self._last_dlo_state = dlo_state

        # 後方互換用 EnemyInfo
        enemies = dlo_state_to_enemy_info(dlo_state)
        self._last_enemies = enemies

        # 敵 DLO を追跡器で更新（ID 割当 + 速度推定）
        tracked_enemies = self._tracker.update(dlo_state.enemy_dlos)
        dlo_state.enemy_dlos = tracked_enemies

        # 予測（1 ステップ先）
        predicted_dlos = self._tracker.predict_all()
        self._last_predicted_dlos = predicted_dlos

        # --- 観測ベクトル構築 ---

        # 1. 自機骨格 (160 dim)
        if skeleton_yx is not None and len(skeleton_yx) == SKELETON_SAMPLE_POINTS:
            skel_norm = skeleton_yx.astype(np.float32).flatten()
            skel_norm[0::2] /= SCREEN_HEIGHT  # y 正規化
            skel_norm[1::2] /= SCREEN_WIDTH   # x 正規化
            skel_norm = skel_norm * 2.0 - 1.0  # [-1, 1]
        else:
            skel_norm = np.zeros(self.SKELETON_DIM, dtype=np.float32)

        # 2. 自機メタ情報 (4 dim): heading, length, vel_x, vel_y
        self_meta = np.zeros(self.SELF_META_DIM, dtype=np.float32)
        if dlo_state.self_dlo is not None:
            self_meta[0] = dlo_state.self_dlo.heading / np.pi  # [-1, 1]
            self_meta[1] = min(dlo_state.self_dlo.length / 500.0, 1.0)
            self_meta[2] = np.clip(dlo_state.self_dlo.velocity[0] / 50.0, -1.0, 1.0)
            self_meta[3] = np.clip(dlo_state.self_dlo.velocity[1] / 50.0, -1.0, 1.0)

        # 3. 敵 DLO メタ情報 (48 dim): top-K 敵 × 6
        cx, cy = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        enemy_dlo_vec = np.zeros(self.ENEMY_DLO_DIM, dtype=np.float32)
        if tracked_enemies:
            # 画面中心からの距離でソート
            dists = [
                np.sqrt((e.center[0] - cx) ** 2 + (e.center[1] - cy) ** 2)
                for e in tracked_enemies
            ]
            sorted_idx = np.argsort(dists)[:TOP_K_ENEMIES]
            for i, idx in enumerate(sorted_idx):
                e = tracked_enemies[idx]
                base = i * 6
                enemy_dlo_vec[base + 0] = np.clip((e.center[0] - cx) / (SCREEN_WIDTH / 2), -1.0, 1.0)
                enemy_dlo_vec[base + 1] = np.clip((e.center[1] - cy) / (SCREEN_HEIGHT / 2), -1.0, 1.0)
                enemy_dlo_vec[base + 2] = e.heading / np.pi  # [-1, 1]
                enemy_dlo_vec[base + 3] = min(e.length / 500.0, 1.0)
                enemy_dlo_vec[base + 4] = np.clip(e.velocity[0] / 50.0, -1.0, 1.0)
                enemy_dlo_vec[base + 5] = np.clip(e.velocity[1] / 50.0, -1.0, 1.0)

        # 4. 最寄り餌の相対位置 (32 dim)
        food_vec = np.zeros(self.FOOD_DIM, dtype=np.float32)
        if len(dlo_state.food_positions) > 0:
            dists = np.sqrt(
                (dlo_state.food_positions[:, 0] - cx) ** 2
                + (dlo_state.food_positions[:, 1] - cy) ** 2
            )
            sorted_idx = np.argsort(dists)[:TOP_M_FOOD]
            for i, idx in enumerate(sorted_idx):
                food_vec[i * 2] = (dlo_state.food_positions[idx, 0] - cx) / (SCREEN_WIDTH / 2)
                food_vec[i * 2 + 1] = (dlo_state.food_positions[idx, 1] - cy) / (SCREEN_HEIGHT / 2)

        # 5. 予測衝突リスク (8 dim): top-K 敵の予測骨格と自機頭の最短距離
        collision_risk = np.zeros(self.COLLISION_RISK_DIM, dtype=np.float32)
        head_yx = skeleton_yx[0] if skeleton_yx is not None and len(skeleton_yx) > 0 else None
        if head_yx is not None and predicted_dlos:
            # 予測敵を画面中心からの距離でソート
            pred_dists_to_center = [
                np.sqrt((p.center[0] - cx) ** 2 + (p.center[1] - cy) ** 2)
                for p in predicted_dlos
            ]
            pred_sorted = np.argsort(pred_dists_to_center)[:TOP_K_ENEMIES]
            for i, idx in enumerate(pred_sorted):
                pred = predicted_dlos[idx]
                if pred.skeleton_yx is not None and len(pred.skeleton_yx) > 0:
                    # 自機頭と予測敵骨格の最短距離
                    d = np.sqrt(
                        (pred.skeleton_yx[:, 0] - head_yx[0]) ** 2
                        + (pred.skeleton_yx[:, 1] - head_yx[1]) ** 2
                    )
                    min_d = float(np.min(d))
                    # 正規化: 近いほど 1.0、遠いほど 0.0
                    collision_risk[i] = np.clip(1.0 - min_d / 300.0, 0.0, 1.0)

        # 自機の認識ベース指標
        length_px = dlo_state.self_dlo.length if dlo_state.self_dlo is not None else 0.0
        area_px = float(np.sum(self_mask > 0)) if self_mask is not None else 0.0

        # 壁検出（画面の赤い境界を視覚検出。JS 不要）
        boundary_proximity = detect_boundary_proximity(frame)

        info = {
            "frame": frame,
            "self_mask": self_mask,
            "skeleton": skeleton_yx,
            "enemies": enemies,
            "dlo_state": dlo_state,
            "predicted_dlos": predicted_dlos,
            "score": 0,
            "length_px": length_px,
            "area_px": area_px,
            "boundary_ratio": boundary_proximity,
        }

        if self._obs_mode == "hybrid":
            # CNN 用画像 + メタデータ (骨格・食物は画像から学習するため除外)
            image = self._preprocess_frame(frame)
            metadata = np.concatenate([self_meta, enemy_dlo_vec, collision_risk])
            metadata = np.clip(metadata, -1.0, 1.0)
            obs = {"image": image, "metadata": metadata}
        else:
            obs = np.concatenate([skel_norm, self_meta, enemy_dlo_vec, food_vec, collision_risk])
            obs = np.clip(obs, -1.0, 1.0)

        return obs, info

    def _compute_reward(self, info: dict) -> float:
        """
        報酬を計算する。

        報酬構成（成長・餌獲得重視設計）:
          +0.1         生存報酬（小さめ。生きてるだけでは大きな報酬にならない）
          +5.0 * delta 成長報酬（自機マスク面積増加。餌を食べて大きくなる＝高報酬）
          +food_approach 餌接近報酬（最寄り餌に近づくと報酬、離れるとペナルティ）
          -penalty     敵近接ペナルティ（100px 以内で線形）
          -penalty     予測衝突リスクペナルティ（80px 以内、DLO ベース）
          -penalty     壁接近ペナルティ（赤い境界検出）
          -10.0        死亡ペナルティ（step() 側で加算）

        Parameters
        ----------
        info : dict
            _get_observation() から得た情報。

        Returns
        -------
        float
            報酬値。
        """
        reward = 0.0

        # 生存報酬（控えめ: 生きてるだけでは稼げない、積極的に餌を取る必要がある）
        r_survival = 0.1
        reward += r_survival
        self._reward_survival += r_survival

        # 成長報酬（主要シグナル: 餌を食べて面積が増加 → 大きな報酬）
        r_growth = 0.0
        current_area = info.get("area_px", 0.0)
        if current_area > 0 and self._prev_area_px > 0:
            delta_area = current_area - self._prev_area_px
            if delta_area > 0:
                # 面積増加に強い報酬（餌獲得が最も重要な行動）
                r_growth = min(delta_area / 100.0, 5.0)
        self._prev_area_px = current_area

        # JS スコアも補助的に使用（取れる場合のみ）
        current_score = info.get("score", 0)
        if current_score > 0 and current_score > self._prev_score:
            r_growth += 2.0 * (current_score - self._prev_score)
        self._prev_score = current_score
        reward += r_growth
        self._reward_growth += r_growth

        # 餌接近報酬（最寄り餌に近づくことを報酬化 → 積極的移動を促進）
        r_food_approach = 0.0
        dlo_state = info.get("dlo_state")
        if dlo_state is not None and len(dlo_state.food_positions) > 0:
            cx, cy = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
            food_dists = np.sqrt(
                (dlo_state.food_positions[:, 0] - cx) ** 2
                + (dlo_state.food_positions[:, 1] - cy) ** 2
            )
            min_food_dist = float(np.min(food_dists))
            if self._prev_min_food_dist > 0:
                # 近づいたら正の報酬、離れたら負の報酬
                delta_dist = self._prev_min_food_dist - min_food_dist
                r_food_approach = np.clip(delta_dist / 100.0, -0.3, 0.5)
            self._prev_min_food_dist = min_food_dist
        reward += r_food_approach
        self._reward_food_approach += r_food_approach

        # 敵近接ペナルティ（現在位置ベース）
        r_enemy = 0.0
        enemies = info.get("enemies")
        if enemies and len(enemies.enemy_centers) > 0:
            cx, cy = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
            dists = np.sqrt(
                (enemies.enemy_centers[:, 0] - cx) ** 2
                + (enemies.enemy_centers[:, 1] - cy) ** 2
            )
            min_dist = np.min(dists)
            if min_dist < 100:
                r_enemy = -(100 - min_dist) / 100.0 * 0.5
        reward += r_enemy
        self._reward_enemy_penalty += r_enemy

        # 予測衝突リスクペナルティ（DLO 予測ベース）
        r_collision = 0.0
        skeleton = info.get("skeleton")
        predicted_dlos = info.get("predicted_dlos", [])
        if skeleton is not None and len(skeleton) > 0 and predicted_dlos:
            head_yx = skeleton[0]
            min_pred_dist = float("inf")
            for pred in predicted_dlos:
                if pred.skeleton_yx is not None and len(pred.skeleton_yx) > 0:
                    d = np.sqrt(
                        (pred.skeleton_yx[:, 0] - head_yx[0]) ** 2
                        + (pred.skeleton_yx[:, 1] - head_yx[1]) ** 2
                    )
                    min_pred_dist = min(min_pred_dist, float(np.min(d)))
            if min_pred_dist < 80:
                r_collision = -(80 - min_pred_dist) / 80.0 * 0.3
        reward += r_collision
        self._reward_collision_penalty += r_collision

        # 壁接近ペナルティ（画面の赤い境界を視覚検出）
        # boundary_ratio: 0.0=赤なし(安全), 1.0=赤が大量(危険)
        r_wall = 0.0
        boundary_ratio = info.get("boundary_ratio", 0.0)
        if boundary_ratio > 0.02:
            r_wall = -boundary_ratio * 10.0 - (boundary_ratio ** 2) * 5.0  # 最大 -15.0
        reward += r_wall
        self._reward_wall_penalty += r_wall

        return reward

    @property
    def last_frame(self) -> np.ndarray | None:
        """最後にキャプチャしたフレーム。"""
        return self._last_frame

    @property
    def last_self_mask(self) -> np.ndarray | None:
        """最後に検出した自機マスク。"""
        return self._last_self_mask

    @property
    def last_skeleton(self) -> np.ndarray | None:
        """最後に抽出した骨格。"""
        return self._last_skeleton

    @property
    def last_enemies(self) -> EnemyInfo | None:
        """最後に検出した敵・餌情報（後方互換）。"""
        return self._last_enemies

    @property
    def last_dlo_state(self) -> DLOState | None:
        """最後の DLO 状態。"""
        return self._last_dlo_state

    @property
    def last_predicted_dlos(self) -> list[DLOInstance]:
        """最後の予測 DLO リスト。"""
        return self._last_predicted_dlos

    @property
    def reward_breakdown(self) -> dict[str, float]:
        """直近エピソードの報酬コンポーネント累計。reset 後も前回値を返す。"""
        return self._last_reward_breakdown

    @property
    def _last_reward_breakdown(self) -> dict[str, float]:
        return getattr(self, '_saved_reward_breakdown', {
            "survival": self._reward_survival,
            "growth": self._reward_growth,
            "food": self._reward_food_approach,
            "enemy": self._reward_enemy_penalty,
            "collision": self._reward_collision_penalty,
            "wall": self._reward_wall_penalty,
        })
