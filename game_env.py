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
)
from capture import capture_screen
from snake_skeleton import extract_snake_skeleton, mask_snake_bgr, largest_connected_component
from config import SNAKE_HSV_LOWER, SNAKE_HSV_UPPER, MIN_SNAKE_AREA
from enemy_detection import detect_all_objects, dlo_state_to_enemy_info, EnemyInfo
from dlo_instance import DLOInstance, DLOState
from dlo_tracker import DLOTracker
from mouse_control import move_to_angle, boost
from browser import is_game_over, restart_game, get_score
from color_detect import auto_detect_snake_color


class SlitherEnv(gym.Env):
    """
    slither.io 用 Gymnasium 環境（DLO 統合版）。

    観測空間 (252 次元):
        自機骨格:       80 × 2 = 160
        自機メタ:       heading + length + velocity(2) = 4
        敵 DLO top-K=8: 各 (center_dx, center_dy, heading, length, vel_dx, vel_dy) = 6 × 8 = 48
        最寄り餌:       16 × 2 = 32
        予測衝突リスク: top-K 敵ごとの最短距離予測 = 8

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

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([360.0, 1.0], dtype=np.float32),
        )

        self._prev_score = 0
        self._step_count = 0
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

        if is_game_over(self.driver):
            restart_game(self.driver)
            time.sleep(2)

        # 自機カラー再検出（リスタート後は色が変わる可能性あり）
        if AUTO_DETECT_COLOR:
            self._hsv_lower, self._hsv_upper = auto_detect_snake_color(capture_screen)

        self._prev_score = get_score(self.driver)
        self._step_count = 0
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

        # 終了判定
        terminated = is_game_over(self.driver)
        truncated = False

        if terminated:
            reward += -10.0
            boost(False)  # ブースト解除

        self._step_count += 1

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> tuple[np.ndarray, dict]:
        """
        現在のゲーム状態から DLO ベースの観測ベクトルを構築する。

        Returns
        -------
        tuple[np.ndarray, dict]
            (正規化された 252 次元観測ベクトル, info dict)
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

        obs = np.concatenate([skel_norm, self_meta, enemy_dlo_vec, food_vec, collision_risk])
        obs = np.clip(obs, -1.0, 1.0)

        info = {
            "frame": frame,
            "self_mask": self_mask,
            "skeleton": skeleton_yx,
            "enemies": enemies,
            "dlo_state": dlo_state,
            "predicted_dlos": predicted_dlos,
            "score": get_score(self.driver),
        }

        return obs, info

    def _compute_reward(self, info: dict) -> float:
        """
        報酬を計算する。DLO 予測を活用した衝突リスクペナルティを含む。

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

        # 生存報酬
        reward += 0.1

        # 餌獲得報酬（スコア増加）
        current_score = info.get("score", 0)
        if current_score > self._prev_score:
            reward += 1.0 * (current_score - self._prev_score)
        self._prev_score = current_score

        # 敵近接ペナルティ（現在位置ベース）
        enemies = info.get("enemies")
        if enemies and len(enemies.enemy_centers) > 0:
            cx, cy = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
            dists = np.sqrt(
                (enemies.enemy_centers[:, 0] - cx) ** 2
                + (enemies.enemy_centers[:, 1] - cy) ** 2
            )
            min_dist = np.min(dists)
            if min_dist < 100:
                reward -= (100 - min_dist) / 100.0  # 近いほどペナルティ大

        # 予測衝突リスクペナルティ（DLO 予測ベース）
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
                reward -= (80 - min_pred_dist) / 80.0 * 0.5  # 予測ベースは控えめ

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
