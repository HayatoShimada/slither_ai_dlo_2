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
    REWARD_GROWTH_SCORE_SCALE,
    REWARD_GROWTH_SCORE_CAP,
    REWARD_GROWTH_TOTAL_CAP,
    REWARD_KILL,
    KILL_DETECT_RADIUS,
    REWARD_IDLE_GRACE_STEPS,
    REWARD_IDLE_PENALTY,
    REWARD_ENEMY_DIST_THRESH,
    REWARD_ENEMY_MAX_PENALTY,
    REWARD_WALL_THRESH,
    REWARD_WALL_MAX,
    REWARD_DEATH,
    REWARD_DEBUG_LOG,
)
from capture import capture_screen
from snake_skeleton import (
    mask_snake_bgr,
    largest_connected_component,
    mask_to_skeleton_binary,
    skeleton_to_ordered_points,
)
from config import SNAKE_HSV_LOWER, SNAKE_HSV_UPPER, MIN_SNAKE_AREA
from enemy_detection import detect_all_objects, dlo_state_to_enemy_info, EnemyInfo
from dlo_instance import DLOInstance, DLOState
from dlo_tracker import DLOTracker
from mouse_control import move_to_angle, boost, set_driver as _set_mouse_driver
from browser import restart_game, is_game_over, get_game_state, get_js_entities
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


def _count_wall_pixels(region: np.ndarray) -> int:
    """
    端領域の赤マスクから壁帯状の連結成分だけのピクセル数を返す。

    壁の赤帯は弧を描いて長く伸びる。赤い敵ヘビや餌はコンパクトな塊。
    連結成分の bounding box の幅・高さのうち大きい方が領域サイズの
    20% 以上であれば壁として採用する。弧は斜めに走る場合もあるため、
    方向を限定せず max(幅, 高さ) で判定する。

    Parameters
    ----------
    region : np.ndarray
        端帯の赤マスク (uint8 0/255)。

    Returns
    -------
    int
        壁と判定された赤ピクセル数。
    """
    if np.count_nonzero(region) == 0:
        return 0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        region, connectivity=8,
    )
    # 長さ閾値: 領域の長辺の 20%
    edge_length = max(region.shape[0], region.shape[1])
    min_span = edge_length * 0.20

    wall_pixels = 0
    for i in range(1, n_labels):  # 0 は背景
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if max(bw, bh) >= min_span:
            wall_pixels += stats[i, cv2.CC_STAT_AREA]
    return wall_pixels


def detect_boundary_proximity(
    frame: np.ndarray,
    *,
    hsv_img: np.ndarray | None = None,
) -> float:
    """
    画面内の赤い境界線を検出し、壁への近さを 0.0~1.0 で返す。

    slither.io のマップ境界は赤い帯として画面に表示される。
    画面の上下左右の端に赤いピクセルがどれだけあるかで判定する。
    形状フィルタ: 連結成分の bounding box の幅・高さのうち大きい方が
    領域サイズの 20% 以上のものだけを壁として採用し、赤い敵ヘビ・餌の
    誤検出を排除する。弧形状の壁にも対応。

    Parameters
    ----------
    hsv_img : np.ndarray | None
        事前計算済み HSV 画像。None の場合は内部で変換する。

    Returns
    -------
    float
        0.0 = 境界なし（安全）, 1.0 = 画面の大部分が赤（危険）
    """
    hsv = hsv_img if hsv_img is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    # 赤色マスク（H が 0-10 または 170-180 の範囲）
    mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = mask1 | mask2

    # 画面の端 (上下左右の 15% 帯) の赤ピクセル比率を計算
    edge = int(min(h, w) * 0.15)
    regions = [
        red_mask[:edge, :],       # 上端
        red_mask[h - edge:, :],   # 下端
        red_mask[:, :edge],       # 左端
        red_mask[:, w - edge:],   # 右端
    ]
    total_pixels = 0
    wall_pixels = 0
    for region in regions:
        total_pixels += region.size
        wall_pixels += _count_wall_pixels(region)

    if total_pixels == 0:
        return 0.0
    ratio = wall_pixels / total_pixels
    # 赤が 1% 以上あれば壁が見えている。10% で最大危険度
    return min(ratio / 0.10, 1.0)


class SlitherEnv(gym.Env):
    """
    slither.io 用 Gymnasium 環境（DLO 統合版）。

    観測空間:
      vector モード (254 次元):
        自機骨格:       80 × 2 = 160
        自機メタ:       heading + length + velocity(2) = 4
        敵 DLO top-K=8: 各 (center_dx, center_dy, heading, length, vel_dx, vel_dy) = 6 × 8 = 48
        最寄り餌:       16 × 2 = 32
        予測衝突リスク: top-K 敵ごとの最短距離予測 = 8
        マップ位置:     JS 経由のマップ中心からの正規化座標 (dx, dy) = 2

      hybrid モード (Dict):
        image:    (84, 84, 1) uint8 グレースケール（VecFrameStack で 4 枚積み重ね）
        metadata: (62,) float32 = self_meta(4) + enemy_dlo(48) + collision_risk(8) + map_pos(2)

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
    MAP_POS_DIM = 2                             # マップ位置 (dx, dy)
    OBS_DIM = SKELETON_DIM + SELF_META_DIM + ENEMY_DLO_DIM + FOOD_DIM + COLLISION_RISK_DIM + MAP_POS_DIM  # 254
    # hybrid モードのメタデータ次元: self_meta + enemy_dlo + collision_risk + map_pos
    METADATA_DIM = SELF_META_DIM + ENEMY_DLO_DIM + COLLISION_RISK_DIM + MAP_POS_DIM  # 62

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
        _set_mouse_driver(driver)
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

        # 行動空間を [-1, 1] に正規化（PPO の初期方策 N(0,1) と相性が良い）
        # action[0]: -1~+1 → 0~360° (角度)
        # action[1]: -1~+1 → 0~1 (ブースト, >0 で ON)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self._prev_score = 0
        self._prev_length_px = 0.0
        self._prev_area_px = 0.0
        self._step_count = 0
        self._steps_since_growth = 0
        self._prev_tracked_enemy_ids: dict[int, DLOInstance] = {}
        # 報酬コンポーネント累計（エピソードログ用）
        self._reward_growth = 0.0
        self._reward_kill = 0.0
        self._reward_idle = 0.0
        self._reward_enemy_danger = 0.0
        self._reward_wall_penalty = 0.0
        self._prev_frame_small: np.ndarray | None = None
        self._stale_frame_count = 0
        self._cached_js_entities: dict = {}
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
            "growth": self._reward_growth,
            "kill": self._reward_kill,
            "idle": self._reward_idle,
            "enemy_danger": self._reward_enemy_danger,
            "wall": self._reward_wall_penalty,
        }

        # ゲームオーバー判定（画像 + JS フォールバック）
        frame = self._capture_browser()
        if detect_game_over(frame) or is_game_over(self.driver):
            restart_game(self.driver)
            time.sleep(0.5)
        self._empty_mask_count = 0

        # 自機カラー再検出（毎エピソード。ゲームごとに色が変わるため必須）
        if AUTO_DETECT_COLOR:
            time.sleep(0.5)  # ゲーム描画が安定するまで待機
            self._hsv_lower, self._hsv_upper = auto_detect_snake_color(self._capture_browser)

        self._prev_score = 0
        self._prev_length_px = 0.0
        self._prev_area_px = 0.0
        self._step_count = 0
        self._steps_since_growth = 0
        self._prev_tracked_enemy_ids = {}
        self._reward_growth = 0.0
        self._reward_kill = 0.0
        self._reward_idle = 0.0
        self._reward_enemy_danger = 0.0
        self._reward_wall_penalty = 0.0
        # 画面フリーズ検出をリセット
        self._prev_frame_small = None
        self._stale_frame_count = 0
        self._cached_js_entities = {}
        # 追跡器をリセット（新エピソードでは ID を引き継がない）
        self._tracker = DLOTracker()

        # JS 動作検証ログ（エピソード開始時にマップ座標が取れるか確認）
        js_state = get_game_state(self.driver)
        br = js_state["boundary_ratio"]
        js_dbg = js_state.get("_debug", "")
        hint = " (JS位置なし→画面の赤で壁ペナルティ)" if br < 0 else ""
        print(
            f"[JS check] boundary_ratio={br:.3f} "
            f"map_dx={js_state.get('map_dx', 'N/A')} "
            f"map_dy={js_state.get('map_dy', 'N/A')} "
            f"score={js_state['score']} playing={js_state['playing']} "
            f"src={js_dbg}{hint}",
            flush=True,
        )

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
        # 正規化行動 [-1,1] → 実際の値に変換
        raw_angle = float(action[0])    # -1 ~ +1
        raw_boost = float(action[1])    # -1 ~ +1
        angle = (raw_angle + 1.0) * 180.0  # -1→0°, 0→180°, +1→360°
        boost_val = raw_boost > 0.0         # 0 より大きければブースト ON

        # マウス操作
        move_to_angle(angle, distance=200)
        boost(boost_val)

        # 操作指示ログ（5ステップごと）
        if self._step_count % 5 == 0:
            print(
                f"  [ACTION] step={self._step_count} raw=({raw_angle:+.2f},{raw_boost:+.2f}) "
                f"angle={angle:.1f}° boost={'ON' if boost_val else 'OFF'}",
                flush=True,
            )

        # ゲームの進行を待つ（短めに設定して学習スループット向上）
        time.sleep(0.02)

        # 観測取得
        obs, info = self._get_observation()

        # 報酬計算
        reward = self._compute_reward(info)

        # 終了判定（JS playing フラグ + フレーム差分による画面フリーズ検出）
        # _get_observation() 内の get_game_state() 結果を再利用（JS 呼び出し削減）
        js_game_over = not info.get("js_playing", True)

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

        # デバッグログ（5ステップごと）
        if self._step_count % 5 == 0 or terminated:
            js_br = info.get("js_boundary_ratio", -1)
            mdx = info.get("map_dx", 0)
            mdy = info.get("map_dy", 0)
            js_dbg = info.get("js_debug", "")
            n_enemy = len(info.get("dlo_state").enemy_dlos) if info.get("dlo_state") else 0
            n_food = len(info.get("dlo_state").food_positions) if info.get("dlo_state") else 0
            print(
                f"  [step {self._step_count}] score={info.get('score',0)} "
                f"js_br={js_br:.2f} map=({mdx:.2f},{mdy:.2f}) "
                f"enemy={n_enemy} food={n_food} src={js_dbg} "
                f"fdiff={frame_diff:.1f} js_over={js_game_over} "
                f"stale={self._stale_frame_count}"
                f"{'  >>> GAME OVER' if terminated else ''}",
                flush=True,
            )
        truncated = False

        if terminated:
            js_br = info.get("js_boundary_ratio", -1.0)
            reward += REWARD_DEATH
            cause = "wall" if js_br > 0.85 else "collision"
            print(
                f"  [DEATH:{cause}] step={self._step_count} js_br={js_br:.3f} "
                f"penalty={REWARD_DEATH}",
                flush=True,
            )
            boost(False)  # ブースト解除

        self._step_count += 1

        return obs, reward, terminated, truncated, info

    def _capture_browser(self) -> np.ndarray:
        """Selenium 経由でブラウザのみをキャプチャする（モニタウィンドウを含まない）。"""
        try:
            png = self.driver.get_screenshot_as_png()
            arr = np.frombuffer(png, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        except Exception:
            pass
        # フォールバック: mss（Selenium が失敗した場合）
        return capture_screen()

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

        vector モード: 正規化された 254 次元ベクトル。
        hybrid モード: {"image": (84,84,1) uint8, "metadata": (62,) float32}。

        Returns
        -------
        tuple[np.ndarray | dict, dict]
            (観測, info dict)
        """
        frame = self._capture_browser()
        self._last_frame = frame

        # HSV 変換を1回だけ実行（全検出処理で共有）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 自機マスク（HSV 再利用、1回だけ計算）
        mask = mask_snake_bgr(frame, self._hsv_lower, self._hsv_upper, hsv_img=hsv)
        self_mask = largest_connected_component(mask, MIN_SNAKE_AREA)
        self._last_self_mask = self_mask

        # 自機骨格抽出（マスクから直接。HSV変換・マスク計算の重複を排除）
        if np.sum(self_mask) > 0:
            skel_bin = mask_to_skeleton_binary(self_mask)
            skeleton_yx = skeleton_to_ordered_points(skel_bin, SKELETON_SAMPLE_POINTS)
        else:
            skeleton_yx = None
        self._last_skeleton = skeleton_yx

        # DLO ベースの全オブジェクト検出（HSV 再利用）
        dlo_state = detect_all_objects(frame, self_mask, skeleton_yx, hsv_img=hsv)

        # JS injection による敵・食物検出（ビジョン検出を補完）
        js_ent = self._cached_js_entities
        if self._step_count % 3 == 0:
            js_ent = get_js_entities(self.driver)
            self._cached_js_entities = js_ent
        if js_ent and js_ent.get("ok"):
            dlo_state = self._merge_js_entities(dlo_state, js_ent)

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

        # 6. JS 経由のゲーム状態（頻度削減: 3ステップに1回 or 自機消失時）
        if self._step_count % 3 == 0 or np.sum(self_mask) == 0:
            js_state = get_game_state(self.driver)
            self._cached_js_state = js_state
        else:
            js_state = getattr(self, "_cached_js_state", None)
            if js_state is None:
                js_state = get_game_state(self.driver)
                self._cached_js_state = js_state
        js_playing = js_state.get("playing", False)

        # 自機の認識ベース指標（ゲームオーバーフレームは誤検出を防ぐため除外）
        length_px = dlo_state.self_dlo.length if dlo_state.self_dlo is not None else 0.0
        area_px = (float(np.sum(self_mask > 0)) if self_mask is not None else 0.0) if js_playing else 0.0

        # 壁検出（JS ワールド座標ベース。ピクセルベースは誤認が多いため廃止）
        js_boundary_ratio = js_state["boundary_ratio"]
        map_dx = js_state.get("map_dx", 0.0)
        map_dy = js_state.get("map_dy", 0.0)
        map_pos_vec = np.array(
            [np.clip(map_dx, -1.0, 1.0), np.clip(map_dy, -1.0, 1.0)],
            dtype=np.float32,
        )

        info = {
            "frame": frame,
            "self_mask": self_mask,
            "skeleton": skeleton_yx,
            "enemies": enemies,
            "dlo_state": dlo_state,
            "predicted_dlos": predicted_dlos,
            "score": js_state["score"],
            "length_px": length_px,
            "area_px": area_px,
            "js_boundary_ratio": js_boundary_ratio,
            "map_dx": map_dx,
            "map_dy": map_dy,
            "js_playing": js_state.get("playing", False),
            "js_debug": js_state.get("_debug", ""),
        }

        if self._obs_mode == "hybrid":
            # CNN 用画像 + メタデータ (骨格・食物は画像から学習するため除外)
            image = self._preprocess_frame(frame)
            metadata = np.concatenate([self_meta, enemy_dlo_vec, collision_risk, map_pos_vec])
            metadata = np.clip(metadata, -1.0, 1.0)
            obs = {"image": image, "metadata": metadata}
        else:
            obs = np.concatenate([skel_norm, self_meta, enemy_dlo_vec, food_vec, collision_risk, map_pos_vec])
            obs = np.clip(obs, -1.0, 1.0)

        return obs, info

    def _merge_js_entities(
        self, dlo_state: DLOState, js_ent: dict,
    ) -> DLOState:
        """
        JS injection で取得した敵・食物データをビジョン検出結果に統合する。

        JS の敵座標はビジョン検出で見逃した敵を補完する。
        ビジョンで既に検出済みの敵（中心距離が近い）は重複追加しない。
        食物はJS検出を優先し、ビジョン検出をフォールバックとする。

        Parameters
        ----------
        dlo_state : DLOState
            ビジョンベース検出結果。
        js_ent : dict
            get_js_entities() の戻り値。

        Returns
        -------
        DLOState
            JS データで補完された DLOState。
        """
        from dlo_instance import compute_heading, compute_length, compute_center_xy

        # --- 食物: JS 検出が利用可能ならそちらを優先 ---
        js_foods = js_ent.get("foods", [])
        if js_foods:
            food_positions = np.array(js_foods, dtype=np.int32)
        else:
            food_positions = dlo_state.food_positions

        # --- 敵: JS 検出をビジョン検出に追加（重複排除） ---
        js_enemies = js_ent.get("enemies", [])
        merged_enemies = list(dlo_state.enemy_dlos)  # ビジョン検出を保持
        dup_thresh = 60.0  # この距離以内なら同一敵と判定 (px)

        for je in js_enemies:
            sx = je.get("sx", 0)
            sy = je.get("sy", 0)

            # ビジョン検出済みの敵と重複チェック
            is_duplicate = False
            for existing in merged_enemies:
                dx = existing.center[0] - sx
                dy = existing.center[1] - sy
                if dx * dx + dy * dy < dup_thresh * dup_thresh:
                    is_duplicate = True
                    break

            if not is_duplicate:
                # JS のみで検出された敵 → 1点骨格の DLOInstance を生成
                skel_yx = np.array([[sy, sx]], dtype=np.int32)
                center = np.array([sx, sy], dtype=np.float64)
                heading = float(je.get("heading", 0.0))
                # sct (segment count) を長さの近似に使用
                length_estimate = float(je.get("length", 0)) * 3.0

                merged_enemies.append(DLOInstance(
                    instance_id=-1,
                    skeleton_yx=skel_yx,
                    heading=heading,
                    length=length_estimate,
                    center=center,
                    contour=None,
                    is_self=False,
                ))

        return DLOState(
            self_dlo=dlo_state.self_dlo,
            enemy_dlos=merged_enemies,
            food_positions=food_positions,
        )

    def _detect_kills(self, dlo_state: DLOState | None) -> int:
        """
        敵キルを検出する。DLO トラッカーの ID 消失 + 自機ヘッド近接で判定。

        Parameters
        ----------
        dlo_state : DLOState or None
            現在フレームの DLO 状態。

        Returns
        -------
        int
            検出されたキル数。
        """
        if dlo_state is None:
            return 0

        current_ids = {e.instance_id for e in dlo_state.enemy_dlos}
        kills = 0
        head_yx = (
            self._last_skeleton[0]
            if self._last_skeleton is not None and len(self._last_skeleton) > 0
            else None
        )

        for eid, prev_enemy in self._prev_tracked_enemy_ids.items():
            if eid not in current_ids and head_yx is not None:
                # 消滅した敵が自機ヘッド近くにいたかチェック
                # center は (x, y) なので (y, x) に変換
                enemy_center_yx = np.array([prev_enemy.center[1], prev_enemy.center[0]])
                dist = float(np.linalg.norm(head_yx - enemy_center_yx))
                if dist < KILL_DETECT_RADIUS:
                    kills += 1
                    print(
                        f"  [KILL] step={self._step_count} enemy_id={eid} "
                        f"dist={dist:.0f}px",
                        flush=True,
                    )

        self._prev_tracked_enemy_ids = {
            e.instance_id: e for e in dlo_state.enemy_dlos
        }
        return kills

    def _compute_reward(self, info: dict) -> float:
        """
        報酬を計算する。シンプル5コンポーネント設計。

        報酬構成:
          +growth       成長報酬（JSスコア増加）
          +kill         キル報酬（敵消滅検出）
          -idle         怠慢ペナルティ（長時間成長なし）
          -enemy_danger 敵危険ペナルティ（近接敵への線形ペナルティ）
          -wall         壁接近ペナルティ（シンプル線形）
          REWARD_DEATH  死亡ペナルティ（step() 側で加算）

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

        # --- 成長報酬（JSスコア増加） ---
        r_growth = 0.0
        is_alive = info.get("js_playing", True)
        current_score = info.get("score", 0) if is_alive else 0
        # スコアが大幅減少（>50%）は JS 読み取りエラーとして無視
        if current_score > 0 and self._prev_score > 0 and current_score < self._prev_score * 0.5:
            current_score = self._prev_score
        if current_score > 0 and current_score > self._prev_score:
            r_growth = min(
                REWARD_GROWTH_SCORE_SCALE * (current_score - self._prev_score),
                REWARD_GROWTH_SCORE_CAP,
            )
            self._steps_since_growth = 0  # 成長したのでリセット
        if current_score > 0:
            self._prev_score = current_score
        r_growth = min(r_growth, REWARD_GROWTH_TOTAL_CAP)
        reward += r_growth
        self._reward_growth += r_growth

        # --- キル報酬（敵消滅検出） ---
        r_kill = 0.0
        dlo_state = info.get("dlo_state")
        kills = self._detect_kills(dlo_state)
        if kills > 0:
            r_kill = REWARD_KILL * kills
        reward += r_kill
        self._reward_kill += r_kill

        # --- 怠慢ペナルティ ---
        r_idle = 0.0
        self._steps_since_growth += 1
        if self._steps_since_growth > REWARD_IDLE_GRACE_STEPS:
            r_idle = -REWARD_IDLE_PENALTY
        reward += r_idle
        self._reward_idle += r_idle

        # --- 敵危険ペナルティ（敵近接 + 予測骨格を統合） ---
        r_enemy_danger = 0.0
        min_enemy_dist = float("inf")

        # 予測骨格を優先（前方予測で早めに回避）
        skeleton = info.get("skeleton")
        predicted_dlos = info.get("predicted_dlos", [])
        if skeleton is not None and len(skeleton) > 0 and predicted_dlos:
            head_yx = skeleton[0]
            for pred in predicted_dlos:
                if pred.skeleton_yx is not None and len(pred.skeleton_yx) > 0:
                    d = np.sqrt(
                        (pred.skeleton_yx[:, 0] - head_yx[0]) ** 2
                        + (pred.skeleton_yx[:, 1] - head_yx[1]) ** 2
                    )
                    min_enemy_dist = min(min_enemy_dist, float(np.min(d)))

        # 予測が無い場合は現在の敵中心位置でフォールバック
        if min_enemy_dist == float("inf"):
            enemies = info.get("enemies")
            if enemies and len(enemies.enemy_centers) > 0:
                cx, cy = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
                dists = np.sqrt(
                    (enemies.enemy_centers[:, 0] - cx) ** 2
                    + (enemies.enemy_centers[:, 1] - cy) ** 2
                )
                min_enemy_dist = float(np.min(dists))

        if min_enemy_dist < REWARD_ENEMY_DIST_THRESH:
            r_enemy_danger = (
                -(REWARD_ENEMY_DIST_THRESH - min_enemy_dist)
                / REWARD_ENEMY_DIST_THRESH
                * REWARD_ENEMY_MAX_PENALTY
            )
        reward += r_enemy_danger
        self._reward_enemy_danger += r_enemy_danger

        # --- 壁接近ペナルティ（シンプル線形） ---
        r_wall = 0.0
        js_boundary_ratio = info.get("js_boundary_ratio", -1.0)
        if js_boundary_ratio >= 0.0 and js_boundary_ratio > REWARD_WALL_THRESH:
            r_wall = (
                -(js_boundary_ratio - REWARD_WALL_THRESH)
                / (1.0 - REWARD_WALL_THRESH)
                * REWARD_WALL_MAX
            )
        reward += r_wall
        self._reward_wall_penalty += r_wall

        # デバッグログ
        if REWARD_DEBUG_LOG and self._step_count % 10 == 0:
            print(
                f"[Reward] step={self._step_count} "
                f"total={reward:+.3f} "
                f"grow={r_growth:+.3f} "
                f"kill={r_kill:+.3f} "
                f"idle={r_idle:+.3f} "
                f"enemy={r_enemy_danger:+.3f} "
                f"wall={r_wall:+.3f} "
                f"score={info.get('score', 0)} "
                f"js_br={js_boundary_ratio:.3f}"
            )

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
            "growth": self._reward_growth,
            "kill": self._reward_kill,
            "idle": self._reward_idle,
            "enemy_danger": self._reward_enemy_danger,
            "wall": self._reward_wall_penalty,
        })
