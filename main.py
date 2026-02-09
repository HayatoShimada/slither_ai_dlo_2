"""
Slither.io RT-DLO パイプラインの第一段階:
  画面キャプチャ → 自機ヘビの骨格抽出 → 可視化

使い方:
  1. Slither.io をブラウザで開き、ゲームを開始する
  2. config.py で SNAKE_HSV_* を自機の色に合わせる
  3. python main.py で実行。'q' で終了。

モード:
  python main.py           -- 骨格可視化 (デフォルト)
  python main.py debug     -- HSV 調整デバッグ
  python main.py bot       -- Docker コンテナ内 自動運転 + 強化学習
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from config import CAPTURE_MONITOR, SNAKE_HSV_LOWER, SNAKE_HSV_UPPER, AUTO_DETECT_COLOR
from capture import capture_screen
from snake_skeleton import (
    extract_snake_skeleton,
    mask_snake_bgr,
    largest_connected_component,
    mask_to_skeleton_binary,
    skeleton_points_for_rt_dlo,
)
from config import MIN_SNAKE_AREA
from color_detect import auto_detect_snake_color


def draw_skeleton_on_frame(bgr: np.ndarray, points_yx: np.ndarray | None) -> np.ndarray:
    """
    骨格座標をフレーム上に描画する。頭を緑、尾を赤、中間を青で表示。
    """
    out = bgr.copy()
    if points_yx is None or len(points_yx) < 2:
        cv2.putText(
            out, "No snake detected - adjust HSV in config.py",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        return out

    pts = points_yx.astype(np.int32)
    for i in range(len(pts) - 1):
        color = (255, 0, 0)  # BGR blue
        if i == 0:
            color = (0, 255, 0)  # 頭: 緑
        elif i == len(pts) - 2:
            color = (0, 0, 255)  # 尾: 赤
        cv2.line(out, (pts[i, 1], pts[i, 0]), (pts[i + 1, 1], pts[i + 1, 0]), color, 2)
    cv2.circle(out, (pts[0, 1], pts[0, 0]), 5, (0, 255, 0), -1)
    cv2.circle(out, (pts[-1, 1], pts[-1, 0]), 5, (0, 0, 255), -1)
    cv2.putText(out, "head", (pts[0, 1] + 6, pts[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
    cv2.putText(out, "tail", (pts[-1, 1] + 6, pts[-1, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
    return out


def run_visualization():
    """メインループ: キャプチャ → 骨格抽出 → 表示。"""
    print("Slither.io 骨格可視化を開始します。")
    print("終了: 表示ウィンドウをアクティブにして 'q' を押すか、Ctrl+C")
    print()

    # 自機カラー自動検出
    if AUTO_DETECT_COLOR:
        hsv_lower, hsv_upper = auto_detect_snake_color(
            lambda: capture_screen(CAPTURE_MONITOR),
        )
    else:
        hsv_lower, hsv_upper = SNAKE_HSV_LOWER, SNAKE_HSV_UPPER

    while True:
        frame = capture_screen(CAPTURE_MONITOR)
        points_yx = extract_snake_skeleton(frame, hsv_lower=hsv_lower, hsv_upper=hsv_upper)

        # RT-DLO 用形式（x, y）も取得可能
        # rt_dlo_points = skeleton_points_for_rt_dlo(points_yx)

        vis = draw_skeleton_on_frame(frame, points_yx)
        cv2.imshow("Slither.io - Snake skeleton", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("終了しました。")


def run_debug_mask():
    """
    デバッグ用: マスク・骨格の中間画像を表示し、HSV 調整の参考にする。
    """
    print("マスク/骨格デバッグ表示。'q' で終了。")

    # 自機カラー自動検出
    if AUTO_DETECT_COLOR:
        hsv_lower, hsv_upper = auto_detect_snake_color(
            lambda: capture_screen(CAPTURE_MONITOR),
        )
    else:
        hsv_lower, hsv_upper = SNAKE_HSV_LOWER, SNAKE_HSV_UPPER

    while True:
        frame = capture_screen(CAPTURE_MONITOR)
        mask = mask_snake_bgr(frame, hsv_lower, hsv_upper)
        mask_cc = largest_connected_component(mask, MIN_SNAKE_AREA)
        skel = mask_to_skeleton_binary(mask_cc)
        skel_vis = (skel * 255).astype(np.uint8)
        skel_bgr = cv2.cvtColor(skel_vis, cv2.COLOR_GRAY2BGR)

        points_yx = extract_snake_skeleton(frame, hsv_lower=hsv_lower, hsv_upper=hsv_upper)
        vis = draw_skeleton_on_frame(frame, points_yx)

        top = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        bottom = np.hstack([cv2.cvtColor(mask_cc, cv2.COLOR_GRAY2BGR), skel_bgr])
        grid = np.vstack([top, bottom])
        # リサイズして画面に収める
        h, w = grid.shape[:2]
        scale = min(1920 / w, 1000 / h, 1.0)
        if scale < 1:
            grid = cv2.resize(grid, (int(w * scale), int(h * scale)))
        cv2.imshow("Debug: raw | mask | mask_cc | skeleton", grid)
        cv2.imshow("Skeleton overlay", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def run_bot():
    """
    Docker コンテナ内での自動運転モード。
    ブラウザ起動 → ゲーム開始 → PPO 学習ループを実行する。
    model.learn() により重み更新が自動で行われる。
    """
    import time

    from browser import create_driver, start_game
    from config import RL_OBS_MODE, CNN_FRAME_STACK
    from game_env import SlitherEnv
    from agent_rl import create_agent, load_model, save_model
    from dlo_tracker import DLOTracker
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    class EpisodeLogCallback(BaseCallback):
        """エピソード終了ごとにログを出力するコールバック。"""

        def __init__(self, save_interval: int = 10000, verbose: int = 0):
            super().__init__(verbose)
            self._episode_count = 0
            self._episode_reward = 0.0
            self._episode_steps = 0
            self._episode_start_time = time.time()
            self._save_interval = save_interval

        def _on_step(self) -> bool:
            self._episode_reward += self.locals["rewards"][0]
            self._episode_steps += 1

            # エピソード終了
            dones = self.locals.get("dones", self.locals.get("done"))
            if dones is not None and (dones[0] if hasattr(dones, '__len__') else dones):
                self._episode_count += 1
                # VecFrameStack → DummyVecEnv → env を貫通して base env を取得
                venv = self.training_env
                while hasattr(venv, "venv"):
                    venv = venv.venv
                env = venv.envs[0]
                base_env = getattr(env, "unwrapped", env)
                score = 0
                try:
                    from browser import get_score
                    score = get_score(env.driver)
                except Exception:
                    pass
                # 認識ベースの成長指標
                length_px = None
                area_px = None
                try:
                    dlo = getattr(base_env, "last_dlo_state", None)
                    if dlo and getattr(dlo, "self_dlo", None) is not None:
                        length_px = dlo.self_dlo.length
                    mask = getattr(base_env, "_last_self_mask", None)
                    if mask is not None:
                        area_px = float(np.sum(mask > 0))
                except Exception:
                    pass
                survival_sec = time.time() - self._episode_start_time
                length_str = f"  length={length_px:.0f}px" if length_px is not None else ""
                area_str = f"  area={area_px:.0f}px²" if area_px is not None else ""
                # 報酬内訳
                rb_str = ""
                try:
                    rb = base_env.reward_breakdown
                    rb_str = (
                        f"  [surv={rb['survival']:+.1f} "
                        f"grow={rb['growth']:+.1f} "
                        f"food={rb.get('food', 0):+.1f} "
                        f"enemy={rb['enemy']:+.1f} "
                        f"coll={rb['collision']:+.1f} "
                        f"wall={rb['wall']:+.1f}]"
                    )
                except Exception:
                    pass
                print(
                    f"[Episode {self._episode_count}] "
                    f"alive={survival_sec:.1f}s  "
                    f"steps={self._episode_steps}  "
                    f"reward={self._episode_reward:+.1f}{rb_str}  "
                    f"score={score}{length_str}{area_str}  "
                    f"total_steps={self.num_timesteps}"
                )
                self._episode_reward = 0.0
                self._episode_steps = 0
                self._episode_start_time = time.time()

            # 定期保存
            if self.num_timesteps % self._save_interval == 0 and self.num_timesteps > 0:
                print(f"[Save] step={self.num_timesteps} -> models/")
                self.model.save("models/slither_ppo")

            return True

    print("=== Slither.io Bot Mode (DLO) ===")
    print(f"Observation mode: {RL_OBS_MODE}")
    print("Starting browser...")

    driver = create_driver()
    try:
        start_game(driver)

        # window.snake の全プロパティをダンプ（初回診断）
        time.sleep(3)
        from browser import dump_snake_properties
        dump_snake_properties(driver)

        # 自機カラー自動検出
        hsv_lower, hsv_upper = None, None
        if AUTO_DETECT_COLOR:
            hsv_lower, hsv_upper = auto_detect_snake_color(capture_screen)

        print("Initializing DLO tracker and environment...")
        tracker = DLOTracker()
        env = SlitherEnv(driver, tracker=tracker, hsv_lower=hsv_lower, hsv_upper=hsv_upper)

        # hybrid モード: VecFrameStack でフレームスタッキング
        if RL_OBS_MODE == "hybrid":
            print(f"Hybrid mode: wrapping with VecFrameStack(n_stack={CNN_FRAME_STACK})")
            venv = DummyVecEnv([lambda e=env: e])
            venv = VecFrameStack(venv, n_stack=CNN_FRAME_STACK, channels_order="last")
            train_env = venv
        else:
            train_env = env

        print("Loading or creating RL agent...")
        model = load_model(train_env)
        if model is None:
            print("No existing model found. Creating new agent...")
            model = create_agent(train_env)

        print("Starting PPO training (weights will be updated)...")
        callback = EpisodeLogCallback(save_interval=10000)
        model.learn(total_timesteps=10_000_000, callback=callback)

    except KeyboardInterrupt:
        print("\nStopping bot...")
    finally:
        save_model(model)
        driver.quit()
        print("Bot stopped.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "vis"
    if mode == "debug":
        run_debug_mask()
    elif mode == "bot":
        run_bot()
    elif mode == "diag":
        from diagnose_detection import run_diagnosis
        image_path = sys.argv[2] if len(sys.argv) > 2 else None
        run_diagnosis(image_path)
    else:
        run_visualization()
