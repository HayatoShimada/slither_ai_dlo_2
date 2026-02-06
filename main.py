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
    ブラウザ起動 → ゲーム開始 → RL 学習ループを実行する。
    """
    from browser import create_driver, start_game
    from game_env import SlitherEnv
    from monitor import update_monitor, RLInfo
    from agent_rl import create_agent, load_model, save_model
    from dlo_tracker import DLOTracker

    print("=== Slither.io Bot Mode (DLO) ===")
    print("Starting browser...")

    driver = create_driver()
    try:
        start_game(driver)

        # 自機カラー自動検出
        hsv_lower, hsv_upper = None, None
        if AUTO_DETECT_COLOR:
            hsv_lower, hsv_upper = auto_detect_snake_color(capture_screen)

        print("Initializing DLO tracker and environment...")
        tracker = DLOTracker()
        env = SlitherEnv(driver, tracker=tracker, hsv_lower=hsv_lower, hsv_upper=hsv_upper)

        print("Loading or creating RL agent...")
        model = load_model(env)
        if model is None:
            print("No existing model found. Creating new agent...")
            model = create_agent(env)

        rl_info = RLInfo()
        obs, info = env.reset()

        print("Starting training loop...")
        while True:
            # エージェントが行動を推論
            action, _states = model.predict(obs, deterministic=False)

            # 環境を1ステップ進める
            obs, reward, terminated, truncated, info = env.step(action)

            # RL 情報更新
            rl_info.reward = reward
            rl_info.total_reward += reward
            rl_info.action_angle = float(action[0])
            rl_info.action_boost = float(action[1]) > 0.5
            rl_info.step_count += 1
            rl_info.reward_history.append(reward)

            # 認識モニタ更新（DLO 情報付き）
            update_monitor(
                frame=info.get("frame", env.last_frame),
                self_mask=info.get("self_mask", env.last_self_mask),
                self_skeleton=info.get("skeleton", env.last_skeleton),
                enemies=info.get("enemies", env.last_enemies),
                rl_info=rl_info,
                dlo_state=info.get("dlo_state", env.last_dlo_state),
                predicted_dlos=info.get("predicted_dlos", env.last_predicted_dlos),
            )

            # ゲームオーバー → リセット
            if terminated or truncated:
                rl_info.episode_count += 1
                print(
                    f"Episode {rl_info.episode_count} ended. "
                    f"Steps: {rl_info.step_count}, "
                    f"Total reward: {rl_info.total_reward:.1f}"
                )
                rl_info.total_reward = 0.0
                obs, info = env.reset()

            # 定期的にモデル保存
            if rl_info.step_count % 10000 == 0 and rl_info.step_count > 0:
                save_model(model)

    except KeyboardInterrupt:
        print("\nStopping bot...")
    finally:
        save_model(model)
        driver.quit()
        cv2.destroyAllWindows()
        print("Bot stopped.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "vis"
    if mode == "debug":
        run_debug_mask()
    elif mode == "bot":
        run_bot()
    else:
        run_visualization()
