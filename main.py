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
  python main.py record    -- 人間プレイを記録 (模倣学習用デモデータ)
  python main.py pretrain  -- デモデータから Behavioral Cloning 事前訓練
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


def run_record():
    """
    人間プレイを記録するモード（模倣学習用）。

    ブラウザを起動し、noVNC (port 6080) 経由で人間がプレイ。
    SlitherEnv と同じ観測パイプラインで obs を取得し、
    マウス位置→角度・ブースト状態と合わせて記録する。
    obs_mode (hybrid/vector) に応じた完全な観測データを保存する。
    """
    import time

    from browser import create_driver, start_game, is_playing, restart_game
    from config import IL_RECORD_INTERVAL, RL_OBS_MODE
    from dlo_tracker import DLOTracker
    from monitor import update_monitor, RLInfo
    from imitation_data import DemoRecorder
    from game_env import SlitherEnv

    def _selenium_capture(drv):
        """Selenium 経由のキャプチャ関数を返す。"""
        def _fn():
            png = drv.get_screenshot_as_png()
            arr = np.frombuffer(png, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return _fn

    print("=== Slither.io Record Mode (模倣学習用) ===")
    print(f"Observation mode: {RL_OBS_MODE}")
    print("noVNC (http://localhost:6080) でブラウザに接続し、プレイしてください。")
    print("操作は自動的に記録されます。Ctrl+C で終了。")
    print()

    driver = create_driver()
    recorder = DemoRecorder(obs_mode=RL_OBS_MODE)
    capture_fn = _selenium_capture(driver)

    try:
        start_game(driver)
        time.sleep(3)

        # 自機カラー自動検出（Selenium 経由、mss 不要）
        hsv_lower, hsv_upper = None, None
        if AUTO_DETECT_COLOR:
            hsv_lower, hsv_upper = auto_detect_snake_color(capture_fn)
        hsv_l = hsv_lower or SNAKE_HSV_LOWER
        hsv_u = hsv_upper or SNAKE_HSV_UPPER

        tracker = DLOTracker()
        env = SlitherEnv(driver, tracker=tracker, hsv_lower=hsv_l, hsv_upper=hsv_u)

        rl_info = RLInfo()
        episode_num = 0

        while True:
            episode_num += 1
            print(f"\n[Record] Episode {episode_num} — プレイしてください...")

            # ゲーム開始待機
            for _ in range(120):
                if is_playing(driver):
                    break
                time.sleep(0.5)

            if not is_playing(driver):
                print("[Record] ゲーム開始待ちタイムアウト。リロードします。")
                restart_game(driver)
                continue

            # 自機カラー再検出（ゲームごとに色が変わる）
            if AUTO_DETECT_COLOR:
                time.sleep(0.5)
                hsv_l, hsv_u = auto_detect_snake_color(env._capture_browser)
                env._hsv_lower = hsv_l
                env._hsv_upper = hsv_u

            env._tracker = DLOTracker()
            recorder.start_episode()
            step = 0
            score = 0

            while True:
                try:
                    # SlitherEnv と同じパイプラインで観測取得
                    obs, info = env._get_observation()

                    # 人間の行動を記録（obs をそのまま渡す）
                    action = recorder.record_step(obs)
                    step += 1
                    score = info.get("score", 0)

                    # ステップログ（20ステップごと）
                    if step % 20 == 0:
                        area = info.get("area_px", 0)
                        print(
                            f"  [record step {step}] score={score} "
                            f"area={area:.0f} angle={action[0]:.0f} "
                            f"boost={action[1]:.0f}",
                            flush=True,
                        )

                    # モニタ更新（5ステップごと）
                    if step % 5 == 0:
                        frame = info.get("frame")
                        if frame is not None:
                            rl_info.step_count = step
                            rl_info.episode_count = episode_num
                            rl_info.human_mode = True
                            update_monitor(
                                frame,
                                info.get("self_mask"),
                                info.get("skeleton"),
                                info.get("enemies"),
                                rl_info,
                                dlo_state=info.get("dlo_state"),
                                predicted_dlos=info.get("predicted_dlos", []),
                            )

                    # ゲームオーバー検出
                    if not info.get("js_playing", True):
                        print(f"[Record] Game Over! score={score}, steps={step}")
                        break

                    time.sleep(IL_RECORD_INTERVAL)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"[Record] Step error: {e}")
                    time.sleep(0.1)

            recorder.end_episode(score=score)

            print("[Record] 3秒後にリスタートします...")
            time.sleep(3)
            restart_game(driver)

    except KeyboardInterrupt:
        print("\n[Record] 記録を終了します。")
        recorder.end_episode(score=0)
    finally:
        driver.quit()
        print("[Record] 完了。")


def run_pretrain():
    """
    デモデータから Behavioral Cloning で事前訓練するモード。

    bot モードと完全に同じモデル構造（VecFrameStack 含む）を作成し、
    単一フレームのデモデータをフレーム複製して学習する。
    これにより bot でそのまま RL ファインチューニングに移行できる。
    """
    from config import RL_OBS_MODE, CNN_FRAME_STACK
    from imitation_learn import pretrain_and_save
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    print("=== Slither.io Pretrain Mode (Behavioral Cloning) ===")
    print(f"Observation mode: {RL_OBS_MODE}")
    print()

    # bot と同じ観測空間のダミー環境を構築
    env = _create_dummy_env()

    if RL_OBS_MODE == "hybrid":
        print(f"Hybrid mode: VecFrameStack(n_stack={CNN_FRAME_STACK})")
        venv = DummyVecEnv([lambda e=env: e])
        venv = VecFrameStack(venv, n_stack=CNN_FRAME_STACK, channels_order="last")
        train_env = venv
    else:
        train_env = env

    model = pretrain_and_save(train_env)
    print("\n事前訓練完了。'python main.py bot' で RL ファインチューニングを開始できます。")


def _create_dummy_env():
    """bot と同じ観測空間・行動空間を持つダミー環境。ブラウザ不要。"""
    import gymnasium as gym
    from gymnasium import spaces
    from config import (
        SKELETON_SAMPLE_POINTS,
        TOP_K_ENEMIES,
        TOP_M_FOOD,
        RL_OBS_MODE,
        CNN_INPUT_SIZE,
    )

    SKELETON_DIM = SKELETON_SAMPLE_POINTS * 2
    SELF_META_DIM = 4
    ENEMY_DLO_DIM = TOP_K_ENEMIES * 6
    FOOD_DIM = TOP_M_FOOD * 2
    COLLISION_RISK_DIM = TOP_K_ENEMIES
    MAP_POS_DIM = 2
    OBS_DIM = SKELETON_DIM + SELF_META_DIM + ENEMY_DLO_DIM + FOOD_DIM + COLLISION_RISK_DIM + MAP_POS_DIM
    METADATA_DIM = SELF_META_DIM + ENEMY_DLO_DIM + COLLISION_RISK_DIM + MAP_POS_DIM

    class DummySlitherEnv(gym.Env):
        metadata = {"render_modes": ["human"]}

        def __init__(self):
            super().__init__()
            if RL_OBS_MODE == "hybrid":
                self.observation_space = spaces.Dict({
                    "image": spaces.Box(0, 255, (CNN_INPUT_SIZE[0], CNN_INPUT_SIZE[1], 1), np.uint8),
                    "metadata": spaces.Box(-1.0, 1.0, (METADATA_DIM,), np.float32),
                })
            else:
                self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
            self.action_space = spaces.Box(
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([360.0, 1.0], dtype=np.float32),
            )

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if RL_OBS_MODE == "hybrid":
                obs = {
                    "image": np.zeros((CNN_INPUT_SIZE[0], CNN_INPUT_SIZE[1], 1), np.uint8),
                    "metadata": np.zeros(METADATA_DIM, np.float32),
                }
            else:
                obs = np.zeros(OBS_DIM, np.float32)
            return obs, {}

        def step(self, action):
            obs, _ = self.reset()
            return obs, 0.0, True, False, {}

    return DummySlitherEnv()


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
    from monitor import update_monitor, RLInfo
    from mouse_control import boost
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
            self._rl_info = RLInfo()
            self._driver = None

        def set_driver(self, driver):
            """Toggle 検出用にドライバ参照を保持する。"""
            self._driver = driver

        def _on_step(self) -> bool:
            self._episode_reward += self.locals["rewards"][0]
            self._episode_steps += 1

            # 人間モード検出（10ステップごとにチェック、JS呼び出しコスト削減）
            if self._episode_steps % 10 == 0 and self._driver is not None:
                if is_human_mode(self._driver):
                    print("[Toggle] Switching to HUMAN mode...")
                    boost(False)  # ブースト解除
                    return False  # model.learn() を停止

            # 認識モニタの更新（3ステップごとに描画負荷を軽減）
            if self._episode_steps % 3 == 0:
                try:
                    venv = self.training_env
                    while hasattr(venv, "venv"):
                        venv = venv.venv
                    base_env = venv.envs[0]
                    base_env = getattr(base_env, "unwrapped", base_env)
                    frame = getattr(base_env, "_last_frame", None)
                    if frame is not None:
                        self._rl_info.reward = float(self.locals["rewards"][0])
                        self._rl_info.total_reward = self._episode_reward
                        self._rl_info.step_count = self._episode_steps
                        self._rl_info.episode_count = self._episode_count
                        self._rl_info.human_mode = False
                        self._rl_info.reward_history.append(float(self.locals["rewards"][0]))
                        update_monitor(
                            frame,
                            getattr(base_env, "_last_self_mask", None),
                            getattr(base_env, "_last_skeleton", None),
                            getattr(base_env, "_last_enemies", None),
                            self._rl_info,
                            dlo_state=getattr(base_env, "_last_dlo_state", None),
                            predicted_dlos=getattr(base_env, "_last_predicted_dlos", []),
                        )
                except Exception:
                    pass

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

    def _run_human_loop(driver, train_env, callback):
        """人間モード: VNCマウスで操作し、観測とモニタ更新のみ行う。"""
        from browser import restart_game as _restart, is_human_mode as _is_human

        # ベース env を取得
        venv = train_env
        while hasattr(venv, "venv"):
            venv = venv.venv
        base_env = venv.envs[0] if hasattr(venv, "envs") else venv
        base_env = getattr(base_env, "unwrapped", base_env)

        rl_info = callback._rl_info
        rl_info.human_mode = True

        while _is_human(driver):
            try:
                obs, info = base_env._get_observation()
                frame = info.get("frame")
                if frame is not None:
                    update_monitor(
                        frame,
                        info.get("self_mask"),
                        info.get("skeleton"),
                        info.get("enemies"),
                        rl_info,
                        dlo_state=info.get("dlo_state"),
                        predicted_dlos=info.get("predicted_dlos", []),
                    )

                # ゲームオーバー検出 → 自動リスタート
                if not info.get("js_playing", True):
                    print("[HUMAN] Game over, restarting...")
                    _restart(driver)
                    inject_toggle_listener(driver)
                    # 自機カラー再検出
                    if AUTO_DETECT_COLOR:
                        hsv_lower, hsv_upper = auto_detect_snake_color(
                            base_env._capture_browser,
                        )
                        base_env._hsv_lower = hsv_lower
                        base_env._hsv_upper = hsv_upper
            except Exception as e:
                print(f"[HUMAN] Error: {e}")

            time.sleep(0.05)

        rl_info.human_mode = False

    print("=== Slither.io Bot Mode (DLO) ===")
    print(f"Observation mode: {RL_OBS_MODE}")
    print("Press Tab in VNC to toggle Human/AI mode.")
    print("Starting browser...")

    driver = create_driver()
    model = None
    try:
        start_game(driver)

        # window.snake の全プロパティをダンプ（初回診断）
        time.sleep(3)
        from browser import dump_snake_properties, inject_toggle_listener, is_human_mode
        dump_snake_properties(driver)
        inject_toggle_listener(driver)

        # 自機カラー自動検出（Selenium 経由、mss 不要）
        def _sel_capture():
            png = driver.get_screenshot_as_png()
            arr = np.frombuffer(png, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        hsv_lower, hsv_upper = None, None
        if AUTO_DETECT_COLOR:
            hsv_lower, hsv_upper = auto_detect_snake_color(_sel_capture)

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

        callback = EpisodeLogCallback(save_interval=10000)
        callback.set_driver(driver)

        # メインループ: AI学習 ↔ 人間プレイ を切り替え
        while True:
            print("[AI Mode] Starting PPO training... (Tab to switch to HUMAN)")
            model.learn(
                total_timesteps=10_000_000,
                callback=callback,
                reset_num_timesteps=False,
            )

            # model.learn() が停止 → 人間モードへ
            if is_human_mode(driver):
                print("=" * 50)
                print("[HUMAN Mode] VNC でマウス操作してください。Tab で AI に戻ります。")
                print("=" * 50)
                _run_human_loop(driver, train_env, callback)
                print("[AI Mode] Resuming training...")
                inject_toggle_listener(driver)  # ページリロード後のために再注入
            else:
                break  # 正常終了（total_timesteps 到達）

    except KeyboardInterrupt:
        print("\nStopping bot...")
    finally:
        if model is not None:
            save_model(model)
        driver.quit()
        print("Bot stopped.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "vis"
    if mode == "debug":
        run_debug_mask()
    elif mode == "bot":
        run_bot()
    elif mode == "record":
        run_record()
    elif mode == "pretrain":
        run_pretrain()
    elif mode == "diag":
        from diagnose_detection import run_diagnosis
        image_path = sys.argv[2] if len(sys.argv) > 2 else None
        run_diagnosis(image_path)
    else:
        run_visualization()
