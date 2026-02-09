"""
模倣学習用デモデータの記録・保存・読み込みモジュール。

人間プレイヤーのゲームプレイを (observation, action) ペアとして記録し、
Behavioral Cloning の教師データとして使用する。

データ形式:
  各エピソードを 1 つの .npz ファイルに保存。
  - obs_metadata: (T, 62) float32  — メタデータ観測（全モード共通）
  - obs_images:   (T, 84, 84) uint8 — グレースケール画像 (hybrid のみ)
  - actions:      (T, 2) float32    — [angle (0~360), boost (0 or 1)]
  - info:         dict              — obs_mode, score, duration 等
"""

from __future__ import annotations

import ctypes
import math
import os
import time
from pathlib import Path

import numpy as np

from config import (
    IL_DEMO_DIR,
    IL_MIN_DEMO_STEPS,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    RL_OBS_MODE,
)


class DemoRecorder:
    """人間プレイヤーの操作を記録するレコーダー。

    マウス位置から角度を逆算し、マウスボタン状態からブーストを判定する。
    観測はゲーム環境と同じ形式で取得する。

    Parameters
    ----------
    save_dir : str
        デモファイルの保存先ディレクトリ。
    obs_mode : str
        観測モード ("vector" or "hybrid")。
    """

    def __init__(self, save_dir: str | None = None, obs_mode: str | None = None):
        self._save_dir = Path(save_dir or IL_DEMO_DIR)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._obs_mode = obs_mode or RL_OBS_MODE

        self._metadata_list: list[np.ndarray] = []
        self._image_list: list[np.ndarray] = []
        self._vector_list: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._episode_start: float = 0.0
        self._episode_count: int = 0

    def start_episode(self) -> None:
        """新しいエピソードの記録を開始する。"""
        self._metadata_list.clear()
        self._image_list.clear()
        self._vector_list.clear()
        self._actions.clear()
        self._episode_start = time.time()

    def record_step(self, observation: np.ndarray | dict) -> np.ndarray:
        """1 ステップ分の操作を記録する。

        現在のマウス位置から行動（角度 + ブースト）を取得し、
        観測とペアにして保存する。

        Parameters
        ----------
        observation : np.ndarray or dict
            SlitherEnv._get_observation() が返す観測。
            dict: {"image": (84,84,1), "metadata": (62,)}
            ndarray: (254,) vector mode

        Returns
        -------
        np.ndarray
            記録された行動 [angle, boost]。
        """
        action = self._get_human_action()

        if isinstance(observation, dict):
            # hybrid モード
            self._metadata_list.append(observation["metadata"].copy())
            # image: (84,84,1) → (84,84) に squeeze して保存（容量削減）
            img = observation["image"]
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]
            self._image_list.append(img.copy())
        else:
            # vector モード
            self._vector_list.append(observation.copy())

        self._actions.append(action.copy())
        return action

    def end_episode(self, score: int = 0) -> str | None:
        """エピソードを終了し、データをファイルに保存する。

        Parameters
        ----------
        score : int
            エピソード終了時のスコア。

        Returns
        -------
        str or None
            保存先のファイルパス。ステップ数が足りない場合は None。
        """
        n_steps = len(self._actions)
        duration = time.time() - self._episode_start

        if n_steps < IL_MIN_DEMO_STEPS:
            print(
                f"[Demo] Episode too short ({n_steps} steps < {IL_MIN_DEMO_STEPS}), "
                f"discarding."
            )
            return None

        self._episode_count += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"demo_{timestamp}_ep{self._episode_count:04d}.npz"
        filepath = self._save_dir / filename

        save_dict: dict = {
            "actions": np.array(self._actions, dtype=np.float32),
            "info": np.array([score, n_steps, duration]),
            "obs_mode": np.array(self._obs_mode),
        }

        if self._metadata_list:
            save_dict["obs_metadata"] = np.array(self._metadata_list, dtype=np.float32)
        if self._image_list:
            save_dict["obs_images"] = np.array(self._image_list, dtype=np.uint8)
        if self._vector_list:
            save_dict["obs_vector"] = np.array(self._vector_list, dtype=np.float32)

        np.savez_compressed(filepath, **save_dict)

        print(
            f"[Demo] Saved: {filename}  "
            f"steps={n_steps}  duration={duration:.1f}s  score={score}  "
            f"mode={self._obs_mode}"
        )
        return str(filepath)

    @staticmethod
    def _get_human_action() -> np.ndarray:
        """現在のマウス状態から行動を取得する。

        Returns
        -------
        np.ndarray
            [angle (0~360), boost (0.0 or 1.0)]
        """
        cx = SCREEN_WIDTH // 2
        cy = SCREEN_HEIGHT // 2

        try:
            import pyautogui
            mx, my = pyautogui.position()
        except Exception:
            return np.array([0.0, 0.0], dtype=np.float32)

        dx = mx - cx
        dy = -(my - cy)  # 画面座標 Y 軸を反転

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad) % 360

        boost_val = _is_mouse_button_pressed()

        return np.array([angle_deg, float(boost_val)], dtype=np.float32)


def _is_mouse_button_pressed() -> bool:
    """X11 経由でマウスボタンが押されているか確認する。

    ctypes で libX11 を直接呼び出す（python-xlib 不要、高速）。

    Returns
    -------
    bool
        いずれかのマウスボタンが押されていれば True。
    """
    try:
        x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
        display = x11.XOpenDisplay(None)
        if not display:
            return False

        root = x11.XDefaultRootWindow(display)
        # XQueryPointer の出力引数
        root_ret = ctypes.c_ulong()
        child_ret = ctypes.c_ulong()
        root_x = ctypes.c_int()
        root_y = ctypes.c_int()
        win_x = ctypes.c_int()
        win_y = ctypes.c_int()
        mask = ctypes.c_uint()

        x11.XQueryPointer(
            display, root,
            ctypes.byref(root_ret), ctypes.byref(child_ret),
            ctypes.byref(root_x), ctypes.byref(root_y),
            ctypes.byref(win_x), ctypes.byref(win_y),
            ctypes.byref(mask),
        )
        x11.XCloseDisplay(display)

        # Button1Mask=256, Button2Mask=512, Button3Mask=1024
        return bool(mask.value & 0x700)
    except Exception:
        return False


class DemoDataset:
    """保存済みデモデータを読み込み、学習用のバッチを提供する。

    obs_mode を自動判別し、hybrid/vector どちらのデモにも対応する。

    Parameters
    ----------
    demo_dir : str
        デモデータのディレクトリ。
    """

    def __init__(self, demo_dir: str | None = None):
        self._demo_dir = Path(demo_dir or IL_DEMO_DIR)
        self._obs_metadata: np.ndarray | None = None
        self._obs_images: np.ndarray | None = None
        self._obs_vector: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._obs_mode: str = "unknown"
        self._loaded = False

    def load(self) -> int:
        """全デモファイルを読み込み、結合する。

        Returns
        -------
        int
            総ステップ数。データが無い場合は 0。
        """
        if not self._demo_dir.exists():
            print(f"[Demo] Directory not found: {self._demo_dir}")
            return 0

        demo_files = sorted(self._demo_dir.glob("demo_*.npz"))
        if not demo_files:
            print(f"[Demo] No demo files found in {self._demo_dir}")
            return 0

        all_metadata = []
        all_images = []
        all_vectors = []
        all_actions = []
        detected_mode = None

        for f in demo_files:
            try:
                data = np.load(f, allow_pickle=True)
                act = data["actions"]
                # info キー（新形式）or metadata キー（旧形式）
                if "info" in data:
                    info = data["info"]
                elif "metadata" in data:
                    info = data["metadata"]
                else:
                    info = np.array([0, len(act), 0.0])

                # obs_mode 判別
                file_mode = str(data["obs_mode"]) if "obs_mode" in data else None
                if file_mode is None:
                    # 旧形式: observations キーがある場合は vector 扱い
                    if "observations" in data:
                        file_mode = "legacy"
                    elif "obs_metadata" in data:
                        file_mode = "hybrid"
                    elif "obs_vector" in data:
                        file_mode = "vector"
                    else:
                        print(f"  WARNING: Unknown format in {f.name}, skipping.")
                        continue

                if detected_mode is None:
                    detected_mode = file_mode
                elif detected_mode != file_mode:
                    print(
                        f"  WARNING: Mixed obs modes ({detected_mode} vs {file_mode}). "
                        f"Skipping {f.name}."
                    )
                    continue

                all_actions.append(act)

                if "obs_metadata" in data:
                    all_metadata.append(data["obs_metadata"])
                if "obs_images" in data:
                    all_images.append(data["obs_images"])
                if "obs_vector" in data:
                    all_vectors.append(data["obs_vector"])
                if "observations" in data and file_mode == "legacy":
                    # 旧形式互換: observations を metadata として扱う
                    all_metadata.append(data["observations"])

                print(
                    f"  Loaded {f.name}: {len(act)} steps, "
                    f"score={int(info[0])}, duration={info[2]:.1f}s, mode={file_mode}"
                )
            except Exception as e:
                print(f"  WARNING: Failed to load {f.name}: {e}")

        if not all_actions:
            return 0

        self._actions = np.concatenate(all_actions, axis=0)
        self._obs_mode = detected_mode or "unknown"

        if all_metadata:
            self._obs_metadata = np.concatenate(all_metadata, axis=0)
        if all_images:
            self._obs_images = np.concatenate(all_images, axis=0)
        if all_vectors:
            self._obs_vector = np.concatenate(all_vectors, axis=0)

        self._loaded = True
        total = len(self._actions)
        print(
            f"[Demo] Total: {len(demo_files)} episodes, "
            f"{total} steps, mode={self._obs_mode}"
        )
        return total

    @property
    def obs_mode(self) -> str:
        """記録時の観測モード。"""
        return self._obs_mode

    @property
    def metadata(self) -> np.ndarray | None:
        """メタデータ観測 (T, 62)。"""
        return self._obs_metadata

    @property
    def images(self) -> np.ndarray | None:
        """画像観測 (T, 84, 84) uint8。"""
        return self._obs_images

    @property
    def vectors(self) -> np.ndarray | None:
        """ベクトル観測 (T, 254) (vector mode)。"""
        return self._obs_vector

    @property
    def observations(self) -> np.ndarray:
        """後方互換: メタデータまたはベクトル観測を返す。"""
        if self._obs_vector is not None:
            return self._obs_vector
        if self._obs_metadata is not None:
            return self._obs_metadata
        raise RuntimeError("No observation data loaded.")

    @property
    def actions(self) -> np.ndarray:
        """行動データ (T, 2)。"""
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._actions

    def summary(self) -> dict:
        """データセットのサマリーを返す。"""
        if not self._loaded:
            return {"loaded": False}
        obs = self.observations
        return {
            "loaded": True,
            "obs_mode": self._obs_mode,
            "total_steps": len(self._actions),
            "obs_dim": obs.shape[1] if obs.ndim == 2 else obs.shape,
            "has_images": self._obs_images is not None,
            "action_dim": self._actions.shape[1],
            "action_angle_range": (
                float(self._actions[:, 0].min()),
                float(self._actions[:, 0].max()),
            ),
            "boost_ratio": float(np.mean(self._actions[:, 1] > 0.5)),
        }
