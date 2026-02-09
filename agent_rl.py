"""
PPO 強化学習エージェントモジュール。
Stable-Baselines3 + PyTorch CUDA で slither.io を学習する。
"""

from __future__ import annotations

import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from config import RL_MODEL_DIR, RL_SAVE_INTERVAL, RL_DEVICE, RL_OBS_MODE, CNN_BATCH_SIZE, CNN_FEATURES_DIM


def create_agent(env) -> PPO:
    """
    PPO エージェントを新規作成する。

    RL_OBS_MODE == "hybrid" の場合は MultiInputPolicy (CNN + MLP) を使用し、
    "vector" の場合は従来の MlpPolicy を使用する。

    Parameters
    ----------
    env : gymnasium.Env
        学習環境（hybrid 時は VecFrameStack でラップ済み）。

    Returns
    -------
    PPO
        初期化された PPO モデル。
    """
    if RL_OBS_MODE == "hybrid":
        policy_kwargs = {
            "features_extractor_kwargs": {"cnn_output_dim": CNN_FEATURES_DIM},
        }
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=10,
            policy_kwargs=policy_kwargs,
            device=RL_DEVICE,
            verbose=1,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=10,
            device=RL_DEVICE,
            verbose=1,
        )
    return model


def load_model(env, path: str | None = None) -> PPO | None:
    """
    保存済みモデルを読み込む。存在しなければ None を返す。

    Parameters
    ----------
    env : gymnasium.Env
        学習環境。
    path : str or None
        モデルファイルのパス。None の場合はデフォルトパスを使用。

    Returns
    -------
    PPO or None
        読み込んだモデル、またはファイルが存在しない場合 None。
    """
    if path is None:
        path = os.path.join(RL_MODEL_DIR, "slither_ppo")

    model_file = path + ".zip" if not path.endswith(".zip") else path
    if os.path.exists(model_file):
        print(f"Loading existing model from {model_file}")
        try:
            model = PPO.load(path, env=env, device="auto")
            return model
        except (ValueError, KeyError) as e:
            print(f"WARNING: Failed to load model (policy mismatch?): {e}")
            print("Creating new agent instead (old model kept on disk).")
            return None

    return None


def save_model(model: PPO, path: str | None = None) -> None:
    """
    モデルを保存する。

    Parameters
    ----------
    model : PPO
        保存するモデル。
    path : str or None
        保存先パス。None の場合はデフォルトパスを使用。
    """
    if path is None:
        path = os.path.join(RL_MODEL_DIR, "slither_ppo")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else RL_MODEL_DIR, exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")


def create_checkpoint_callback() -> CheckpointCallback:
    """
    定期的にモデルを保存するコールバックを作成する。

    Returns
    -------
    CheckpointCallback
        チェックポイントコールバック。
    """
    os.makedirs(RL_MODEL_DIR, exist_ok=True)
    return CheckpointCallback(
        save_freq=RL_SAVE_INTERVAL,
        save_path=RL_MODEL_DIR,
        name_prefix="slither_ppo_checkpoint",
    )


def train(model: PPO, total_timesteps: int = 100_000) -> PPO:
    """
    モデルを学習する。

    Parameters
    ----------
    model : PPO
        学習するモデル。
    total_timesteps : int
        学習する総ステップ数。

    Returns
    -------
    PPO
        学習後のモデル。
    """
    callback = create_checkpoint_callback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    save_model(model)
    return model
