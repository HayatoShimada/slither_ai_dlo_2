"""
Behavioral Cloning (BC) による模倣学習モジュール。

デモデータから PPO ポリシーネットワークを教師あり学習で事前訓練する。
SB3 の PPO モデルのポリシーを直接最適化するため、学習後はそのまま
RL ファインチューニングに使用できる。

観測空間の整合:
  record は単一フレームの obs を保存する。
  bot の hybrid モードは VecFrameStack(n_stack=4) でフレームを積み重ねる。
  BC では単一フレームを N 回複製して N-frame stacked obs を擬似的に構成し、
  ポリシーの full forward パスを通す。これにより bot と同一のモデル構造を
  事前訓練でき、そのまま RL ファインチューニングに移行できる。

使い方:
  1. demos/ にデモデータ (.npz) を配置
  2. pretrain_bc() でポリシーを事前訓練
  3. models/slither_ppo.zip に保存 → bot モードで RL 継続学習
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO

from config import (
    IL_BC_EPOCHS,
    IL_BC_BATCH_SIZE,
    IL_BC_LR,
    CNN_FRAME_STACK,
)
from imitation_data import DemoDataset


def _is_hybrid_policy(policy) -> bool:
    """ポリシーが MultiInputPolicy (Dict 観測) かどうか判定する。"""
    from stable_baselines3.common.torch_layers import CombinedExtractor
    return isinstance(policy.features_extractor, CombinedExtractor)


def _replicate_frames(
    metadata: torch.Tensor,
    images: torch.Tensor | None,
    n_stack: int,
    policy,
    device: torch.device,
) -> dict:
    """単一フレームのデータを n_stack 分複製して Dict 観測を構築する。

    VecFrameStack 後の観測空間に合わせる:
      image:    (batch, n_stack, H, W) float32  ← VecTransposeImage 後の CHW 形式
      metadata: (batch, metadata_dim * n_stack) float32

    Parameters
    ----------
    metadata : torch.Tensor
        (batch, metadata_dim) 単一フレームのメタデータ。
    images : torch.Tensor or None
        (batch, H, W) 単一フレームのグレースケール画像。None の場合ゼロ画像。
    n_stack : int
        フレームスタック数。
    policy : BasePolicy
        SB3 ポリシー。
    device : torch.device
        デバイス。

    Returns
    -------
    dict
        ポリシーが期待する Dict 観測。
    """
    batch_size = metadata.shape[0]
    obs_space = policy.observation_space

    # metadata: (batch, dim) → (batch, dim * n_stack) に複製
    stacked_metadata = metadata.repeat(1, n_stack)

    # image: obs_space["image"] から shape を取得
    # VecTransposeImage 後: (C, H, W) = (n_stack, 84, 84)
    img_shape = obs_space["image"].shape  # (n_stack, H, W)
    if images is not None:
        # (batch, H, W) → (batch, 1, H, W) → (batch, n_stack, H, W) に複製
        img = images.unsqueeze(1).repeat(1, n_stack, 1, 1).float()
    else:
        img = torch.zeros((batch_size, *img_shape), dtype=torch.float32, device=device)

    return {
        "image": img.to(device),
        "metadata": stacked_metadata.to(device),
    }


def pretrain_bc(
    model: PPO,
    dataset: DemoDataset,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
) -> dict:
    """Behavioral Cloning でポリシーを事前訓練する。

    PPO ポリシーの actor ネットワークに対して、デモデータの
    (observation, action) ペアで教師あり学習を行う。

    角度は周期性を考慮して (sin, cos) に変換してから MSE 損失を計算する。
    ブーストは BCE (Binary Cross Entropy) 損失を使用する。

    Parameters
    ----------
    model : PPO
        SB3 の PPO モデル。ポリシーが直接更新される。
    dataset : DemoDataset
        読み込み済みのデモデータセット。
    epochs : int or None
        エポック数。None の場合は config のデフォルト。
    batch_size : int or None
        バッチサイズ。None の場合は config のデフォルト。
    lr : float or None
        学習率。None の場合は config のデフォルト。

    Returns
    -------
    dict
        学習結果。
    """
    epochs = epochs or IL_BC_EPOCHS
    batch_size = batch_size or IL_BC_BATCH_SIZE
    lr = lr or IL_BC_LR

    device = model.device
    policy = model.policy
    is_hybrid = _is_hybrid_policy(policy)

    # --- データ準備 ---
    act_np = dataset.actions  # (T, 2) — [angle_deg, boost]

    # 角度を sin/cos に変換（周期性対応）
    angle_rad = np.deg2rad(act_np[:, 0])
    angle_sin = torch.tensor(np.sin(angle_rad), dtype=torch.float32)
    angle_cos = torch.tensor(np.cos(angle_rad), dtype=torch.float32)
    boost_label = torch.tensor((act_np[:, 1] > 0.5).astype(np.float32))

    if is_hybrid:
        # hybrid: metadata + images (optional)
        meta_np = dataset.metadata
        if meta_np is None:
            raise ValueError(
                "Hybrid mode requires metadata in demo data. "
                "Re-record demos with hybrid obs_mode."
            )
        obs_meta = torch.tensor(meta_np, dtype=torch.float32)
        img_np = dataset.images
        obs_img = torch.tensor(img_np, dtype=torch.uint8) if img_np is not None else None

        if obs_img is not None:
            torch_ds = TensorDataset(obs_meta, obs_img, angle_sin, angle_cos, boost_label)
        else:
            torch_ds = TensorDataset(obs_meta, angle_sin, angle_cos, boost_label)
    else:
        # vector
        obs_np = dataset.observations
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32)
        torch_ds = TensorDataset(obs_tensor, angle_sin, angle_cos, boost_label)

    dataloader = DataLoader(torch_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # --- ポリシーネットワークの取得 ---
    policy.train()

    # actor ネットワークのパラメータを最適化
    optimized_params = (
        list(policy.features_extractor.parameters())
        + list(policy.mlp_extractor.parameters())
        + list(policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(optimized_params, lr=lr)

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    mode_str = "hybrid (frame-replicated Dict obs)" if is_hybrid else "vector"
    data_steps = len(act_np)
    obs_info = f"metadata({meta_np.shape[1]})" if is_hybrid else f"vector({obs_np.shape[1]})"
    has_img_str = f" + image({img_np.shape[1]}x{img_np.shape[2]})" if (is_hybrid and img_np is not None) else ""

    print(f"\n=== Behavioral Cloning ===")
    print(f"  Mode:       {mode_str}")
    print(f"  Data:       {data_steps} steps")
    print(f"  Obs:        {obs_info}{has_img_str}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print(f"  Device:     {device}")
    if is_hybrid:
        print(f"  FrameStack: {CNN_FRAME_STACK}x replicate")
    print()

    start_time = time.time()
    best_loss = float("inf")
    final_angle_loss = 0.0
    final_boost_loss = 0.0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_angle_loss = 0.0
        epoch_boost_loss = 0.0
        n_batches = 0

        for batch_data in dataloader:
            if is_hybrid and obs_img is not None:
                meta_b, img_b, sin_b, cos_b, boost_b = batch_data
                img_b = img_b.to(device)
            elif is_hybrid:
                meta_b, sin_b, cos_b, boost_b = batch_data
                img_b = None
            else:
                obs_b, sin_b, cos_b, boost_b = batch_data
                obs_b = obs_b.to(device)

            sin_b = sin_b.to(device)
            cos_b = cos_b.to(device)
            boost_b = boost_b.to(device)

            # forward pass
            if is_hybrid:
                meta_b = meta_b.to(device)
                dict_obs = _replicate_frames(meta_b, img_b, CNN_FRAME_STACK, policy, device)
                features = policy.extract_features(dict_obs, policy.features_extractor)
            else:
                features = policy.extract_features(obs_b, policy.features_extractor)

            latent_pi, _ = policy.mlp_extractor(features)
            action_mean = policy.action_net(latent_pi)

            # 損失計算
            pred_sin = torch.sin(action_mean[:, 0])
            pred_cos = torch.cos(action_mean[:, 0])
            loss_angle = mse_loss_fn(pred_sin, sin_b) + mse_loss_fn(pred_cos, cos_b)
            loss_boost = bce_loss_fn(action_mean[:, 1], boost_b)
            loss = loss_angle + loss_boost * 0.5

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_angle_loss += loss_angle.item()
            epoch_boost_loss += loss_boost.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_angle = epoch_angle_loss / max(n_batches, 1)
        avg_boost = epoch_boost_loss / max(n_batches, 1)
        final_angle_loss = avg_angle
        final_boost_loss = avg_boost

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            print(
                f"  [Epoch {epoch:3d}/{epochs}] "
                f"loss={avg_loss:.4f}  "
                f"angle={avg_angle:.4f}  boost={avg_boost:.4f}  "
                f"best={best_loss:.4f}  "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.time() - start_time
    policy.eval()

    print(f"\n  BC training complete: {elapsed:.1f}s")
    print(f"  Best loss: {best_loss:.4f}")

    return {
        "epochs": epochs,
        "final_loss": best_loss,
        "angle_loss": final_angle_loss,
        "boost_loss": final_boost_loss,
        "elapsed_sec": elapsed,
    }


def pretrain_and_save(
    env,
    demo_dir: str | None = None,
    model_path: str | None = None,
    **bc_kwargs,
) -> PPO:
    """デモデータを読み込み、BC 事前訓練し、モデルを保存する一括処理。

    Parameters
    ----------
    env : gymnasium.Env
        環境（観測空間・行動空間の定義に必要）。
    demo_dir : str or None
        デモデータのディレクトリ。
    model_path : str or None
        モデル保存先パス。
    **bc_kwargs
        pretrain_bc() への追加引数。

    Returns
    -------
    PPO
        BC 事前訓練済みの PPO モデル。
    """
    from agent_rl import create_agent, load_model, save_model

    # デモデータ読み込み
    dataset = DemoDataset(demo_dir)
    n_steps = dataset.load()
    if n_steps == 0:
        raise ValueError("No demonstration data found. Record demos first.")

    summary = dataset.summary()
    print(f"\n[Dataset Summary]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # モデル読み込み or 新規作成
    model = load_model(env)
    if model is None:
        print("No existing model. Creating new agent for BC pre-training...")
        model = create_agent(env)
    else:
        print("Loaded existing model. BC will fine-tune the policy.")

    # BC 学習
    result = pretrain_bc(model, dataset, **bc_kwargs)

    # 保存
    save_model(model, model_path)
    print(
        f"\n[BC Complete] "
        f"loss={result['final_loss']:.4f}  "
        f"angle_loss={result['angle_loss']:.4f}  "
        f"boost_loss={result['boost_loss']:.4f}  "
        f"time={result['elapsed_sec']:.1f}s"
    )

    return model
