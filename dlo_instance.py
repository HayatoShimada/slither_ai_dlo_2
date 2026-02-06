"""
DLO (Deformable Linear Object) データ構造。
全てのヘビ（自機＋敵）を統一的に表現するデータクラス群。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DLOInstance:
    """1本のヘビ（自機 or 敵）を DLO として表現する。

    Attributes
    ----------
    instance_id : int
        追跡用の一意な ID。
    skeleton_yx : np.ndarray
        骨格座標 (N, 2)、各行は (y, x)。numpy インデックス順。
    heading : float
        頭の向き (rad)。atan2 基準。
    length : float
        骨格全長 (px)。隣接点間距離の合計。
    center : np.ndarray
        重心座標 (2,) (x, y)。OpenCV / 表示用座標系。
    contour : np.ndarray | None
        輪郭 (描画用)。cv2.findContours の出力形式。
    is_self : bool
        自機フラグ。True なら自機ヘビ。
    velocity : np.ndarray
        重心速度ベクトル (2,) (dx, dy) px/frame。
    skeleton_velocity : np.ndarray | None
        各骨格点の速度 (N, 2) (dy, dx) px/frame。
    """

    instance_id: int
    skeleton_yx: np.ndarray
    heading: float
    length: float
    center: np.ndarray
    contour: np.ndarray | None
    is_self: bool
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    skeleton_velocity: np.ndarray | None = None


@dataclass
class DLOState:
    """1フレームの全 DLO インスタンス + 餌。

    Attributes
    ----------
    self_dlo : DLOInstance | None
        自機ヘビの DLO。検出できない場合は None。
    enemy_dlos : list[DLOInstance]
        敵ヘビの DLO リスト。
    food_positions : np.ndarray
        餌の座標 (M, 2) (x, y)。
    """

    self_dlo: DLOInstance | None
    enemy_dlos: list[DLOInstance]
    food_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int32),
    )


def compute_heading(skeleton_yx: np.ndarray) -> float:
    """骨格の先頭2点から頭の向き (rad) を算出する。

    Parameters
    ----------
    skeleton_yx : np.ndarray
        骨格座標 (N, 2)、各行は (y, x)。

    Returns
    -------
    float
        向き (rad)。atan2(-dy, dx) で画面座標系に合わせる。
    """
    if len(skeleton_yx) < 2:
        return 0.0
    dy = skeleton_yx[1, 0] - skeleton_yx[0, 0]
    dx = skeleton_yx[1, 1] - skeleton_yx[0, 1]
    return float(np.arctan2(-dy, dx))


def compute_length(skeleton_yx: np.ndarray) -> float:
    """骨格の全長 (px) を算出する。

    Parameters
    ----------
    skeleton_yx : np.ndarray
        骨格座標 (N, 2)。

    Returns
    -------
    float
        隣接点間距離の合計。
    """
    if len(skeleton_yx) < 2:
        return 0.0
    diffs = np.diff(skeleton_yx, axis=0)
    return float(np.sum(np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)))


def compute_center_xy(skeleton_yx: np.ndarray) -> np.ndarray:
    """骨格点の重心を (x, y) で返す。

    Parameters
    ----------
    skeleton_yx : np.ndarray
        骨格座標 (N, 2)、各行は (y, x)。

    Returns
    -------
    np.ndarray
        重心 (2,) (x, y)。
    """
    mean_yx = np.mean(skeleton_yx, axis=0)
    return np.array([mean_yx[1], mean_yx[0]], dtype=np.float64)
