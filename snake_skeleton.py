"""
Slither.io の自機ヘビを画像から抽出し、細線化して骨格（座標列）を得るモジュール。
RT-DLO や制御アルゴリズムへの入力形式を出力する。
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from config import (
    SNAKE_HSV_LOWER,
    SNAKE_HSV_UPPER,
    MORPH_KERNEL_SIZE,
    MIN_SNAKE_AREA,
    SKELETON_SAMPLE_POINTS,
)


def mask_snake_bgr(
    bgr: np.ndarray,
    hsv_lower: tuple,
    hsv_upper: tuple,
    *,
    hsv_img: np.ndarray | None = None,
) -> np.ndarray:
    """
    BGR 画像からヘビの色領域のマスク（0/255）を返す。

    Parameters
    ----------
    hsv_img : np.ndarray | None
        事前計算済み HSV 画像。None の場合は内部で変換する。
    """
    hsv = hsv_img if hsv_img is not None else cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # 赤は H が 0 と 180 付近にまたがる場合の例（必要なら config で有効化）
    # mask_red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    # mask = cv2.bitwise_or(mask, mask_red2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def largest_connected_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    マスクから最大の連結成分だけを残す（自機を1匹と仮定）。
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    # 0 は背景。1 以降で面積最大のラベルを選ぶ
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return np.zeros_like(mask)
    max_idx = np.argmax(areas)
    if areas[max_idx] < min_area:
        return np.zeros_like(mask)
    label_id = max_idx + 1
    out = np.where(labels == label_id, 255, 0).astype(np.uint8)
    return out


def mask_to_skeleton_binary(mask: np.ndarray) -> np.ndarray:
    """
    マスク（0/255）を細線化し、1ピクセル幅の骨格（0/1 の二値）を返す。
    """
    binary = (mask > 0).astype(np.uint8)
    skel = skeletonize(binary).astype(np.uint8)
    return skel


def skeleton_to_ordered_points(skel: np.ndarray, num_points: int) -> np.ndarray | None:
    """
    骨格画像から、頭→尾の順の座標列を返す。
    端点を2つ検出し、一方を頭・もう一方を尾として、経路に沿ってサンプリングする。

    Returns
    -------
    (N, 2) の np.ndarray (y, x) または None（骨格が空など）
    """
    if np.sum(skel) == 0:
        return None

    # 端点: 周囲8近傍で自分を含めて1の数が2のピクセル（先端）
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    endpoints = np.where((skel > 0) & (neighbor_count == 2))
    if len(endpoints[0]) < 2:
        # 輪になっているなど
        endpoints = np.where(skel > 0)
        if len(endpoints[0]) == 0:
            return None
        # 適当に端として左上・右下を使う
        pts = np.column_stack([endpoints[0], endpoints[1]])
        head_idx = np.argmin(pts[:, 0] + pts[:, 1])
        tail_idx = np.argmax(pts[:, 0] + pts[:, 1])
        head = tuple(pts[head_idx][::-1].astype(int))  # (x,y) for cv2
        tail = tuple(pts[tail_idx][::-1].astype(int))
    else:
        pts_xy = np.column_stack([endpoints[1], endpoints[0]])  # (x, y)
        head = tuple(pts_xy[0].astype(int))
        tail = tuple(pts_xy[1].astype(int))

    # 骨格上の全点を取得し、head から tail まで最短経路で並べる（BFS/DFS）
    skel_pts = np.column_stack(np.where(skel > 0))  # (row, col) = (y, x)
    head_rc = (head[1], head[0])
    tail_rc = (tail[1], tail[0])

    path = _trace_skeleton_path(skel, head_rc, tail_rc)
    if path is None or len(path) < 2:
        path = skel_pts

    path = np.array(path)  # (N, 2) row, col
    # 等間隔にリサンプル
    n = len(path)
    indices = np.linspace(0, n - 1, num=min(num_points, n), dtype=int)
    sampled = path[indices]
    return sampled  # (num_points, 2)  y, x


def _trace_skeleton_path(skel: np.ndarray, start: tuple, end: tuple) -> list | None:
    """
    骨格上を start から end までたどり、座標リストを返す。BFS。
    """
    from collections import deque

    h, w = skel.shape
    queue = deque([start])
    parent = {start: None}
    seen = {start}

    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            break
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and skel[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

    if end not in parent:
        return None
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def extract_snake_skeleton(
    bgr: np.ndarray,
    hsv_lower: tuple | None = None,
    hsv_upper: tuple | None = None,
    num_points: int | None = None,
) -> np.ndarray | None:
    """
    1枚の BGR 画像から自機ヘビの骨格座標を抽出する。
    RT-DLO や可視化の入力として使える。

    Parameters
    ----------
    bgr : np.ndarray
        BGR 画像 (H, W, 3)
    hsv_lower, hsv_upper : tuple, optional
        省略時は config の値を使用
    num_points : int, optional
        骨格のサンプル点数。省略時は config の SKELETON_SAMPLE_POINTS

    Returns
    -------
    np.ndarray or None
        shape (N, 2), 各行は (y, x)。抽出失敗時は None。
    """
    hsv_lower = hsv_lower or SNAKE_HSV_LOWER
    hsv_upper = hsv_upper or SNAKE_HSV_UPPER
    num_points = num_points or SKELETON_SAMPLE_POINTS

    mask = mask_snake_bgr(bgr, hsv_lower, hsv_upper)
    mask = largest_connected_component(mask, MIN_SNAKE_AREA)
    if np.sum(mask) == 0:
        return None

    skel = mask_to_skeleton_binary(mask)
    points = skeleton_to_ordered_points(skel, num_points)
    return points


def skeleton_points_for_rt_dlo(points: np.ndarray) -> np.ndarray:
    """
    RT-DLO が期待する形式（例: (N, 2) の x, y 順）に変換する。
    ここでは (y, x) -> (x, y) にし、必要ならスケールを揃える。
    """
    if points is None or len(points) == 0:
        return np.zeros((0, 2))
    return points[:, ::-1].astype(np.float64)  # (x, y)
