"""
敵ヘビ・餌の検出モジュール。
背景除去 + 自機マスク除外で、残りの前景を敵と餌に分類する。
DLO 統合: 敵の各連結成分に対して骨格抽出を行い、DLOInstance として返す。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from config import (
    BG_HSV_LOWER,
    BG_HSV_UPPER,
    ENEMY_MIN_AREA,
    ENEMY_SKELETON_POINTS,
    FOOD_MAX_AREA,
)
from dlo_instance import (
    DLOInstance,
    DLOState,
    compute_center_xy,
    compute_heading,
    compute_length,
)
from snake_skeleton import mask_to_skeleton_binary, skeleton_to_ordered_points


@dataclass
class EnemyInfo:
    """敵・餌の検出結果を保持するデータクラス（後方互換用）。"""

    enemy_contours: list = field(default_factory=list)
    enemy_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    food_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))


def dlo_state_to_enemy_info(state: DLOState) -> EnemyInfo:
    """DLOState から EnemyInfo に変換する（後方互換用）。

    Parameters
    ----------
    state : DLOState
        DLO ベースの検出結果。

    Returns
    -------
    EnemyInfo
        従来形式の敵・餌検出結果。
    """
    contours = []
    centers_list = []
    for dlo in state.enemy_dlos:
        if dlo.contour is not None:
            contours.append(dlo.contour)
        centers_list.append([int(dlo.center[0]), int(dlo.center[1])])

    enemy_centers = (
        np.array(centers_list, dtype=np.int32)
        if centers_list
        else np.zeros((0, 2), dtype=np.int32)
    )

    return EnemyInfo(
        enemy_contours=contours,
        enemy_centers=enemy_centers,
        food_positions=state.food_positions,
    )


def detect_background_mask(bgr: np.ndarray) -> np.ndarray:
    """
    slither.io の背景（暗い六角格子）を HSV で検出する。

    Parameters
    ----------
    bgr : np.ndarray
        BGR 画像 (H, W, 3)。

    Returns
    -------
    np.ndarray
        背景マスク (0/255)。背景部分が 255。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(BG_HSV_LOWER, dtype=np.uint8)
    upper = np.array(BG_HSV_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # ノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _extract_enemy_skeleton(component_mask: np.ndarray) -> np.ndarray | None:
    """敵の連結成分マスクから骨格座標を抽出する。

    Parameters
    ----------
    component_mask : np.ndarray
        1 つの敵連結成分のマスク (0/255)。

    Returns
    -------
    np.ndarray | None
        骨格座標 (N, 2) (y, x)。抽出失敗時は None。
    """
    skel = mask_to_skeleton_binary(component_mask)
    if np.sum(skel) == 0:
        return None
    points = skeleton_to_ordered_points(skel, ENEMY_SKELETON_POINTS)
    return points


def detect_all_objects(
    bgr: np.ndarray,
    self_mask: np.ndarray,
    self_skeleton_yx: np.ndarray | None = None,
) -> DLOState:
    """
    背景と自機を除外し、全ての検出物を DLO インスタンスとして返す。

    自機は渡された骨格情報から DLOInstance を構築する。
    敵は各連結成分に対して骨格抽出を試み、DLOInstance として返す。
    餌の座標もまとめて DLOState に格納する。

    Parameters
    ----------
    bgr : np.ndarray
        BGR 画像 (H, W, 3)。
    self_mask : np.ndarray
        自機ヘビのマスク (0/255)。
    self_skeleton_yx : np.ndarray | None
        自機の骨格座標 (N, 2) (y, x)。None の場合、自機 DLO は None。

    Returns
    -------
    DLOState
        全 DLO インスタンス + 餌の座標。
    """
    bg_mask = detect_background_mask(bgr)

    # 前景 = 背景でも自機でもない領域
    foreground = cv2.bitwise_and(
        cv2.bitwise_not(bg_mask),
        cv2.bitwise_not(self_mask),
    )

    # ノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

    # 連結成分解析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground, connectivity=8,
    )

    enemy_dlos: list[DLOInstance] = []
    food_positions_list: list[list[int]] = []

    for i in range(1, num_labels):  # 0 は背景
        area = stats[i, cv2.CC_STAT_AREA]
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])

        if area >= ENEMY_MIN_AREA:
            # 敵ヘビ: マスク → 骨格抽出 → DLOInstance
            component_mask = (labels == i).astype(np.uint8) * 255

            # 輪郭抽出（描画用）
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            contour = contours[0] if contours else None

            # 骨格抽出
            skel_yx = _extract_enemy_skeleton(component_mask)

            if skel_yx is not None and len(skel_yx) >= 2:
                center = compute_center_xy(skel_yx)
                heading = compute_heading(skel_yx)
                length = compute_length(skel_yx)
            else:
                # 骨格抽出失敗 → 重心のみの 1 点 DLO
                skel_yx = np.array([[cy, cx]], dtype=np.int32)
                center = np.array([cx, cy], dtype=np.float64)
                heading = 0.0
                length = 0.0

            enemy_dlos.append(DLOInstance(
                instance_id=-1,  # 追跡器が割り当てる
                skeleton_yx=skel_yx,
                heading=heading,
                length=length,
                center=center,
                contour=contour,
                is_self=False,
            ))
        elif area > 10:  # 極小ノイズは無視
            food_positions_list.append([cx, cy])

    # 自機 DLO
    self_dlo: DLOInstance | None = None
    if self_skeleton_yx is not None and len(self_skeleton_yx) >= 2:
        self_dlo = DLOInstance(
            instance_id=-1,
            skeleton_yx=self_skeleton_yx,
            heading=compute_heading(self_skeleton_yx),
            length=compute_length(self_skeleton_yx),
            center=compute_center_xy(self_skeleton_yx),
            contour=None,
            is_self=True,
        )

    food_positions = (
        np.array(food_positions_list, dtype=np.int32)
        if food_positions_list
        else np.zeros((0, 2), dtype=np.int32)
    )

    return DLOState(
        self_dlo=self_dlo,
        enemy_dlos=enemy_dlos,
        food_positions=food_positions,
    )


def detect_enemies_and_food(bgr: np.ndarray, self_mask: np.ndarray) -> EnemyInfo:
    """
    背景と自機を除外し、残りの前景から敵ヘビと餌を検出する。
    後方互換ラッパー: 内部で detect_all_objects() を呼び、EnemyInfo に変換する。

    Parameters
    ----------
    bgr : np.ndarray
        BGR 画像 (H, W, 3)。
    self_mask : np.ndarray
        自機ヘビのマスク (0/255)。

    Returns
    -------
    EnemyInfo
        敵の輪郭・中心座標、餌の位置を含む検出結果。
    """
    state = detect_all_objects(bgr, self_mask)
    return dlo_state_to_enemy_info(state)


def detect_danger_zones(
    enemy_contours: list, margin: int = 30,
) -> list[np.ndarray]:
    """
    敵の輪郭に安全マージンを加えた危険エリアの輪郭リストを返す。

    Parameters
    ----------
    enemy_contours : list
        敵ヘビの輪郭リスト (cv2.findContours の出力)。
    margin : int
        安全マージン (ピクセル)。

    Returns
    -------
    list[np.ndarray]
        膨張した危険エリアの輪郭リスト。
    """
    danger_zones = []
    for contour in enemy_contours:
        # 輪郭を膨張させてマージンを追加
        hull = cv2.convexHull(contour)
        # 面積に基づいてバウンディングボックスを拡張
        x, y, w, h = cv2.boundingRect(hull)
        expanded = np.array([
            [[x - margin, y - margin]],
            [[x + w + margin, y - margin]],
            [[x + w + margin, y + h + margin]],
            [[x - margin, y + h + margin]],
        ], dtype=np.int32)
        danger_zones.append(expanded)
    return danger_zones
