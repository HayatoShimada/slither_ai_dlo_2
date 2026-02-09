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
    SHAPE_CIRCULARITY_THRESH,
    SHAPE_ASPECT_RATIO_THRESH,
    SHAPE_USE_FOR_CLASSIFICATION,
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


def _create_ui_mask(h: int, w: int) -> np.ndarray:
    """slither.io の UI 領域をマスクする (255=UIエリア)。

    Leaderboard (top-right), score text (bottom-left), minimap (bottom-right)
    をマスクして DLO 検出から除外する。
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    # Leaderboard (top-right): x>65%, y<30%
    mask[0:int(h * 0.30), int(w * 0.65):] = 255
    # Score text (bottom-left): x<25%, y>82%
    mask[int(h * 0.82):, 0:int(w * 0.25)] = 255
    # Minimap (bottom-right): x>78%, y>72%
    mask[int(h * 0.72):, int(w * 0.78):] = 255
    return mask


def detect_background_mask(
    bgr: np.ndarray,
    *,
    hsv_img: np.ndarray | None = None,
) -> np.ndarray:
    """
    slither.io の背景（暗い六角格子 + 壁外 + UI 領域）を検出する。

    slither.io の六角格子は彩度が高い暗青色 (H≈106, S≈143, V≈36)。
    壁外の赤い領域も暗い (V≈40)。主に明度 (V) で背景を判定する。
    食物やヘビは V>80 で光っているため区別できる。

    Parameters
    ----------
    bgr : np.ndarray
        BGR 画像 (H, W, 3)。
    hsv_img : np.ndarray | None
        事前計算済み HSV 画像。None の場合は内部で変換する。

    Returns
    -------
    np.ndarray
        背景マスク (0/255)。背景部分が 255。
    """
    hsv = hsv_img if hsv_img is not None else cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    # Layer 1: 暗いピクセル (V <= 60) → 六角格子 (V≈36) + 壁外領域 (V≈40)
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))

    # Layer 2: 低彩度・低明度 (従来のフォールバック)
    low_sat_mask = cv2.inRange(
        hsv,
        np.array(BG_HSV_LOWER, dtype=np.uint8),
        np.array(BG_HSV_UPPER, dtype=np.uint8),
    )

    # Layer 3: 壁境界のグローライン (H≈0/180 赤, S>220, V=60-150)
    # 壁のグロー線は S≈250 で非常に高彩度。通常のヘビ (S<220) とは区別可能。
    wall_glow1 = cv2.inRange(hsv, (0, 220, 60), (10, 255, 160))
    wall_glow2 = cv2.inRange(hsv, (170, 220, 60), (180, 255, 160))
    wall_glow_mask = wall_glow1 | wall_glow2

    # Layer 4: UI 領域マスク
    ui_mask = _create_ui_mask(h, w)

    mask = dark_mask | low_sat_mask | wall_glow_mask | ui_mask

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


def _classify_by_shape(contour: np.ndarray, area: float) -> str:
    """連結成分の形状特徴から敵ヘビか餌かを判定する。

    判定基準:
    - 円形度 (circularity): 4π × area / perimeter². 円=1.0, 細長い=低い値
    - アスペクト比: 最小外接矩形の長辺/短辺. ヘビ=高い, 餌=~1.0

    Parameters
    ----------
    contour : np.ndarray
        輪郭 (cv2.findContours の出力)。
    area : float
        面積 (px²)。

    Returns
    -------
    str
        "enemy", "food", or "ambiguous" (面積ベースにフォールバック)。
    """
    if contour is None or len(contour) < 5:
        return "ambiguous"

    # 円形度
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter > 0:
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0.0

    # アスペクト比（最小外接矩形）
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if min(w, h) > 0:
        aspect_ratio = max(w, h) / min(w, h)
    else:
        aspect_ratio = 1.0

    # 分類ロジック
    # 高円形度 & 低アスペクト比 → 餌
    if circularity > SHAPE_CIRCULARITY_THRESH and aspect_ratio < SHAPE_ASPECT_RATIO_THRESH:
        return "food"
    # 低円形度 or 高アスペクト比 → ヘビ
    if circularity < 0.3 or aspect_ratio > SHAPE_ASPECT_RATIO_THRESH:
        return "enemy"

    return "ambiguous"


def detect_all_objects(
    bgr: np.ndarray,
    self_mask: np.ndarray,
    self_skeleton_yx: np.ndarray | None = None,
    *,
    hsv_img: np.ndarray | None = None,
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
    hsv_img : np.ndarray | None
        事前計算済み HSV 画像。None の場合は内部で変換する。

    Returns
    -------
    DLOState
        全 DLO インスタンス + 餌の座標。
    """
    bg_mask = detect_background_mask(bgr, hsv_img=hsv_img)

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

        if area <= 10:  # 極小ノイズは無視
            continue

        # マスク・輪郭を先に計算（形状分類と敵DLO構築の両方で使う）
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = contours[0] if contours else None

        # 面積による初期分類
        is_enemy_by_area = area >= ENEMY_MIN_AREA

        # 形状ベース分類（有効時）
        shape_class = "ambiguous"
        if SHAPE_USE_FOR_CLASSIFICATION and area > 30 and contour is not None:
            shape_class = _classify_by_shape(contour, float(area))

        # 最終分類: 形状判定 > 面積判定
        if shape_class == "food":
            is_enemy = False
        elif shape_class == "enemy":
            is_enemy = True
        else:
            is_enemy = is_enemy_by_area

        if is_enemy:

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
        else:
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
