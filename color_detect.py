"""
自機ヘビのカラー自動検出モジュール。

ゲーム開始直後のフレームから画面中心付近の色相分布を分析し、
自機の HSV 範囲を動的に推定する。slither.io ではカメラが常に
自機の頭を画面中心に追従するため、中心 ROI 内の前景ピクセルは
高確率で自機の体色である。

検出に失敗した場合は config.py のデフォルト値にフォールバックする。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from config import (
    SNAKE_HSV_LOWER,
    SNAKE_HSV_UPPER,
    COLOR_DETECT_ROI_SIZE,
    COLOR_DETECT_HUE_MARGIN,
    COLOR_DETECT_MIN_FG_PIXELS,
)


def detect_snake_hsv(
    frame: np.ndarray,
    roi_size: int | None = None,
    hue_margin: int | None = None,
    min_fg_pixels: int | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """
    BGR フレームから自機ヘビの HSV 範囲を推定する。

    画面中心付近の ROI 内で背景を除外し、残った前景ピクセルの
    色相 (H) 分布からピーク hue を検出して HSV 範囲を構築する。

    Parameters
    ----------
    frame : np.ndarray
        BGR 画像 (H, W, 3)
    roi_size : int, optional
        中心 ROI の辺長 (px)。デフォルトは config の COLOR_DETECT_ROI_SIZE。
    hue_margin : int, optional
        ピーク hue からの許容幅。デフォルトは config の COLOR_DETECT_HUE_MARGIN。
    min_fg_pixels : int, optional
        前景ピクセルの最低数。これ未満なら検出失敗。

    Returns
    -------
    tuple[tuple, tuple] or None
        (hsv_lower, hsv_upper)。検出失敗時は None。
    """
    roi_size = roi_size or COLOR_DETECT_ROI_SIZE
    hue_margin = hue_margin or COLOR_DETECT_HUE_MARGIN
    min_fg_pixels = min_fg_pixels or COLOR_DETECT_MIN_FG_PIXELS

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half = roi_size // 2

    # ROI 切り出し
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    roi = frame[y1:y2, x1:x2]

    # HSV 変換
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 背景除外: 彩度と明度が低いピクセルは背景
    fg_mask = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 60)
    fg_hues = hsv[fg_mask, 0]

    if len(fg_hues) < min_fg_pixels:
        return None

    # H チャネルのヒストグラム (0-179, OpenCV の H 範囲)
    hist = cv2.calcHist([fg_hues.reshape(-1, 1)], [0], None, [180], [0, 180])
    hist = hist.flatten()

    # ピーク検出
    peak_hue = int(np.argmax(hist))

    # 赤の特別処理: H ≈ 0 or H ≈ 179 の場合
    # 赤はヒストグラムの両端にまたがるため、巡回的に扱う
    if peak_hue <= hue_margin or peak_hue >= 180 - hue_margin:
        # 赤系: 0 付近と 170-179 付近の合計で判定
        red_low_count = float(np.sum(hist[:hue_margin + 1]))
        red_high_count = float(np.sum(hist[180 - hue_margin:]))
        total_red = red_low_count + red_high_count

        # 赤ピクセルが全前景の 30% 以上なら赤と判定
        if total_red / len(fg_hues) > 0.3:
            # 赤は2つの範囲が必要だが、単一範囲で表現するため
            # H=0~margin と H=(180-margin)~180 をカバー
            # OpenCV の inRange は wrap しないので、lower > upper は不可
            # → lower=(0, S, V), upper=(margin, 255, 255) を返し、
            #   呼び出し側で2番目の範囲を追加する方式もあるが、
            #   ここでは広めの範囲で対応
            return (
                (0, 60, 60),
                (hue_margin, 255, 255),
            )

    # 通常色: ピーク ± マージン
    h_lower = max(0, peak_hue - hue_margin)
    h_upper = min(179, peak_hue + hue_margin)

    return (
        (h_lower, 60, 60),
        (h_upper, 255, 255),
    )


def auto_detect_snake_color(
    capture_fn,
    num_frames: int = 3,
    frame_interval: float = 0.3,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    複数フレームをキャプチャして安定した HSV 範囲を返す。

    Parameters
    ----------
    capture_fn : callable
        フレームをキャプチャする関数。引数なしで BGR ndarray を返す。
    num_frames : int
        キャプチャするフレーム数。
    frame_interval : float
        フレーム間の待機時間 (秒)。

    Returns
    -------
    tuple[tuple, tuple]
        (hsv_lower, hsv_upper)。検出失敗時は config のデフォルト値。
    """
    results: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []

    for i in range(num_frames):
        if i > 0:
            time.sleep(frame_interval)
        frame = capture_fn()
        result = detect_snake_hsv(frame)
        if result is not None:
            results.append(result)

    if not results:
        print("[color_detect] 自動検出失敗。デフォルト HSV を使用します。")
        return SNAKE_HSV_LOWER, SNAKE_HSV_UPPER

    # 複数フレームで得られた範囲を統合（最も広い包含範囲）
    all_h_lower = [r[0][0] for r in results]
    all_h_upper = [r[1][0] for r in results]

    h_lower = min(all_h_lower)
    h_upper = max(all_h_upper)

    # S, V は固定 (60, 60) ~ (255, 255)
    hsv_lower = (h_lower, 60, 60)
    hsv_upper = (h_upper, 255, 255)

    print(f"[color_detect] 自機カラー検出完了: HSV lower={hsv_lower}, upper={hsv_upper}")
    return hsv_lower, hsv_upper
