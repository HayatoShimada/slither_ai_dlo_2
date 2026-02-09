"""
起動中のXvfbからmssでスクリーンショットを取り、DLO検出精度を診断する。
使い方: DISPLAY=localhost:99 python diagnose_snapshot.py
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import sys

# mssコンテキストをリセット
import capture
capture._sctx = None

from capture import capture_screen
from config import (
    MIN_SNAKE_AREA, BG_HSV_LOWER, BG_HSV_UPPER,
    ENEMY_MIN_AREA, SHAPE_CIRCULARITY_THRESH, SHAPE_ASPECT_RATIO_THRESH,
)
from snake_skeleton import mask_snake_bgr, largest_connected_component
from enemy_detection import detect_background_mask, _classify_by_shape
from color_detect import auto_detect_snake_color


def diagnose_frame(frame, hsv_lower, hsv_upper, save_dir="/tmp/diag"):
    os.makedirs(save_dir, exist_ok=True)
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imwrite(f"{save_dir}/01_raw.png", frame)

    self_mask = mask_snake_bgr(frame, hsv_lower, hsv_upper, hsv_img=hsv)
    self_mask_cc = largest_connected_component(self_mask, MIN_SNAKE_AREA)
    cv2.imwrite(f"{save_dir}/02_self_mask.png", self_mask)
    cv2.imwrite(f"{save_dir}/03_self_mask_cc.png", self_mask_cc)

    bg_mask = detect_background_mask(frame, hsv_img=hsv)
    cv2.imwrite(f"{save_dir}/04_bg_mask.png", bg_mask)

    foreground = cv2.bitwise_and(cv2.bitwise_not(bg_mask), cv2.bitwise_not(self_mask_cc))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"{save_dir}/05_foreground.png", foreground)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    vis = frame.copy()
    enemy_count = food_count = noise_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx_i, cy_i = int(centroids[i][0]), int(centroids[i][1])
        if area <= 10:
            noise_count += 1
            continue

        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] if contours else None

        circularity = aspect_ratio = 0.0
        shape_class = "ambiguous"
        if contour is not None and len(contour) >= 5:
            perimeter = cv2.arcLength(contour, closed=True)
            circularity = 4.0 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 1.0
            shape_class = _classify_by_shape(contour, float(area))

        area_class = "enemy" if area >= ENEMY_MIN_AREA else "food"
        final = shape_class if shape_class != "ambiguous" else area_class

        if final == "enemy":
            color, enemy_count = (0, 0, 255), enemy_count + 1
        else:
            color, food_count = (0, 255, 255), food_count + 1

        label_text = f"{'E' if final=='enemy' else 'F'} a={area} c={circularity:.2f} ar={aspect_ratio:.1f} [{shape_class}]"
        if contour is not None:
            cv2.drawContours(vis, [contour], -1, color, 1)
        cv2.putText(vis, label_text, (cx_i - 80, cy_i - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    self_contours, _ = cv2.findContours(self_mask_cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, self_contours, -1, (0, 255, 0), 2)
    cv2.imwrite(f"{save_dir}/06_classification.png", vis)

    # 壁検出
    mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = mask1 | mask2
    cv2.imwrite(f"{save_dir}/07_red_mask.png", red_mask)

    edge = int(min(h, w) * 0.15)
    edge_mask = np.zeros_like(red_mask)
    edge_mask[:edge, :] = red_mask[:edge, :]
    edge_mask[h-edge:, :] = red_mask[h-edge:, :]
    edge_mask[:, :edge] = red_mask[:, :edge]
    edge_mask[:, w-edge:] = red_mask[:, w-edge:]
    cv2.imwrite(f"{save_dir}/08_edge_red.png", edge_mask)

    total_edge = 2 * edge * w + 2 * edge * (h - 2 * edge)
    red_count = int(np.count_nonzero(edge_mask))
    bnd = red_count / total_edge if total_edge > 0 else 0

    print(f"\n=== 診断: {save_dir} ===")
    print(f"自機: lower={hsv_lower} upper={hsv_upper} area={np.sum(self_mask_cc>0)}px")
    print(f"背景: {np.sum(bg_mask>0)*100/(h*w):.1f}%  前景: {np.sum(foreground>0)}px")
    print(f"検出: 敵={enemy_count} 餌={food_count} ノイズ={noise_count}")
    print(f"壁: 赤全体={np.sum(red_mask>0)} エッジ赤={red_count} ratio={bnd:.4f}")


if __name__ == "__main__":
    frame = capture_screen()
    # 自機カラー自動検出
    hsv_lower, hsv_upper = auto_detect_snake_color(capture_screen)
    for i in range(3):
        import time
        frame = capture_screen()
        diagnose_frame(frame, hsv_lower, hsv_upper, f"/tmp/diag/frame_{i}")
        time.sleep(1)
    print("\n完了。/tmp/diag/ の画像を確認してください。")
