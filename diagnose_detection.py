"""
餌・敵検出の診断ツール。

連結成分ごとに分類結果（餌/敵/ノイズ）、面積、形状特徴、平均色を
可視化し、面積閾値による誤分類がないか検証する。

使い方:
  python main.py diag          -- ライブキャプチャで連続診断（'q' で終了）
  python main.py diag <image>  -- 保存画像で単発診断（結果を diag_out/ に保存）
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import (
    CAPTURE_MONITOR,
    SNAKE_HSV_LOWER,
    SNAKE_HSV_UPPER,
    AUTO_DETECT_COLOR,
    MIN_SNAKE_AREA,
    BG_HSV_LOWER,
    BG_HSV_UPPER,
    ENEMY_MIN_AREA,
    FOOD_MAX_AREA,
)
from capture import capture_screen
from snake_skeleton import mask_snake_bgr, largest_connected_component
from enemy_detection import detect_background_mask
from color_detect import auto_detect_snake_color


@dataclass
class ComponentInfo:
    """1つの連結成分の分析結果。"""

    label_id: int
    area: int
    cx: int
    cy: int
    bx: int
    by: int
    bw: int
    bh: int
    aspect_ratio: float          # max(w,h)/min(w,h)  細長い=大
    circularity: float           # 4π*area / perimeter²  円=1.0
    mean_hsv: tuple[float, float, float]
    classification: str          # "enemy", "food", "noise"
    is_borderline: bool          # 閾値 ±50% 以内

    @property
    def elongation(self) -> float:
        """bounding box の縦横比。1.0=正方形、大=細長い。"""
        return self.aspect_ratio


def analyze_frame(
    frame: np.ndarray,
    hsv_lower: tuple = SNAKE_HSV_LOWER,
    hsv_upper: tuple = SNAKE_HSV_UPPER,
) -> tuple[np.ndarray, list[ComponentInfo], dict]:
    """
    1フレームに対して餌/敵分類の診断を実行する。

    Returns
    -------
    tuple[np.ndarray, list[ComponentInfo], dict]
        (診断画像 2x2 grid, 全コンポーネント情報, 統計サマリ)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    # --- パイプライン再現 ---
    self_mask_raw = mask_snake_bgr(frame, hsv_lower, hsv_upper, hsv_img=hsv)
    self_mask = largest_connected_component(self_mask_raw, MIN_SNAKE_AREA)
    bg_mask = detect_background_mask(frame, hsv_img=hsv)

    foreground = cv2.bitwise_and(
        cv2.bitwise_not(bg_mask),
        cv2.bitwise_not(self_mask),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground, connectivity=8,
    )

    # --- 各コンポーネント分析 ---
    components: list[ComponentInfo] = []
    borderline_lo = int(ENEMY_MIN_AREA * 0.5)   # 150
    borderline_hi = int(ENEMY_MIN_AREA * 1.5)   # 450

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # 形状特徴
        mn = max(min(bw, bh), 1)
        mx = max(bw, bh)
        aspect_ratio = mx / mn

        # 真円度
        comp_mask = ((labels == i) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

        # 平均色
        pixels_hsv = hsv[comp_mask > 0]
        mean_h = float(np.mean(pixels_hsv[:, 0]))
        mean_s = float(np.mean(pixels_hsv[:, 1]))
        mean_v = float(np.mean(pixels_hsv[:, 2]))

        if area >= ENEMY_MIN_AREA:
            classification = "enemy"
        elif area > 10:
            classification = "food"
        else:
            classification = "noise"

        is_borderline = borderline_lo <= area <= borderline_hi

        components.append(ComponentInfo(
            label_id=i,
            area=area,
            cx=cx, cy=cy,
            bx=bx, by=by, bw=bw, bh=bh,
            aspect_ratio=aspect_ratio,
            circularity=circularity,
            mean_hsv=(mean_h, mean_s, mean_v),
            classification=classification,
            is_borderline=is_borderline,
        ))

    # --- 統計サマリ ---
    enemies = [c for c in components if c.classification == "enemy"]
    foods = [c for c in components if c.classification == "food"]
    borderlines = [c for c in components if c.is_borderline]
    noises = [c for c in components if c.classification == "noise"]

    stats_summary = {
        "enemy_count": len(enemies),
        "food_count": len(foods),
        "noise_count": len(noises),
        "borderline_count": len(borderlines),
        "enemy_areas": sorted([c.area for c in enemies]),
        "food_areas": sorted([c.area for c in foods]),
        "enemy_aspect_ratios": [round(c.aspect_ratio, 1) for c in enemies],
        "food_aspect_ratios": [round(c.aspect_ratio, 1) for c in foods],
        "enemy_circularity": [round(c.circularity, 2) for c in enemies],
        "food_circularity": [round(c.circularity, 2) for c in foods],
        "enemy_mean_hsv": [(round(c.mean_hsv[0]), round(c.mean_hsv[1]), round(c.mean_hsv[2])) for c in enemies],
        "food_mean_hsv": [(round(c.mean_hsv[0]), round(c.mean_hsv[1]), round(c.mean_hsv[2])) for c in foods],
    }

    # --- 可視化 ---
    diag_img = _draw_diagnostic(
        frame, self_mask, foreground, hsv, components, stats_summary, h, w,
    )

    return diag_img, components, stats_summary


def _draw_diagnostic(
    frame: np.ndarray,
    self_mask: np.ndarray,
    foreground: np.ndarray,
    hsv: np.ndarray,
    components: list[ComponentInfo],
    stats_summary: dict,
    h: int,
    w: int,
) -> np.ndarray:
    """4パネルの診断画像を生成する。"""

    # --- パネル1: 分類オーバーレイ ---
    p1 = frame.copy()

    # 自機マスク（緑半透明）
    green = np.zeros_like(p1)
    green[:, :, 1] = self_mask
    p1 = cv2.addWeighted(p1, 0.7, green, 0.3, 0)

    for c in components:
        if c.classification == "noise":
            continue

        if c.classification == "enemy":
            color = (0, 0, 255)    # 赤
        else:
            color = (0, 255, 255)  # 黄

        # ボーダーラインは太枠で強調
        thickness = 3 if c.is_borderline else 1
        cv2.rectangle(p1, (c.bx, c.by), (c.bx + c.bw, c.by + c.bh), color, thickness)

        # ラベル
        label = f"{'E' if c.classification == 'enemy' else 'F'} {c.area}"
        if c.is_borderline:
            label += " !"
        cv2.putText(p1, label, (c.cx + 5, c.cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        cv2.circle(p1, (c.cx, c.cy), 2, color, -1)

    # ボーダーライン警告
    bl = stats_summary["borderline_count"]
    if bl > 0:
        cv2.putText(p1, f"BORDERLINE: {bl} items (area {int(ENEMY_MIN_AREA*0.5)}-{int(ENEMY_MIN_AREA*1.5)})",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- パネル2: 成分カラーマップ + 形状情報 ---
    p2 = np.zeros((h, w, 3), dtype=np.uint8)

    for c in components:
        if c.classification == "enemy":
            color = (0, 0, 255)
        elif c.classification == "food":
            color = (0, 255, 255)
        else:
            color = (60, 60, 60)

        # ボーダーラインは白
        if c.is_borderline:
            color = (255, 255, 255)

        # 成分ピクセルを塗る（高速: bounding box 内で labelID チェック不要、全体でやる）
        cv2.rectangle(p2, (c.bx, c.by), (c.bx + c.bw, c.by + c.bh), color, -1 if c.area < 50 else 2)

        if c.classification != "noise":
            label = f"a={c.area} ar={c.aspect_ratio:.1f} ci={c.circularity:.2f}"
            cv2.putText(p2, label, (c.cx - 30, c.cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)

    # --- パネル3: 面積ヒストグラム ---
    p3 = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_area_histogram(p3, components, w, h)

    # --- パネル4: 形状散布図 (面積 vs 真円度) + 統計テキスト ---
    p4 = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_scatter_and_stats(p4, components, stats_summary, w, h)

    # リサイズ & 結合
    ph, pw = h // 2, w // 2
    panels = []
    titles = [
        "1: Classification (R=enemy Y=food White=borderline)",
        "2: Components + Shape",
        "3: Area Histogram",
        "4: Area vs Circularity + Stats",
    ]
    for panel, title in zip([p1, p2, p3, p4], titles):
        small = cv2.resize(panel, (pw, ph))
        cv2.putText(small, title, (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        panels.append(small)

    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    return np.vstack([top, bottom])


def _draw_area_histogram(panel: np.ndarray, components: list[ComponentInfo], w: int, h: int) -> None:
    """面積ヒストグラムを描画する。餌と敵を色分け。"""
    margin = 40
    gx, gy = margin, margin
    gw, gh = w - margin * 2, h - margin * 2

    if not components or gw < 10 or gh < 10:
        cv2.putText(panel, "No components detected", (gx, gy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return

    # 面積でビニング（0-1000, 50刻み）
    max_area = min(max(c.area for c in components if c.classification != "noise"), 2000)
    bin_size = max(max_area // 30, 10)
    n_bins = max_area // bin_size + 1

    food_hist = [0] * n_bins
    enemy_hist = [0] * n_bins

    for c in components:
        if c.classification == "noise":
            continue
        b = min(c.area // bin_size, n_bins - 1)
        if c.classification == "food":
            food_hist[b] += 1
        else:
            enemy_hist[b] += 1

    max_count = max(max(food_hist), max(enemy_hist), 1)

    # 軸
    cv2.line(panel, (gx, gy + gh), (gx + gw, gy + gh), (100, 100, 100), 1)
    cv2.line(panel, (gx, gy), (gx, gy + gh), (100, 100, 100), 1)

    bar_w = max(gw // n_bins - 1, 1)

    for i in range(n_bins):
        x = gx + i * (gw // n_bins)

        # 餌（黄色）
        fh = int(food_hist[i] / max_count * gh)
        if fh > 0:
            cv2.rectangle(panel, (x, gy + gh - fh), (x + bar_w, gy + gh), (0, 200, 200), -1)

        # 敵（赤）
        eh = int(enemy_hist[i] / max_count * gh)
        if eh > 0:
            cv2.rectangle(panel, (x, gy + gh - fh - eh), (x + bar_w, gy + gh - fh), (0, 0, 220), -1)

    # 閾値ライン
    threshold_x = gx + int(ENEMY_MIN_AREA / bin_size * (gw / n_bins))
    if gx < threshold_x < gx + gw:
        cv2.line(panel, (threshold_x, gy), (threshold_x, gy + gh), (0, 255, 0), 2)
        cv2.putText(panel, f"threshold={ENEMY_MIN_AREA}",
                    (threshold_x + 3, gy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # X 軸ラベル
    for i in range(0, n_bins, max(n_bins // 8, 1)):
        x = gx + i * (gw // n_bins)
        cv2.putText(panel, str(i * bin_size), (x, gy + gh + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    cv2.putText(panel, "area (px)", (gx + gw // 2 - 30, gy + gh + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(panel, "Y=food  R=enemy  |=threshold", (gx, gy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)


def _draw_scatter_and_stats(
    panel: np.ndarray,
    components: list[ComponentInfo],
    stats_summary: dict,
    w: int,
    h: int,
) -> None:
    """面積 vs 真円度の散布図 + 統計テキスト。"""
    # 左半分: 散布図
    margin = 40
    gx, gy = margin, margin
    gw = w // 2 - margin * 2
    gh = h - margin * 2

    non_noise = [c for c in components if c.classification != "noise"]

    if non_noise and gw > 10 and gh > 10:
        max_area = min(max(c.area for c in non_noise), 2000)

        cv2.line(panel, (gx, gy + gh), (gx + gw, gy + gh), (100, 100, 100), 1)
        cv2.line(panel, (gx, gy), (gx, gy + gh), (100, 100, 100), 1)

        # 閾値ライン（縦）
        thr_x = gx + int(ENEMY_MIN_AREA / max(max_area, 1) * gw)
        cv2.line(panel, (thr_x, gy), (thr_x, gy + gh), (0, 255, 0), 1)

        for c in non_noise:
            px = gx + int(min(c.area, max_area) / max(max_area, 1) * gw)
            py = gy + gh - int(min(c.circularity, 1.0) * gh)

            if c.classification == "enemy":
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
            if c.is_borderline:
                color = (255, 255, 255)
            cv2.circle(panel, (px, py), 3, color, -1)

        cv2.putText(panel, "area ->", (gx + gw // 2, gy + gh + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        cv2.putText(panel, "circ", (gx - 30, gy + gh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # 右半分: 統計テキスト
    tx = w // 2 + 10
    ty = 40
    lh = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = 0.35
    col = (200, 200, 200)

    def put(text: str, color=col) -> None:
        nonlocal ty
        cv2.putText(panel, text, (tx, ty), font, sc, color, 1)
        ty += lh

    put(f"Enemy: {stats_summary['enemy_count']}   "
        f"Food: {stats_summary['food_count']}   "
        f"Noise: {stats_summary['noise_count']}")
    put(f"Borderline ({int(ENEMY_MIN_AREA*0.5)}-{int(ENEMY_MIN_AREA*1.5)}px): "
        f"{stats_summary['borderline_count']}", (0, 0, 255) if stats_summary['borderline_count'] > 0 else col)

    ty += 5
    put("--- Enemy ---", (0, 0, 255))
    ea = stats_summary["enemy_areas"]
    if ea:
        put(f"  area: min={min(ea)} max={max(ea)} med={int(np.median(ea))}")
        ear = stats_summary["enemy_aspect_ratios"]
        put(f"  aspect_ratio: min={min(ear)} max={max(ear)} med={np.median(ear):.1f}")
        ecr = stats_summary["enemy_circularity"]
        put(f"  circularity: min={min(ecr)} max={max(ecr)} med={np.median(ecr):.2f}")
        ehsv = stats_summary["enemy_mean_hsv"]
        if ehsv:
            hs = [x[0] for x in ehsv]
            put(f"  hue: min={min(hs):.0f} max={max(hs):.0f} med={np.median(hs):.0f}")
    else:
        put("  (none)")

    ty += 5
    put("--- Food ---", (0, 255, 255))
    fa = stats_summary["food_areas"]
    if fa:
        put(f"  area: min={min(fa)} max={max(fa)} med={int(np.median(fa))}")
        far = stats_summary["food_aspect_ratios"]
        put(f"  aspect_ratio: min={min(far)} max={max(far)} med={np.median(far):.1f}")
        fcr = stats_summary["food_circularity"]
        put(f"  circularity: min={min(fcr)} max={max(fcr)} med={np.median(fcr):.2f}")
        fhsv = stats_summary["food_mean_hsv"]
        if fhsv:
            hs = [x[0] for x in fhsv]
            put(f"  hue: min={min(hs):.0f} max={max(hs):.0f} med={np.median(hs):.0f}")
    else:
        put("  (none)")

    ty += 5
    put("--- Confusion Risk ---", (255, 255, 255))
    # 真円度による判別可能性の分析
    ec = stats_summary["enemy_circularity"]
    fc = stats_summary["food_circularity"]
    if ec and fc:
        e_med = np.median(ec)
        f_med = np.median(fc)
        if f_med > e_med + 0.1:
            put(f"  food circ ({f_med:.2f}) > enemy ({e_med:.2f})", (0, 255, 0))
            put("  -> circularity can help distinguish")
        else:
            put(f"  food circ ({f_med:.2f}) ~ enemy ({e_med:.2f})", (0, 165, 255))
            put("  -> circularity overlap, area-only risky")

    # 面積の重なり分析
    if ea and fa:
        max_food = max(fa)
        min_enemy = min(ea)
        if max_food >= ENEMY_MIN_AREA * 0.7:
            put(f"  WARNING: large food ({max_food}px) near threshold", (0, 0, 255))
        if min_enemy <= ENEMY_MIN_AREA * 1.3:
            put(f"  WARNING: small enemy ({min_enemy}px) near threshold", (0, 0, 255))
        if max_food < ENEMY_MIN_AREA * 0.5 and min_enemy > ENEMY_MIN_AREA * 1.5:
            put("  OK: clear separation between food and enemy", (0, 255, 0))


def print_summary(stats_summary: dict) -> None:
    """統計サマリをコンソールに出力する。"""
    print("=" * 60)
    print("検出診断サマリ")
    print("=" * 60)
    print(f"  敵:  {stats_summary['enemy_count']} 個")
    print(f"  餌:  {stats_summary['food_count']} 個")
    print(f"  ノイズ: {stats_summary['noise_count']} 個")
    print(f"  ボーダーライン (面積 {int(ENEMY_MIN_AREA*0.5)}-{int(ENEMY_MIN_AREA*1.5)}): "
          f"{stats_summary['borderline_count']} 個")
    print()

    ea = stats_summary["enemy_areas"]
    fa = stats_summary["food_areas"]

    if ea:
        print(f"  敵 面積:  {min(ea)}-{max(ea)} (中央値 {int(np.median(ea))})")
        print(f"  敵 縦横比: {stats_summary['enemy_aspect_ratios']}")
        print(f"  敵 真円度: {stats_summary['enemy_circularity']}")
    if fa:
        print(f"  餌 面積:  {min(fa)}-{max(fa)} (中央値 {int(np.median(fa))})")
    if ea and fa:
        max_food = max(fa)
        min_enemy = min(ea)
        gap = min_enemy - max_food
        print(f"\n  閾値({ENEMY_MIN_AREA}) 周辺の間隙: 餌max={max_food}, 敵min={min_enemy}, gap={gap}")
        if gap < 50:
            print("  ⚠ 餌と敵の面積が閾値付近で近接 → 誤分類リスクあり")
        else:
            print("  ✓ 面積で明確に分離できている")

    print("=" * 60)


def run_diagnosis(image_path: str | None = None) -> None:
    """
    検出診断を実行する。

    Parameters
    ----------
    image_path : str | None
        画像ファイルパス。None の場合はライブキャプチャ。
    """
    # HSV 設定
    hsv_lower, hsv_upper = SNAKE_HSV_LOWER, SNAKE_HSV_UPPER

    if image_path is not None:
        # 保存画像で診断
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"画像を読み込めません: {image_path}")
            return

        diag_img, components, stats_summary = analyze_frame(frame, hsv_lower, hsv_upper)
        print_summary(stats_summary)

        os.makedirs("diag_out", exist_ok=True)
        out_path = "diag_out/diagnosis.png"
        cv2.imwrite(out_path, diag_img)
        print(f"診断画像を保存: {out_path}")

        # ウィンドウ表示も試みる
        try:
            cv2.imshow("Detection Diagnosis", diag_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass
        return

    # ライブキャプチャモード
    print("餌/敵 検出診断モード (ライブ)。'q' で終了、's' でスナップショット保存。")

    if AUTO_DETECT_COLOR:
        hsv_lower, hsv_upper = auto_detect_snake_color(
            lambda: capture_screen(CAPTURE_MONITOR),
        )

    frame_count = 0
    os.makedirs("diag_out", exist_ok=True)

    while True:
        frame = capture_screen(CAPTURE_MONITOR)
        diag_img, components, stats_summary = analyze_frame(frame, hsv_lower, hsv_upper)

        # 10フレームごとにコンソールログ
        if frame_count % 30 == 0:
            print_summary(stats_summary)

        cv2.imshow("Detection Diagnosis", diag_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = int(time.time())
            cv2.imwrite(f"diag_out/diag_{ts}.png", diag_img)
            cv2.imwrite(f"diag_out/raw_{ts}.png", frame)
            print(f"スナップショット保存: diag_out/diag_{ts}.png")

        frame_count += 1

    cv2.destroyAllWindows()
