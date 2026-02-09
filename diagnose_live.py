"""
ライブゲーム画面のDLO検出精度を診断するスクリプト。
Selenium経由でフレームをキャプチャし、検出パイプラインの各段階を可視化して保存する。
"""
from __future__ import annotations
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, MIN_SNAKE_AREA,
    BG_HSV_LOWER, BG_HSV_UPPER, ENEMY_MIN_AREA, FOOD_MAX_AREA,
    SHAPE_CIRCULARITY_THRESH, SHAPE_ASPECT_RATIO_THRESH,
)
from snake_skeleton import mask_snake_bgr, largest_connected_component
from enemy_detection import detect_all_objects, detect_background_mask, _classify_by_shape
from color_detect import auto_detect_snake_color


def capture_from_existing_chrome():
    """既存のChromeに接続してキャプチャする。"""
    options = Options()
    options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/google-chrome-stable")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={SCREEN_WIDTH},{SCREEN_HEIGHT}")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-gpu-sandbox")
    options.add_argument("--disable-extensions")
    driver = webdriver.Chrome(options=options)
    driver.get("http://slither.io")
    import time
    time.sleep(5)
    return driver


def diagnose_frame(frame: np.ndarray, hsv_lower, hsv_upper, save_dir="/tmp/diag"):
    """1フレームを詳細診断する。"""
    os.makedirs(save_dir, exist_ok=True)
    h, w = frame.shape[:2]

    # 1. 原画像を保存
    cv2.imwrite(f"{save_dir}/01_raw.png", frame)

    # 2. HSV変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3. 自機マスク
    self_mask = mask_snake_bgr(frame, hsv_lower, hsv_upper, hsv_img=hsv)
    self_mask_cc = largest_connected_component(self_mask, MIN_SNAKE_AREA)
    cv2.imwrite(f"{save_dir}/02_self_mask.png", self_mask)
    cv2.imwrite(f"{save_dir}/03_self_mask_cc.png", self_mask_cc)

    # 4. 背景マスク
    bg_mask = detect_background_mask(frame, hsv_img=hsv)
    cv2.imwrite(f"{save_dir}/04_bg_mask.png", bg_mask)

    # 5. 前景（背景でも自機でもない）
    foreground = cv2.bitwise_and(
        cv2.bitwise_not(bg_mask),
        cv2.bitwise_not(self_mask_cc),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"{save_dir}/05_foreground.png", foreground)

    # 6. 連結成分解析 + 分類
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    vis = frame.copy()
    enemy_count = 0
    food_count = 0
    noise_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx_i = int(centroids[i][0])
        cy_i = int(centroids[i][1])

        if area <= 10:
            noise_count += 1
            continue

        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] if contours else None

        # 形状分類
        shape_class = "ambiguous"
        circularity = 0.0
        aspect_ratio = 1.0
        if contour is not None and len(contour) >= 5:
            perimeter = cv2.arcLength(contour, closed=True)
            if perimeter > 0:
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if min(rw, rh) > 0:
                aspect_ratio = max(rw, rh) / min(rw, rh)
            shape_class = _classify_by_shape(contour, float(area))

        # 面積ベース分類
        area_class = "enemy" if area >= ENEMY_MIN_AREA else "food"

        # 最終分類
        if shape_class == "food":
            final = "food"
        elif shape_class == "enemy":
            final = "enemy"
        else:
            final = area_class

        # 描画
        if final == "enemy":
            color = (0, 0, 255)  # 赤
            enemy_count += 1
            label_text = f"E a={area} c={circularity:.2f} ar={aspect_ratio:.1f}"
        else:
            color = (0, 255, 255)  # 黄
            food_count += 1
            label_text = f"F a={area} c={circularity:.2f} ar={aspect_ratio:.1f}"

        if contour is not None:
            cv2.drawContours(vis, [contour], -1, color, 1)
        cv2.putText(vis, label_text, (cx_i - 50, cy_i - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # 自機を緑で表示
    self_contours, _ = cv2.findContours(self_mask_cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, self_contours, -1, (0, 255, 0), 2)

    cv2.imwrite(f"{save_dir}/06_classification.png", vis)

    # 7. 壁検出診断
    wall_vis = frame.copy()
    # 赤マスク
    mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = mask1 | mask2
    cv2.imwrite(f"{save_dir}/07_red_mask.png", red_mask)

    # エッジ領域の赤
    edge = int(min(h, w) * 0.15)
    edge_mask = np.zeros_like(red_mask)
    edge_mask[:edge, :] = red_mask[:edge, :]
    edge_mask[h-edge:, :] = red_mask[h-edge:, :]
    edge_mask[:, :edge] = red_mask[:, :edge]
    edge_mask[:, w-edge:] = red_mask[:, w-edge:]
    cv2.imwrite(f"{save_dir}/08_edge_red.png", edge_mask)

    total_edge = edge * w * 2 + edge * (h - 2 * edge) * 2
    red_count = int(np.count_nonzero(edge_mask))
    boundary_ratio = red_count / total_edge if total_edge > 0 else 0

    print(f"\n=== 診断結果 ===")
    print(f"画像サイズ: {w}x{h}")
    print(f"自機HSV: lower={hsv_lower}, upper={hsv_upper}")
    print(f"自機マスク面積: {np.sum(self_mask_cc > 0)} px")
    print(f"背景マスク面積: {np.sum(bg_mask > 0)} px ({np.sum(bg_mask > 0)*100/(h*w):.1f}%)")
    print(f"前景面積: {np.sum(foreground > 0)} px")
    print(f"連結成分数: {num_labels - 1}")
    print(f"  敵: {enemy_count}")
    print(f"  餌: {food_count}")
    print(f"  ノイズ(<10px): {noise_count}")
    print(f"壁検出:")
    print(f"  赤ピクセル総数: {np.sum(red_mask > 0)}")
    print(f"  エッジ赤ピクセル: {red_count}")
    print(f"  boundary_ratio: {boundary_ratio:.4f}")
    print(f"\n診断画像保存先: {save_dir}/")
    print(f"  01_raw.png - 原画像")
    print(f"  02_self_mask.png - 自機マスク(HSV)")
    print(f"  03_self_mask_cc.png - 自機マスク(最大連結成分)")
    print(f"  04_bg_mask.png - 背景マスク")
    print(f"  05_foreground.png - 前景(敵+餌)")
    print(f"  06_classification.png - 分類結果(赤=敵, 黄=餌, 緑=自機)")
    print(f"  07_red_mask.png - 赤色マスク(壁検出用)")
    print(f"  08_edge_red.png - エッジ赤色(壁判定対象)")


if __name__ == "__main__":
    import sys
    # 既存のSeleniumセッションを使う代わりに、直接画像ファイルから診断可能
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        frame = cv2.imread(sys.argv[1])
        hsv_lower = (35, 80, 80)
        hsv_upper = (85, 255, 255)
        diagnose_frame(frame, hsv_lower, hsv_upper)
    else:
        # Seleniumで新しいブラウザを起動
        driver = capture_from_existing_chrome()
        import time

        # ニックネーム入力してゲーム開始
        from browser import start_game
        # start_game はすでに呼んでいるので、フレームをキャプチャ
        time.sleep(2)

        def _capture():
            png = driver.get_screenshot_as_png()
            arr = np.frombuffer(png, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # 自機カラー検出
        hsv_lower, hsv_upper = auto_detect_snake_color(_capture)

        # 5フレーム取得して診断
        for i in range(5):
            frame = _capture()
            diagnose_frame(frame, hsv_lower, hsv_upper, f"/tmp/diag/frame_{i}")
            time.sleep(1)

        driver.quit()
