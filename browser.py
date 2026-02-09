"""
Selenium 経由で Chromium を制御し、slither.io を自動操作するモジュール。
Docker コンテナ内の Xvfb に実描画する (headless=False)。
"""

from __future__ import annotations

import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import GAME_URL, NICKNAME, SCREEN_WIDTH, SCREEN_HEIGHT


def create_driver() -> webdriver.Chrome:
    """
    Chromium WebDriver を生成して返す。

    Xvfb 上に実描画するため headless=False。
    Docker コンテナ内で安全に動作するフラグを設定。

    Returns
    -------
    webdriver.Chrome
        設定済みの WebDriver インスタンス。
    """
    options = Options()
    options.binary_location = "/usr/bin/google-chrome-stable"
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={SCREEN_WIDTH},{SCREEN_HEIGHT}")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-gpu-sandbox")
    options.add_argument("--disable-extensions")

    # Selenium 4.6+ は chromedriver を自動ダウンロード・管理する
    driver = webdriver.Chrome(options=options)
    return driver


def start_game(driver: webdriver.Chrome) -> None:
    """
    slither.io に遷移し、ニックネーム入力 → Play ボタンクリックでゲームを開始する。

    Parameters
    ----------
    driver : webdriver.Chrome
        create_driver() で生成した WebDriver。
    """
    driver.get(GAME_URL)

    # ページ読み込み待機
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, "nick"))
    )
    time.sleep(1)  # アセット読み込み余裕

    # ニックネーム入力
    nick_input = driver.find_element(By.ID, "nick")
    nick_input.clear()
    nick_input.send_keys(NICKNAME)

    # Play ボタンクリック (CSS セレクタ → JS フォールバック)
    try:
        play_btn = driver.find_element(By.CSS_SELECTOR, ".play-btn, .btn-play, #playh")
        play_btn.click()
    except Exception:
        driver.execute_script(
            "var btn = document.querySelector('.play-btn') || "
            "document.querySelector('.btn-play') || "
            "document.getElementById('playh'); "
            "if(btn) btn.click(); else if(window.play) window.play();"
        )

    # ゲーム開始を待機
    for _ in range(60):
        if is_playing(driver):
            print("Game started successfully.")
            return
        time.sleep(0.5)

    print("WARNING: Game start not confirmed, proceeding anyway.")


def is_playing(driver: webdriver.Chrome) -> bool:
    """
    ゲームがプレイ中かどうかを JS で確認する。

    Returns
    -------
    bool
        プレイ中なら True。
    """
    try:
        return driver.execute_script("return window.playing || false;")
    except Exception:
        return False


def is_game_over(driver: webdriver.Chrome) -> bool:
    """
    ゲームオーバーかどうかを判定する。

    Returns
    -------
    bool
        ゲームオーバーなら True。
    """
    return not is_playing(driver)


def restart_game(driver: webdriver.Chrome) -> None:
    """
    ゲームオーバー後にページをフルリロードして再プレイする。
    ボタンクリックは不安定なため、常にフルリロードで確実にリスタートする。
    """
    print("Restarting game (full reload)...")
    start_game(driver)


def get_game_state(driver: webdriver.Chrome) -> dict:
    """
    1回の JS 呼び出しでスコア・マップ位置・境界比率をまとめて取得する。
    Selenium のラウンドトリップを最小化するためバッチ化。

    Returns
    -------
    dict
        {"score": int, "boundary_ratio": float, "playing": bool}
        取得失敗時はデフォルト値。
    """
    try:
        result = driver.execute_script("""
            var out = {score: 0, boundary_ratio: -1, playing: false};
            out.playing = !!(window.playing);

            var s = window.snake;
            if (!s) return out;

            // スコア: pts.length > fam > sct の優先順
            if (s.pts && s.pts.length > 0) out.score = s.pts.length;
            else if (s.fam && s.fam > 0) out.score = Math.floor(s.fam);
            else if (s.sct && s.sct > 0) out.score = s.sct;

            // マップ位置 → 境界比率
            var x = s.xx || s.x || 0;
            var y = s.yy || s.y || 0;
            if (x > 0 && y > 0) {
                var grd = window.grd || 21600;
                var dx = x - grd;
                var dy = y - grd;
                var dist = Math.sqrt(dx*dx + dy*dy);
                out.boundary_ratio = Math.min(dist / grd, 1.0);
            }
            return out;
        """)
        if result and isinstance(result, dict):
            return {
                "score": int(result.get("score", 0)),
                "boundary_ratio": float(result.get("boundary_ratio", -1.0)),
                "playing": bool(result.get("playing", False)),
            }
    except Exception:
        pass
    return {"score": 0, "boundary_ratio": -1.0, "playing": False}


def get_map_boundary_ratio(driver: webdriver.Chrome) -> float:
    """後方互換ラッパー。"""
    return get_game_state(driver)["boundary_ratio"]


def dump_snake_properties(driver: webdriver.Chrome) -> None:
    """window.snake の全プロパティをログに出力する（初回診断用）。"""
    try:
        result = driver.execute_script("""
            if (!window.snake) return 'window.snake is null/undefined';
            var props = {};
            for (var k in window.snake) {
                var v = window.snake[k];
                var t = typeof v;
                if (t === 'number' || t === 'boolean' || t === 'string') {
                    props[k] = v;
                } else if (Array.isArray(v)) {
                    props[k] = 'Array(' + v.length + ')';
                } else if (v === null) {
                    props[k] = 'null';
                } else {
                    props[k] = t;
                }
            }
            return JSON.stringify(props, null, 2);
        """)
        print(f"[DEBUG] window.snake properties:\n{result}")
    except Exception as e:
        print(f"[DEBUG] Failed to dump snake properties: {e}")


def get_score(driver: webdriver.Chrome) -> int:
    """後方互換ラッパー。"""
    return get_game_state(driver)["score"]
