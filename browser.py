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
    options.binary_location = "/usr/bin/chromium-browser"
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={SCREEN_WIDTH},{SCREEN_HEIGHT}")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-gpu-sandbox")
    options.add_argument("--disable-extensions")

    service = Service("/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=options)
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
    time.sleep(2)  # アセット読み込み余裕

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
    ゲームオーバー後に再プレイする。

    Parameters
    ----------
    driver : webdriver.Chrome
        既存の WebDriver。
    """
    print("Restarting game...")
    time.sleep(2)

    # 再プレイボタンをクリック、または JS でリスタート
    try:
        driver.execute_script(
            "var btn = document.querySelector('.play-btn') || "
            "document.querySelector('.btn-play') || "
            "document.getElementById('playh'); "
            "if(btn) btn.click(); else if(window.play) window.play();"
        )
    except Exception:
        # ページをリロードして最初からやり直す
        start_game(driver)
        return

    for _ in range(60):
        if is_playing(driver):
            print("Game restarted successfully.")
            return
        time.sleep(0.5)

    # フォールバック: フルリロード
    print("Restart via button failed, reloading page...")
    start_game(driver)


def get_score(driver: webdriver.Chrome) -> int:
    """
    現在のスコア（ヘビの長さ）を JS で取得する。

    Returns
    -------
    int
        スコア値。取得失敗時は 0。
    """
    try:
        score = driver.execute_script(
            "return window.snake ? window.snake.sct : 0;"
        )
        return int(score) if score else 0
    except Exception:
        return 0
