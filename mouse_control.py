"""
JavaScript dispatchEvent 経由でブラウザ内のマウスを制御するモジュール。
execute_script() は JS コンテキスト内で完結するため、
ウィンドウのフォーカス奪取やシステムマウスへの影響が一切ない。

注: CDP Input.dispatchMouseEvent はブラウザを前面に持ってくる副作用がある。
"""

from __future__ import annotations

import math

from config import SCREEN_WIDTH, SCREEN_HEIGHT

_driver = None
_boost_active = False
_last_x: int = SCREEN_WIDTH // 2
_last_y: int = SCREEN_HEIGHT // 2


def set_driver(driver) -> None:
    """
    Selenium WebDriver を設定する。
    move_to_angle / boost 等の呼び出し前に必ず実行すること。

    Parameters
    ----------
    driver : webdriver.Chrome
        Selenium WebDriver インスタンス。
    """
    global _driver
    _driver = driver


def move_to_angle(angle_deg: float, distance: float = 200) -> None:
    """
    画面中心から指定角度・距離の位置にカーソルを移動する。

    Parameters
    ----------
    angle_deg : float
        移動方向の角度 (度)。0=右, 90=上, 180=左, 270=下。
    distance : float
        中心からの距離 (ピクセル)。
    """
    global _last_x, _last_y
    if _driver is None:
        return

    cx = SCREEN_WIDTH // 2
    cy = SCREEN_HEIGHT // 2

    rad = math.radians(angle_deg)
    # Y 軸はスクリーン座標で下向きが正なので反転
    dx = distance * math.cos(rad)
    dy = -distance * math.sin(rad)

    target_x = int(cx + dx)
    target_y = int(cy + dy)

    # 画面範囲にクランプ
    target_x = max(0, min(SCREEN_WIDTH - 1, target_x))
    target_y = max(0, min(SCREEN_HEIGHT - 1, target_y))

    _last_x, _last_y = target_x, target_y
    _dispatch_mouse_move(target_x, target_y)


def move_to_position(x: int, y: int) -> None:
    """
    絶対座標にカーソルを移動する。座標は画面範囲にクランプされる。

    Parameters
    ----------
    x : int
        X 座標。
    y : int
        Y 座標。
    """
    global _last_x, _last_y
    if _driver is None:
        return

    x = max(0, min(SCREEN_WIDTH - 1, x))
    y = max(0, min(SCREEN_HEIGHT - 1, y))
    _last_x, _last_y = x, y
    _dispatch_mouse_move(x, y)


def boost(active: bool) -> None:
    """
    加速の ON/OFF を切り替える。mousedown/mouseup で制御。

    Parameters
    ----------
    active : bool
        True で加速開始、False で加速終了。
    """
    global _boost_active
    if _driver is None:
        return
    if active == _boost_active:
        return
    _boost_active = active
    _dispatch_mouse_button(active, _last_x, _last_y)


def _dispatch_mouse_move(x: int, y: int) -> None:
    """JS dispatchEvent で mousemove をブラウザ内に送信する。

    1. slither.io 内部のマウス追跡変数 (xm/ym) を直接設定
    2. MouseEvent を canvas に dispatch（イベントリスナー用）
    execute_script はウィンドウフォーカスに影響しない。
    """
    try:
        _driver.execute_script(
            """
            window.xm = arguments[0];
            window.ym = arguments[1];
            var c = document.querySelector('canvas') || document.body;
            c.dispatchEvent(new MouseEvent('mousemove', {
                clientX: arguments[0], clientY: arguments[1],
                bubbles: true, cancelable: true, view: window
            }));
            """,
            x, y,
        )
    except Exception:
        pass


def _dispatch_mouse_button(pressed: bool, x: int, y: int) -> None:
    """JS dispatchEvent で mousedown/mouseup をブラウザ内に送信する。"""
    try:
        _driver.execute_script(
            """
            var type = arguments[0] ? 'mousedown' : 'mouseup';
            var c = document.querySelector('canvas') || document.body;
            c.dispatchEvent(new MouseEvent(type, {
                clientX: arguments[1], clientY: arguments[2],
                button: 0, buttons: arguments[0] ? 1 : 0,
                bubbles: true, cancelable: true, view: window
            }));
            """,
            pressed, x, y,
        )
    except Exception:
        pass
