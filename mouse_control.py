"""
pyautogui を使ったマウス制御モジュール。
Xvfb 上の仮想ディスプレイに対してカーソル移動・クリックを行う。
"""

from __future__ import annotations

import math

import pyautogui

from config import SCREEN_WIDTH, SCREEN_HEIGHT

# 安全装置を無効化（仮想ディスプレイなのでフェイルセーフ不要）
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01


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

    pyautogui.moveTo(target_x, target_y)


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
    x = max(0, min(SCREEN_WIDTH - 1, x))
    y = max(0, min(SCREEN_HEIGHT - 1, y))
    pyautogui.moveTo(x, y)


def boost(active: bool) -> None:
    """
    加速の ON/OFF を切り替える。マウスボタンの押下/解放で制御。

    Parameters
    ----------
    active : bool
        True で加速開始、False で加速終了。
    """
    if active:
        pyautogui.mouseDown()
    else:
        pyautogui.mouseUp()
