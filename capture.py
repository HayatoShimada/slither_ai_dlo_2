"""
ブラウザ（または任意の画面）をリアルタイムでキャプチャするモジュール。
mss を使用して高速にスクリーンショットを取得する。
"""

from __future__ import annotations

import numpy as np

try:
    import mss
    import mss.tools
except ImportError:
    mss = None


def capture_screen(monitor: int | dict | None = None) -> np.ndarray:
    """
    画面をキャプチャし、BGR の numpy 配列で返す（OpenCV と互換）。

    Parameters
    ----------
    monitor : int or dict or None
        - None: 全画面を1枚でキャプチャ
        - 1, 2, ...: mss のモニター番号（1 がメイン）
        - dict: {"left", "top", "width", "height"} で領域指定

    Returns
    -------
    np.ndarray
        shape (H, W, 3), dtype uint8, BGR.
    """
    if mss is None:
        raise ImportError("mss をインストールしてください: pip install mss")

    with mss.mss() as sctx:
        if isinstance(monitor, dict):
            target = monitor
        elif isinstance(monitor, int):
            target = sctx.monitors[monitor]
        else:
            target = sctx.monitors[0]

        shot = sctx.grab(target)
        # mss は BGRA。OpenCV は BGR なので A を落とす。
        frame = np.array(shot)[:, :, :3]
        return frame


def list_monitors() -> list[dict]:
    """利用可能なモニター一覧を返す（デバッグ・設定用）。"""
    if mss is None:
        return []
    with mss.mss() as sctx:
        return list(sctx.monitors)
