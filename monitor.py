"""
認識モニタリングウィンドウモジュール。
VNC 経由で確認できる 2x2 グリッド表示で、認識状態をリアルタイム表示する。

DLO 統合: 敵パネルに骨格線・ID・速度矢印・予測位置を描画する。

レイアウト:
┌─────────────────┬─────────────────┐
│ 自機検出        │ 敵 DLO 検出     │
│ (緑=自機マスク) │ (赤=骨格,橙=予測)│
├─────────────────┼─────────────────┤
│ 統合オーバーレイ │ RL 状態表示     │
│ (全DLO + 餌)    │ (報酬,行動,学習) │
└─────────────────┴─────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from config import SCREEN_WIDTH, SCREEN_HEIGHT
from dlo_instance import DLOInstance, DLOState


# モニタパネルのサイズ（元画像の半分）
PANEL_W = SCREEN_WIDTH // 2
PANEL_H = SCREEN_HEIGHT // 2


@dataclass
class RLInfo:
    """RL 状態情報を保持するデータクラス。"""

    reward: float = 0.0
    total_reward: float = 0.0
    action_angle: float = 0.0
    action_boost: bool = False
    step_count: int = 0
    episode_count: int = 0
    reward_history: list[float] = field(default_factory=list)
    human_mode: bool = False


def _resize_panel(image: np.ndarray) -> np.ndarray:
    """画像をパネルサイズにリサイズする。"""
    return cv2.resize(image, (PANEL_W, PANEL_H))


def _draw_dlo_skeleton(
    panel: np.ndarray,
    dlo: DLOInstance,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """DLO の骨格線をパネルに描画する（in-place）。"""
    if dlo.skeleton_yx is None or len(dlo.skeleton_yx) < 2:
        return
    pts = dlo.skeleton_yx.astype(np.int32)
    for i in range(len(pts) - 1):
        cv2.line(
            panel,
            (pts[i, 1], pts[i, 0]),
            (pts[i + 1, 1], pts[i + 1, 0]),
            color, thickness,
        )


def _draw_dlo_id_label(
    panel: np.ndarray,
    dlo: DLOInstance,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """DLO の ID ラベルを頭位置近くに描画する（in-place）。"""
    if dlo.skeleton_yx is None or len(dlo.skeleton_yx) == 0:
        return
    head = dlo.skeleton_yx[0].astype(int)
    cv2.putText(
        panel,
        f"#{dlo.instance_id}",
        (head[1] + 5, head[0] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
    )


def _draw_velocity_arrow(
    panel: np.ndarray,
    dlo: DLOInstance,
    color: tuple[int, int, int] = (0, 200, 200),
    scale: float = 5.0,
) -> None:
    """DLO の速度ベクトルを矢印で描画する（in-place）。"""
    if dlo.velocity is None or np.linalg.norm(dlo.velocity) < 0.5:
        return
    cx, cy = int(dlo.center[0]), int(dlo.center[1])
    dx, dy = dlo.velocity[0] * scale, dlo.velocity[1] * scale
    cv2.arrowedLine(
        panel,
        (cx, cy),
        (int(cx + dx), int(cy + dy)),
        color,
        2,
        tipLength=0.3,
    )


def _draw_self_panel(
    frame: np.ndarray,
    self_mask: np.ndarray | None,
    skeleton_yx: np.ndarray | None,
) -> np.ndarray:
    """パネル1: 自機マスク（緑）+ 骨格線（青）。"""
    panel = frame.copy()

    if self_mask is not None:
        green_overlay = np.zeros_like(panel)
        green_overlay[:, :, 1] = self_mask  # 緑チャネル
        panel = cv2.addWeighted(panel, 0.7, green_overlay, 0.3, 0)

    if skeleton_yx is not None and len(skeleton_yx) >= 2:
        pts = skeleton_yx.astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(
                panel,
                (pts[i, 1], pts[i, 0]),
                (pts[i + 1, 1], pts[i + 1, 0]),
                (255, 100, 0), 2,
            )
        cv2.circle(panel, (pts[0, 1], pts[0, 0]), 5, (0, 255, 0), -1)

    cv2.putText(panel, "Self Snake", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return _resize_panel(panel)


def _draw_enemy_panel(
    frame: np.ndarray,
    enemy_dlos: list[DLOInstance],
    predicted_dlos: list[DLOInstance],
    food_positions: np.ndarray | None,
) -> np.ndarray:
    """パネル2: 敵 DLO 骨格（赤）+ 予測位置（橙）+ ID + 速度矢印 + 餌（黄丸）。"""
    panel = frame.copy()

    # 予測位置（橙、破線風に thin で描画）
    for pred in predicted_dlos:
        if pred.is_self:
            continue
        _draw_dlo_skeleton(panel, pred, color=(0, 140, 255), thickness=1)

    # 現在位置（赤）
    for dlo in enemy_dlos:
        _draw_dlo_skeleton(panel, dlo, color=(0, 0, 255), thickness=2)
        # 頭を丸で強調
        if dlo.skeleton_yx is not None and len(dlo.skeleton_yx) > 0:
            head = dlo.skeleton_yx[0].astype(int)
            cv2.circle(panel, (head[1], head[0]), 4, (0, 0, 255), -1)
        _draw_dlo_id_label(panel, dlo, color=(0, 0, 255))
        _draw_velocity_arrow(panel, dlo)

    # 餌（黄丸）
    if food_positions is not None and len(food_positions) > 0:
        for pos in food_positions:
            cv2.circle(panel, (int(pos[0]), int(pos[1])), 4, (0, 255, 255), -1)

    cv2.putText(panel, "Enemy DLOs & Food", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return _resize_panel(panel)


def _draw_overlay_panel(
    frame: np.ndarray,
    self_mask: np.ndarray | None,
    skeleton_yx: np.ndarray | None,
    enemy_dlos: list[DLOInstance],
    predicted_dlos: list[DLOInstance],
    food_positions: np.ndarray | None,
) -> np.ndarray:
    """パネル3: 自機骨格 + 全敵骨格 + 予測骨格 + 餌。"""
    panel = frame.copy()

    # 自機マスク（緑半透明）
    if self_mask is not None:
        green_overlay = np.zeros_like(panel)
        green_overlay[:, :, 1] = self_mask
        panel = cv2.addWeighted(panel, 0.7, green_overlay, 0.3, 0)

    # 自機骨格（青）
    if skeleton_yx is not None and len(skeleton_yx) >= 2:
        pts = skeleton_yx.astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(
                panel,
                (pts[i, 1], pts[i, 0]),
                (pts[i + 1, 1], pts[i + 1, 0]),
                (255, 100, 0), 2,
            )

    # 敵予測骨格（橙）
    for pred in predicted_dlos:
        if pred.is_self:
            continue
        _draw_dlo_skeleton(panel, pred, color=(0, 140, 255), thickness=1)

    # 敵現在骨格（赤）
    for dlo in enemy_dlos:
        _draw_dlo_skeleton(panel, dlo, color=(0, 0, 255), thickness=2)
        _draw_velocity_arrow(panel, dlo, scale=3.0)

    # 餌（黄）
    if food_positions is not None and len(food_positions) > 0:
        for pos in food_positions:
            cv2.circle(panel, (int(pos[0]), int(pos[1])), 4, (0, 255, 255), -1)

    cv2.putText(panel, "Combined DLO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return _resize_panel(panel)


def _draw_rl_panel(rl_info: RLInfo) -> np.ndarray:
    """パネル4: RL の報酬推移、現在の行動、学習ステップ数。"""
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    y = 30
    line_h = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (200, 200, 200)

    cv2.putText(panel, "RL Status", (10, y), font, 0.6, (0, 200, 255), 2)
    y += line_h + 5

    # モード表示
    if rl_info.human_mode:
        cv2.putText(panel, "Mode: HUMAN", (10, y), font, scale, (0, 255, 0), 2)
    else:
        cv2.putText(panel, "Mode: AI", (10, y), font, scale, (0, 200, 255), 1)
    y += line_h

    cv2.putText(panel, f"Episode: {rl_info.episode_count}", (10, y), font, scale, color, 1)
    y += line_h
    cv2.putText(panel, f"Step: {rl_info.step_count}", (10, y), font, scale, color, 1)
    y += line_h
    cv2.putText(panel, f"Reward: {rl_info.reward:+.2f}", (10, y), font, scale, color, 1)
    y += line_h
    cv2.putText(panel, f"Total: {rl_info.total_reward:+.1f}", (10, y), font, scale, color, 1)
    y += line_h
    cv2.putText(
        panel,
        f"Action: {rl_info.action_angle:.0f}deg {'BOOST' if rl_info.action_boost else ''}",
        (10, y), font, scale, color, 1,
    )
    y += line_h + 10

    # 報酬推移グラフ（簡易折れ線）
    history = rl_info.reward_history
    if len(history) > 1:
        graph_x = 10
        graph_y = y
        graph_w = PANEL_W - 20
        graph_h = PANEL_H - y - 10

        if graph_h > 20:
            cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (50, 50, 50), 1)

            # 直近 200 ステップ分を表示
            recent = history[-200:]
            min_r = min(recent)
            max_r = max(recent)
            r_range = max_r - min_r if max_r != min_r else 1.0

            points = []
            for i, r in enumerate(recent):
                px = graph_x + int(i * graph_w / max(len(recent) - 1, 1))
                py = graph_y + graph_h - int((r - min_r) / r_range * graph_h)
                points.append((px, py))

            for i in range(len(points) - 1):
                cv2.line(panel, points[i], points[i + 1], (0, 200, 255), 1)

    return panel


def update_monitor(
    frame: np.ndarray,
    self_mask: np.ndarray | None,
    self_skeleton: np.ndarray | None,
    enemies: object | None,
    rl_info: RLInfo | None,
    dlo_state: DLOState | None = None,
    predicted_dlos: list[DLOInstance] | None = None,
) -> None:
    """
    認識モニタウィンドウを更新する。

    Parameters
    ----------
    frame : np.ndarray
        現在のキャプチャフレーム (BGR)。
    self_mask : np.ndarray or None
        自機ヘビのマスク (0/255)。
    self_skeleton : np.ndarray or None
        自機骨格座標 (N, 2) (y, x)。
    enemies : EnemyInfo or None
        敵・餌の検出結果（後方互換、dlo_state が None の場合に使用）。
    rl_info : RLInfo or None
        RL 状態情報。
    dlo_state : DLOState or None
        DLO ベースの検出結果。指定時は enemies より優先。
    predicted_dlos : list[DLOInstance] or None
        予測 DLO リスト。
    """
    preds = predicted_dlos if predicted_dlos is not None else []

    if dlo_state is not None:
        # DLO ベースの描画
        enemy_dlos = dlo_state.enemy_dlos
        food_positions = dlo_state.food_positions
    else:
        # 後方互換: EnemyInfo からフォールバック
        enemy_dlos = []
        food_positions = enemies.food_positions if enemies else None

    # 4 パネル生成
    p1 = _draw_self_panel(frame, self_mask, self_skeleton)
    p2 = _draw_enemy_panel(frame, enemy_dlos, preds, food_positions)
    p3 = _draw_overlay_panel(frame, self_mask, self_skeleton, enemy_dlos, preds, food_positions)
    p4 = _draw_rl_panel(rl_info if rl_info else RLInfo())

    # 2x2 グリッドに結合
    top = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    grid = np.vstack([top, bottom])

    cv2.imshow("Recognition Monitor", grid)
    cv2.waitKey(1)
