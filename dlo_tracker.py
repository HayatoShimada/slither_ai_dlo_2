"""
DLO フレーム間追跡・速度推定・変形予測モジュール。
RT-DLO の時系列追跡に相当する機能を OpenCV ベースで実装する。

- フレーム間マッチング: 重心距離 + 骨格形状類似度 → Hungarian 法
- 速度推定: 重心差分 + 骨格点差分、指数移動平均で平滑化
- 変形予測: 現在の骨格 + skeleton_velocity で 1 ステップ先を線形外挿
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import DLO_MAX_LOST_FRAMES, DLO_VELOCITY_ALPHA, DLO_MATCH_MAX_DIST
from dlo_instance import DLOInstance


@dataclass
class _Track:
    """内部追跡状態。"""

    instance_id: int
    last_dlo: DLOInstance
    lost_frames: int = 0
    velocity_ema: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64),
    )
    skeleton_velocity_ema: np.ndarray | None = None


class DLOTracker:
    """DLO インスタンスのフレーム間追跡器。

    Parameters
    ----------
    max_lost_frames : int
        ID を消失させるまでの猶予フレーム数。
    velocity_alpha : float
        速度の指数移動平均係数 (0~1)。大きいほど最新値重視。
    match_max_dist : float
        マッチングの最大距離 (px)。これを超えるペアは対応付けない。
    """

    def __init__(
        self,
        max_lost_frames: int = DLO_MAX_LOST_FRAMES,
        velocity_alpha: float = DLO_VELOCITY_ALPHA,
        match_max_dist: float = DLO_MATCH_MAX_DIST,
    ) -> None:
        self._tracks: dict[int, _Track] = {}
        self._next_id: int = 0
        self._max_lost_frames = max_lost_frames
        self._velocity_alpha = velocity_alpha
        self._match_max_dist = match_max_dist

    def update(self, current_dlos: list[DLOInstance]) -> list[DLOInstance]:
        """新フレームの DLO を受け取り、ID 割当・速度計算済みの DLO を返す。

        Parameters
        ----------
        current_dlos : list[DLOInstance]
            現フレームで検出された DLO インスタンス（ID 未割当でよい）。

        Returns
        -------
        list[DLOInstance]
            ID が割り当てられ、velocity / skeleton_velocity が設定された DLO リスト。
        """
        if not self._tracks:
            # 初回: 全て新規 ID を発行
            result = []
            for dlo in current_dlos:
                track = self._create_track(dlo)
                dlo.instance_id = track.instance_id
                dlo.velocity = track.velocity_ema.copy()
                result.append(dlo)
            return result

        # コスト行列を構築
        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(current_dlos)

        if n_tracks == 0 and n_dets == 0:
            return []

        if n_dets == 0:
            # 全トラック失跡
            self._age_all_tracks()
            return []

        if n_tracks == 0:
            # 全て新規
            result = []
            for dlo in current_dlos:
                track = self._create_track(dlo)
                dlo.instance_id = track.instance_id
                dlo.velocity = track.velocity_ema.copy()
                result.append(dlo)
            return result

        cost_matrix = np.full((n_tracks, n_dets), 1e6, dtype=np.float64)
        for i, tid in enumerate(track_ids):
            prev = self._tracks[tid].last_dlo
            for j, det in enumerate(current_dlos):
                dist = _center_distance(prev, det)
                if dist < self._match_max_dist:
                    shape_cost = _shape_distance(prev.skeleton_yx, det.skeleton_yx)
                    cost_matrix[i, j] = dist + shape_cost * 0.5

        # Hungarian 法でマッチング
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        result: list[DLOInstance] = []

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= 1e6:
                continue
            tid = track_ids[r]
            track = self._tracks[tid]
            dlo = current_dlos[c]

            # 速度更新
            self._update_track_velocity(track, dlo)

            # DLO に追跡情報を反映
            dlo.instance_id = track.instance_id
            dlo.velocity = track.velocity_ema.copy()
            dlo.skeleton_velocity = (
                track.skeleton_velocity_ema.copy()
                if track.skeleton_velocity_ema is not None
                else None
            )

            track.last_dlo = dlo
            track.lost_frames = 0
            matched_tracks.add(tid)
            matched_dets.add(c)
            result.append(dlo)

        # 未マッチのトラック → lost_frames 加算、期限切れで削除
        for tid in track_ids:
            if tid not in matched_tracks:
                self._tracks[tid].lost_frames += 1
                if self._tracks[tid].lost_frames > self._max_lost_frames:
                    del self._tracks[tid]

        # 未マッチの検出 → 新規トラック
        for j, dlo in enumerate(current_dlos):
            if j not in matched_dets:
                track = self._create_track(dlo)
                dlo.instance_id = track.instance_id
                dlo.velocity = track.velocity_ema.copy()
                result.append(dlo)

        return result

    def predict_all(self) -> list[DLOInstance]:
        """全追跡中 DLO の次フレーム位置を線形外挿で予測する。

        Returns
        -------
        list[DLOInstance]
            予測された DLO インスタンスのリスト。instance_id は元と同じ。
        """
        predictions = []
        for track in self._tracks.values():
            if track.lost_frames > 0:
                continue
            pred = self._predict_next(track)
            if pred is not None:
                predictions.append(pred)
        return predictions

    def _create_track(self, dlo: DLOInstance) -> _Track:
        """新規トラックを作成して登録する。"""
        tid = self._next_id
        self._next_id += 1
        track = _Track(
            instance_id=tid,
            last_dlo=dlo,
        )
        self._tracks[tid] = track
        return track

    def _update_track_velocity(self, track: _Track, new_dlo: DLOInstance) -> None:
        """トラックの速度を指数移動平均で更新する。"""
        alpha = self._velocity_alpha
        prev = track.last_dlo

        # 重心速度
        d_center = new_dlo.center - prev.center
        track.velocity_ema = alpha * d_center + (1 - alpha) * track.velocity_ema

        # 骨格点速度
        if prev.skeleton_yx is not None and new_dlo.skeleton_yx is not None:
            n_new = len(new_dlo.skeleton_yx)
            if len(prev.skeleton_yx) == n_new:
                d_skel = (new_dlo.skeleton_yx - prev.skeleton_yx).astype(np.float64)
            else:
                # 点数が変わった場合はリサンプルして差分を計算
                idx = np.linspace(0, len(prev.skeleton_yx) - 1, n_new, dtype=int)
                d_skel = (new_dlo.skeleton_yx - prev.skeleton_yx[idx]).astype(np.float64)

            if (
                track.skeleton_velocity_ema is None
                or len(track.skeleton_velocity_ema) != n_new
            ):
                track.skeleton_velocity_ema = d_skel
            else:
                track.skeleton_velocity_ema = (
                    alpha * d_skel + (1 - alpha) * track.skeleton_velocity_ema
                )

    def _age_all_tracks(self) -> None:
        """全トラックの lost_frames を加算し、期限切れを削除する。"""
        to_delete = []
        for tid, track in self._tracks.items():
            track.lost_frames += 1
            if track.lost_frames > self._max_lost_frames:
                to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]

    def _predict_next(self, track: _Track) -> DLOInstance | None:
        """1 ステップ先の DLO 位置を線形外挿で予測する。"""
        dlo = track.last_dlo
        if dlo.skeleton_yx is None or len(dlo.skeleton_yx) < 2:
            return None

        # 骨格点の予測
        if track.skeleton_velocity_ema is not None:
            pred_skel = dlo.skeleton_yx + track.skeleton_velocity_ema
        else:
            pred_skel = dlo.skeleton_yx.copy()

        # 重心の予測
        pred_center = dlo.center + track.velocity_ema

        from dlo_instance import compute_heading, compute_length

        return DLOInstance(
            instance_id=dlo.instance_id,
            skeleton_yx=pred_skel,
            heading=compute_heading(pred_skel),
            length=compute_length(pred_skel),
            center=pred_center,
            contour=None,
            is_self=dlo.is_self,
            velocity=track.velocity_ema.copy(),
            skeleton_velocity=track.skeleton_velocity_ema,
        )


def _center_distance(a: DLOInstance, b: DLOInstance) -> float:
    """2つの DLO の重心間ユークリッド距離を返す。"""
    return float(np.linalg.norm(a.center - b.center))


def _shape_distance(skel_a: np.ndarray, skel_b: np.ndarray) -> float:
    """骨格形状の類似度を簡易 Frechet 風に算出する。

    両骨格を同じ点数にリサンプルし、対応点間距離の平均を返す。
    """
    if skel_a is None or skel_b is None:
        return 0.0
    if len(skel_a) == 0 or len(skel_b) == 0:
        return 0.0

    # 同じ点数にリサンプル（少ない方に合わせる）
    n = min(len(skel_a), len(skel_b))
    idx_a = np.linspace(0, len(skel_a) - 1, n, dtype=int)
    idx_b = np.linspace(0, len(skel_b) - 1, n, dtype=int)
    a_pts = skel_a[idx_a].astype(np.float64)
    b_pts = skel_b[idx_b].astype(np.float64)

    # 重心を引いて形状のみで比較
    a_centered = a_pts - np.mean(a_pts, axis=0)
    b_centered = b_pts - np.mean(b_pts, axis=0)

    diffs = a_centered - b_centered
    return float(np.mean(np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)))
