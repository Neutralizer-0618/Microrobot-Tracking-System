from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from micromotor_tracker.utils.config import TrackingConfig


@dataclass
class TrackState:
    track_id: int
    last_frame: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)

    def predict(self, frame_index: int) -> Tuple[float, float]:
        dt = max(frame_index - self.last_frame, 0)
        return self.x + self.vx * dt, self.y + self.vy * dt


def _distance_cost(active: List[TrackState], detections: pd.DataFrame, frame_index: int) -> np.ndarray:
    if not active or detections.empty:
        return np.empty((0, 0))
    cost = np.zeros((len(active), len(detections)), dtype=float)
    for i, track in enumerate(active):
        px, py = track.predict(frame_index)
        det_xy = detections[["x", "y"]].to_numpy(dtype=float)
        cost[i] = np.sqrt((det_xy[:, 0] - px) ** 2 + (det_xy[:, 1] - py) ** 2)
    return cost


def _append_row(track: TrackState, row: Dict[str, float]) -> None:
    track.history.append(row)
    track.last_frame = int(row["frame"])
    track.x = float(row["x"])
    track.y = float(row["y"])


def run_tracking(detections: pd.DataFrame, config: TrackingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detections.empty:
        return pd.DataFrame(), pd.DataFrame()

    detections = detections.sort_values("frame").reset_index(drop=True)
    active: Dict[int, TrackState] = {}
    completed: List[TrackState] = []
    next_track_id = 1

    for frame_index in sorted(detections["frame"].unique().tolist()):
        frame_detections = detections[detections["frame"] == frame_index].reset_index(drop=True)
        active_tracks = list(active.values())
        assigned_track_ids = set()
        assigned_detection_indices = set()

        if active_tracks and not frame_detections.empty:
            cost = _distance_cost(active_tracks, frame_detections, frame_index)
            rows, cols = linear_sum_assignment(cost)
            for row_index, col_index in zip(rows.tolist(), cols.tolist()):
                distance = cost[row_index, col_index]
                if distance > config.max_link_distance:
                    continue
                track = active_tracks[row_index]
                detection = frame_detections.iloc[col_index].to_dict()
                gap = int(frame_index - track.last_frame)
                if gap > 1:
                    prev_x, prev_y = track.x, track.y
                    target_x, target_y = float(detection["x"]), float(detection["y"])
                    for step in range(1, gap):
                        alpha = step / gap
                        interpolated = detection.copy()
                        interpolated["frame"] = int(track.last_frame + step)
                        interpolated["x"] = prev_x + (target_x - prev_x) * alpha
                        interpolated["y"] = prev_y + (target_y - prev_y) * alpha
                        interpolated["interpolated"] = True
                        interpolated["confidence"] = float(detection.get("confidence", 0.5))
                        interpolated["source_detection_id"] = detection.get("detection_id")
                        interpolated["track_id"] = track.track_id
                        _append_row(track, interpolated)
                dt = max(frame_index - track.last_frame, 1)
                detection["track_id"] = track.track_id
                detection["interpolated"] = False
                detection["source_detection_id"] = detection.get("detection_id")
                track.vx = (float(detection["x"]) - track.x) / dt
                track.vy = (float(detection["y"]) - track.y) / dt
                _append_row(track, detection)
                assigned_track_ids.add(track.track_id)
                assigned_detection_indices.add(col_index)

        for detection_index, detection in frame_detections.iterrows():
            if detection_index in assigned_detection_indices:
                continue
            det = detection.to_dict()
            track = TrackState(
                track_id=next_track_id,
                last_frame=int(det["frame"]),
                x=float(det["x"]),
                y=float(det["y"]),
            )
            det["track_id"] = next_track_id
            det["interpolated"] = False
            det["source_detection_id"] = det.get("detection_id")
            _append_row(track, det)
            active[next_track_id] = track
            assigned_track_ids.add(next_track_id)
            next_track_id += 1

        stale_ids = []
        for track_id, track in active.items():
            if frame_index - track.last_frame > config.max_frame_gap:
                completed.append(track)
                stale_ids.append(track_id)
        for track_id in stale_ids:
            active.pop(track_id, None)

    completed.extend(active.values())

    track_rows: List[Dict[str, float]] = []
    track_meta: List[Dict[str, float]] = []
    for track in completed:
        history = pd.DataFrame(track.history).sort_values("frame")
        non_interpolated = history[history["interpolated"] == False]  # noqa: E712
        if len(non_interpolated) < config.min_track_length:
            continue
        median_area = float(non_interpolated["area"].median()) if "area" in non_interpolated.columns else 0.0
        if median_area < config.min_object_area or median_area > config.max_object_area:
            continue
        interpolated_fraction = float(history["interpolated"].mean()) if not history.empty else 0.0
        mean_confidence = float(non_interpolated["confidence"].mean()) if "confidence" in non_interpolated.columns else 0.0
        low_confidence = mean_confidence < config.low_confidence_threshold or (
            interpolated_fraction > config.max_interpolated_fraction
        )
        history["track_id"] = track.track_id
        history["low_confidence_track"] = low_confidence
        history["mean_track_confidence"] = mean_confidence
        history["interpolated_fraction"] = interpolated_fraction
        track_rows.extend(history.to_dict(orient="records"))
        track_meta.append(
            {
                "track_id": track.track_id,
                "mean_track_confidence": mean_confidence,
                "interpolated_fraction": interpolated_fraction,
                "low_confidence_track": low_confidence,
                "median_area": median_area,
                "non_interpolated_frames": int(len(non_interpolated)),
            }
        )

    return (
        pd.DataFrame(track_rows).sort_values(["track_id", "frame"]).reset_index(drop=True) if track_rows else pd.DataFrame(),
        pd.DataFrame(track_meta).sort_values("track_id").reset_index(drop=True) if track_meta else pd.DataFrame(),
    )

