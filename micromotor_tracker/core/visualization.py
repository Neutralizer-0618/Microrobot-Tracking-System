from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _track_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(value) for value in rng.integers(50, 255, size=3))


def draw_detection_overlay(
    frame: np.ndarray,
    detections: pd.DataFrame,
    roi_box: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    canvas = frame.copy()
    for _, row in detections.iterrows():
        x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{row.get('confidence', 0.0):.2f}"
        cv2.putText(canvas, label, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    if roi_box is not None:
        x, y, w, h = roi_box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 200, 0), 2)
    return canvas


def draw_track_overlay(
    frame: np.ndarray,
    track_rows: pd.DataFrame,
    frame_index: int,
    roi_box: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    canvas = frame.copy()
    history = track_rows[track_rows["frame"] <= frame_index]
    current = track_rows[track_rows["frame"] == frame_index]
    for track_id, group in history.groupby("track_id"):
        color = _track_color(int(track_id))
        points = group[["x", "y"]].to_numpy(dtype=int)
        if len(points) >= 2:
            cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, color, 2)
    for _, row in current.iterrows():
        color = _track_color(int(row["track_id"]))
        center = (int(row["x"]), int(row["y"]))
        cv2.circle(canvas, center, 5, color, -1)
        cv2.putText(
            canvas,
            f"ID {int(row['track_id'])}",
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    if roi_box is not None:
        x, y, w, h = roi_box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 200, 0), 2)
    return canvas


def save_trajectory_overlay(base_frame: np.ndarray, track_rows: pd.DataFrame, output_path: str) -> str:
    canvas = base_frame.copy()
    for track_id, group in track_rows.groupby("track_id"):
        color = _track_color(int(track_id))
        points = group[["x", "y"]].to_numpy(dtype=int)
        if len(points) >= 2:
            cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, color, 2)
        if len(points) >= 1:
            cv2.putText(
                canvas,
                f"ID {int(track_id)}",
                tuple(points[-1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(output_path, canvas)
    return output_path


def save_speed_histogram(track_stats: pd.DataFrame, output_path: str) -> str:
    plt.figure(figsize=(7, 4))
    series = track_stats["mean_speed_um_s"].dropna()
    if not series.empty:
        plt.hist(series, bins=min(20, max(5, len(series))), color="#15616d", edgecolor="white")
        plt.xlabel("Mean speed (um/s)")
    else:
        plt.text(0.5, 0.5, "Calibration unavailable", ha="center", va="center")
    plt.ylabel("Track count")
    plt.title("Speed distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_speed_time_plot(speed_table: pd.DataFrame, output_path: str) -> str:
    plt.figure(figsize=(8, 4.5))
    for track_id, group in speed_table.groupby("track_id"):
        y = group["instantaneous_speed_um_s"] if group["instantaneous_speed_um_s"].notna().any() else group["instantaneous_speed_px_s"]
        plt.plot(group["time_s"], y, linewidth=1.2, alpha=0.8, label=f"Track {int(track_id)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed")
    plt.title("Per-track speed vs time")
    if speed_table["track_id"].nunique() <= 12:
        plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_msd_plot(msd_table: pd.DataFrame, output_path: str) -> Optional[str]:
    if msd_table.empty:
        return None
    plt.figure(figsize=(6.5, 4))
    plt.plot(msd_table["lag_time_s"], msd_table["msd"], color="#78290f", linewidth=2)
    plt.xlabel("Lag time (s)")
    plt.ylabel("MSD")
    plt.title("Mean squared displacement")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def write_annotated_video(
    video_source,
    track_rows: pd.DataFrame,
    output_path: str,
    fps: float,
    roi_box: Optional[tuple[int, int, int, int]] = None,
) -> str:
    width = video_source.metadata.width
    height = video_source.metadata.height
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame_index, frame in enumerate(video_source.iter_frames()):
            overlay = draw_track_overlay(frame, track_rows, frame_index, roi_box=roi_box)
            writer.write(overlay)
    finally:
        writer.release()
    return output_path

