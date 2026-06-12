from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from micromotor_tracker.utils.config import AnalysisConfig


def compute_track_statistics(
    track_rows: pd.DataFrame,
    fps: float,
    micron_per_pixel: Optional[float],
    analysis_config: AnalysisConfig,
    track_meta: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if track_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if fps <= 0:
        raise ValueError("FPS must be greater than zero for motility analysis.")

    per_track = []
    speed_rows = []
    for track_id, group in track_rows.sort_values(["track_id", "frame"]).groupby("track_id"):
        group = group.reset_index(drop=True)
        dx = group["x"].diff()
        dy = group["y"].diff()
        dt = group["frame"].diff() / fps
        distance_px = np.sqrt(dx.fillna(0) ** 2 + dy.fillna(0) ** 2)
        speed_px_s = distance_px / dt.replace(0, np.nan)
        speed_px_s = speed_px_s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if micron_per_pixel:
            distance_um = distance_px * micron_per_pixel
            speed_um_s = speed_px_s * micron_per_pixel
            net_displacement_um = float(
                np.sqrt((group["x"].iloc[-1] - group["x"].iloc[0]) ** 2 + (group["y"].iloc[-1] - group["y"].iloc[0]) ** 2)
                * micron_per_pixel
            )
        else:
            distance_um = pd.Series([np.nan] * len(group))
            speed_um_s = pd.Series([np.nan] * len(group))
            net_displacement_um = np.nan

        path_length_px = float(distance_px.sum())
        net_displacement_px = float(
            np.sqrt((group["x"].iloc[-1] - group["x"].iloc[0]) ** 2 + (group["y"].iloc[-1] - group["y"].iloc[0]) ** 2)
        )
        duration_s = float(max(group["frame"].iloc[-1] - group["frame"].iloc[0], 0) / fps)
        mean_speed_um_s = float(speed_um_s.mean()) if micron_per_pixel else np.nan
        mean_speed_px_s = float(speed_px_s.mean())
        net_rate_um_s = net_displacement_um / duration_s if micron_per_pixel and duration_s > 0 else np.nan
        if analysis_config.active_mode == "net_displacement_rate":
            active = bool(
                duration_s >= analysis_config.min_active_duration_s
                and np.isfinite(net_rate_um_s)
                and net_rate_um_s >= analysis_config.active_net_displacement_threshold_um_s
            )
        else:
            active = bool(
                duration_s >= analysis_config.min_active_duration_s
                and np.isfinite(mean_speed_um_s)
                and mean_speed_um_s >= analysis_config.active_speed_threshold_um_s
            )

        per_track.append(
            {
                "track_id": track_id,
                "frames_tracked": int(len(group)),
                "start_frame": int(group["frame"].iloc[0]),
                "end_frame": int(group["frame"].iloc[-1]),
                "track_duration_s": duration_s,
                "mean_speed_px_s": mean_speed_px_s,
                "median_speed_px_s": float(speed_px_s.median()),
                "max_speed_px_s": float(speed_px_s.max()),
                "mean_speed_um_s": mean_speed_um_s,
                "median_speed_um_s": float(speed_um_s.median()) if micron_per_pixel else np.nan,
                "max_speed_um_s": float(speed_um_s.max()) if micron_per_pixel else np.nan,
                "total_path_length_px": path_length_px,
                "total_path_length_um": float(distance_um.sum()) if micron_per_pixel else np.nan,
                "net_displacement_px": net_displacement_px,
                "net_displacement_um": net_displacement_um,
                "straightness": float(net_displacement_px / path_length_px) if path_length_px > 0 else 0.0,
                "directionality_ratio": float(net_displacement_px / path_length_px) if path_length_px > 0 else 0.0,
                "low_confidence_track": bool(group["low_confidence_track"].iloc[0]) if "low_confidence_track" in group else False,
                "mean_track_confidence": float(group["mean_track_confidence"].iloc[0]) if "mean_track_confidence" in group else np.nan,
                "interpolated_fraction": float(group["interpolated_fraction"].iloc[0]) if "interpolated_fraction" in group else 0.0,
                "active_motion": active,
                "active_metric_value_um_s": net_rate_um_s if analysis_config.active_mode == "net_displacement_rate" else mean_speed_um_s,
            }
        )

        group_speed = pd.DataFrame(
            {
                "track_id": track_id,
                "frame": group["frame"],
                "time_s": group["frame"] / fps,
                "instantaneous_speed_px_s": speed_px_s,
                "instantaneous_speed_um_s": speed_um_s,
            }
        )
        speed_rows.append(group_speed)

    track_stats = pd.DataFrame(per_track).sort_values("track_id").reset_index(drop=True)
    speed_table = pd.concat(speed_rows, ignore_index=True) if speed_rows else pd.DataFrame()
    population = compute_population_summary(track_stats, track_rows)
    if track_meta is not None and not track_meta.empty:
        track_stats = track_stats.merge(track_meta, on="track_id", how="left", suffixes=("", "_meta"))
    return track_stats, speed_table, population


def compute_population_summary(track_stats: pd.DataFrame, track_rows: pd.DataFrame) -> pd.DataFrame:
    if track_stats.empty:
        return pd.DataFrame(
            [
                {
                    "total_detections": 0,
                    "valid_tracks": 0,
                    "mean_speed_um_s": np.nan,
                    "median_speed_um_s": np.nan,
                    "speed_std_um_s": np.nan,
                    "moving_fraction": np.nan,
                    "inactive_fraction": np.nan,
                }
            ]
        )
    moving_fraction = float(track_stats["active_motion"].mean())
    inactive_fraction = 1.0 - moving_fraction
    return pd.DataFrame(
        [
            {
                "total_detections": int(len(track_rows)),
                "valid_tracks": int(track_stats["track_id"].nunique()),
                "mean_speed_um_s": float(track_stats["mean_speed_um_s"].mean()),
                "median_speed_um_s": float(track_stats["median_speed_um_s"].median()),
                "speed_std_um_s": float(track_stats["mean_speed_um_s"].std(ddof=0)),
                "moving_fraction": moving_fraction,
                "inactive_fraction": inactive_fraction,
                "low_confidence_tracks": int(track_stats["low_confidence_track"].sum()),
            }
        ]
    )


def compute_msd(track_rows: pd.DataFrame, fps: float, micron_per_pixel: Optional[float]) -> pd.DataFrame:
    if track_rows.empty or fps <= 0:
        return pd.DataFrame()
    values = []
    for track_id, group in track_rows.sort_values(["track_id", "frame"]).groupby("track_id"):
        coords = group[["x", "y"]].to_numpy(dtype=float)
        if len(coords) < 4:
            continue
        scale = micron_per_pixel if micron_per_pixel else 1.0
        coords = coords * scale
        for lag in range(1, len(coords) // 2 + 1):
            diffs = coords[lag:] - coords[:-lag]
            msd = float(np.mean(np.sum(diffs**2, axis=1)))
            values.append({"track_id": track_id, "lag_frames": lag, "lag_time_s": lag / fps, "msd": msd})
    if not values:
        return pd.DataFrame()
    msd_table = pd.DataFrame(values)
    return (
        msd_table.groupby(["lag_frames", "lag_time_s"], as_index=False)["msd"].mean().sort_values("lag_frames").reset_index(drop=True)
    )

