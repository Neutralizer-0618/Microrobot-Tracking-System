from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from micromotor_tracker.core.analysis import compute_msd
from micromotor_tracker.core.visualization import (
    save_msd_plot,
    save_speed_histogram,
    save_speed_time_plot,
    save_trajectory_overlay,
    write_annotated_video,
)


def create_output_dir(base_dir: str, source_name: str) -> Path:
    safe_name = Path(source_name).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{safe_name}_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_results(
    *,
    video_source,
    fps: float,
    micron_per_pixel: Optional[float],
    calibration_metadata: Dict,
    analysis_parameters: Dict,
    detections: pd.DataFrame,
    track_rows: pd.DataFrame,
    track_stats: pd.DataFrame,
    population_summary: pd.DataFrame,
    speed_table: pd.DataFrame,
    roi_box: Optional[tuple[int, int, int, int]],
    output_dir: str,
) -> Dict[str, Optional[str]]:
    target_dir = create_output_dir(output_dir, video_source.metadata.source_name)
    detections_path = target_dir / "per_frame_detections.csv"
    track_stats_path = target_dir / "per_track_statistics.csv"
    population_path = target_dir / "population_summary.csv"
    calibration_path = target_dir / "calibration_metadata.json"
    parameters_path = target_dir / "analysis_parameters.json"
    annotated_video_path = target_dir / "annotated_tracking_video.mp4"
    overlay_path = target_dir / "trajectory_overlay.png"
    histogram_path = target_dir / "speed_histogram.png"
    speed_time_path = target_dir / "per_track_speed_vs_time.png"
    msd_plot_path = target_dir / "msd_plot.png"

    detections.to_csv(detections_path, index=False)
    track_stats.to_csv(track_stats_path, index=False)
    population_summary.to_csv(population_path, index=False)
    speed_table.to_csv(target_dir / "per_track_speed_timeseries.csv", index=False)
    with calibration_path.open("w", encoding="utf-8") as handle:
        json.dump(calibration_metadata, handle, indent=2)
    with parameters_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis_parameters, handle, indent=2)

    base_frame = video_source.get_frame(0)
    write_annotated_video(video_source, track_rows, str(annotated_video_path), fps=fps, roi_box=roi_box)
    save_trajectory_overlay(base_frame, track_rows, str(overlay_path))
    save_speed_histogram(track_stats, str(histogram_path))
    save_speed_time_plot(speed_table, str(speed_time_path))

    msd_path = None
    msd_table = compute_msd(track_rows, fps=fps, micron_per_pixel=micron_per_pixel)
    if not msd_table.empty:
        msd_table.to_csv(target_dir / "msd_values.csv", index=False)
        msd_path = save_msd_plot(msd_table, str(msd_plot_path))

    return {
        "output_dir": str(target_dir),
        "per_frame_detections_csv": str(detections_path),
        "per_track_statistics_csv": str(track_stats_path),
        "population_summary_csv": str(population_path),
        "calibration_metadata_json": str(calibration_path),
        "analysis_parameters_json": str(parameters_path),
        "annotated_tracking_video_mp4": str(annotated_video_path),
        "trajectory_overlay_png": str(overlay_path),
        "speed_histogram_png": str(histogram_path),
        "per_track_speed_vs_time_png": str(speed_time_path),
        "msd_plot_png": msd_path,
    }
