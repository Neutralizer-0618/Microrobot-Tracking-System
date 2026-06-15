from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class DetectionConfig:
    threshold_method: str = "otsu"
    manual_threshold: int = 128
    adaptive_block_size: int = 31
    adaptive_c: int = 5
    blur_kernel: int = 5
    use_background_subtraction: bool = True
    background_blur_kernel: int = 31
    min_area: float = 20.0
    max_area: float = 5000.0
    min_circularity: float = 0.05
    max_circularity: float = 1.5
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0
    min_confidence: float = 0.2
    morph_open_iterations: int = 1
    morph_close_iterations: int = 1
    detector_mode: str = "contour"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackingConfig:
    max_link_distance: float = 40.0
    max_frame_gap: int = 3
    min_track_length: int = 5
    min_object_area: float = 20.0
    max_object_area: float = 5000.0
    low_confidence_threshold: float = 0.45
    max_interpolated_fraction: float = 0.4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisConfig:
    active_speed_threshold_um_s: float = 5.0
    min_active_duration_s: float = 1.0
    active_mode: str = "mean_speed"
    active_net_displacement_threshold_um_s: float = 5.0
    speed_window_frames: int = 20
    speed_bin_min_um_s: float = 0.0
    speed_bin_max_um_s: float = 50.0
    speed_bin_width_um_s: float = 10.0
    motility_speed_threshold_um_s: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationConfig:
    line_length_pixels: Optional[float] = None
    real_length_um: Optional[float] = None
    micron_per_pixel: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def serialize_parameters(
    detection: DetectionConfig,
    tracking: TrackingConfig,
    analysis: AnalysisConfig,
    calibration: Optional[CalibrationConfig] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "detection": detection.to_dict(),
        "tracking": tracking.to_dict(),
        "analysis": analysis.to_dict(),
    }
    if calibration:
        payload["calibration"] = calibration.to_dict()
    if extra:
        payload["extra"] = extra
    return payload
