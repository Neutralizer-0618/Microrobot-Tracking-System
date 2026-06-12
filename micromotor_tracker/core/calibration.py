from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from micromotor_tracker.utils.geometry import line_length


@dataclass
class CalibrationResult:
    point_a: Tuple[float, float]
    point_b: Tuple[float, float]
    line_length_pixels: float
    real_length_um: float
    micron_per_pixel: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "point_a_x": self.point_a[0],
            "point_a_y": self.point_a[1],
            "point_b_x": self.point_b[0],
            "point_b_y": self.point_b[1],
            "line_length_pixels": self.line_length_pixels,
            "real_length_um": self.real_length_um,
            "micron_per_pixel": self.micron_per_pixel,
        }


def compute_calibration(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    real_length_um: float,
) -> CalibrationResult:
    pixels = line_length(point_a, point_b)
    if pixels <= 0:
        raise ValueError("Calibration line length must be greater than zero pixels.")
    if real_length_um <= 0:
        raise ValueError("Real-world calibration length must be greater than zero micrometers.")
    micron_per_pixel = real_length_um / pixels
    return CalibrationResult(
        point_a=point_a,
        point_b=point_b,
        line_length_pixels=pixels,
        real_length_um=float(real_length_um),
        micron_per_pixel=float(micron_per_pixel),
    )

