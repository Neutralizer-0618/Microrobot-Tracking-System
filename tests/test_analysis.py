import unittest

import pandas as pd

from micromotor_tracker.core.analysis import compute_track_statistics
from micromotor_tracker.utils.config import AnalysisConfig


class AnalysisTests(unittest.TestCase):
    def test_compute_track_statistics_basic(self):
        track_rows = pd.DataFrame(
            [
                {"track_id": 1, "frame": 0, "x": 0.0, "y": 0.0, "low_confidence_track": False, "mean_track_confidence": 0.9, "interpolated_fraction": 0.0},
                {"track_id": 1, "frame": 1, "x": 3.0, "y": 4.0, "low_confidence_track": False, "mean_track_confidence": 0.9, "interpolated_fraction": 0.0},
                {"track_id": 1, "frame": 2, "x": 6.0, "y": 8.0, "low_confidence_track": False, "mean_track_confidence": 0.9, "interpolated_fraction": 0.0},
            ]
        )
        stats, speeds, population = compute_track_statistics(
            track_rows=track_rows,
            fps=1.0,
            micron_per_pixel=2.0,
            analysis_config=AnalysisConfig(active_speed_threshold_um_s=5.0, min_active_duration_s=1.0, speed_window_frames=2),
        )
        self.assertEqual(len(stats), 1)
        self.assertEqual(round(stats.loc[0, "mean_speed_px_s"], 2), 3.33)
        self.assertEqual(round(stats.loc[0, "net_displacement_um"], 2), 20.0)
        self.assertEqual(len(speeds), 1)
        self.assertEqual(round(speeds.loc[0, "speed_um_s"], 2), 10.0)
        self.assertEqual(population.loc[0, "valid_tracks"], 1)


if __name__ == "__main__":
    unittest.main()
