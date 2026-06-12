import unittest

from micromotor_tracker.core.calibration import compute_calibration


class CalibrationTests(unittest.TestCase):
    def test_compute_calibration(self):
        result = compute_calibration((0, 0), (3, 4), 10.0)
        self.assertEqual(round(result.line_length_pixels, 5), 5.0)
        self.assertEqual(round(result.micron_per_pixel, 5), 2.0)


if __name__ == "__main__":
    unittest.main()
