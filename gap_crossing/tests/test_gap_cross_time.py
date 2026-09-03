"""Tests for temporal gap stopping states."""

import unittest

import numpy as np

from gap_crossing import gap_cross_time


class SustainedStopTests(unittest.TestCase):
    """Check stopping behavior after a global signal loss."""

    def test_speed_below_threshold_for_required_duration_is_a_stop(self):
        """A continuous low-speed interval after loss is a stop."""
        time_s = np.arange(0.0, 2.1, 1 / 60)
        speed_mm_s = np.full(len(time_s), 2.0)
        speed_mm_s[18:49] = 0.9

        self.assertTrue(hasattr(gap_cross_time, "has_sustained_stop"))
        self.assertTrue(
            gap_cross_time.has_sustained_stop(
                time_s, speed_mm_s, loss_time_s=0.0,
                threshold_mm_s=1.0, duration_s=0.5, window_s=2.0,
            )
        )

    def test_short_low_speed_period_is_not_a_stop(self):
        """A low-speed period shorter than the required duration is walking."""
        time_s = np.arange(0.0, 2.1, 1 / 60)
        speed_mm_s = np.full(len(time_s), 2.0)
        speed_mm_s[18:48] = 0.9

        self.assertTrue(hasattr(gap_cross_time, "has_sustained_stop"))
        self.assertFalse(
            gap_cross_time.has_sustained_stop(
                time_s, speed_mm_s, loss_time_s=0.0,
                threshold_mm_s=1.0, duration_s=0.5, window_s=2.0,
            )
        )

    def test_low_speed_before_loss_does_not_define_a_stop(self):
        """Only the post-loss window defines the stopping state."""
        time_s = np.arange(-1.0, 2.1, 1 / 60)
        speed_mm_s = np.full(len(time_s), 2.0)
        speed_mm_s[time_s < 0.0] = 0.9

        self.assertTrue(hasattr(gap_cross_time, "has_sustained_stop"))
        self.assertFalse(
            gap_cross_time.has_sustained_stop(
                time_s, speed_mm_s, loss_time_s=0.0,
                threshold_mm_s=1.0, duration_s=0.5, window_s=2.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
