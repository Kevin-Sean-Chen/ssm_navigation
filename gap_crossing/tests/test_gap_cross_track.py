"""Tests for robust gap-boundary fitting."""

import unittest

import numpy as np
import pandas as pd

from gap_crossing.gap_cross_track import (
    get_cross_after_regain_by_attempt,
    get_plot_position_signal,
    get_regain_transition_durations,
    get_robust_gap_edges,
    make_second_order_transition_matrices,
)


class RobustGapBoundaryTests(unittest.TestCase):
    """Check that sparse signal points do not move a gap boundary."""

    def test_sparse_signal_inside_gap_does_not_move_edges(self):
        """One positive sample inside a well-supported gap is an outlier."""
        x_centers = np.arange(0.1, 10.0, 0.2)
        all_count = np.full(len(x_centers), 100)
        odor_count = all_count.copy()
        in_gap = (x_centers >= 4.0) & (x_centers < 6.0)
        odor_count[in_gap] = 0
        odor_count[np.flatnonzero(x_centers >= 5.0)[0]] = 1

        left_edge, right_edge = get_robust_gap_edges(
            x_centers,
            odor_count,
            all_count,
            gap_center_x_mm=5.0,
            search_half_window_mm=3.0,
            min_gap_width_mm=1.0,
            max_gap_width_mm=4.0,
        )

        self.assertAlmostEqual(left_edge, 4.0)
        self.assertAlmostEqual(right_edge, 6.0)


class PlotSampleTests(unittest.TestCase):
    """Check that the geometry plot uses a bounded, repeatable sample."""

    def test_default_plot_sample_shows_dense_bounded_data(self):
        """The default view shows 800 tracks with 400 frames each."""
        tracks = [
            {
                "track_id": f"track{track_id}",
                "xy": np.zeros((401, 2)),
                "signal": np.zeros(401),
            }
            for track_id in range(801)
        ]

        _, _, details = get_plot_position_signal(tracks)

        self.assertEqual(details["displayed_track_count"], 800)
        self.assertEqual(details["displayed_frame_count"], 320000)

    def test_plot_sample_limits_tracks_and_frames(self):
        """The sample includes evenly spaced tracks and frames."""
        tracks = [
            {
                "track_id": f"track{track_id}",
                "xy": np.column_stack((np.full(5, track_id), np.arange(5))),
                "signal": np.arange(5) % 2,
            }
            for track_id in range(3)
        ]

        xy, signal, details = get_plot_position_signal(
            tracks,
            max_tracks=2,
            max_frames_per_track=2,
        )

        self.assertEqual(len(xy), 4)
        self.assertEqual(len(signal), 4)
        self.assertEqual(details["raw_track_count"], 3)
        self.assertEqual(details["displayed_track_count"], 2)
        self.assertEqual(details["raw_frame_count"], 15)
        self.assertEqual(details["displayed_frame_count"], 4)
        self.assertEqual(details["track_ids"], ["track0", "track2"])
        np.testing.assert_array_equal(xy[:, 0], [0, 0, 2, 2])


class CrossAfterRegainTests(unittest.TestCase):
    """Check conditional crossing summaries by attempt number."""

    def test_summary_groups_current_attempts_after_regain(self):
        """Each eligible track contributes one binary cross outcome."""
        events = pd.DataFrame(
            {
                "track_id": ["a", "b", "c", "d"],
                "track_attempt": [2, 2, 2, 3],
                "previous_outcome": ["regain", "regain", "cross", "regain"],
                "is_cross": [1, 0, 1, 1],
            }
        )

        summary = get_cross_after_regain_by_attempt(events)

        self.assertEqual(summary["track_attempt"].to_list(), [2, 3])
        np.testing.assert_allclose(summary["mean_cross"], [0.5, 1.0])
        np.testing.assert_allclose(summary["std_cross"], [np.sqrt(0.5), 0.0])
        np.testing.assert_allclose(summary["sem_cross"], [0.5, 0.0])
        self.assertEqual(summary["track_count"].to_list(), [2, 1])


class RegainTransitionTimingTests(unittest.TestCase):
    """Check elapsed time from regain events to the next attempt."""

    def test_keeps_only_consecutive_attempts_after_regain(self):
        """Each duration is the next same-track attempt after a regain."""
        events = pd.DataFrame(
            {
                "track_id": ["a", "a", "a", "a", "a", "b", "b"],
                "attempt_time_s": [1.0, 4.0, 6.0, 8.0, 12.0, 2.0, 5.0],
                "outcome": ["regain", "cross", "regain", "regain", "abort", "cross", "regain"],
            }
        )

        durations = get_regain_transition_durations(events)

        self.assertEqual(
            durations["transition"].to_list(),
            ["regain → cross", "regain → regain", "regain → abort"],
        )
        np.testing.assert_allclose(durations["elapsed_s"], [3.0, 2.0, 4.0])


class SecondOrderTransitionTests(unittest.TestCase):
    """Check transition matrices conditioned on an earlier outcome."""

    def test_triplets_enter_the_matrix_for_their_two_back_outcome(self):
        """Each matrix keeps only pairs after its conditioning outcome."""
        events = pd.DataFrame(
            {
                "track_id": ["a", "a", "a", "a", "b", "b", "b"],
                "attempt_time_s": [1, 2, 3, 4, 1, 2, 3],
                "outcome": [
                    "cross", "regain", "abort", "cross",
                    "abort", "cross", "cross",
                ],
            }
        )

        matrices, counts = make_second_order_transition_matrices(events)

        self.assertEqual(counts, {"cross": 1, "regain": 1, "abort": 1})
        self.assertEqual(matrices["cross"].loc["abort", "regain"], 1.0)
        self.assertEqual(matrices["regain"].loc["cross", "abort"], 1.0)
        self.assertEqual(matrices["abort"].loc["cross", "cross"], 1.0)


if __name__ == "__main__":
    unittest.main()
