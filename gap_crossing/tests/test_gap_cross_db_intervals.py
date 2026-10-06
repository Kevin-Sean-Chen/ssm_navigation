"""Tests for consecutive gap-crossing decision intervals."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gap_crossing.memory_analysis import gap_cross_db_intervals as interval_analysis


class IntervalTableTests(unittest.TestCase):
    """Check interval and adjacent-pair construction within tracks."""

    def test_attempt_times_make_ordered_intervals_and_pairs(self):
        """Input row order does not change chronological interval pairs."""
        events = pd.DataFrame(
            {
                "track_id": ["a", "a", "a", "a"],
                "attempt_time_s": [15.0, 1.0, 7.0, 3.0],
            }
        )

        intervals = interval_analysis.make_interval_table(events)
        pairs = interval_analysis.make_adjacent_pair_table(intervals)

        np.testing.assert_allclose(intervals["interval_s"], [2.0, 4.0, 8.0])
        self.assertEqual(intervals["interval_number"].to_list(), [1, 2, 3])
        np.testing.assert_allclose(pairs["previous_interval_s"], [2.0, 4.0])
        np.testing.assert_allclose(pairs["next_interval_s"], [4.0, 8.0])

    def test_two_attempt_track_has_no_adjacent_pair(self):
        """One interval is not enough for an interval pair."""
        events = pd.DataFrame(
            {"track_id": ["a", "a"], "attempt_time_s": [1.0, 3.0]}
        )

        intervals = interval_analysis.make_interval_table(events)
        pairs = interval_analysis.make_adjacent_pair_table(intervals)

        self.assertEqual(len(intervals), 1)
        self.assertTrue(pairs.empty)

    def test_zero_interval_does_not_join_surrounding_intervals(self):
        """Filtering a zero interval does not create a new adjacency."""
        events = pd.DataFrame(
            {
                "track_id": ["a", "a", "a", "a"],
                "attempt_time_s": [1.0, 3.0, 3.0, 7.0],
            }
        )

        intervals = interval_analysis.make_interval_table(events)
        pairs = interval_analysis.make_adjacent_pair_table(intervals)

        self.assertEqual(intervals["interval_number"].to_list(), [1, 3])
        self.assertTrue(pairs.empty)

    def test_nonfinite_attempt_time_excludes_its_track(self):
        """An unknown attempt order cannot create an interval bridge."""
        events = pd.DataFrame(
            {
                "track_id": ["a", "a", "a", "a", "a"],
                "attempt_time_s": [1.0, 3.0, np.nan, 7.0, 15.0],
            }
        )

        intervals = interval_analysis.make_interval_table(events)
        pairs = interval_analysis.make_adjacent_pair_table(intervals)

        self.assertTrue(intervals.empty)
        self.assertTrue(pairs.empty)

    def test_missing_event_field_raises_clear_error(self):
        """Events without times cannot produce decision intervals."""
        events = pd.DataFrame({"track_id": ["a"]})

        with self.assertRaisesRegex(RuntimeError, "attempt_time_s"):
            interval_analysis.make_interval_table(events)


class IntervalCorrelationTests(unittest.TestCase):
    """Check observed and shuffled lag-1 correlations."""

    @staticmethod
    def make_positive_intervals():
        """Return two tracks with strong positive serial order."""
        return pd.DataFrame(
            {
                "track_id": ["a"] * 4 + ["b"] * 4,
                "interval_number": [1, 2, 3, 4] * 2,
                "interval_s": [1.0, 2.0, 4.0, 8.0, 1.5, 3.0, 6.0, 12.0],
            }
        )

    def test_shuffle_preserves_each_track_interval_multiset(self):
        """A shuffle changes order without moving values between tracks."""
        intervals = self.make_positive_intervals()

        first = interval_analysis.shuffle_intervals_within_tracks(
            intervals, np.random.default_rng(7)
        )
        second = interval_analysis.shuffle_intervals_within_tracks(
            intervals, np.random.default_rng(7)
        )

        pd.testing.assert_frame_equal(first, second)
        for track_id, original in intervals.groupby("track_id"):
            shuffled = first.loc[first["track_id"].eq(track_id)]
            self.assertEqual(
                shuffled["interval_number"].to_list(),
                original["interval_number"].to_list(),
            )
            np.testing.assert_allclose(
                np.sort(shuffled["interval_s"]),
                np.sort(original["interval_s"]),
            )

    def test_analysis_matches_observed_spearman_and_builds_null(self):
        """The result contains the observed rho and requested finite null."""
        intervals = self.make_positive_intervals()
        pairs = interval_analysis.make_adjacent_pair_table(intervals)
        expected_rho = spearmanr(
            pairs["previous_interval_s"], pairs["next_interval_s"]
        ).statistic

        result = interval_analysis.analyze_interval_correlation(
            intervals, permutation_count=50, random_seed=3
        )

        self.assertAlmostEqual(result.observed_rho, expected_rho)
        self.assertEqual(result.null_rho.shape, (50,))
        self.assertTrue(np.isfinite(result.null_rho).all())
        self.assertAlmostEqual(result.null_median_rho, np.median(result.null_rho))
        self.assertAlmostEqual(
            result.excess_rho, result.observed_rho - result.null_median_rho
        )

    def test_permutation_p_value_is_two_sided_around_null_median(self):
        """The finite-sample p value compares centered absolute effects."""
        result = interval_analysis.analyze_interval_correlation(
            self.make_positive_intervals(), permutation_count=40, random_seed=5
        )
        extreme_count = np.count_nonzero(
            np.abs(result.null_rho - result.null_median_rho)
            >= abs(result.observed_rho - result.null_median_rho)
        )

        self.assertEqual(result.p_value, (1 + extreme_count) / 41)
        self.assertGreater(result.p_value, 0)
        self.assertLessEqual(result.p_value, 1)

    def test_analysis_rejects_too_few_pairs(self):
        """Two pooled pairs are not enough for this first-pass test."""
        intervals = pd.DataFrame(
            {
                "track_id": ["a", "a", "a"],
                "interval_number": [1, 2, 3],
                "interval_s": [1.0, 2.0, 3.0],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "three adjacent interval pairs"):
            interval_analysis.analyze_interval_correlation(intervals)

    def test_analysis_rejects_constant_previous_intervals(self):
        """Constant previous values cannot define Spearman correlation."""
        intervals = pd.DataFrame(
            {
                "track_id": ["a", "a", "b", "b", "c", "c"],
                "interval_number": [1, 2] * 3,
                "interval_s": [1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "Previous intervals are constant"):
            interval_analysis.analyze_interval_correlation(intervals)

    def test_analysis_rejects_constant_next_intervals(self):
        """Constant next values cannot define Spearman correlation."""
        intervals = pd.DataFrame(
            {
                "track_id": ["a", "a", "b", "b", "c", "c"],
                "interval_number": [1, 2] * 3,
                "interval_s": [2.0, 1.0, 3.0, 1.0, 4.0, 1.0],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "Next intervals are constant"):
            interval_analysis.analyze_interval_correlation(intervals)

    def test_analysis_bounds_attempts_to_make_finite_null(self):
        """Repeated undefined shuffles stop with a clear error."""
        intervals = pd.DataFrame(
            {
                "track_id": ["a", "a", "b", "b", "c", "c"],
                "interval_number": [1, 2] * 3,
                "interval_s": [1.0, 2.0, 3.0, 1.0, 4.0, 1.0],
            }
        )

        class SortingGenerator:
            @staticmethod
            def permutation(values):
                return np.sort(values)

        with patch.object(
            interval_analysis.np.random,
            "default_rng",
            return_value=SortingGenerator(),
        ):
            with self.assertRaisesRegex(RuntimeError, "finite shuffled correlations"):
                interval_analysis.analyze_interval_correlation(
                    intervals, permutation_count=2
                )


class TrackIntervalCorrelationTests(unittest.TestCase):
    """Check lag-1 correlations for individual tracks."""

    def test_analysis_keeps_only_eligible_tracks(self):
        """Short and constant tracks do not produce track estimates."""
        intervals = pd.DataFrame(
            {
                "track_id": ["eligible"] * 11 + ["short"] * 10 + ["constant"] * 11,
                "interval_number": list(range(1, 12))
                + list(range(1, 11))
                + list(range(1, 12)),
                "interval_s": list(range(1, 12))
                + list(range(1, 11))
                + [2.0] * 11,
            }
        )

        result = interval_analysis.analyze_track_correlations(
            intervals,
            min_pair_count=10,
            permutation_count=20,
            random_seed=4,
        )

        self.assertEqual(result["track_id"].to_list(), ["eligible"])
        self.assertEqual(result.loc[0, "pair_count"], 10)
        self.assertAlmostEqual(result.loc[0, "observed_rho"], 1.0)
        self.assertTrue(np.isfinite(result.loc[0, "null_median_rho"]))
        self.assertAlmostEqual(
            result.loc[0, "excess_rho"],
            result.loc[0, "observed_rho"] - result.loc[0, "null_median_rho"],
        )

    def test_analysis_rejects_nonpositive_limits(self):
        """Invalid track-analysis limits stop before permutation."""
        intervals = IntervalCorrelationTests.make_positive_intervals()

        with self.assertRaisesRegex(ValueError, "Minimum pair count"):
            interval_analysis.analyze_track_correlations(
                intervals, min_pair_count=0
            )
        with self.assertRaisesRegex(ValueError, "Permutation count"):
            interval_analysis.analyze_track_correlations(
                intervals, permutation_count=0
            )


class IntervalPlotTests(unittest.TestCase):
    """Check the compact interval-correlation report."""

    def test_plot_shows_pairs_and_shuffled_null(self):
        """The figure contains pooled and individual-track comparisons."""
        intervals = IntervalCorrelationTests.make_positive_intervals()
        result = interval_analysis.analyze_interval_correlation(
            intervals, permutation_count=20, random_seed=2
        )
        track_results = pd.DataFrame(
            {
                "track_id": ["a", "b"],
                "pair_count": [10, 14],
                "observed_rho": [0.4, -0.1],
                "null_median_rho": [-0.1, -0.05],
                "excess_rho": [0.5, -0.05],
            }
        )

        figure = interval_analysis.plot_interval_correlation(
            intervals, result, track_results
        )

        self.assertEqual(len(figure.axes), 3)
        self.assertEqual(figure.axes[0].get_xscale(), "log")
        self.assertEqual(figure.axes[0].get_yscale(), "log")
        self.assertGreater(len(figure.axes[0].collections), 0)
        self.assertGreater(len(figure.axes[1].patches), 0)
        self.assertGreaterEqual(len(figure.axes[1].lines), 2)
        self.assertGreater(len(figure.axes[2].collections), 0)
        self.assertGreaterEqual(len(figure.axes[2].lines), 1)
        interval_analysis.analysis.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
