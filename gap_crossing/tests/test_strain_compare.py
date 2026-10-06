"""Tests for shared strain-comparison statistics."""

import unittest

import numpy as np
import pandas as pd

from gap_crossing import strain_compare as compare


def make_recordings(rows):
    """Return recording rows with session fields."""
    return pd.DataFrame([
        {
            "source_file": source_file,
            "dataset_label": label,
            "recording_year": 2026,
            "recording_month": 8,
            "recording_day": day,
            "recording_experimenter": "kevin",
            "recording_vial": vial,
        }
        for source_file, label, day, vial in rows
    ])


class SessionTests(unittest.TestCase):
    """Check session grouping."""

    def test_same_day_and_vial_in_two_strains_are_separate_sessions(self):
        """Strain is part of the session key."""
        recordings = make_recordings([
            ("a1", "empty", 19, 0), ("a2", "empty", 19, 0), ("b1", "fc2", 19, 0),
        ])
        tracks = [{"source_file": name} for name in ("a1", "a2", "b1")]

        sessions = compare.group_tracks_by_session(tracks, recordings)

        self.assertEqual(
            {session_id: len(session_tracks) for session_id, (_, session_tracks) in sessions.items()},
            {"empty | 2026-08-19 | kevin | vial 0": 2, "fc2 | 2026-08-19 | kevin | vial 0": 1},
        )

    def test_reference_strain_comes_first(self):
        """The reference label leads the strain order."""
        self.assertEqual(compare.get_strain_order(["a", "b", "c"], "c"), ["c", "a", "b"])
        with self.assertRaises(ValueError):
            compare.get_strain_order(["a"], "missing")


class StatisticTests(unittest.TestCase):
    """Check bootstrap and permutation statistics."""

    def test_bootstrap_interval_contains_mean(self):
        """The interval brackets the session mean."""
        mean, low, high = compare.bootstrap_mean([1.0, 2.0, 3.0, 4.0, np.nan], np.random.default_rng(0), 2000)

        self.assertAlmostEqual(mean, 2.5)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_single_session_has_no_interval(self):
        """One session gives a mean without an interval."""
        mean, low, high = compare.bootstrap_mean([2.0], np.random.default_rng(0))

        self.assertEqual(mean, 2.0)
        self.assertTrue(np.isnan(low) and np.isnan(high))

    def test_permutation_separates_distinct_strains(self):
        """Identical strains give a large p value; separated strains a small one."""
        rng = np.random.default_rng(0)
        same = compare.permutation_difference([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], rng, 2000)
        apart = compare.permutation_difference([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], rng, 2000)

        self.assertAlmostEqual(same[0], 0.0)
        self.assertGreater(same[1], 0.9)
        self.assertAlmostEqual(apart[0], 10.0)
        self.assertLess(apart[1], 0.02)

    def test_summary_and_comparison_keep_group_columns(self):
        """Group columns survive in strain summaries and comparisons."""
        sessions = pd.DataFrame({
            "dataset_label": ["empty"] * 3 + ["fc2"] * 3,
            "metric": ["speed"] * 6,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        rng = np.random.default_rng(0)

        summary = compare.summarize_by_strain(sessions, "value", ["metric"], rng, 500)
        comparison = compare.compare_to_reference(sessions, "value", "empty", ["metric"], rng, 500)

        self.assertEqual(summary["mean"].tolist(), [2.0, 5.0])
        self.assertEqual(summary["session_count"].tolist(), [3, 3])
        self.assertEqual(comparison["metric"].tolist(), ["speed"])
        self.assertAlmostEqual(comparison["difference"].iloc[0], 3.0)

    def test_benjamini_hochberg_matches_hand_values(self):
        """q values follow the step-up rule and skip NaN p values."""
        q_values = compare.benjamini_hochberg([0.01, 0.04, np.nan, 0.03, 0.5])

        np.testing.assert_allclose(q_values, [0.04, 0.04 * 4 / 3, np.nan, 0.04 * 4 / 3, 0.5])

    def test_session_histogram_sums_to_one(self):
        """In-range values give bin fractions that sum to 1."""
        fractions = compare.session_histogram([0.5, 1.5, 1.5, np.nan], [0, 1, 2])

        np.testing.assert_allclose(fractions, [1 / 3, 2 / 3])


if __name__ == "__main__":
    unittest.main()
