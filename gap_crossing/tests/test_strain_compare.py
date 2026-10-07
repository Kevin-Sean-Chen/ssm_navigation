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


class BlockTests(unittest.TestCase):
    """Check trial blocks within sessions."""

    @staticmethod
    def make_trial_recordings(trials):
        """Return one session with the given trial numbers, one file per trial."""
        recordings = make_recordings([(f"t{trial}", "empty", 19, 0) for trial in trials])
        recordings["recording_trial"] = trials
        return recordings

    def test_blocks_follow_trial_order_and_skip_numbering_gaps(self):
        """Consecutive ranked trials form blocks even when trial numbers jump."""
        trials = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22]
        lookup = compare.make_unit_lookup(self.make_trial_recordings(trials), 5, 3)

        blocks = {trial: lookup[f"t{trial}"]["block"] for trial in trials}
        self.assertEqual([blocks[trial] for trial in trials], [1] * 5 + [2] * 5 + [3] * 4)
        self.assertEqual(lookup["t22"]["block_trials"], 4)
        self.assertEqual(lookup["t2"]["unit_id"], "empty | 2026-08-19 | kevin | vial 0 | trial block 1")

    def test_short_last_block_joins_the_previous_block(self):
        """17 trials in blocks of 5 give 5, 5, 7 and keep every trial."""
        trials = list(range(1, 18))
        recordings = self.make_trial_recordings(trials)
        tracks = [{"source_file": f"t{trial}"} for trial in trials]

        units = compare.group_tracks_by_unit(tracks, recordings, 5, 3)

        self.assertEqual(
            [(unit["block"], unit["block_trials"], len(unit_tracks)) for unit, unit_tracks in units.values()],
            [(1, 5, 5), (2, 5, 5), (3, 7, 7)],
        )

    def test_block_sizes_for_common_trial_counts(self):
        """Remainders below min_block_trials merge; others stay their own block."""
        expected = {
            15: [5, 5, 5],
            16: [5, 5, 6],
            17: [5, 5, 7],
            18: [5, 5, 5, 3],
            20: [5, 5, 5, 5],
            4: [4],
            2: [2],
        }
        for trial_count, sizes in expected.items():
            with self.subTest(trial_count=trial_count):
                blocks = compare.assign_trial_blocks(trial_count, 5, 3)
                self.assertEqual(np.bincount(blocks)[1:].tolist(), sizes)

    def test_session_mode_keeps_one_unit_per_session(self):
        """Without blocks, the unit is the session."""
        lookup = compare.make_unit_lookup(self.make_trial_recordings([1, 2, 3]))

        self.assertEqual({unit["unit_id"] for unit in lookup.values()}, {"empty | 2026-08-19 | kevin | vial 0"})
        self.assertEqual(lookup["t1"]["block_trials"], 3)


class ClusteredStatisticTests(unittest.TestCase):
    """Check statistics that treat sessions as the independent samples."""

    @staticmethod
    def make_blocked_sessions():
        """Return 2 + 2 sessions of 5 identical blocks; strains differ by session."""
        rows = []
        for label, session_means in [("empty", [1.0, 2.0]), ("fc2", [3.0, 4.0])]:
            for index, mean in enumerate(session_means):
                rows.extend(
                    {"dataset_label": label, "session_id": f"{label}{index}", "value": mean}
                    for _ in range(5)
                )
        return pd.DataFrame(rows)

    def test_blocks_do_not_shrink_the_permutation_p_value(self):
        """With 2 vs 2 sessions, whole-session shuffles cannot give p below 1/3."""
        units = self.make_blocked_sessions()
        rng = np.random.default_rng(0)
        reference = units.loc[units["dataset_label"] == "empty"]
        other = units.loc[units["dataset_label"] == "fc2"]

        naive = compare.permutation_difference(reference["value"], other["value"], rng, 2000)
        clustered = compare.permutation_difference_clustered(
            reference["value"], reference["session_id"], other["value"], other["session_id"], rng, 2000
        )

        self.assertAlmostEqual(clustered[0], 2.0)
        self.assertLess(naive[1], 0.01)
        self.assertGreater(clustered[1], 0.25)

    def test_one_unit_per_session_matches_the_plain_permutation(self):
        """Clustering with one unit per cluster gives the plain test."""
        rng = np.random.default_rng(1)
        reference, other = rng.normal(0, 1, 8), rng.normal(1, 1, 8)
        plain = compare.permutation_difference(reference, other, np.random.default_rng(2), 5000)
        clustered = compare.permutation_difference_clustered(
            reference, [f"a{i}" for i in range(8)], other, [f"b{i}" for i in range(8)],
            np.random.default_rng(2), 5000,
        )

        self.assertAlmostEqual(plain[0], clustered[0])
        self.assertAlmostEqual(plain[1], clustered[1], delta=0.02)

    def test_clustered_interval_is_wider_than_block_level_interval(self):
        """Resampling sessions keeps the interval from shrinking with more blocks."""
        units = self.make_blocked_sessions()
        empty = units.loc[units["dataset_label"] == "empty"]
        rng = np.random.default_rng(0)

        _, naive_low, naive_high = compare.bootstrap_mean(empty["value"], rng, 4000)
        mean, low, high = compare.bootstrap_clustered_mean(empty["value"], empty["session_id"], rng, 4000)

        self.assertAlmostEqual(mean, 1.5)
        self.assertGreater(high - low, naive_high - naive_low)
        self.assertTrue(np.isnan(compare.bootstrap_clustered_mean([1.0, 2.0], ["s", "s"], rng)[1]))

    def test_summary_counts_sessions_and_units(self):
        """Summaries report independent sessions and plotted units."""
        summary = compare.summarize_by_strain(
            self.make_blocked_sessions(), "value", rng=np.random.default_rng(0),
            samples=500, cluster_column="session_id",
        )

        self.assertEqual(summary["session_count"].tolist(), [2, 2])
        self.assertEqual(summary["unit_count"].tolist(), [10, 10])

    def test_variance_components_attribute_spread_to_sessions(self):
        """Identical blocks within sessions put all spread between sessions."""
        variance = compare.variance_components(self.make_blocked_sessions(), "value")

        np.testing.assert_allclose(variance["between_fraction"], [1.0, 1.0])
        np.testing.assert_allclose(variance["within_session_sd"], [0.0, 0.0])


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
