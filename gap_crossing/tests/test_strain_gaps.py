"""Tests for the strain comparison of gap-crossing behavior."""

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gap_crossing import strain_gaps as gaps


def make_events(source_file, sequences, start_s=0.0):
    """Return attempt rows; sequences maps track_id -> [(time_s, outcome), ...]."""
    return pd.DataFrame([
        {
            "track_id": track_id,
            "parent_track_id": track_id.split("::seg")[0],
            "source_file": source_file,
            "attempt_time_s": start_s + time_s,
            "outcome": outcome,
        }
        for track_id, attempts in sequences.items()
        for time_s, outcome in attempts
    ])


def make_recording(source_file, label, day, trial=1):
    """Return one recording row with session fields."""
    return {
        "source_file": source_file, "dataset_label": label, "recording_year": 2026,
        "recording_month": 8, "recording_day": day, "recording_experimenter": "kevin",
        "recording_vial": 0, "recording_trial": trial,
    }


def get_value(metrics, metric, category):
    """Return the single value of one metric and category."""
    return metrics.loc[(metrics["metric"] == metric) & (metrics["category"] == category), "value"].item()


class TransitionTests(unittest.TestCase):
    """Check consecutive-attempt pairs."""

    def test_transitions_stay_within_tracks_and_follow_time(self):
        """Pairs are time-ordered and never join two tracks."""
        events = make_events("a", {
            "a::t0": [(5.0, "cross"), (1.0, "regain"), (2.0, "regain")],
            "a::t1": [(3.0, "abort"), (7.0, "cross")],
        })

        transitions = gaps.make_transition_table(events)

        self.assertEqual(
            transitions[["track_id", "transition", "interval_s"]].values.tolist(),
            [["a::t0", "regain->regain", 1.0], ["a::t0", "regain->cross", 3.0],
             ["a::t1", "abort->cross", 4.0]],
        )

    def test_zero_interval_keeps_transition_but_not_interval(self):
        """Simultaneous attempts count as a transition with no interval."""
        events = make_events("a", {"a::t0": [(1.0, "regain"), (1.0, "cross")]})

        transitions = gaps.make_transition_table(events)

        self.assertEqual(transitions["transition"].tolist(), ["regain->cross"])
        self.assertTrue(np.isnan(transitions["interval_s"].iloc[0]))


class UnitMetricTests(unittest.TestCase):
    """Check per-unit metrics."""

    def test_metric_values(self):
        """Fractions, intervals, and conditional probabilities match hand counts."""
        sequence = [(0, "regain"), (1, "regain"), (3, "cross"), (4, "cross"), (6, "regain"), (7, "abort")]
        events = make_events("a", {"a::t0": sequence, "a::t0::seg1": [(20, "cross"), (30, "cross")]})
        events = gaps.add_unit_columns(events, pd.DataFrame([make_recording("a", "empty", 19)]))

        with (
            patch.object(gaps, "MIN_UNIT_ATTEMPTS", 1),
            patch.object(gaps, "MIN_UNIT_INTERVALS", 1),
            patch.object(gaps, "MIN_UNIT_TRANSITIONS", 1),
        ):
            metrics = gaps.analyze_unit(events)

        # Two sequences of one parent track: 8 attempts per track.
        self.assertEqual(get_value(metrics, "attempts_per_track", "all"), 8.0)
        self.assertAlmostEqual(get_value(metrics, "outcome_fraction", "cross"), 4 / 8)
        self.assertAlmostEqual(get_value(metrics, "outcome_fraction", "abort"), 1 / 8)
        # Intervals: 1, 2, 1, 2, 1 and 10.
        self.assertAlmostEqual(get_value(metrics, "interval_mean", "all"), 17 / 6)
        self.assertAlmostEqual(get_value(metrics, "interval_mean", "regain->regain"), 1.0)
        self.assertAlmostEqual(get_value(metrics, "interval_mean", "cross->cross"), 5.5)
        # After regain: regain, cross, abort. After cross: cross, regain, cross.
        self.assertAlmostEqual(get_value(metrics, "transition_probability", "regain->regain"), 1 / 3)
        self.assertAlmostEqual(get_value(metrics, "transition_probability", "regain->cross"), 1 / 3)
        self.assertAlmostEqual(get_value(metrics, "transition_probability", "cross->cross"), 2 / 3)

    def test_too_few_counts_give_nan(self):
        """Metrics below their minimum counts are NaN."""
        events = make_events("a", {"a::t0": [(0, "regain"), (1, "cross")]})
        events = gaps.add_unit_columns(events, pd.DataFrame([make_recording("a", "empty", 19)]))

        metrics = gaps.analyze_unit(events)

        self.assertEqual(get_value(metrics, "attempts_per_track", "all"), 2.0)
        self.assertTrue(metrics.loc[metrics["metric"] != "attempts_per_track", "value"].isna().all())


class RunTests(unittest.TestCase):
    """Check a synthetic multi-strain run."""

    def tearDown(self):
        plt.close("all")

    def make_data(self):
        """Return events and recordings of two strains, two days, four trials each."""
        patterns = {
            "empty": ["regain", "cross", "cross", "regain", "regain", "cross"],
            "fc2": ["regain", "regain", "regain", "abort", "regain", "regain"],
        }
        events = []
        recordings = []
        for label, outcomes in patterns.items():
            for day in (19, 20):
                for trial in range(1, 5):
                    source = f"2026/{label}_{day}_{trial}"
                    recordings.append(make_recording(source, label, day, trial))
                    attempts = [(2.0 * index, outcome) for index, outcome in enumerate(outcomes)]
                    events.append(make_events(source, {f"{source}::track0": attempts}))
        return pd.concat(events, ignore_index=True), pd.DataFrame(recordings)

    def test_session_tables_comparisons_and_plot(self):
        """Each session gives one unit; strain differences match the patterns."""
        events, recordings = self.make_data()
        rng = np.random.default_rng(0)

        with (
            patch.object(gaps, "BOOTSTRAP_SAMPLES", 200),
            patch.object(gaps, "PERMUTATION_COUNT", 200),
        ):
            tables = gaps.analyze_units(events, recordings)
            summary = gaps.summarize(tables["unit_metrics"], rng)
            comparisons = gaps.compare_strains(tables["unit_metrics"], "empty", rng)

        metrics = tables["unit_metrics"]
        self.assertEqual(metrics["unit_id"].nunique(), 4)
        self.assertEqual(len(tables["transitions"]), 16 * 5)
        cross = comparisons.loc[
            (comparisons["metric"] == "outcome_fraction") & (comparisons["category"] == "cross")
        ].iloc[0]
        self.assertAlmostEqual(cross["difference"], -0.5)
        self.assertEqual((cross["session_count"], cross["reference_session_count"]), (2, 2))
        self.assertIn("q_value", comparisons)

        figure = gaps.plot_gap_metrics(metrics, summary, ["empty", "fc2"], {"empty": "C0", "fc2": "C1"})
        self.assertEqual(len(figure.axes), len(gaps.METRIC_CATEGORIES))

    def test_block_mode_nests_blocks_in_sessions(self):
        """Block mode gives two blocks per session and drops none."""
        events, recordings = self.make_data()

        with (
            patch.object(gaps, "SAMPLE_UNIT", "block"),
            patch.object(gaps, "TRIALS_PER_BLOCK", 2),
            patch.object(gaps, "MIN_BLOCK_TRIALS", 2),
        ):
            tables = gaps.analyze_units(events, recordings)

        metrics = tables["unit_metrics"]
        self.assertEqual(metrics["session_id"].nunique(), 4)
        self.assertEqual(metrics["unit_id"].nunique(), 8)
        self.assertEqual(len(tables["events"]), len(events))

    def test_short_last_block_is_dropped(self):
        """Events in a block below MIN_BLOCK_TRIALS are left out."""
        events, recordings = self.make_data()

        with (
            patch.object(gaps, "SAMPLE_UNIT", "block"),
            patch.object(gaps, "TRIALS_PER_BLOCK", 3),
            patch.object(gaps, "MIN_BLOCK_TRIALS", 2),
        ):
            tables = gaps.analyze_units(events, recordings)

        self.assertEqual(sorted(tables["events"]["block"].unique()), [1])
        self.assertEqual(len(tables["events"]), len(events) * 3 // 4)


if __name__ == "__main__":
    unittest.main()
