"""Tests for session-scale gap-crossing learning summaries."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd


from gap_crossing import gap_cross_db_vial as vial_analysis


def make_events(session_trials):
    """Return one regain-following event for each session trial."""
    rows = []
    for session, outcomes in session_trials.items():
        for trial, outcome in enumerate(outcomes, start=1):
            rows.append(
                {
                    "recording_year": 2026,
                    "recording_month": 5,
                    "recording_day": session,
                    "recording_experimenter": "kevin",
                    "recording_vial": 0,
                    "recording_trial": trial,
                    "previous_outcome": "regain",
                    "outcome": outcome,
                    "is_cross": int(outcome == "cross"),
                }
            )
    return pd.DataFrame(rows)


class TrialLearningTests(unittest.TestCase):
    """Check trial-scale session learning summaries."""

    def test_trial_summary_reduces_session_count_at_later_trials(self):
        """Only sessions that have a trial contribute to that trial rank."""
        events = make_events(
            {
                1: ["cross", "regain", "cross", "regain", "cross", "regain"],
                2: ["regain", "cross", "regain", "cross"],
            }
        )

        summary = vial_analysis.make_trial_regain_cross_summary(events)

        self.assertEqual(summary["recording_trial"].to_list(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(summary["available_session_count"].to_list(), [2, 2, 2, 2, 1, 1])
        self.assertEqual(summary["session_count"].to_list(), [2, 2, 2, 2, 1, 1])
        np.testing.assert_allclose(summary["mean_cross"], [0.5] * 4 + [1.0, 0.0])
        np.testing.assert_allclose(summary["sem_cross"], [0.5] * 4 + [0.0, 0.0])

    def test_first_last_matrices_exclude_sessions_with_fewer_than_six_trials(self):
        """First and last windows use separate trials from eligible sessions."""
        events = make_events(
            {
                1: ["cross", "cross", "regain", "abort", "abort", "cross"],
                2: ["cross", "cross", "cross", "cross", "cross"],
            }
        )

        matrices, session_counts = vial_analysis.make_first_last_trial_transition_matrices(
            events
        )

        self.assertEqual(session_counts, {"first": 1, "last": 1})
        self.assertAlmostEqual(matrices["first"].loc["cross", "regain"], 2 / 3)
        self.assertAlmostEqual(matrices["first"].loc["regain", "regain"], 1 / 3)
        self.assertAlmostEqual(matrices["last"].loc["cross", "regain"], 1 / 3)
        self.assertAlmostEqual(matrices["last"].loc["abort", "regain"], 2 / 3)

    def test_transition_matrix_plot_has_no_pandas_deprecation_warning(self):
        """The matrix plot uses the current pandas element map API."""
        events = make_events(
            {1: ["cross", "cross", "regain", "abort", "abort", "cross"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            vial_analysis.plot_first_last_trial_transition_matrices(events)
        vial_analysis.analysis.plt.close("all")


class TrackExportTests(unittest.TestCase):
    """Check the collaborator track export."""

    def test_timestamped_track_path_keeps_date_and_time(self):
        """A new run gets a readable timestamp in its export name."""
        timestamp = datetime(2026, 9, 29, 14, 35, 6)

        output_path = vial_analysis.make_timestamped_track_path(timestamp)

        self.assertEqual(
            output_path,
            Path("saved_data")
            / "gap_cross"
            / "gap_crossing_tracks_2026-09-29_14-35-06.joblib",
        )

    @staticmethod
    def make_valid_export():
        """Return one valid export for validation and save tests."""
        tracks = [
            {
                "track_id": "track-a",
                "time_s": np.array([0.0, 1.0, 2.0]),
                "xy": np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]),
                "signal": np.array([0.0, 255.0, 0.0]),
            }
        ]
        events = pd.DataFrame(
            {
                "track_id": ["track-a", "track-a"],
                "attempt_time_s": [0.5, 1.5],
                "outcome": ["regain", "cross"],
            }
        )
        return vial_analysis.make_track_export(tracks, events)

    def test_export_contains_query_parameters_and_retained_track_arrays(self):
        """The export keeps only tracks that have retained attempts."""
        tracks = [
            {
                "track_id": "track-a",
                "time_s": np.array([0.0, 1.0, 2.0]),
                "xy": np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]),
                "signal": np.array([0.0, 255.0, 0.0]),
            },
            {
                "track_id": "track-b",
                "time_s": np.array([0.0, 1.0]),
                "xy": np.array([[30.0, 40.0], [31.0, 41.0]]),
                "signal": np.array([255.0, 0.0]),
            },
        ]
        events = pd.DataFrame(
            {
                "track_id": ["track-a", "track-a"],
                "attempt_time_s": [1.5, 0.5],
                "outcome": ["cross", "regain"],
            }
        )

        export = vial_analysis.make_track_export(tracks, events)

        self.assertEqual(set(export), {"metadata", "tracks"})
        self.assertEqual(
            export["metadata"]["query"],
            {
                "database_location": vial_analysis.pooled.DATABASE_LOCATION,
                "data_location": vial_analysis.pooled.DATA_LOCATION,
                "query_filters": vial_analysis.pooled.QUERY_FILTERS,
                "query_periods": vial_analysis.pooled.QUERY_PERIODS,
                "max_experiments": vial_analysis.pooled.MAX_EXPERIMENTS,
            },
        )
        self.assertEqual(
            export["metadata"]["analysis_parameters"],
            {
                "frame_rate_hz": vial_analysis.analysis.FRAME_RATE_HZ,
                "min_track_s": vial_analysis.analysis.MIN_TRACK_S,
                "pre_signal_s": vial_analysis.analysis.PRE_SIGNAL_S,
                "outcome_s": vial_analysis.analysis.OUTCOME_S,
                "distance_mm": vial_analysis.analysis.DISTANCE_MM,
                "min_attempts_per_track": vial_analysis.analysis.MIN_ATTEMPTS_PER_TRACK,
                "attempt_ribbon_half_width_mm": (
                    vial_analysis.analysis.ATTEMPT_RIBBON_HALF_WIDTH_MM
                ),
                "gap_geometry_method": vial_analysis.analysis.GAP_GEOMETRY_METHOD,
            },
        )
        self.assertEqual(len(export["tracks"]), 1)
        track = export["tracks"][0]
        self.assertEqual(
            set(track),
            {"t_s", "x_mm", "y_mm", "signal", "attempt_t_s", "attempt_state"},
        )
        np.testing.assert_array_equal(track["t_s"], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(track["x_mm"], [10.0, 11.0, 12.0])
        np.testing.assert_array_equal(track["y_mm"], [20.0, 21.0, 22.0])
        np.testing.assert_array_equal(track["signal"], [0.0, 255.0, 0.0])
        np.testing.assert_array_equal(track["attempt_t_s"], [0.5, 1.5])
        np.testing.assert_array_equal(track["attempt_state"], ["regain", "cross"])

    def test_validation_rejects_invalid_track_data(self):
        """Invalid sample and attempt arrays do not produce an export."""
        invalid_changes = {
            "continuous length mismatch": (
                "x_mm",
                np.array([10.0, 11.0]),
            ),
            "non-increasing continuous time": (
                "t_s",
                np.array([0.0, 0.0, 2.0]),
            ),
            "non-increasing attempt time": (
                "attempt_t_s",
                np.array([0.5, 0.5]),
            ),
            "unsupported state": (
                "attempt_state",
                np.array(["regain", "stop"]),
            ),
            "attempt outside track": (
                "attempt_t_s",
                np.array([-0.5, 1.5]),
            ),
            "empty attempts": (
                "attempt_t_s",
                np.array([], dtype=float),
            ),
        }

        for case, (field, value) in invalid_changes.items():
            with self.subTest(case=case):
                export = deepcopy(self.make_valid_export())
                export["tracks"][0][field] = value
                if case == "empty attempts":
                    export["tracks"][0]["attempt_state"] = np.array([], dtype=str)

                with self.assertRaises(ValueError):
                    vial_analysis.validate_track_export(export)

    def test_disabled_save_still_validates(self):
        """A disabled file write still rejects an invalid export."""
        export = self.make_valid_export()

        self.assertIsNone(vial_analysis.save_track_export(export, None))
        export["tracks"][0]["attempt_state"][0] = "stop"
        with self.assertRaises(ValueError):
            vial_analysis.save_track_export(export, None)

    def test_enabled_save_writes_reloadable_joblib(self):
        """An enabled file write preserves metadata and track arrays."""
        export = self.make_valid_export()
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "nested" / "gap_crossing_tracks.joblib"

            saved_path = vial_analysis.save_track_export(export, output_path)
            loaded = joblib.load(output_path)

        self.assertEqual(saved_path, output_path)
        self.assertEqual(loaded["metadata"], export["metadata"])
        self.assertEqual(len(loaded["tracks"]), 1)
        for field in export["tracks"][0]:
            np.testing.assert_array_equal(
                loaded["tracks"][0][field], export["tracks"][0][field]
            )


if __name__ == "__main__":
    unittest.main()
