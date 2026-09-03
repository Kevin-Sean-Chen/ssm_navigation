"""Tests for session-scale gap-crossing learning summaries."""

import unittest
import warnings

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


if __name__ == "__main__":
    unittest.main()
