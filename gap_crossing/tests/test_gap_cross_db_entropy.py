"""Tests for vial-date outcome entropy analysis."""

import unittest
import warnings

import numpy as np
import pandas as pd


from gap_crossing.memory_analysis import gap_cross_db_entropy as entropy_analysis


def make_events():
    """Return two valid sessions and one short-track session."""
    rows = []
    session_tracks = {
        1: {
            "a": ["cross", "regain", "cross", "regain", "cross", "regain"],
            "short": ["cross", "cross"],
        },
        2: {"b": ["cross", "cross", "cross", "cross", "cross", "cross"]},
        3: {
            "c": ["cross", "regain"],
            "d": ["cross", "regain"],
        },
    }
    for day, tracks in session_tracks.items():
        for track_id, outcomes in tracks.items():
            for attempt_time_s, outcome in enumerate(outcomes):
                rows.append(
                    {
                        "recording_year": 2026,
                        "recording_month": 5,
                        "recording_day": day,
                        "recording_experimenter": "kevin",
                        "recording_vial": 0,
                        "track_id": track_id,
                        "attempt_time_s": attempt_time_s,
                        "outcome": outcome,
                    }
                )
    return pd.DataFrame(rows)


def make_two_back_dependent_events():
    """Return one session whose next outcome needs the two-back outcome."""
    outcomes = ["cross", "abort", "cross", "regain"] * 25
    rows = []
    for attempt_time_s, outcome in enumerate(outcomes):
        rows.append(
            {
                "recording_year": 2026,
                "recording_month": 5,
                "recording_day": 4,
                "recording_experimenter": "kevin",
                "recording_vial": 0,
                "track_id": "two_back_track",
                "attempt_time_s": attempt_time_s,
                "outcome": outcome,
            }
        )
    return pd.DataFrame(rows)


class OutcomeEntropyTests(unittest.TestCase):
    """Check entropy estimates from within-track outcome sequences."""

    def test_alternating_sequence_has_one_bit_unconditional_entropy(self):
        """The alternating next outcome is deterministic after one state."""
        sequences = [["cross", "regain", "cross", "regain", "cross", "regain"]]

        self.assertAlmostEqual(
            entropy_analysis.get_conditional_entropy(sequences, history_length=0), 1.0
        )
        self.assertAlmostEqual(
            entropy_analysis.get_conditional_entropy(sequences, history_length=1), 0.0
        )
        self.assertAlmostEqual(
            entropy_analysis.get_conditional_entropy(sequences, history_length=2), 0.0
        )
        self.assertAlmostEqual(
            entropy_analysis.get_conditional_entropy(sequences, history_length=3), 0.0
        )
        self.assertAlmostEqual(
            entropy_analysis.get_conditional_entropy(sequences, history_length=4), 0.0
        )

    def test_session_summary_excludes_tracks_without_five_attempts(self):
        """Short tracks do not change a complete entropy decay curve."""
        sessions = entropy_analysis.make_session_entropy_summary(make_events())
        summary = entropy_analysis.make_entropy_decay_summary(sessions)

        self.assertEqual(sessions["recording_day"].to_list(), [1, 2])
        self.assertEqual(summary["session_count"].to_list(), [2, 2, 2, 2, 2])
        np.testing.assert_allclose(summary["mean_entropy_bits"], [0.5, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(summary["sem_entropy_bits"], [0.5, 0.0, 0.0, 0.0, 0.0])

    def test_markov_null_reproduces_deterministic_first_order_sessions(self):
        """A matched Markov null has no excess higher-order information."""
        null_summary = entropy_analysis.make_markov_null_entropy_summary(
            make_events(), simulation_count=20, random_seed=0
        )

        self.assertEqual(len(null_summary), 10)
        alternating = null_summary.loc[
            null_summary["recording_day"].eq(1)
        ].sort_values("history_length")
        np.testing.assert_allclose(
            alternating["observed_entropy_bits"],
            alternating["null_mean_entropy_bits"],
        )
        np.testing.assert_allclose(alternating["excess_information_bits"], 0.0)

    def test_markov_null_plot_has_observed_and_control_axes(self):
        """The control comparison leaves the observed entropy figure separate."""
        figure = entropy_analysis.plot_markov_null_comparison(
            make_events(), simulation_count=5, random_seed=0
        )

        self.assertEqual(len(figure.axes), 2)
        entropy_analysis.analysis.plt.close("all")

    def test_second_order_control_has_no_excess_for_deterministic_markov_tracks(self):
        """A first-order sequence has no two-back deviation above its null."""
        comparison = entropy_analysis.make_second_order_markov_null_comparison(
            make_events(), simulation_count=20, random_seed=0
        )

        np.testing.assert_allclose(
            comparison.session_summary["excess_total_variation"], 0.0
        )
        for residual_matrix in comparison.residual_matrices.values():
            np.testing.assert_allclose(residual_matrix, 0.0)

    def test_second_order_control_detects_two_back_dependent_sequence(self):
        """A two-back-dependent sequence exceeds its first-order Markov null."""
        comparison = entropy_analysis.make_second_order_markov_null_comparison(
            make_two_back_dependent_events(), simulation_count=100, random_seed=0
        )

        self.assertGreater(
            comparison.session_summary.loc[0, "excess_total_variation"], 0.1
        )

    def test_second_order_control_plot_shows_one_matrix_per_two_back_outcome(self):
        """The matrix control visualizes all three two-back outcomes."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            figure = entropy_analysis.plot_second_order_markov_null_comparison(
                make_events(), simulation_count=5, random_seed=0
            )

        self.assertEqual(len(figure.axes), 4)
        entropy_analysis.analysis.plt.close("all")


if __name__ == "__main__":
    unittest.main()
