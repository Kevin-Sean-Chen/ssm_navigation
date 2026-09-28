"""Tests for gap-crossing fixed, Markov, and outcome-memory models."""

import unittest

import numpy as np
import pandas as pd

from gap_crossing.memory_analysis import gap_cross_db_memory as memory_analysis


SESSION_COLUMNS = [
    "recording_year",
    "recording_month",
    "recording_day",
    "recording_experimenter",
    "recording_vial",
]


def make_events(session_sequences):
    """Return ordered events from session track outcome sequences."""
    rows = []
    for day, tracks in session_sequences.items():
        for track_id, outcomes in tracks.items():
            for attempt_time_s, outcome in enumerate(outcomes):
                rows.append(
                    {
                        "recording_year": 2026,
                        "recording_month": 5,
                        "recording_day": day,
                        "recording_experimenter": "kevin",
                        "recording_vial": 0,
                        "track_id": f"{day}_{track_id}",
                        "attempt_time_s": float(2 * attempt_time_s),
                        "outcome": outcome,
                    }
                )
    return pd.DataFrame(rows)


class TransitionTableTests(unittest.TestCase):
    """Check elapsed-time transition construction."""

    def test_transition_table_adds_elapsed_time_and_attempt_memory(self):
        """Each transition keeps elapsed time and prior-outcome memory."""
        events = make_events(
            {1: {"a": ["cross", "regain", "abort", "cross"]}}
        )

        transitions = memory_analysis.make_transition_table(events)

        self.assertEqual(
            transitions["current_outcome"].to_list(), ["cross", "regain", "abort"]
        )
        self.assertEqual(
            transitions["next_outcome"].to_list(), ["regain", "abort", "cross"]
        )
        self.assertEqual(
            transitions["previous_outcome"].to_list(), [None, "cross", "regain"]
        )
        np.testing.assert_allclose(transitions["elapsed_s"], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(
            transitions["log_elapsed_s"], np.log1p([2.0, 2.0, 2.0])
        )
        np.testing.assert_allclose(transitions["memory_cross"], [0.0, 0.5, 0.25])
        np.testing.assert_allclose(transitions["memory_regain"], [0.0, 0.0, 0.5])
        np.testing.assert_allclose(transitions["memory_abort"], [0.0, 0.0, 0.0])

    def test_second_order_features_encode_current_and_previous_outcomes(self):
        """The saturated model distinguishes each two-outcome context."""
        events = make_events(
            {1: {"a": ["cross", "regain", "abort", "cross"]}}
        )
        transitions = memory_analysis.make_transition_table(events).dropna(
            subset=["previous_outcome"]
        )

        features = memory_analysis.make_model_features(
            transitions, "markov_second_order"
        )

        self.assertEqual(
            features.columns.to_list(),
            [
                "current_regain",
                "current_abort",
                "previous_regain",
                "previous_abort",
                "current_regain__previous_regain",
                "current_regain__previous_abort",
                "current_abort__previous_regain",
                "current_abort__previous_abort",
            ],
        )
        np.testing.assert_allclose(features.to_numpy(), [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1, 0]])

    def test_transition_table_excludes_nonpositive_elapsed_time(self):
        """A zero or negative interval does not become a model row."""
        events = make_events({1: {"a": ["cross", "regain", "abort"]}})
        events.loc[events["outcome"] == "regain", "attempt_time_s"] = 0.0

        transitions = memory_analysis.make_transition_table(events)

        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions.loc[0, "current_outcome"], "regain")
        self.assertEqual(transitions.loc[0, "next_outcome"], "abort")
        self.assertEqual(transitions.loc[0, "elapsed_s"], 4.0)

    def test_transition_table_uses_outcome_specific_retention(self):
        """Cross, regain, and abort histories can decay at different rates."""
        events = make_events(
            {1: {"a": ["cross", "regain", "abort", "cross"]}}
        )

        transitions = memory_analysis.make_transition_table(
            events,
            memory_retention={"cross": 0.2, "regain": 0.8, "abort": 0.5},
        )

        np.testing.assert_allclose(transitions["memory_cross"], [0.0, 0.8, 0.16])
        np.testing.assert_allclose(transitions["memory_regain"], [0.0, 0.0, 0.2])
        np.testing.assert_allclose(transitions["memory_abort"], [0.0, 0.0, 0.0])


class SessionModelTests(unittest.TestCase):
    """Check held-out vial-date model comparison."""

    def make_three_session_events(self):
        """Return sessions with all next-outcome classes in each training set."""
        return make_events(
            {
                1: {"a": ["cross", "regain", "abort", "cross", "regain", "abort"]},
                2: {"b": ["regain", "abort", "cross", "regain", "abort", "cross"]},
                3: {"c": ["abort", "cross", "regain", "abort", "cross", "regain"]},
            }
        )

    def test_each_session_is_scored_once_per_model(self):
        """Held-out scores keep vial-date sessions as independent samples."""
        scores = memory_analysis.evaluate_leave_one_session_out(
            self.make_three_session_events()
        )

        self.assertEqual(len(scores), 24)
        self.assertEqual(
            set(scores["model"]), set(memory_analysis.DESCRIPTION_MODEL_ORDER)
        )
        self.assertEqual(scores.groupby(SESSION_COLUMNS).size().to_list(), [8, 8, 8])
        self.assertEqual(
            scores.loc[scores["sample"] == "all"].groupby(SESSION_COLUMNS).size().to_list(),
            [3, 3, 3],
        )
        self.assertTrue(np.isfinite(scores["test_nll_bits"]).all())
        self.assertNotIn("tau_s", scores.columns)
        self.assertTrue(
            scores.loc[
                scores["model"] == "markov_memory", "memory_retention"
            ].isin(memory_analysis.MEMORY_RETENTION_GRID).all()
        )

    def test_two_back_comparison_scores_all_models_on_the_same_rows(self):
        """The trace and second-order models use identical held-out rows."""
        scores = memory_analysis.evaluate_leave_one_session_out(
            self.make_three_session_events()
        )
        two_back_scores = scores.loc[scores["sample"] == "two_back"]

        self.assertEqual(len(two_back_scores), 15)
        self.assertEqual(
            set(two_back_scores["model"]), set(memory_analysis.DESCRIPTION_MODEL_ORDER)
        )
        self.assertEqual(
            two_back_scores.groupby(SESSION_COLUMNS).size().to_list(), [5, 5, 5]
        )

    def test_memory_retention_is_selected_by_inner_session_scores(self):
        """Each retention gets one inner held-out score per training session."""
        selection = memory_analysis.select_memory_retention(
            self.make_three_session_events()
        )

        self.assertIn(selection.retention, memory_analysis.MEMORY_RETENTION_GRID)
        self.assertEqual(
            len(selection.scores),
            3 * len(memory_analysis.MEMORY_RETENTION_GRID),
        )
        self.assertEqual(
            selection.scores.groupby("memory_retention").size().to_list(),
            [3] * len(memory_analysis.MEMORY_RETENTION_GRID),
        )
        self.assertTrue(np.isfinite(selection.scores["test_nll_bits"]).all())

    def test_information_gain_uses_fixed_and_markov_baselines(self):
        """Lower held-out loss gives positive gain over the correct baseline."""
        scores = pd.DataFrame(
            {
                "recording_year": [2026, 2026, 2026],
                "recording_month": [5, 5, 5],
                "recording_day": [1, 1, 1],
                "recording_experimenter": ["kevin", "kevin", "kevin"],
                "recording_vial": [0, 0, 0],
                "model": ["fixed", "markov", "markov_memory"],
                "test_nll_bits": [1.0, 0.8, 0.6],
            }
        )

        gains = memory_analysis.add_information_gain(scores)

        fixed = gains.loc[gains["model"] == "fixed"].iloc[0]
        markov = gains.loc[gains["model"] == "markov"].iloc[0]
        memory = gains.loc[gains["model"] == "markov_memory"].iloc[0]
        self.assertEqual(fixed["gain_over_fixed_bits"], 0.0)
        self.assertEqual(markov["gain_over_markov_bits"], 0.0)
        self.assertAlmostEqual(memory["gain_over_fixed_bits"], 0.4)
        self.assertAlmostEqual(memory["gain_over_markov_bits"], 0.2)

    def test_memory_model_plot(self):
        """The revision produces a held-out outcome-memory comparison plot."""
        events = self.make_three_session_events()

        comparison_figure = memory_analysis.plot_memory_model_comparison(events)

        self.assertEqual(len(comparison_figure.axes), 1)
        memory_analysis.analysis.plt.close("all")


if __name__ == "__main__":
    unittest.main()
