"""Tests for fixed, Markov, and cumulative crossing-history models."""

import unittest

import numpy as np
import pandas as pd

from gap_crossing.memory_analysis import gap_cross_db_drift as drift_analysis


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
            for attempt_index, outcome in enumerate(outcomes):
                rows.append(
                    {
                        "recording_year": 2026,
                        "recording_month": 5,
                        "recording_day": day,
                        "recording_experimenter": "kevin",
                        "recording_vial": 0,
                        "track_id": f"{day}_{track_id}",
                        "attempt_time_s": float(2 * attempt_index),
                        "outcome": outcome,
                    }
                )
    return pd.DataFrame(rows)


class TransitionTableTests(unittest.TestCase):
    """Check construction of ordered transition rows."""

    def test_prior_cross_fraction_uses_only_earlier_same_track_outcomes(self):
        """Each target gets the crossing fraction observed before it."""
        events = make_events(
            {
                1: {
                    "a": ["cross", "regain", "abort", "cross"],
                    "b": ["abort", "cross", "regain"],
                }
            }
        ).sample(frac=1.0, random_state=4)

        transitions = drift_analysis.make_transition_table(events)

        self.assertEqual(
            transitions["current_outcome"].to_list(),
            ["cross", "regain", "abort", "abort", "cross"],
        )
        self.assertEqual(
            transitions["next_outcome"].to_list(),
            ["regain", "abort", "cross", "cross", "regain"],
        )
        np.testing.assert_allclose(
            transitions["prior_cross_fraction"],
            [1.0, 0.5, 1.0 / 3.0, 0.0, 0.5],
        )


class ModelFeatureTests(unittest.TestCase):
    """Check the fixed, Markov, and history-model inputs."""

    def test_cross_fraction_adds_one_scaled_feature_without_state_interactions(self):
        """Crossing history has one shared slope across current outcomes."""
        transitions = pd.DataFrame(
            {
                "current_outcome": ["cross", "regain", "abort"],
                "prior_cross_fraction": [0.0, 0.5, 1.0],
            }
        )

        fixed = drift_analysis.make_model_features(transitions, "fixed")
        stationary = drift_analysis.make_model_features(transitions, "markov")
        with_history = drift_analysis.make_model_features(
            transitions,
            "markov_cross_fraction",
            covariate_mean=0.5,
            covariate_std=0.5,
        )

        self.assertEqual(fixed.columns.to_list(), ["constant"])
        np.testing.assert_allclose(fixed["constant"], 0.0)
        self.assertEqual(
            stationary.columns.to_list(), ["current_regain", "current_abort"]
        )
        self.assertEqual(
            with_history.columns.to_list(),
            ["current_regain", "current_abort", "prior_cross_fraction"],
        )
        np.testing.assert_allclose(
            with_history.to_numpy(),
            [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        )

    def test_cross_fraction_scale_comes_from_the_training_transitions(self):
        """The fitted model stores its training-only history scale."""
        transitions = pd.DataFrame(
            {
                "current_outcome": ["cross", "regain", "abort"],
                "next_outcome": ["regain", "abort", "cross"],
                "prior_cross_fraction": [0.0, 0.5, 1.0],
            }
        )

        fitted = drift_analysis.fit_transition_model(
            transitions, "markov_cross_fraction"
        )

        self.assertEqual(fitted.covariate_mean, 0.5)
        self.assertAlmostEqual(fitted.covariate_std, np.sqrt(1.0 / 6.0))

    def test_separated_alpha_and_beta_reconstruct_history_probabilities(self):
        """Reference-coded parameters reproduce the fitted softmax model."""
        events = make_events(
            {
                1: {"a": ["cross", "regain", "abort", "cross", "regain", "abort"]},
                2: {"b": ["regain", "abort", "cross", "regain", "abort", "cross"]},
            }
        )
        transitions = drift_analysis.make_transition_table(events)
        fitted = drift_analysis.fit_transition_model(
            transitions, "markov_cross_fraction"
        )

        alpha, beta = drift_analysis.separate_alpha_beta(fitted)

        self.assertEqual(
            alpha.index.to_list(), drift_analysis.analysis.OUTCOME_ORDER
        )
        self.assertEqual(
            alpha.columns.to_list(), drift_analysis.analysis.OUTCOME_ORDER
        )
        np.testing.assert_allclose(alpha.loc["cross"], 0.0)
        self.assertEqual(beta.index.to_list(), drift_analysis.analysis.OUTCOME_ORDER)
        self.assertEqual(beta.loc["cross"], 0.0)

        scaled_fraction = 0.75
        prior_cross_fraction = (
            fitted.covariate_mean + scaled_fraction * fitted.covariate_std
        )
        grid = pd.DataFrame(
            {
                "current_outcome": ["regain"],
                "prior_cross_fraction": [prior_cross_fraction],
            }
        )
        fitted_probability = drift_analysis.predict_probabilities(fitted, grid)[0]
        logits = alpha["regain"].to_numpy() + beta.to_numpy() * scaled_fraction
        reconstructed = np.exp(logits - logits.max())
        reconstructed /= reconstructed.sum()
        class_probability = pd.Series(
            fitted_probability, index=fitted.classifier.classes_
        ).reindex(drift_analysis.analysis.OUTCOME_ORDER)
        np.testing.assert_allclose(reconstructed, class_probability)


class SessionModelTests(unittest.TestCase):
    """Check held-out vial-day model comparison."""

    def make_three_session_events(self):
        """Return sessions with all next outcomes in each training set."""
        return make_events(
            {
                1: {"a": ["cross", "regain", "abort", "cross", "regain", "abort"]},
                2: {"b": ["regain", "abort", "cross", "regain", "abort", "cross"]},
                3: {"c": ["abort", "cross", "regain", "abort", "cross", "regain"]},
            }
        )

    def test_each_session_is_scored_once_per_model(self):
        """Each held-out session gets fixed, Markov, and history scores."""
        scores = drift_analysis.evaluate_leave_one_session_out(
            self.make_three_session_events()
        )

        self.assertEqual(len(scores), 9)
        self.assertEqual(
            set(scores["model"]),
            {"fixed", "markov", "markov_cross_fraction"},
        )
        self.assertEqual(scores.groupby(SESSION_COLUMNS).size().to_list(), [3, 3, 3])
        self.assertTrue(np.isfinite(scores["test_nll_bits"]).all())

    def test_information_gain_uses_fixed_and_markov_baselines(self):
        """Each model has the correct cumulative and incremental gain."""
        scores = pd.DataFrame(
            {
                "recording_year": [2026, 2026, 2026],
                "recording_month": [5, 5, 5],
                "recording_day": [1, 1, 1],
                "recording_experimenter": ["kevin", "kevin", "kevin"],
                "recording_vial": [0, 0, 0],
                "model": ["fixed", "markov", "markov_cross_fraction"],
                "test_nll_bits": [1.4, 1.2, 0.9],
            }
        )

        gains = drift_analysis.add_information_gain(scores)

        fixed = gains.loc[gains["model"] == "fixed"].iloc[0]
        stationary = gains.loc[gains["model"] == "markov"].iloc[0]
        with_history = gains.loc[
            gains["model"] == "markov_cross_fraction"
        ].iloc[0]
        self.assertEqual(fixed["gain_over_fixed_bits"], 0.0)
        self.assertAlmostEqual(stationary["gain_over_fixed_bits"], 0.2)
        self.assertAlmostEqual(with_history["gain_over_fixed_bits"], 0.5)
        self.assertEqual(stationary["gain_over_markov_bits"], 0.0)
        self.assertAlmostEqual(with_history["gain_over_markov_bits"], 0.3)

    def test_parameter_tables_include_the_fitted_markov_matrix(self):
        """Reported fixed and Markov probabilities have valid normalization."""
        fixed_probability, markov_matrix, alpha, beta = (
            drift_analysis.fit_parameter_tables(self.make_three_session_events())
        )

        self.assertEqual(
            fixed_probability.index.to_list(), drift_analysis.analysis.OUTCOME_ORDER
        )
        self.assertAlmostEqual(fixed_probability.sum(), 1.0)
        self.assertEqual(
            markov_matrix.index.to_list(), drift_analysis.analysis.OUTCOME_ORDER
        )
        self.assertEqual(
            markov_matrix.columns.to_list(), drift_analysis.analysis.OUTCOME_ORDER
        )
        np.testing.assert_allclose(markov_matrix.sum(axis=0), 1.0)
        self.assertEqual(alpha.shape, (3, 3))
        self.assertEqual(beta.shape, (3,))

    def test_model_comparison_plot_reports_held_out_gain(self):
        """The comparison figure shows session-held-out history gain."""
        scores = drift_analysis.evaluate_leave_one_session_out(
            self.make_three_session_events()
        )

        figure = drift_analysis.plot_model_comparison(raw_scores=scores)

        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(
            figure.axes[0].get_ylabel(),
            "Held-out gain over fixed (bits / transition)",
        )
        drift_analysis.analysis.plt.close(figure)

    def test_cross_fraction_plot_conditions_on_current_outcome(self):
        """The descriptive figure has one panel per current outcome."""
        figure = drift_analysis.plot_cross_fraction_probabilities(
            self.make_three_session_events()
        )

        self.assertEqual(len(figure.axes), 3)
        self.assertTrue(
            all(len(axis.lines) == 3 for axis in figure.axes)
        )
        self.assertEqual(
            figure.axes[-1].get_xlabel(),
            "Cumulative prior crossing fraction",
        )
        drift_analysis.analysis.plt.close(figure)

    def test_fitted_parameter_plot_separates_markov_alpha_and_beta(self):
        """One figure shows fixed, Markov, and both history parameters."""
        figure = drift_analysis.plot_fitted_parameters(
            self.make_three_session_events()
        )

        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(
            [axis.get_title() for axis in figure.axes],
            [
                "Fixed outcome probabilities",
                "Stationary Markov probabilities",
                "Crossing-history baseline α",
                "Crossing-history β",
            ],
        )
        self.assertEqual(len(figure.axes[0].patches), 3)
        self.assertEqual(len(figure.axes[3].patches), 3)
        drift_analysis.analysis.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
