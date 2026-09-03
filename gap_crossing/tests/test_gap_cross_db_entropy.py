"""Tests for vial-date outcome entropy analysis."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
