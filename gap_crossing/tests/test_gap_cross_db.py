"""Tests for database-backed gap-crossing data saves."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from optogui.analysis import load_saved_experiment_data


from gap_crossing import gap_cross_db as database_analysis


class LoadedRecordingSaveTests(unittest.TestCase):
    """Check the optional save of database-loaded recordings."""

    def test_save_loaded_recordings_writes_optogui_joblib(self):
        """A configured output path keeps the exact loaded recording list."""
        recordings = [
            {
                "sql": {"id": 17, "path": "2025/kevin/example"},
                "data": {"trjn": np.array([1, 1], dtype=int)},
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "gap_cross_snapshot"

            saved_path = database_analysis.save_loaded_recordings(
                recordings, output_path
            )

            self.assertEqual(saved_path, output_path.with_suffix(".joblib"))
            loaded = load_saved_experiment_data(saved_path)
            self.assertEqual(loaded[0]["sql"]["id"], 17)
            np.testing.assert_array_equal(loaded[0]["data"]["trjn"], [1, 1])


class DatabaseTrackTests(unittest.TestCase):
    """Check kinematic fields on database-loaded tracks."""

    def test_make_tracks_keeps_recorded_heading(self):
        """The quality plot receives theta from each accepted track."""
        frame_count = 601
        recordings = [
            {
                "sql": {"path": "2025/kevin/example"},
                "data": {
                    "trjn": np.zeros(frame_count, dtype=int),
                    "headx_smooth": np.zeros(frame_count),
                    "heady_smooth": np.zeros(frame_count),
                    "signal": np.zeros(frame_count),
                    "t": np.arange(frame_count, dtype=float) / 60,
                    "vx_smooth": np.ones(frame_count),
                    "vy_smooth": np.zeros(frame_count),
                    "spd_smooth": np.ones(frame_count),
                    "theta": np.linspace(0, 359, frame_count),
                },
            }
        ]

        tracks = database_analysis.make_tracks(recordings)

        np.testing.assert_allclose(tracks[0]["theta"], np.linspace(0, 359, frame_count))


if __name__ == "__main__":
    unittest.main()
