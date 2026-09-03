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


if __name__ == "__main__":
    unittest.main()
