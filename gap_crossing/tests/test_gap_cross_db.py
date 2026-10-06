"""Tests for the pooled gap-crossing analysis of saved datasets."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gap_crossing import gap_cross_db as database_analysis
from gap_crossing import run_io
from gap_crossing.tests.test_load_db import make_recording
from gap_crossing import load_db


class DatabaseDiagnosticPlotTests(unittest.TestCase):
    """Check diagnostic plots that precede gap detection."""

    def tearDown(self):
        """Close figures created by each test."""
        plt.close("all")

    def test_run_displays_diagnostics_before_gap_detection(self):
        """Both diagnostic figures display before gap detection starts."""
        frame_count = 601
        recording = make_recording(
            path="2026/kevin/example",
            headx_smooth=np.linspace(0, 10, frame_count),
            heady_smooth=np.linspace(1, 2, frame_count),
            signal=np.arange(frame_count) % 2,
        )
        figure_count_before_run = len(plt.get_fignums())
        run_order = []

        def fail_gap_detection(_tracks):
            run_order.append("gap detection")
            raise RuntimeError("detection failed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(os.environ, {run_io.OUTPUT_ROOT_ENV: temporary_directory}):
                with run_io.dataset_run(__file__, "example") as dataset_dir:
                    run_io.save_dataset(
                        dataset_dir,
                        tracks=load_db.make_tracks([recording]),
                        recordings=load_db.make_recording_metadata([recording]),
                        experiments=pd.DataFrame([{"id": 1}]),
                        failed=pd.DataFrame(),
                        query={"query_filters": {}},
                        params={},
                    )
                with (
                    patch.object(database_analysis, "DATASET_DIRS", {"example": dataset_dir}),
                    patch.object(
                        database_analysis.analysis.plt,
                        "show",
                        side_effect=lambda: run_order.append("plot display"),
                    ),
                    patch.object(
                        database_analysis.analysis,
                        "get_gap_geometry",
                        side_effect=fail_gap_detection,
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "detection failed"):
                        database_analysis.run()

        self.assertEqual(
            (
                len(plt.get_fignums()) - figure_count_before_run,
                run_order,
            ),
            (2, ["plot display", "gap detection"]),
        )

    def test_database_source_plots_without_saving(self):
        """Database mode loads through load_db and writes no run folder."""
        frame_count = 601
        recording = make_recording(
            path="2026/kevin/example",
            headx_smooth=np.linspace(0, 10, frame_count),
            heady_smooth=np.linspace(1, 2, frame_count),
            signal=np.arange(frame_count) % 2,
        )
        tracks = load_db.make_tracks([recording])
        recordings = load_db.make_recording_metadata([recording])
        run_order = []

        def fail_gap_detection(_tracks):
            run_order.append("gap detection")
            raise RuntimeError("detection failed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.dict(os.environ, {run_io.OUTPUT_ROOT_ENV: temporary_directory}),
                patch.object(database_analysis, "DATA_SOURCE", "database"),
                patch.object(
                    database_analysis.load_db,
                    "load_tracks_from_database",
                    return_value=(tracks, recordings, {"example": {"path": None}}),
                ),
                patch.object(
                    database_analysis.analysis.plt,
                    "show",
                    side_effect=lambda: run_order.append("plot display"),
                ),
                patch.object(
                    database_analysis.analysis,
                    "get_gap_geometry",
                    side_effect=fail_gap_detection,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "detection failed"):
                    database_analysis.run()
                written = list(Path(temporary_directory).iterdir())

        self.assertEqual(run_order, ["plot display", "gap detection"])
        self.assertEqual(written, [])


class DataSourceTests(unittest.TestCase):
    """Check the data-source switch."""

    def test_unknown_data_source_is_an_error(self):
        """A misspelled DATA_SOURCE names the valid choices."""
        with self.assertRaisesRegex(ValueError, "dataset"):
            with database_analysis.open_run(__file__, "databse", {}, []):
                pass

    def test_database_run_saves_nothing(self):
        """Tables and exports are skipped in database mode."""
        run_info = database_analysis.DatabaseRun([], pd.DataFrame(), {})

        self.assertIsNone(run_info.save_table("events", pd.DataFrame({"a": [1]})))
        self.assertIsNone(run_info.export_path("tracks.joblib"))
        self.assertIsNone(run_info.path)


if __name__ == "__main__":
    unittest.main()
