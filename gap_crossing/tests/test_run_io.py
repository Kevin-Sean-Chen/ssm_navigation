"""Tests for gap-crossing run folders and saved datasets."""

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gap_crossing import run_io


def make_track(source_file, local_id=0, frame_count=5):
    """Return one small processed track."""
    return {
        "track_id": f"{source_file}::track{local_id}",
        "source_file": source_file,
        "xy": np.zeros((frame_count, 2)),
        "time_s": np.arange(frame_count, dtype=float),
        "signal": np.zeros(frame_count),
    }


def make_recordings(source_file, genotype_file="a.yaml", rig_name="or42b"):
    """Return one recording metadata row."""
    return pd.DataFrame([{
        "source_file": source_file,
        "recording_genotype_file": genotype_file,
        "recording_rig_name": rig_name,
        "recording_camera_model": "Grasshopper3",
        "recording_camera_resolution": "2016x1248",
        "recording_vial": 0,
    }])


def make_query(genotype_file="a.yaml", day=19):
    """Return one stored query."""
    return {
        "database_location": "server",
        "query_filters": {"experimenter": "kevin", "genotype_file": genotype_file},
        "query_periods": [{"year": 2026, "month": [8], "day": [day]}],
        "matrix_fields": ["t", "signal"],
    }


class OutputRootTestCase(unittest.TestCase):
    """Point the run_io output root at a temporary folder."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temporary_directory.name)
        self.environment = patch.dict(
            os.environ, {run_io.OUTPUT_ROOT_ENV: str(self.output_root)}
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary_directory.cleanup()

    def save_dataset(self, label, source_file, genotype_file="a.yaml", rig_name="or42b",
                     day=19, params=None):
        """Write one dataset folder and return its path."""
        with run_io.dataset_run(__file__, label) as run_dir:
            run_io.save_dataset(
                run_dir,
                tracks=[make_track(source_file)],
                recordings=make_recordings(source_file, genotype_file, rig_name),
                experiments=pd.DataFrame([{"id": 1, "path": source_file}]),
                failed=pd.DataFrame(),
                query=make_query(genotype_file, day),
                params=params or {"min_track_s": 10},
            )
        return run_dir


class RunFolderTests(OutputRootTestCase):
    """Check run-folder names and the no-overwrite rule."""

    def test_run_folder_name_starts_with_timestamp(self):
        """A run folder name joins the timestamp and a safe label."""
        timestamp = run_io.make_timestamp(datetime(2026, 10, 6, 14, 5, 9))

        run_dir = run_io.make_run_dir("datasets", "gap ribbon/empty", timestamp)

        self.assertEqual(
            run_dir, self.output_root / "datasets" / "2026-10-06_140509_gap_ribbon_empty"
        )
        self.assertTrue(run_dir.is_dir())

    def test_existing_run_folder_is_an_error(self):
        """A second run with the same name does not reuse the folder."""
        run_io.make_run_dir("analyses", "memory", "2026-10-06_140509")

        with self.assertRaises(FileExistsError):
            run_io.make_run_dir("analyses", "memory", "2026-10-06_140509")

    def test_new_files_never_replace_existing_files(self):
        """JSON writes refuse an existing path."""
        path = self.output_root / "settings.json"
        run_io.write_json(path, {"a": 1})

        with self.assertRaises(FileExistsError):
            run_io.write_json(path, {"a": 2})
        self.assertEqual(run_io.read_json(path), {"a": 1})


class SettingsTests(unittest.TestCase):
    """Check automatic settings collection."""

    def test_collect_settings_keeps_uppercase_values_only(self):
        """Settings become JSON values; functions and lowercase names are skipped."""
        module = types.ModuleType("example_settings")
        module.PRE_SIGNAL_S = 5
        module.ROOT_DIR = Path("C:/data")
        module.EDGES_MM = np.array([1.5, 2.5])
        module.PERIODS = ({"year": 2026},)
        module.helper = 3
        module.RUN = lambda: None

        settings = run_io.collect_settings(module)

        self.assertEqual(
            settings,
            {"example_settings": {
                "PRE_SIGNAL_S": 5,
                "ROOT_DIR": str(Path("C:/data")),
                "EDGES_MM": [1.5, 2.5],
                "PERIODS": [{"year": 2026}],
            }},
        )
        json.dumps(settings)


class DatasetTests(OutputRootTestCase):
    """Check dataset folders and their reload."""

    def test_dataset_round_trip_keeps_tracks_tables_and_provenance(self):
        """A saved dataset reloads with its tracks, tables, and run records."""
        run_dir = self.save_dataset("empty", "2026/kevin/a")

        dataset = run_io.load_dataset(run_dir)

        self.assertEqual(dataset.tracks[0]["track_id"], "2026/kevin/a::track0")
        np.testing.assert_array_equal(dataset.tracks[0]["time_s"], np.arange(5.0))
        self.assertEqual(dataset.recordings["recording_rig_name"].tolist(), ["or42b"])
        self.assertEqual(dataset.query, make_query())
        self.assertTrue(dataset.failed.empty)
        self.assertEqual(dataset.provenance["run_type"], "dataset")
        for key in ("git_commit", "git_dirty", "optogui_commit", "packages", "created"):
            self.assertIn(key, dataset.provenance)
        self.assertEqual(run_io.read_json(run_dir / "status.json")["status"], "completed")
        self.assertTrue((run_dir / "code" / Path(__file__).name).is_file())
        self.assertIn("Dataset folder:", (run_dir / "log.txt").read_text(encoding="utf-8"))

    def test_failed_run_keeps_folder_and_records_error(self):
        """An error inside a run leaves a failed status, not a deleted folder."""
        with self.assertRaisesRegex(RuntimeError, "no tracks"):
            with run_io.dataset_run(__file__, "broken") as run_dir:
                raise RuntimeError("no tracks")

        status = run_io.read_json(run_dir / "status.json")
        self.assertEqual(status["status"], "failed")
        self.assertIn("no tracks", status["error"])

    def test_missing_dataset_files_are_reported(self):
        """Loading a folder without dataset files names the missing files."""
        empty_dir = self.output_root / "not_a_dataset"
        empty_dir.mkdir()

        with self.assertRaisesRegex(FileNotFoundError, "tracks.joblib"):
            run_io.load_dataset(empty_dir)


class CompatibilityTests(OutputRootTestCase):
    """Check which datasets may be combined."""

    def test_genotype_may_differ(self):
        """Strains with identical other settings combine with labeled tracks."""
        dataset_dirs = {
            "empty": self.save_dataset("empty", "2026/kevin/a", "a.yaml"),
            "fc2": self.save_dataset("fc2", "2026/kevin/b", "b.yaml"),
        }

        datasets, warnings = run_io.load_datasets(dataset_dirs)
        tracks, recordings = run_io.combine_datasets(datasets)

        self.assertEqual(warnings, [])
        self.assertEqual([track["dataset_label"] for track in tracks], ["empty", "fc2"])
        self.assertEqual(recordings["dataset_label"].tolist(), ["empty", "fc2"])

    def test_other_query_differences_are_errors(self):
        """A different query period stops the combination."""
        dataset_dirs = {
            "empty": self.save_dataset("empty", "2026/kevin/a", "a.yaml", day=19),
            "fc2": self.save_dataset("fc2", "2026/kevin/b", "b.yaml", day=20),
        }

        with self.assertRaisesRegex(ValueError, "query_periods"):
            run_io.load_datasets(dataset_dirs)

    def test_track_filter_differences_are_errors(self):
        """Different track filters stop the combination."""
        dataset_dirs = {
            "empty": self.save_dataset("empty", "2026/kevin/a", "a.yaml"),
            "fc2": self.save_dataset("fc2", "2026/kevin/b", "b.yaml", params={"min_track_s": 5}),
        }

        with self.assertRaisesRegex(ValueError, "min_track_s"):
            run_io.load_datasets(dataset_dirs)

    def test_mixed_camera_setups_are_errors(self):
        """Recordings from two rigs need an orientation check first."""
        dataset_dirs = {
            "empty": self.save_dataset("empty", "2026/kevin/a", "a.yaml", rig_name="or42b"),
            "fc2": self.save_dataset("fc2", "2026/kevin/b", "b.yaml", rig_name="other"),
        }

        with self.assertRaisesRegex(ValueError, "recording_rig_name"):
            run_io.load_datasets(dataset_dirs)

    def test_same_recording_in_two_datasets_is_an_error(self):
        """A recording cannot be counted twice."""
        dataset_dirs = {
            "first": self.save_dataset("first", "2026/kevin/a", "a.yaml"),
            "second": self.save_dataset("second", "2026/kevin/a", "b.yaml"),
        }

        with self.assertRaisesRegex(ValueError, "more than one dataset"):
            run_io.load_datasets(dataset_dirs)

    def test_empty_dataset_dirs_is_an_error(self):
        """An analysis without datasets explains what to set."""
        with self.assertRaisesRegex(ValueError, "DATASET_DIRS"):
            run_io.load_datasets({})


class AnalysisRunTests(OutputRootTestCase):
    """Check analysis folders."""

    def test_analysis_run_records_datasets_settings_and_tables(self):
        """An analysis folder points to its dataset and stores its settings."""
        dataset_dir = self.save_dataset("empty", "2026/kevin/a")
        settings = types.ModuleType("example_analysis")
        settings.__file__ = __file__
        settings.OUTCOME_S = 5

        with run_io.analysis_run(__file__, {"empty": dataset_dir}, [settings]) as run_info:
            table_path = run_info.save_table(
                "events", pd.DataFrame({"outcome": ["cross"], "xy": [np.zeros((2, 2))]})
            )
            export_path = run_info.export_path("tracks.joblib")

        provenance = run_io.read_json(run_info.path / "provenance.json")
        self.assertEqual(provenance["run_type"], "analysis")
        self.assertEqual(provenance["datasets"]["empty"]["path"], str(dataset_dir))
        self.assertEqual(
            run_io.read_json(run_info.path / "params.json"),
            {"example_analysis": {"OUTCOME_S": 5}},
        )
        self.assertEqual(pd.read_csv(table_path).columns.tolist(), ["outcome"])
        self.assertEqual(export_path, run_info.path / "exports" / "tracks.joblib")
        self.assertEqual(len(run_info.tracks), 1)
        self.assertEqual(
            run_io.read_json(run_info.path / "status.json")["status"], "completed"
        )

    def test_save_figures_numbers_each_figure_once(self):
        """Figures are saved once each, in order, without replacing files."""
        dataset_dir = self.save_dataset("empty", "2026/kevin/a")
        saved_names = []

        def fake_savefig(path, **_kwargs):
            Path(path).write_text("figure", encoding="utf-8")
            saved_names.append(Path(path).name)

        first = types.SimpleNamespace(
            get_suptitle=lambda: "Speed", axes=[], number=1, savefig=fake_savefig
        )
        second = types.SimpleNamespace(
            get_suptitle=lambda: "", axes=[], number=2, savefig=fake_savefig
        )
        figures = {1: first, 2: second}
        with run_io.analysis_run(__file__, {"empty": dataset_dir}) as run_info:
            with (
                patch.object(run_io.plt, "get_fignums", return_value=[1]),
                patch.object(run_io.plt, "figure", side_effect=figures.get),
            ):
                run_info.save_figures()
            with (
                patch.object(run_io.plt, "get_fignums", return_value=[1, 2]),
                patch.object(run_io.plt, "figure", side_effect=figures.get),
            ):
                run_info.save_figures()

        self.assertEqual(
            saved_names,
            ["01_Speed.png", "01_Speed.pdf", "02_figure2.png", "02_figure2.pdf"],
        )


if __name__ == "__main__":
    unittest.main()
