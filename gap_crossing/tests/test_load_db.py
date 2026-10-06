"""Tests for loading gap-crossing datasets from the database."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gap_crossing import load_db, run_io


def make_recording(frame_count=601, path="2025/kevin/example", jumps=None, **fields):
    """Return one loaded recording with one track."""
    data = {
        "trjn": np.zeros(frame_count, dtype=int),
        "headx_smooth": np.zeros(frame_count),
        "heady_smooth": np.zeros(frame_count),
        "signal": np.zeros(frame_count),
        "t": np.arange(frame_count, dtype=float) / 60,
        "vx_smooth": np.ones(frame_count),
        "vy_smooth": np.zeros(frame_count),
        "spd_smooth": np.ones(frame_count),
        "theta": np.linspace(0, 359, frame_count),
        "theta_smooth": np.linspace(1, 358, frame_count),
        "dtheta_smooth": np.full(frame_count, 2.0),
        "jumps": np.zeros(frame_count, dtype=bool) if jumps is None else jumps,
    }
    data.update(fields)
    return {"sql": {"path": path, "rig_name": "or42b"}, "data": data}


class TrackTests(unittest.TestCase):
    """Check processed tracks."""

    def test_make_tracks_keeps_heading_turn_rate_and_jumps(self):
        """Accepted tracks carry theta, theta_smooth, dtheta_smooth, and jumps."""
        tracks = load_db.make_tracks([make_recording()])

        self.assertEqual(len(tracks), 1)
        np.testing.assert_allclose(tracks[0]["theta"], np.linspace(0, 359, 601))
        np.testing.assert_allclose(tracks[0]["theta_smooth"], np.linspace(1, 358, 601))
        np.testing.assert_allclose(tracks[0]["dtheta_smooth"], 2.0)
        self.assertEqual(tracks[0]["jumps"].dtype, bool)
        self.assertEqual(tracks[0]["track_id"], "2025/kevin/example::track0")

    def test_short_and_fast_tracks_are_rejected(self):
        """Track length and speed filters use the named settings."""
        short = make_recording(frame_count=load_db.MIN_TRACK_S * load_db.FRAME_RATE_HZ)
        fast = make_recording(vx_smooth=np.full(601, float(load_db.MAX_SPEED_MM_S)))

        self.assertEqual(load_db.make_tracks([short, fast]), [])

    def test_jump_filter_is_optional(self):
        """Tracks with jumps stay unless DROP_TRACKS_WITH_JUMPS is set."""
        jumps = np.zeros(601, dtype=bool)
        jumps[10] = True
        recording = make_recording(jumps=jumps)

        self.assertEqual(len(load_db.make_tracks([recording])), 1)
        with patch.object(load_db, "DROP_TRACKS_WITH_JUMPS", True):
            self.assertEqual(load_db.make_tracks([recording]), [])

    def test_recording_without_new_fields_is_skipped(self):
        """A recording without a required matrix field gives no tracks."""
        recording = make_recording()
        del recording["data"]["dtheta_smooth"]

        self.assertEqual(load_db.make_tracks([recording]), [])


class QueryTests(unittest.TestCase):
    """Check query settings."""

    def test_unknown_query_field_is_an_error(self):
        """A misspelled field stops the run before it widens the query."""
        with patch.object(load_db, "QUERY_FILTERS", {"experimentr": "kevin"}):
            with self.assertRaisesRegex(ValueError, "experimentr"):
                load_db.validate_query_fields(["experimenter", "genotype_file", "year", "month", "day"])

    def test_known_query_fields_pass(self):
        """Filters and period fields that exist in the database pass."""
        fields = {"experimenter", "stim_protocol", "genotype_file", "year", "month", "day", "vial"}
        load_db.validate_query_fields(fields)

    def test_query_record_differs_only_by_genotype(self):
        """Datasets from one batch share every query setting but the genotype."""
        first = load_db.make_query_record("a.yaml")
        second = load_db.make_query_record("b.yaml")

        self.assertEqual(first["query_filters"]["genotype_file"], "a.yaml")
        first["query_filters"].pop("genotype_file")
        second["query_filters"].pop("genotype_file")
        self.assertEqual(first, second)


class RunTests(unittest.TestCase):
    """Check one load run without the database."""

    def test_run_writes_one_dataset_per_genotype(self):
        """Each genotype gets its own dataset folder that run_io can combine."""
        def fake_select(query_filters):
            return pd.DataFrame([{"id": 1, "path": f"2026/{query_filters['genotype_file']}"}])

        def fake_load(experiments):
            path = experiments["path"].iloc[0]
            return [make_recording(path=path)], pd.DataFrame()

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.dict(os.environ, {run_io.OUTPUT_ROOT_ENV: temporary_directory}),
                patch.object(load_db, "GENOTYPE_FILES", ["a.yaml", "b.yaml"]),
                patch.object(load_db, "get_available_fields", return_value=[
                    "experimenter", "stim_protocol", "genotype_file", "year", "month", "day", "vial",
                ]),
                patch.object(load_db, "select_experiments", side_effect=fake_select),
                patch.object(load_db, "load_recordings", side_effect=fake_load),
            ):
                dataset_dirs = load_db.run()
                datasets, _ = run_io.load_datasets(
                    {Path(name).stem: path for name, path in dataset_dirs.items()}
                )

        self.assertEqual(list(datasets), ["a", "b"])
        self.assertEqual(
            datasets["a"].provenance["batch_id"], datasets["b"].provenance["batch_id"]
        )
        self.assertEqual(datasets["b"].query["query_filters"]["genotype_file"], "b.yaml")
        self.assertEqual(datasets["a"].params, load_db.make_track_params())

    def test_load_from_database_labels_genotypes_and_saves_nothing(self):
        """Database mode returns labeled tracks and writes no folder."""
        def fake_select(query_filters):
            return pd.DataFrame([{"id": 1, "path": f"2026/{query_filters['genotype_file']}"}])

        def fake_load(experiments):
            return [make_recording(path=experiments["path"].iloc[0])], pd.DataFrame()

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.dict(os.environ, {run_io.OUTPUT_ROOT_ENV: temporary_directory}),
                patch.object(load_db, "GENOTYPE_FILES", ["a.yaml", "b.yaml"]),
                patch.object(load_db, "get_available_fields", return_value=[
                    "experimenter", "stim_protocol", "genotype_file", "year", "month", "day",
                ]),
                patch.object(load_db, "select_experiments", side_effect=fake_select),
                patch.object(load_db, "load_recordings", side_effect=fake_load),
            ):
                tracks, recordings, sources = load_db.load_tracks_from_database()
                written = list(Path(temporary_directory).iterdir())

        self.assertEqual(written, [])
        self.assertEqual([track["dataset_label"] for track in tracks], ["a", "b"])
        self.assertEqual(recordings["dataset_label"].tolist(), ["a", "b"])
        self.assertEqual(sources["b"]["query"], load_db.make_query_record("b.yaml"))
        self.assertIsNone(sources["a"]["path"])

    def test_genotype_without_records_stops_before_loading(self):
        """No dataset folder is written when one genotype matches nothing."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.dict(os.environ, {run_io.OUTPUT_ROOT_ENV: temporary_directory}),
                patch.object(load_db, "GENOTYPE_FILES", ["a.yaml"]),
                patch.object(load_db, "get_available_fields", return_value=[
                    "experimenter", "stim_protocol", "genotype_file", "year", "month", "day",
                ]),
                patch.object(load_db, "select_experiments", return_value=pd.DataFrame()),
            ):
                with self.assertRaisesRegex(RuntimeError, "a.yaml"):
                    load_db.run()
                self.assertEqual(list(Path(temporary_directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
