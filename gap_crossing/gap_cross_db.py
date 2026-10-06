"""Pooled gap-crossing analysis of database recordings.

Set DATA_SOURCE to choose where tracks come from:
    "dataset":  read saved dataset folders in DATASET_DIRS (run load_db.py
                first). Each run writes an analysis folder with figures,
                tables, and provenance.
    "database": query and load with the settings in load_db.py, then plot.
                Nothing is saved.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

try:
    from gap_crossing import gap_cross_track as analysis
    from gap_crossing import load_db, run_io
except ModuleNotFoundError:
    import gap_cross_track as analysis
    import load_db
    import run_io


# %% Data source settings
DATA_SOURCE = "dataset"  # "dataset" or "database"
# Label -> dataset folder written by load_db.py. Used when DATA_SOURCE is "dataset".
DATASET_DIRS: dict[str, Path] = {
    # "GMOCLKir_empty": Path(r"...\gap_crossing\datasets\2026-10-06_120000_gap_ribbon_GMOCLKir_empty"),
}
DATA_SOURCES = ("dataset", "database")


# %% Data source
@dataclass
class DatabaseRun:
    """Tracks loaded straight from the database; saves nothing."""

    tracks: list[dict]
    recordings: pd.DataFrame
    sources: dict
    path: None = None

    def show(self) -> None:
        """Display open figures."""
        analysis.plt.show()

    def save_table(self, _name, _table) -> None:
        """Skip table saves in database mode."""
        return None

    def export_path(self, _filename) -> None:
        """Return no export path in database mode."""
        return None


@contextmanager
def open_run(script_path, data_source, dataset_dirs, settings_modules):
    """Yield tracks from saved datasets or from the database."""
    if data_source == "dataset":
        with run_io.analysis_run(script_path, dataset_dirs, settings_modules) as run_info:
            yield run_info
    elif data_source == "database":
        print("Loading from the database with the settings in load_db.py; nothing is saved.")
        yield DatabaseRun(*load_db.load_tracks_from_database())
    else:
        raise ValueError(f"DATA_SOURCE must be one of {DATA_SOURCES}, not {data_source!r}.")


# %% Events
def make_events(tracks, recordings):
    """Return gap geometry and attempt events with recording metadata."""
    geometry = analysis.get_gap_geometry(tracks)
    print(f"Gap regions: {len(geometry)}")
    events = analysis.make_event_table(tracks, geometry)
    events = events.merge(recordings, on="source_file")
    return geometry, events


# %% Run
def run():
    """Load tracks and run the pooled gap-crossing analysis."""
    settings_modules = [analysis, sys.modules[__name__]]
    with open_run(__file__, DATA_SOURCE, DATASET_DIRS, settings_modules) as run_info:
        tracks = run_info.tracks
        print(f"Valid tracks: {len(tracks)}")
        if not tracks:
            raise RuntimeError("No valid tracks.")

        analysis.plot_kinematic_histograms(tracks)
        analysis.plot_tracks_with_signal(tracks)
        run_info.show()
        geometry, events = make_events(tracks, run_info.recordings)
        run_info.save_table("gap_geometry", geometry)
        run_info.save_table("events", events)
        print(f"Repeated-attempt tracks: {events['track_id'].nunique()}")
        print(f"Valid attempts: {len(events)}")
        print(events.groupby("outcome").size().reindex(analysis.OUTCOME_ORDER, fill_value=0))

        analysis.plot_gap_geometry(tracks, geometry)
        analysis.plot_centerline_profiles(tracks, geometry)
        analysis.plot_event_summary(events)
        analysis.plot_within_track_summary(events)
        analysis.plot_cross_after_regain_by_attempt(events)
        analysis.plot_regain_transition_durations(events)
        speed_events = analysis.get_post_entry_speed(events)
        print(f"Events with post-entry speed: {len(speed_events)}")
        analysis.plot_post_entry_speed(speed_events)
        analysis.plot_transition_model(events)
        analysis.plot_second_order_transition_matrices(events)
        analysis.plot_early_late_transition_matrices(events)
        analysis.plot_motif_enrichment(events)
        analysis.plot_event_paths(events)
        analysis.evaluate_track_model(events)
        run_info.show()


if __name__ == "__main__":
    run()
