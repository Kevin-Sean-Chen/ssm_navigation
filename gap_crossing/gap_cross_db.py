"""Database-backed gap-crossing analysis.

This script queries optogui recordings, loads the selected matrices, and runs
the analysis in gap_cross_track. Edit QUERY_FILTERS before a full data load.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from optogui.analysis import (
    load_experiment_data,
    query_experiments,
    save_experiment_data,
)

try:
    from gap_crossing import gap_cross_track as analysis
except ModuleNotFoundError:
    import gap_cross_track as analysis


# %% Database settings
DATABASE_LOCATION = "server"
DATA_LOCATION = "server"
QUERY_FILTERS = {
    "experimenter": "kevin",
    # "day": 18,
    # "vial": [0, 1],
    "genotype_file": "117_GMUCR.yaml",
    "stim_protocol": "users.kevin.intermittent_gaps_ribbon",
}
QUERY_PERIODS = [
    # {"year": 2025, "month": [12]},
    {"year": 2026, "month": [5,6,7, 8]},
]
# QUERY_PERIODS = [
#     {"year": 2025, "month": [12], "day": 15}, ### debugging error
# ]
MAX_EXPERIMENTS = None
# Set this path to save loaded recordings as an Optogui Joblib file.
SAVE_LOADED_RECORDINGS_PATH: Path | None = None
SAVE_LOADED_RECORDINGS_PATH = Path(
    r"saved_data\gap_cross\kevin_2026_5678_gap_ribbons"
)

MATRIX_FIELDS = [
    "headx_smooth",
    "heady_smooth",
    "signal",
    "t",
    "vx_smooth",
    "vy_smooth",
    "spd_smooth",
]
RECORDING_FIELDS = [
    "path",
    "year",
    "month",
    "day",
    "experimenter",
    "vial",
    "trial",
    "stim_name",
    "stim_protocol",
    "genotype",
]


@contextmanager
def suppress_loader_logs():
    """Temporarily suppress logs from one-record public loader calls."""
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)


# %% Load tracks
def select_experiments():
    """Return the database records selected for analysis."""
    query_results = [
        query_experiments(
            db_location=DATABASE_LOCATION,
            order_by="id",
            **QUERY_FILTERS,
            **period,
        )
        for period in QUERY_PERIODS
    ]
    experiments = pd.concat(query_results, ignore_index=True).drop_duplicates("id")
    experiments = experiments.sort_values("id").reset_index(drop=True)
    if experiments.empty:
        raise RuntimeError("No database records matched QUERY_FILTERS.")
    if MAX_EXPERIMENTS is not None:
        experiments = experiments.head(MAX_EXPERIMENTS).copy()
    return experiments


def load_recordings(experiments):
    """Load recordings and return failed query rows without stopping the run."""
    loaded_recordings = []
    failure_rows = []
    for _, experiment in tqdm(
        experiments.iterrows(),
        total=len(experiments),
        desc="Loading experiments",
        unit="experiment",
    ):
        experiment_frame = experiment.to_frame().T
        try:
            with suppress_loader_logs():
                loaded = load_experiment_data(
                    experiment_frame,
                    data_location=DATA_LOCATION,
                    load_obj=False,
                    load_flies=False,
                    load_shapes_proj=False,
                    load_shapes_screen=False,
                    load_data=True,
                    matrix_fields=MATRIX_FIELDS,
                    combine=False,
                    skip_errors=True,
                    n_jobs=1,
                    show_progress=False,
                )
        except Exception as error:
            failure_rows.append({
                **experiment.to_dict(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
            tqdm.write(
                f"Skipping experiment {experiment.get('id')}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        if loaded:
            loaded_recordings.extend(loaded)
        else:
            failure_rows.append({
                **experiment.to_dict(),
                "error_type": "NoDataLoaded",
                "error_message": "The loader returned no data.",
            })
            tqdm.write(
                f"Skipping experiment {experiment.get('id')}: no data loaded"
            )
    return loaded_recordings, pd.DataFrame(failure_rows)


def save_loaded_recordings(loaded_recordings, output_path):
    """Save loaded recordings when one output path is configured."""
    if output_path is None:
        return None
    return save_experiment_data(loaded_recordings, output_path)


def make_recording_metadata(loaded_recordings):
    """Return one metadata row for each loaded recording."""
    rows = []
    for recording in loaded_recordings:
        sql = recording["sql"]
        source_file = str(sql["path"])
        rows.append(
            {
                "source_file": source_file,
                **{f"recording_{field}": sql.get(field) for field in RECORDING_FIELDS},
            }
        )
    return pd.DataFrame(rows).drop_duplicates("source_file")


def make_tracks(loaded_recordings):
    """Return valid tracks with recording-level identity."""
    tracks = []
    min_frames = int(analysis.MIN_TRACK_S * analysis.FRAME_RATE_HZ)
    required = {"trjn", *MATRIX_FIELDS}

    for recording in loaded_recordings:
        data = recording["data"]
        sql = recording["sql"]
        source_file = str(sql["path"])
        missing = required.difference(data)
        if missing:
            print(f"Skip file with missing data: {source_file} | {sorted(missing)}")
            continue

        for local_id in np.unique(data["trjn"]):
            index = np.flatnonzero(data["trjn"] == local_id)
            if len(index) <= min_frames:
                continue

            xy = np.column_stack((data["headx_smooth"][index], data["heady_smooth"][index]))
            signal = np.asarray(data["signal"][index]).squeeze().astype(float)
            time_s = np.asarray(data["t"][index]).squeeze().astype(float)
            velocity = np.column_stack((data["vx_smooth"][index], data["vy_smooth"][index]))
            speed_smooth = np.asarray(data["spd_smooth"][index]).squeeze().astype(float)

            if (
                xy.shape != (len(index), 2)
                or signal.shape != (len(index),)
                or time_s.shape != (len(index),)
                or speed_smooth.shape != (len(index),)
                or not np.isfinite(velocity).all()
            ):
                continue

            speed = np.linalg.norm(velocity, axis=1)
            if np.nanmean(speed) <= 0.1 or np.nanmax(speed) >= 50:
                continue

            tracks.append(
                {
                    "track_id": f"{source_file}::track{local_id}",
                    "source_file": source_file,
                    "xy": xy,
                    "signal": signal,
                    "time_s": time_s,
                    "velocity": velocity,
                    "speed_smooth": speed_smooth,
                }
            )
    return tracks


# %% Run
def run():
    """Query, load, and analyze the selected recordings."""
    experiments = select_experiments()
    print(f"Database records: {len(experiments)}")
    print(experiments[RECORDING_FIELDS].to_string(index=False))

    loaded_recordings, failed_experiments = load_recordings(experiments)
    print(f"Loaded recordings: {len(loaded_recordings)}")
    print(f"Failed recordings: {len(failed_experiments)}")
    if not failed_experiments.empty:
        print(failed_experiments.to_string(index=False))
    save_loaded_recordings(loaded_recordings, SAVE_LOADED_RECORDINGS_PATH)
    tracks = make_tracks(loaded_recordings)
    print(f"Valid tracks: {len(tracks)}")
    if not tracks:
        raise RuntimeError("No valid tracks. Check QUERY_FILTERS and matrix fields.")

    geometry = analysis.get_gap_geometry(tracks)
    print(f"Gap regions: {len(geometry)}")
    events = analysis.make_event_table(tracks, geometry)
    events = events.merge(make_recording_metadata(loaded_recordings), on="source_file")
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
    analysis.plt.show()


if __name__ == "__main__":
    run()
