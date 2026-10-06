"""Load gap-crossing tracks from the optogui database into dataset folders.

Edit the settings, then run this script. Each genotype in GENOTYPE_FILES gets
its own dataset folder under the run_io output root. All genotypes use the same
query filters, periods, matrix fields, and track filters. Analysis scripts read
these folders through their DATASET_DIRS setting.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from optogui.analysis import (
    get_available_fields,
    load_experiment_data,
    query_experiments,
)

try:
    from gap_crossing import run_io
except ModuleNotFoundError:
    import run_io


# %% Query settings
DATASET_LABEL = "gap_ribbon"
GENOTYPE_FILES = [
    "GMOCLKir_empty.yaml",
    # "GMOCLKir_FC2.yaml",
    # "GMOCLKir_86861.yaml",
    # "117_GMUCR.yaml",
    # "OCLKir_GMUCR.yaml",
]
DATABASE_LOCATION = "server"
DATA_LOCATION = "server"
QUERY_FILTERS = {
    "experimenter": "kevin",
    # "vial": [0, 1],
    "stim_protocol": "users.kevin.intermittent_gaps_ribbon",
}
QUERY_PERIODS = [
    # {"year": 2025, "month": [12]},
    # {"year": 2026, "month": [5, 6, 7, 8, 9]},
    {"year": 2026, "month": [8], "day": [19]},
]
MAX_EXPERIMENTS = None

MATRIX_FIELDS = [
    "headx_smooth",
    "heady_smooth",
    "signal",
    "t",
    "vx_smooth",
    "vy_smooth",
    "spd_smooth",
    "theta",
    "theta_smooth",
    "dtheta_smooth",
    "jumps",
]
RECORDING_FIELDS = [
    "id",
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
    "genotype_file",
    "rig_name",
    "camera_model",
    "camera_resolution",
]

# %% Track filters
FRAME_RATE_HZ = 60
MIN_TRACK_S = 10
MIN_MEAN_SPEED_MM_S = 0.1
MAX_SPEED_MM_S = 50
# About one third of tracks contain at least one jump frame. Tracks keep their
# jumps array so analyses can mask those frames instead.
DROP_TRACKS_WITH_JUMPS = False


@contextmanager
def suppress_loader_logs():
    """Temporarily suppress logs from one-record public loader calls."""
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)


# %% Query
def validate_query_fields(available_fields):
    """Raise when a query field is not a database field.

    optogui skips unknown fields with a warning, which would widen the query.
    """
    requested = {"genotype_file", *QUERY_FILTERS}
    for period in QUERY_PERIODS:
        requested.update(period)
    unknown = sorted(requested.difference(available_fields))
    if unknown:
        raise ValueError(f"Unknown database query fields: {unknown}")


def make_query_filters(genotype_file):
    """Return the shared query filters for one genotype."""
    return {**QUERY_FILTERS, "genotype_file": genotype_file}


def make_query_record(genotype_file):
    """Return the query settings stored in one dataset folder."""
    return {
        "database_location": DATABASE_LOCATION,
        "data_location": DATA_LOCATION,
        "query_filters": make_query_filters(genotype_file),
        "query_periods": QUERY_PERIODS,
        "max_experiments": MAX_EXPERIMENTS,
        "matrix_fields": MATRIX_FIELDS,
        "recording_fields": RECORDING_FIELDS,
    }


def make_track_params():
    """Return the track filters stored in one dataset folder."""
    return {
        "frame_rate_hz": FRAME_RATE_HZ,
        "min_track_s": MIN_TRACK_S,
        "min_mean_speed_mm_s": MIN_MEAN_SPEED_MM_S,
        "max_speed_mm_s": MAX_SPEED_MM_S,
        "drop_tracks_with_jumps": DROP_TRACKS_WITH_JUMPS,
    }


def select_experiments(query_filters):
    """Return the database records selected by the filters and periods."""
    query_results = [
        query_experiments(
            db_location=DATABASE_LOCATION,
            order_by="id",
            **query_filters,
            **period,
        )
        for period in QUERY_PERIODS
    ]
    experiments = pd.concat(query_results, ignore_index=True)
    if experiments.empty:
        return experiments
    experiments = experiments.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    if MAX_EXPERIMENTS is not None:
        experiments = experiments.head(MAX_EXPERIMENTS).copy()
    return experiments


# %% Load tracks
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
    min_frames = int(MIN_TRACK_S * FRAME_RATE_HZ)
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
            velocity = np.column_stack((data["vx_smooth"][index], data["vy_smooth"][index]))
            series = {
                name: np.asarray(data[field][index]).squeeze().astype(dtype)
                for name, field, dtype in (
                    ("signal", "signal", float),
                    ("time_s", "t", float),
                    ("speed_smooth", "spd_smooth", float),
                    ("theta", "theta", float),
                    ("theta_smooth", "theta_smooth", float),
                    ("dtheta_smooth", "dtheta_smooth", float),
                    ("jumps", "jumps", bool),
                )
            }
            if (
                xy.shape != (len(index), 2)
                or any(values.shape != (len(index),) for values in series.values())
                or not np.isfinite(velocity).all()
            ):
                continue
            if DROP_TRACKS_WITH_JUMPS and series["jumps"].any():
                continue

            speed = np.linalg.norm(velocity, axis=1)
            if np.nanmean(speed) <= MIN_MEAN_SPEED_MM_S or np.nanmax(speed) >= MAX_SPEED_MM_S:
                continue

            tracks.append(
                {
                    "track_id": f"{source_file}::track{local_id}",
                    "source_file": source_file,
                    "xy": xy,
                    "velocity": velocity,
                    **series,
                }
            )
    return tracks


# %% Run
def select_genotype_experiments():
    """Return database records for each genotype; every genotype must match."""
    if not GENOTYPE_FILES:
        raise ValueError("Set GENOTYPE_FILES to at least one genotype file.")
    validate_query_fields(get_available_fields())

    experiments_by_genotype = {
        genotype_file: select_experiments(make_query_filters(genotype_file))
        for genotype_file in GENOTYPE_FILES
    }
    for genotype_file, experiments in experiments_by_genotype.items():
        print(f"{genotype_file}: {len(experiments)} database records")
    empty = [name for name, experiments in experiments_by_genotype.items() if experiments.empty]
    if empty:
        raise RuntimeError(f"No database records matched these genotype files: {empty}")
    return experiments_by_genotype


def load_genotype_tracks(genotype_file, experiments):
    """Load one genotype's records and return tracks, metadata, and failures."""
    print(f"Genotype file: {genotype_file}")
    print(f"Database records: {len(experiments)}")
    loaded_recordings, failed_experiments = load_recordings(experiments)
    print(f"Loaded recordings: {len(loaded_recordings)}")
    print(f"Failed recordings: {len(failed_experiments)}")
    if not failed_experiments.empty:
        print(failed_experiments.to_string(index=False))
    tracks = make_tracks(loaded_recordings)
    print(f"Valid tracks: {len(tracks)}")
    if not tracks:
        raise RuntimeError(f"No valid tracks for {genotype_file}. Check the query and matrix fields.")
    return tracks, make_recording_metadata(loaded_recordings), failed_experiments


def load_tracks_from_database():
    """Query and load every genotype without saving anything.

    Returns tracks and recording metadata labeled by genotype-file stem, plus
    the query and track filters of each genotype.
    """
    tracks = []
    recordings = []
    sources = {}
    for genotype_file, experiments in select_genotype_experiments().items():
        label = run_io.sanitize_label(Path(genotype_file).stem)
        genotype_tracks, genotype_recordings, _ = load_genotype_tracks(genotype_file, experiments)
        tracks.extend({**track, "dataset_label": label} for track in genotype_tracks)
        recordings.append(genotype_recordings.assign(dataset_label=label))
        sources[label] = {
            "path": None,
            "query": make_query_record(genotype_file),
            "params": make_track_params(),
        }
    return tracks, pd.concat(recordings, ignore_index=True), sources


def load_genotype_dataset(genotype_file, experiments, timestamp):
    """Load one genotype and write its dataset folder."""
    label = f"{DATASET_LABEL}_{Path(genotype_file).stem}"
    extra = {
        "batch_id": f"{timestamp}_{run_io.sanitize_label(DATASET_LABEL)}",
        "batch_genotype_files": GENOTYPE_FILES,
        "dataset_label": label,
    }
    with run_io.dataset_run(__file__, label, timestamp, extra) as run_dir:
        tracks, recordings, failed_experiments = load_genotype_tracks(genotype_file, experiments)
        run_io.save_dataset(
            run_dir,
            tracks=tracks,
            recordings=recordings,
            experiments=experiments,
            failed=failed_experiments,
            query=make_query_record(genotype_file),
            params=make_track_params(),
        )
    return run_dir


def run():
    """Query every genotype, then load and save one dataset per genotype."""
    experiments_by_genotype = select_genotype_experiments()
    timestamp = run_io.make_timestamp()
    dataset_dirs = {
        genotype_file: load_genotype_dataset(genotype_file, experiments, timestamp)
        for genotype_file, experiments in experiments_by_genotype.items()
    }
    print("Dataset folders:")
    for genotype_file, run_dir in dataset_dirs.items():
        print(f'    "{Path(genotype_file).stem}": Path(r"{run_dir}"),')
    return dataset_dirs


if __name__ == "__main__":
    run()
