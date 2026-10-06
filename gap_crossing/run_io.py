"""Run folders, provenance, and saved datasets for gap-crossing analyses.

A load run writes one dataset folder per strain. An analysis run reads one or
more dataset folders and writes one analysis folder. Each folder name starts
with its creation time. This module creates new files only. It never deletes,
moves, or replaces a file.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import types

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Settings
DEFAULT_OUTPUT_ROOT = Path(
    r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui"
) / "gap_crossing"
OUTPUT_ROOT_ENV = "GAP_CROSSING_OUTPUT_ROOT"
REPO_ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP_FORMAT = "%Y-%m-%d_%H%M%S"
RUN_KINDS = ("datasets", "analyses")
ALLOWED_DATASET_DIFFERENCES = ("genotype_file",)
CAMERA_SETUP_FIELDS = (
    "recording_rig_name",
    "recording_camera_model",
    "recording_camera_resolution",
)
PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "joblib",
    "optogui",
)
FIGURE_FORMATS = ("png", "pdf")
FIGURE_DPI = 150

TRACKS_FILE = "tracks.joblib"
RECORDINGS_FILE = "recordings.csv"
EXPERIMENTS_FILE = "experiments.csv"
FAILED_FILE = "failed.csv"
QUERY_FILE = "query.json"
PARAMS_FILE = "params.json"
PROVENANCE_FILE = "provenance.json"
STATUS_FILE = "status.json"
LOG_FILE = "log.txt"
CODE_DIR = "code"


# %% Paths and JSON
def get_output_root() -> Path:
    """Return the output root, using the environment override when set."""
    override = os.environ.get(OUTPUT_ROOT_ENV)
    return Path(override) if override else DEFAULT_OUTPUT_ROOT


def make_timestamp(now: datetime | None = None) -> str:
    """Return the timestamp that starts a run folder name."""
    return (datetime.now() if now is None else now).strftime(TIMESTAMP_FORMAT)


def sanitize_label(label: str) -> str:
    """Return a label that is safe in a Windows folder name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("._-")
    if not cleaned:
        raise ValueError(f"Label {label!r} has no usable characters.")
    return cleaned


def make_run_dir(kind: str, label: str, timestamp: str | None = None) -> Path:
    """Create and return a new run folder; an existing folder is an error."""
    if kind not in RUN_KINDS:
        raise ValueError(f"Run kind must be one of {RUN_KINDS}, not {kind!r}.")
    timestamp = make_timestamp() if timestamp is None else timestamp
    run_dir = get_output_root() / kind / f"{timestamp}_{sanitize_label(label)}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    return run_dir


def require_new_path(path: Path) -> Path:
    """Return path, or raise when a file already exists there."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace an existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value):
    """Convert settings and metadata to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def write_json(path: Path, data) -> Path:
    """Write a new JSON file."""
    path = require_new_path(path)
    with open(path, "x", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2)
        file.write("\n")
    return path


def read_json(path: Path):
    """Read one JSON file."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def collect_settings(*modules: types.ModuleType) -> dict:
    """Return the uppercase settings of each module, keyed by module name."""
    settings = {}
    for module in modules:
        values = {}
        for name, value in vars(module).items():
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                continue
            if isinstance(value, (types.ModuleType, types.FunctionType, type)):
                continue
            values[name] = to_jsonable(value)
        name = module.__name__
        if name == "__main__" and getattr(module, "__file__", None):
            name = Path(module.__file__).stem
        settings[name] = values
    return settings


# %% Provenance
def _run_git(repo_dir: Path, *args: str) -> str | None:
    """Return git output, or None when git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def get_git_state(repo_dir: Path) -> dict:
    """Return commit, branch, dirty state, changed files, and diff text."""
    commit = _run_git(repo_dir, "rev-parse", "HEAD")
    if commit is None:
        return {
            "repo": str(repo_dir), "commit": None, "branch": None,
            "dirty": None, "status": None, "diff": None,
        }
    status = _run_git(repo_dir, "status", "--porcelain") or ""
    return {
        "repo": str(repo_dir),
        "commit": commit.strip(),
        "branch": (_run_git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD") or "").strip(),
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "diff": _run_git(repo_dir, "diff", "HEAD") or "",
    }


def get_optogui_repo() -> Path | None:
    """Return the optogui checkout used by this environment, when available."""
    try:
        import optogui
    except ImportError:
        return None
    for parent in Path(optogui.__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def get_package_versions() -> dict:
    """Return installed versions of the main analysis packages."""
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def make_provenance(run_type: str, script_path: Path, extra: dict | None = None) -> tuple[dict, dict]:
    """Return provenance and the diff text for each repository."""
    script_path = Path(script_path).resolve()
    repo_state = get_git_state(REPO_ROOT)
    optogui_repo = get_optogui_repo()
    optogui_state = get_git_state(optogui_repo) if optogui_repo else None
    try:
        script = str(script_path.relative_to(REPO_ROOT))
    except ValueError:
        script = str(script_path)

    provenance = {
        "run_type": run_type,
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": script,
        "git_commit": repo_state["commit"],
        "git_branch": repo_state["branch"],
        "git_dirty": repo_state["dirty"],
        "git_status": repo_state["status"],
        "optogui_commit": optogui_state["commit"] if optogui_state else None,
        "optogui_dirty": optogui_state["dirty"] if optogui_state else None,
        "python_version": sys.version,
        "packages": get_package_versions(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command_line": sys.argv,
        **(extra or {}),
    }
    diffs = {
        "git_diff.patch": repo_state["diff"],
        "optogui_git_diff.patch": optogui_state["diff"] if optogui_state else None,
    }
    return provenance, diffs


def snapshot_code(run_dir: Path, script_paths, diffs: dict) -> Path:
    """Copy the scripts and write non-empty git diffs into the code folder."""
    code_dir = Path(run_dir) / CODE_DIR
    code_dir.mkdir(exist_ok=False)
    for script_path in dict.fromkeys(Path(path).resolve() for path in script_paths):
        shutil.copy2(script_path, require_new_path(code_dir / script_path.name))
    for name, diff in diffs.items():
        if diff:
            require_new_path(code_dir / name).write_text(diff, encoding="utf-8")
    return code_dir


class _Tee:
    """Write text to the console and a log file."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def tee_log(run_dir: Path):
    """Copy printed output to log.txt in the run folder."""
    path = require_new_path(Path(run_dir) / LOG_FILE)
    with open(path, "x", encoding="utf-8") as log_file:
        console = sys.stdout
        sys.stdout = _Tee(console, log_file)
        try:
            yield path
        finally:
            sys.stdout = console


@contextmanager
def _recorded_run(run_dir: Path):
    """Log output and write status.json when the run ends."""
    started = datetime.now()
    with tee_log(run_dir):
        try:
            yield
        except BaseException as error:
            write_json(run_dir / STATUS_FILE, {
                "status": "failed",
                "started": started.isoformat(timespec="seconds"),
                "ended": datetime.now().isoformat(timespec="seconds"),
                "error": f"{type(error).__name__}: {error}",
            })
            raise
    write_json(run_dir / STATUS_FILE, {
        "status": "completed",
        "started": started.isoformat(timespec="seconds"),
        "ended": datetime.now().isoformat(timespec="seconds"),
    })


# %% Datasets
@dataclass
class Dataset:
    """One saved dataset folder."""

    path: Path
    tracks: list[dict]
    recordings: pd.DataFrame
    experiments: pd.DataFrame
    failed: pd.DataFrame
    query: dict
    params: dict
    provenance: dict


@contextmanager
def dataset_run(script_path: Path, label: str, timestamp: str | None = None, extra: dict | None = None):
    """Create one dataset folder with provenance, code, log, and status."""
    run_dir = make_run_dir("datasets", label, timestamp)
    provenance, diffs = make_provenance("dataset", script_path, extra)
    write_json(run_dir / PROVENANCE_FILE, provenance)
    snapshot_code(run_dir, [script_path, __file__], diffs)
    with _recorded_run(run_dir):
        print(f"Dataset folder: {run_dir}")
        yield run_dir


def save_dataset(
    run_dir: Path,
    *,
    tracks: list[dict],
    recordings: pd.DataFrame,
    experiments: pd.DataFrame,
    failed: pd.DataFrame,
    query: dict,
    params: dict,
) -> Path:
    """Write processed tracks and their tables into a dataset folder."""
    run_dir = Path(run_dir)
    joblib.dump(
        {"tracks": tracks, "recordings": recordings},
        require_new_path(run_dir / TRACKS_FILE),
        compress=3,
    )
    recordings.to_csv(require_new_path(run_dir / RECORDINGS_FILE), index=False)
    experiments.to_csv(require_new_path(run_dir / EXPERIMENTS_FILE), index=False)
    failed.to_csv(require_new_path(run_dir / FAILED_FILE), index=False)
    write_json(run_dir / QUERY_FILE, query)
    write_json(run_dir / PARAMS_FILE, params)
    return run_dir


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV table; an empty file gives an empty table."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def upcast_track(track: dict) -> dict:
    """Return the track with float32 arrays converted to float64.

    Datasets store float32 to save space. Analysis code runs in float64 so
    results match a float64 load; float32 arithmetic shifts gap edges.
    """
    return {
        name: values.astype(np.float64)
        if isinstance(values, np.ndarray) and values.dtype == np.float32 else values
        for name, values in track.items()
    }


def load_dataset(dataset_dir: Path) -> Dataset:
    """Load one dataset folder; float32 track arrays are returned as float64."""
    dataset_dir = Path(dataset_dir)
    required = [TRACKS_FILE, QUERY_FILE, PARAMS_FILE, PROVENANCE_FILE]
    missing = [name for name in required if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset folder {dataset_dir} is missing {missing}.")
    stored = joblib.load(dataset_dir / TRACKS_FILE)
    return Dataset(
        path=dataset_dir,
        tracks=[upcast_track(track) for track in stored["tracks"]],
        recordings=stored["recordings"],
        experiments=_read_csv(dataset_dir / EXPERIMENTS_FILE),
        failed=_read_csv(dataset_dir / FAILED_FILE),
        query=read_json(dataset_dir / QUERY_FILE),
        params=read_json(dataset_dir / PARAMS_FILE),
        provenance=read_json(dataset_dir / PROVENANCE_FILE),
    )


def _comparable_query(query: dict, allowed_differences) -> dict:
    """Return the query without fields that may differ between datasets."""
    query = deepcopy(query)
    for key in allowed_differences:
        query.pop(key, None)
        query.get("query_filters", {}).pop(key, None)
    return query


def _differing_keys(left: dict, right: dict) -> list[str]:
    """Return top-level and query-filter keys whose values differ."""
    keys = []
    for key in sorted(set(left) | set(right)):
        if key == "query_filters" and isinstance(left.get(key), dict):
            nested_left, nested_right = left.get(key, {}), right.get(key, {})
            keys.extend(
                f"query_filters.{name}"
                for name in sorted(set(nested_left) | set(nested_right))
                if nested_left.get(name) != nested_right.get(name)
            )
        elif left.get(key) != right.get(key):
            keys.append(key)
    return keys


def check_dataset_compatibility(
    datasets: dict[str, Dataset],
    allowed_differences=ALLOWED_DATASET_DIFFERENCES,
) -> list[str]:
    """Raise when datasets were loaded differently; return code warnings."""
    labels = list(datasets)
    reference_label = labels[0]
    reference = datasets[reference_label]
    reference_query = _comparable_query(reference.query, allowed_differences)
    problems = []
    warnings = []
    for label in labels[1:]:
        dataset = datasets[label]
        query_keys = _differing_keys(
            reference_query, _comparable_query(dataset.query, allowed_differences)
        )
        if query_keys:
            problems.append(f"{label} query differs from {reference_label} in {query_keys}")
        param_keys = _differing_keys(reference.params, dataset.params)
        if param_keys:
            problems.append(f"{label} params differ from {reference_label} in {param_keys}")
        for key in ("git_commit", "optogui_commit"):
            if dataset.provenance.get(key) != reference.provenance.get(key):
                warnings.append(
                    f"{label} was loaded with a different {key} than {reference_label}."
                )

    source_counts = pd.concat(
        [pd.Series(dataset.recordings["source_file"].unique()) for dataset in datasets.values()],
        ignore_index=True,
    ).value_counts()
    duplicated = source_counts[source_counts > 1].index.tolist()
    if duplicated:
        problems.append(f"Recordings appear in more than one dataset: {duplicated[:5]}")

    recordings = pd.concat([dataset.recordings for dataset in datasets.values()], ignore_index=True)
    for column in CAMERA_SETUP_FIELDS:
        if column not in recordings:
            warnings.append(f"Camera setup check skipped: recordings have no {column}.")
            continue
        values = recordings[column].dropna().astype(str).unique().tolist()
        if len(values) > 1:
            problems.append(
                f"Recordings mix camera setups in {column}: {values}. Standardize "
                "arena orientation before combining them."
            )

    if problems:
        raise ValueError("Datasets cannot be combined:\n- " + "\n- ".join(problems))
    return warnings


def load_datasets(
    dataset_dirs: dict[str, Path],
    allowed_differences=ALLOWED_DATASET_DIFFERENCES,
) -> tuple[dict[str, Dataset], list[str]]:
    """Load labeled dataset folders and check that they can be combined."""
    if not dataset_dirs:
        raise ValueError("Set DATASET_DIRS to at least one dataset folder.")
    datasets = {
        sanitize_label(label): load_dataset(path) for label, path in dataset_dirs.items()
    }
    return datasets, check_dataset_compatibility(datasets, allowed_differences)


def combine_datasets(datasets: dict[str, Dataset]) -> tuple[list[dict], pd.DataFrame]:
    """Return all tracks and recording rows labeled by dataset."""
    tracks = []
    recordings = []
    for label, dataset in datasets.items():
        tracks.extend({**track, "dataset_label": label} for track in dataset.tracks)
        recordings.append(dataset.recordings.assign(dataset_label=label))
    return tracks, pd.concat(recordings, ignore_index=True)


# %% Analysis runs
def _figure_title(figure) -> str:
    """Return a short title for a figure file name."""
    suptitle = figure.get_suptitle()
    if suptitle:
        return suptitle
    for axis in figure.axes:
        if axis.get_title():
            return axis.get_title()
    return f"figure{figure.number}"


def _scalar_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose cells hold arrays or lists."""
    keep = [
        column for column in table.columns
        if not table[column].map(lambda value: isinstance(value, (np.ndarray, list, dict))).any()
    ]
    return table[keep]


@dataclass
class AnalysisRun:
    """One analysis folder and the datasets it reads."""

    path: Path
    datasets: dict[str, Dataset]
    tracks: list[dict]
    recordings: pd.DataFrame
    saved_figures: set = field(default_factory=set)

    @property
    def sources(self) -> dict:
        """Return the path, query, and track filters of each dataset."""
        return {
            label: {"path": str(dataset.path), "query": dataset.query, "params": dataset.params}
            for label, dataset in self.datasets.items()
        }

    def save_figures(self) -> list[Path]:
        """Save each open figure that this run has not saved yet."""
        figure_dir = self.path / "figures"
        figure_dir.mkdir(exist_ok=True)
        saved = []
        for number in plt.get_fignums():
            figure = plt.figure(number)
            if id(figure) in self.saved_figures:
                continue
            index = len(self.saved_figures) + 1
            stem = f"{index:02d}_{sanitize_label(_figure_title(figure))[:60]}"
            for extension in FIGURE_FORMATS:
                path = require_new_path(figure_dir / f"{stem}.{extension}")
                figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
                saved.append(path)
            self.saved_figures.add(id(figure))
        return saved

    def show(self) -> None:
        """Save open figures, then display them."""
        self.save_figures()
        plt.show()

    def save_table(self, name: str, table: pd.DataFrame) -> Path:
        """Save one table as CSV, leaving out array-valued columns."""
        path = require_new_path(self.path / "tables" / f"{sanitize_label(name)}.csv")
        _scalar_columns(table).to_csv(path, index=False)
        return path

    def export_path(self, filename: str) -> Path:
        """Return a new path in the exports folder."""
        return require_new_path(self.path / "exports" / filename)


@contextmanager
def analysis_run(
    script_path: Path,
    dataset_dirs: dict[str, Path],
    settings_modules=(),
    allowed_differences=ALLOWED_DATASET_DIFFERENCES,
):
    """Load datasets, create one analysis folder, and record the run."""
    datasets, warnings = load_datasets(dataset_dirs, allowed_differences)
    tracks, recordings = combine_datasets(datasets)
    run_dir = make_run_dir("analyses", Path(script_path).stem)
    provenance, diffs = make_provenance(
        "analysis",
        script_path,
        {
            "datasets": {
                label: {"path": str(dataset.path), "provenance": dataset.provenance}
                for label, dataset in datasets.items()
            },
            "allowed_dataset_differences": list(allowed_differences),
            "compatibility_warnings": warnings,
        },
    )
    write_json(run_dir / PROVENANCE_FILE, provenance)
    write_json(run_dir / PARAMS_FILE, collect_settings(*settings_modules))
    module_files = [module.__file__ for module in settings_modules]
    snapshot_code(run_dir, [script_path, *module_files, __file__], diffs)
    with _recorded_run(run_dir):
        print(f"Analysis folder: {run_dir}")
        for label, dataset in datasets.items():
            print(f"Dataset {label}: {dataset.path} ({len(dataset.tracks)} tracks)")
        for warning in warnings:
            print(f"Warning: {warning}")
        yield AnalysisRun(run_dir, datasets, tracks, recordings)
