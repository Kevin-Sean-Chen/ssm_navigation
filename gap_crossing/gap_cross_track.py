# -*- coding: utf-8 -*-
"""Within-track gap-crossing analysis.

An attempt is a downwind-to-upwind crossing of a loss boundary with odor in
the prior PRE_SIGNAL_S seconds. The first qualifying odor return in the next
OUTCOME_S seconds defines the outcome:
    cross:  return at least DISTANCE_MM upwind of the boundary.
    regain: return at least DISTANCE_MM downwind of the boundary.
    abort:  no qualifying return.
"""

from pathlib import Path
from collections import Counter
from itertools import product

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from natsort import natsorted
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# %% Settings
ROOT_DIR = Path(
    r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-12-15\kevin"
)
EXPERIMENT_TEXT = "same"
FORBIDDEN_TEXT = []
TARGET_FILE = "exp_matrix.joblib"

GAP_GEOMETRY_METHOD = "detected"  # "manual" or "detected"
MANUAL_LOSS_X_MM = np.array([76, 128, 181, 232]) +2
FRAME_RATE_HZ = 60
MIN_TRACK_S = 10
PRE_SIGNAL_S = 5
OUTCOME_S = 5
DISTANCE_MM = 3
MIN_ATTEMPTS_PER_TRACK = 2
ATTEMPT_RIBBON_HALF_WIDTH_MM = 7
POST_SPEED_S = 1
MIN_POST_SPEED_SAMPLES = 30
MOTIF_PERMUTATIONS = 1000
MOTIF_RANDOM_SEED = 0
MIN_TRANSITIONS_FOR_PHASE = 3

# Detection settings. Use these only with GAP_GEOMETRY_METHOD = "detected".
GRID_MM = 0.2
SMOOTH_X_MM = .5
SMOOTH_Y_MM = .5
EXPECTED_RIBBON_COUNT = 4
EXPECTED_GAPS_PER_RIBBON = 4
GAP_FIT_SEARCH_HALF_WINDOW_MM = 5
MIN_GAP_WIDTH_MM = 2
MAX_GAP_WIDTH_MM = 12

OUTCOME_ORDER = ["cross", "regain", "abort"]
OUTCOME_COLOR = {"cross": "C0", "regain": "C2", "abort": "C3"}

PLOT_MAX_TRACKS = 1000
PLOT_MAX_FRAMES_PER_TRACK = 300
sns.set_style("white")
sns.set_context("talk")


# %% Load tracks
def get_target_files(root_dir):
    """Return sorted experiment files that match the settings."""
    files = []
    for folder in root_dir.iterdir():
        if not folder.is_dir():
            continue
        target = folder / TARGET_FILE
        if (
            target.is_file()
            and EXPERIMENT_TEXT in str(folder)
            and not any(text in str(folder) for text in FORBIDDEN_TEXT)
        ):
            files.append(target)
    return [Path(path) for path in natsorted(files)]


def load_tracks(target_files):
    """Load valid tracks and give every track a unique ID."""
    tracks = []
    min_frames = int(MIN_TRACK_S * FRAME_RATE_HZ)

    for file_index, file_path in enumerate(target_files):
        try:
            data = joblib.load(file_path)
        except Exception as error:
            print(f"Skip unreadable file: {file_path} | {error}")
            continue

        required = {
            "trjn", "headx_smooth", "heady_smooth", "signal", "t",
            "vx_smooth", "vy_smooth", 'spd_smooth',
        }
        missing = required.difference(data)
        if missing:
            print(f"Skip file with missing data: {file_path} | {sorted(missing)}")
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
                    "track_id": f"file{file_index}_track{local_id}",
                    "source_file": str(file_path),
                    "xy": xy,
                    "signal": signal,
                    "time_s": time_s,
                    "velocity": velocity,
                    "speed_smooth": speed_smooth,
                }
            )
    return tracks


# %% Gap geometry
def get_odor_xy(tracks):
    """Return finite positions where odor signal is present."""
    xy, odor = get_position_signal(tracks)
    if not np.any(odor):
        raise RuntimeError("No finite odor samples for gap detection.")
    return xy[odor]


def get_position_signal(tracks):
    """Return finite positions and their binary odor state."""
    xy = np.concatenate([track["xy"] for track in tracks])
    signal = np.concatenate([track["signal"] for track in tracks])
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(signal)
    return xy[valid], signal[valid] > 0


def get_plot_position_signal(
    tracks,
    max_tracks=PLOT_MAX_TRACKS,
    max_frames_per_track=PLOT_MAX_FRAMES_PER_TRACK,
):
    """Return a bounded, deterministic sample for the geometry plot."""
    if max_tracks < 1 or max_frames_per_track < 1:
        raise ValueError("Plot sample limits must be positive.")

    raw_track_count = len(tracks)
    raw_frame_count = sum(len(track["signal"]) for track in tracks)
    track_count = min(raw_track_count, max_tracks)
    track_indices = np.linspace(0, raw_track_count - 1, track_count, dtype=int)
    plot_xy = []
    plot_signal = []
    track_ids = []

    for track_index in track_indices:
        track = tracks[track_index]
        frame_count = min(len(track["signal"]), max_frames_per_track)
        frame_indices = np.linspace(
            0, len(track["signal"]) - 1, frame_count, dtype=int
        )
        xy = np.asarray(track["xy"])[frame_indices]
        signal = np.asarray(track["signal"])[frame_indices]
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(signal)
        plot_xy.append(xy[valid])
        plot_signal.append(signal[valid])
        track_ids.append(track["track_id"])

    xy = np.concatenate(plot_xy)
    signal = np.concatenate(plot_signal)
    details = {
        "raw_track_count": raw_track_count,
        "displayed_track_count": track_count,
        "raw_frame_count": raw_frame_count,
        "displayed_frame_count": len(xy),
        "track_ids": track_ids,
    }
    return xy, signal, details


    """Estimate odor probability along one ribbon centerline strip."""
def get_centerline_signal_trace(xy, odor, y_min, y_max, x_edges):
    all_count, odor_count = get_centerline_signal_counts(
        xy, odor, y_min, y_max, x_edges
    )
    smooth_odor = gaussian_filter1d(odor_count.astype(float), SMOOTH_X_MM / GRID_MM)
    smooth_all = gaussian_filter1d(all_count.astype(float), SMOOTH_X_MM / GRID_MM)
    return np.divide(smooth_odor, smooth_all, out=np.zeros_like(smooth_odor), where=smooth_all > 0)


def get_centerline_signal_counts(xy, odor, y_min, y_max, x_edges):
    """Return total and odor sample counts in one ribbon centerline strip."""
    in_strip = (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)
    all_count, _ = np.histogram(xy[in_strip, 0], bins=x_edges)
    odor_count, _ = np.histogram(xy[in_strip & odor, 0], bins=x_edges)
    return all_count, odor_count


def get_robust_gap_edges(
    x_centers,
    odor_count,
    all_count,
    gap_center_x_mm,
    search_half_window_mm,
    min_gap_width_mm,
    max_gap_width_mm,
):
    """Fit one gap interval by weighted agreement with binary signal data."""
    x_centers = np.asarray(x_centers, dtype=float)
    odor_count = np.asarray(odor_count, dtype=float)
    all_count = np.asarray(all_count, dtype=float)
    if not (len(x_centers) and x_centers.shape == odor_count.shape == all_count.shape):
        raise ValueError("x centers and sample counts must have the same nonzero length.")
    if np.any(odor_count < 0) or np.any(all_count < odor_count):
        raise ValueError("odor counts must be between zero and all sample counts.")

    x_edges = np.empty(len(x_centers) + 1, dtype=float)
    x_edges[1:-1] = 0.5 * (x_centers[:-1] + x_centers[1:])
    x_edges[0] = x_centers[0] - 0.5 * (x_centers[1] - x_centers[0])
    x_edges[-1] = x_centers[-1] + 0.5 * (x_centers[-1] - x_centers[-2])

    # A positive value favors odor. A negative value favors a gap. Counts
    # preserve the evidence strength, so a sparse outlier cannot set an edge.
    gap_delta = 2 * odor_count - all_count
    best = None
    for start_index in range(len(x_centers)):
        left_edge = x_edges[start_index]
        if not gap_center_x_mm - search_half_window_mm <= left_edge <= gap_center_x_mm:
            continue
        for stop_index in range(start_index + 1, len(x_centers) + 1):
            right_edge = x_edges[stop_index]
            width = right_edge - left_edge
            if right_edge < gap_center_x_mm:
                continue
            if right_edge > gap_center_x_mm + search_half_window_mm:
                break
            if not min_gap_width_mm <= width <= max_gap_width_mm:
                continue
            score = gap_delta[start_index:stop_index].sum()
            candidate = (score, left_edge, right_edge)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise RuntimeError("No valid gap interval near the detected center.")
    _, left_edge, right_edge = best
    return left_edge, right_edge


def select_strongest_peaks(trace, expected_count, invert=False):
    """Return the strongest, well-spaced peaks in a one-dimensional trace."""
    values = -trace if invert else trace
    min_separation = max(1, len(trace) // (2 * expected_count))
    peaks, properties = find_peaks(values, distance=min_separation, prominence=0)
    if len(peaks) < expected_count:
        raise RuntimeError(
            f"Found {len(peaks)} peaks, expected {expected_count}. "
            "Check the selected experiment data."
        )
    strongest = np.argsort(properties["prominences"])[-expected_count:]
    return np.sort(peaks[strongest])


def get_half_height_edges(trace, valley_index):
    """Return the signal edges around one valley at its local half height."""
    maxima, _ = find_peaks(trace)
    left_maxima = maxima[maxima < valley_index]
    right_maxima = maxima[maxima > valley_index]
    if not len(left_maxima) or not len(right_maxima):
        return None

    left_peak = left_maxima[-1]
    right_peak = right_maxima[0]
    baseline = min(trace[left_peak], trace[right_peak])
    level = trace[valley_index] + 0.5 * (baseline - trace[valley_index])

    left_candidates = np.flatnonzero(trace[:valley_index + 1] >= level)
    right_candidates = np.flatnonzero(trace[valley_index:] >= level) + valley_index
    if not len(left_candidates) or not len(right_candidates):
        return None
    return left_candidates[-1], right_candidates[0]


def make_manual_geometry():
    """Make four global loss boundaries for comparison with detected gaps."""
    return pd.DataFrame(
        {
            "geometry_id": np.arange(len(MANUAL_LOSS_X_MM)),
            "ribbon_id": np.nan,
            "gap_id": np.arange(len(MANUAL_LOSS_X_MM)),
            "gap_label": [f"G{gap + 1}" for gap in range(len(MANUAL_LOSS_X_MM))],
            "ribbon_center_y_mm": np.nan,
            "y_min_mm": np.nan,
            "y_max_mm": np.nan,
            "attempt_y_min_mm": np.nan,
            "attempt_y_max_mm": np.nan,
            "regain_edge_x_mm": np.nan,
            "loss_edge_x_mm": MANUAL_LOSS_X_MM,
            "gap_width_mm": np.nan,
        }
    )


def detect_gap_geometry(tracks):
    """Detect gap edges from four odor-ribbon centerline traces."""
    xy, odor = get_position_signal(tracks)
    odor_xy = xy[odor]
    if not len(odor_xy):
        raise RuntimeError("No finite odor samples for gap detection.")

    x_min = np.floor(odor_xy[:, 0].min())
    x_max = np.ceil(odor_xy[:, 0].max())
    y_min = np.floor(odor_xy[:, 1].min())
    y_max = np.ceil(odor_xy[:, 1].max())
    x_edges = np.arange(x_min, x_max + GRID_MM, GRID_MM)
    y_edges = np.arange(y_min, y_max + GRID_MM, GRID_MM)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    y_count, _ = np.histogram(odor_xy[:, 1], bins=y_edges)
    y_density = gaussian_filter1d(y_count.astype(float), SMOOTH_Y_MM / GRID_MM)
    ribbon_peaks = select_strongest_peaks(y_density, EXPECTED_RIBBON_COUNT)
    ribbon_y = y_centers[ribbon_peaks]
    rows = []

    for ribbon_id, (peak_index, center_y) in enumerate(zip(ribbon_peaks, ribbon_y)):
        _, _, left_ip, right_ip = peak_widths(y_density, [peak_index], rel_height=0.5)
        ribbon_y_min = np.interp(left_ip[0], np.arange(len(y_centers)), y_centers)
        ribbon_y_max = np.interp(right_ip[0], np.arange(len(y_centers)), y_centers)
        all_count, odor_count = get_centerline_signal_counts(
            xy, odor, ribbon_y_min, ribbon_y_max, x_edges
        )
        smooth_odor = gaussian_filter1d(
            odor_count.astype(float), SMOOTH_X_MM / GRID_MM
        )
        smooth_all = gaussian_filter1d(
            all_count.astype(float), SMOOTH_X_MM / GRID_MM
        )
        x_density = np.divide(
            smooth_odor, smooth_all, out=np.zeros_like(smooth_odor), where=smooth_all > 0
        )
        valley_indices = select_strongest_peaks(
            x_density, EXPECTED_GAPS_PER_RIBBON, invert=True
        )
        for gap_id, valley_index in enumerate(valley_indices):
            regain_edge, loss_edge = get_robust_gap_edges(
                x_centers,
                odor_count,
                all_count,
                x_centers[valley_index],
                GAP_FIT_SEARCH_HALF_WINDOW_MM,
                MIN_GAP_WIDTH_MM,
                MAX_GAP_WIDTH_MM,
            )
            rows.append(
                {
                    "geometry_id": len(rows),
                    "ribbon_id": ribbon_id,
                    "gap_id": gap_id,
                    "gap_label": f"R{ribbon_id + 1}-G{gap_id + 1}",
                    "ribbon_center_y_mm": center_y,
                    "y_min_mm": ribbon_y_min,
                    "y_max_mm": ribbon_y_max,
                    "attempt_y_min_mm": center_y - ATTEMPT_RIBBON_HALF_WIDTH_MM,
                    "attempt_y_max_mm": center_y + ATTEMPT_RIBBON_HALF_WIDTH_MM,
                    "gap_center_x_mm": x_centers[valley_index],
                    "regain_edge_x_mm": regain_edge,
                    "loss_edge_x_mm": loss_edge,
                    "gap_width_mm": loss_edge - regain_edge,
                }
            )

    geometry = pd.DataFrame(rows)
    print("Detected centerline gap geometry:")
    print(geometry[["gap_label", "ribbon_center_y_mm", "gap_center_x_mm", "loss_edge_x_mm", "gap_width_mm"]])
    return geometry


def get_gap_geometry(tracks):
    """Return manual or data-driven gap geometry."""
    if GAP_GEOMETRY_METHOD == "manual":
        return make_manual_geometry()
    if GAP_GEOMETRY_METHOD == "detected":
        return detect_gap_geometry(tracks)
    raise ValueError("GAP_GEOMETRY_METHOD must be 'manual' or 'detected'.")


def plot_gap_geometry(tracks, geometry):
    """Overlay the selected gap geometry on the pooled odor map."""
    xy, signal, details = get_plot_position_signal(tracks)
    odor = signal > 0

    fig, axis = plt.subplots(figsize=(11, 7))
    axis.plot(xy[:, 0], xy[:, 1], "k,", alpha=0.2, label="displayed samples")
    axis.plot(xy[odor, 0], xy[odor, 1], "r,", alpha=0.8, label="odor signal")
    for _, gap in geometry.iterrows():
        if np.isfinite(gap["attempt_y_min_mm"]):
            rectangle = Rectangle(
                (gap["regain_edge_x_mm"], gap["attempt_y_min_mm"]),
                gap["loss_edge_x_mm"] - gap["regain_edge_x_mm"],
                gap["attempt_y_max_mm"] - gap["attempt_y_min_mm"],
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
            )
            axis.add_patch(rectangle)
            axis.text(gap["loss_edge_x_mm"], gap["attempt_y_max_mm"], gap["gap_label"], color="cyan", fontsize=8)
            axis.plot(gap["gap_center_x_mm"], gap["ribbon_center_y_mm"], "co", ms=3)
        else:
            axis.axvline(gap["loss_edge_x_mm"], color="cyan", lw=1.5)
    axis.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=(
            f"Gap geometry: {GAP_GEOMETRY_METHOD} "
            f"({details['displayed_frame_count']:,}/{details['raw_frame_count']:,} frames; "
            f"{details['displayed_track_count']}/{details['raw_track_count']} tracks)"
        ),
    )
    axis.legend(markerscale=8)
    fig.tight_layout()


def plot_centerline_profiles(tracks, geometry):
    """Plot the four centerline odor traces used for detected gaps."""
    if GAP_GEOMETRY_METHOD != "detected":
        return

    xy, odor = get_position_signal(tracks)
    odor_xy = xy[odor]
    x_edges = np.arange(
        np.floor(odor_xy[:, 0].min()), np.ceil(odor_xy[:, 0].max()) + GRID_MM, GRID_MM
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    fig, axes = plt.subplots(EXPECTED_RIBBON_COUNT, 1, figsize=(12, 9), sharex=True)

    for ribbon_id, axis in enumerate(np.ravel(axes)):
        ribbon_gaps = geometry.loc[geometry["ribbon_id"] == ribbon_id]
        center_y = ribbon_gaps["ribbon_center_y_mm"].iloc[0]
        half_width = 0.5 * (
            ribbon_gaps["y_max_mm"].iloc[0] - ribbon_gaps["y_min_mm"].iloc[0]
        )
        density = get_centerline_signal_trace(
            xy, odor, center_y - half_width, center_y + half_width, x_edges
        )
        axis.plot(x_centers, density, color="black")
        for _, gap in ribbon_gaps.iterrows():
            axis.axvline(gap["gap_center_x_mm"], color="cyan", ls="--", label="gap center")
            axis.axvline(gap["regain_edge_x_mm"], color="C2", ls=":", label="upwind edge")
            axis.axvline(gap["loss_edge_x_mm"], color="C3", ls=":", label="loss edge")
        axis.set(ylabel=f"R{ribbon_id + 1}\nP(odor)", title=f"y center = {center_y:.1f} mm")
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("x (mm)")
    fig.tight_layout()


# %% Event definitions
def classify_outcome(xy, signal, time_s, attempt_index, boundary_x):
    """Return outcome and outcome index for one fixed-boundary attempt.

    The first signal frame that is at least DISTANCE_MM from the boundary
    decides the outcome. Signal returns in the boundary band do not decide it.
    """
    last_index = np.searchsorted(
        time_s, time_s[attempt_index] + OUTCOME_S, side="right"
    )
    future = np.arange(attempt_index + 1, min(last_index, len(time_s)))
    if not len(future):
        return "abort", np.nan

    odor = future[signal[future] > 0]
    qualified = odor[np.abs(xy[odor, 0] - boundary_x) >= DISTANCE_MM]
    if not len(qualified):
        return "abort", np.nan

    outcome_index = qualified[0]
    if xy[outcome_index, 0] <= boundary_x - DISTANCE_MM:
        return "cross", outcome_index
    return "regain", outcome_index


def find_attempts(track, geometry):
    """Find valid attempts and assign an outcome to each one."""
    xy = track["xy"]
    signal = track["signal"].copy()
    signal[~np.isfinite(signal)] = 0
    time_s = track["time_s"]
    velocity = track["velocity"]
    rows = []

    for _, gap in geometry.iterrows():
        boundary_x = gap["loss_edge_x_mm"]
        candidate = np.flatnonzero(
            (xy[:-1, 0] > boundary_x) & (xy[1:, 0] <= boundary_x)
        )
        for attempt_index in candidate:
            if np.isfinite(gap["ribbon_center_y_mm"]) and not (
                abs(xy[attempt_index, 1] - gap["ribbon_center_y_mm"])
                <= ATTEMPT_RIBBON_HALF_WIDTH_MM
            ):
                continue
            start_time = time_s[attempt_index] - PRE_SIGNAL_S
            past = np.flatnonzero(
                (time_s >= start_time) & (time_s < time_s[attempt_index])
            )
            if not len(past) or not np.any(signal[past] > 0):
                continue

            outcome, outcome_index = classify_outcome(
                xy, signal, time_s, attempt_index, boundary_x
            )
            rows.append(
                {
                    "track_id": track["track_id"],
                    "source_file": track["source_file"],
                    "geometry_id": int(gap["geometry_id"]),
                    "ribbon_id": gap["ribbon_id"],
                    "gap_id": int(gap["gap_id"]),
                    "gap_order": int(gap["gap_id"]) + 1,
                    "gap_label": gap["gap_label"],
                    "boundary_x_mm": boundary_x,
                    "attempt_index": attempt_index,
                    "attempt_time_s": time_s[attempt_index],
                    "entry_index": attempt_index + 1,
                    "entry_time_s": time_s[attempt_index + 1],
                    "outcome": outcome,
                    "outcome_index": outcome_index,
                    "outcome_time_s": (
                        time_s[int(outcome_index)] if np.isfinite(outcome_index) else np.nan
                    ),
                    "outcome_x_mm": (
                        xy[int(outcome_index), 0] if np.isfinite(outcome_index) else np.nan
                    ),
                    "xy": xy,
                    "time_s": time_s,
                    "speed_smooth": track["speed_smooth"],
                }
            )
    return rows


def make_event_table(tracks, geometry):
    """Create time-ordered event rows and within-track predictors."""
    rows = [row for track in tracks for row in find_attempts(track, geometry)]
    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError("No valid attempts. Check the settings and input data.")

    events = events.sort_values(["track_id", "attempt_time_s"]).reset_index(drop=True)
    events["track_attempt"] = events.groupby("track_id").cumcount() + 1
    events["track_attempt_count"] = events.groupby("track_id")["track_id"].transform("size")
    events = events.loc[events["track_attempt_count"] >= MIN_ATTEMPTS_PER_TRACK].copy()

    events["is_cross"] = (events["outcome"] == "cross").astype(int)
    events["is_regain"] = (events["outcome"] == "regain").astype(int)
    events["is_abort"] = (events["outcome"] == "abort").astype(int)
    events["previous_outcome"] = events.groupby("track_id")["outcome"].shift(1)
    events["previous_cross"] = events.groupby("track_id")["is_cross"].shift(1)
    events["previous_cross_count"] = (
        events.groupby("track_id")["is_cross"].cumsum() - events["is_cross"]
    )
    return events


def get_cross_after_regain_by_attempt(events):
    """Return current-cross statistics for attempts after a regain."""
    after_regain = events.loc[events["previous_outcome"] == "regain"]
    summary = (
        after_regain.groupby("track_attempt")["is_cross"]
        .agg(mean_cross="mean", track_count="size", std_cross="std")
        .reset_index()
    )
    summary["std_cross"] = summary["std_cross"].fillna(0.0)
    summary["sem_cross"] = summary["std_cross"] / np.sqrt(summary["track_count"])
    return summary


def get_regain_transition_durations(events):
    """Return elapsed times from a regain to the next same-track attempt."""
    ordered = events.sort_values(["track_id", "attempt_time_s"]).copy()
    ordered["previous_outcome"] = ordered.groupby("track_id")["outcome"].shift(1)
    ordered["previous_attempt_time_s"] = (
        ordered.groupby("track_id")["attempt_time_s"].shift(1)
    )
    durations = ordered.loc[
        ordered["previous_outcome"].eq("regain")
        & ordered["outcome"].isin(OUTCOME_ORDER)
    ].copy()
    durations["elapsed_s"] = (
        durations["attempt_time_s"] - durations["previous_attempt_time_s"]
    )
    durations = durations.loc[np.isfinite(durations["elapsed_s"])].copy()
    durations["transition"] = "regain → " + durations["outcome"]
    return durations


# %% Summary plots
def plot_event_summary(events):
    """Plot event counts and outcomes by loss boundary."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gap_order = list(range(1, EXPECTED_GAPS_PER_RIBBON + 1))
    sns.countplot(data=events, x="gap_order", order=gap_order, hue="outcome",
                  hue_order=OUTCOME_ORDER, palette=OUTCOME_COLOR, ax=axes[0])
    axes[0].set(xlabel="Gap order (upwind to downwind)", ylabel="Attempt count",
                title="Valid attempts, all ribbons pooled")

    by_gap = (
        events.groupby(["gap_order", "outcome"]).size().unstack(fill_value=0)
        .reindex(columns=OUTCOME_ORDER, fill_value=0)
        .reindex(gap_order, fill_value=0)
    )
    by_gap = by_gap.div(by_gap.sum(axis=1), axis=0)
    by_gap.plot(kind="bar", stacked=True, color=[OUTCOME_COLOR[x] for x in OUTCOME_ORDER],
                ax=axes[1], legend=False)
    axes[1].set(xlabel="Gap order (upwind to downwind)", ylabel="Outcome fraction",
                ylim=(0, 1), title="Outcome by gap, all ribbons pooled")

    counts = events[["track_id", "track_attempt_count"]].drop_duplicates()
    axes[2].hist(counts["track_attempt_count"], bins=np.arange(1.5, counts["track_attempt_count"].max() + 1.5),
                 color="0.3")
    axes[2].set(xlabel="Attempts per track", ylabel="Track count", title="Repeated-attempt tracks")
    fig.tight_layout()


def plot_within_track_summary(events):
    """Plot outcome change by attempt order and previous outcome."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    maximum_order = min(8, events["track_attempt"].max())
    for gap_order in range(1, EXPECTED_GAPS_PER_RIBBON + 1):
        ordered = events.loc[
            (events["gap_order"] == gap_order)
            & (events["track_attempt"] <= maximum_order)
        ]
        rates = ordered.groupby("track_attempt")["is_cross"].agg(["mean", "count"])
        if not len(rates):
            continue
        error = np.sqrt(rates["mean"] * (1 - rates["mean"]) / rates["count"])
        axes[0, 0].errorbar(
            rates.index, rates["mean"], yerr=error, fmt="o-", label=f"Gap {gap_order}"
        )
    axes[0, 0].set(
        xlabel="Attempt number within track", ylabel="P(cross)", ylim=(0, 1),
        title="Within-track crossing by gap"
    )
    axes[0, 0].legend(fontsize=9)

    pooled = events.loc[events["track_attempt"] <= maximum_order]
    for outcome_column, outcome_name in [
        ("is_cross", "cross"),
        ("is_regain", "regain"),
        ("is_abort", "abort"),
    ]:
        rates = pooled.groupby("track_attempt")[outcome_column].agg(["mean", "count"])
        error = np.sqrt(rates["mean"] * (1 - rates["mean"]) / rates["count"])
        axes[0, 1].errorbar(
            rates.index, rates["mean"], yerr=error, fmt="o-",
            color=OUTCOME_COLOR[outcome_name], label=outcome_name
        )
    axes[0, 1].set(
        xlabel="Attempt number within track", ylabel="Outcome probability", ylim=(0, 1),
        title="Outcome probability, all gaps pooled"
    )
    axes[0, 1].legend(fontsize=9)

    prior = events.dropna(subset=["previous_outcome"])
    sns.barplot(data=prior, x="previous_outcome", y="is_cross", order=OUTCOME_ORDER,
                palette=OUTCOME_COLOR, ax=axes[1, 0])
    axes[1, 0].set(xlabel="Previous outcome", ylabel="P(current cross)", ylim=(0, 1),
                title="Effect of previous outcome")

    first_last = events.groupby("track_id")["is_cross"].agg(["first", "last"])
    for _, row in first_last.iterrows():
        axes[1, 1].plot([1, 2], [row["first"], row["last"]], color="0.7", alpha=0.5)
    if len(first_last):
        axes[1, 1].plot(
            [1, 2], [first_last["first"].mean(), first_last["last"].mean()],
            "o-", color="black", lw=3
        )
    axes[1, 1].set(xlabel="Attempt", ylabel="Cross outcome", xticks=[1, 2],
                xticklabels=["First", "Last"], yticks=[0, 1], ylim=(-0.1, 1.1),
                title="First to last attempt")
    fig.tight_layout()


def plot_cross_after_regain_by_attempt(events):
    """Plot crossing probability after a regain by current attempt number."""
    summary = get_cross_after_regain_by_attempt(events)
    if summary.empty:
        print("No attempts follow a regain event.")
        return

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        summary["track_attempt"],
        summary["mean_cross"],
        yerr=summary["sem_cross"],
        fmt="o-",
        color=OUTCOME_COLOR["cross"],
        capsize=4,
    )
    for _, result in summary.iterrows():
        axis.annotate(
            f"n={int(result['track_count'])}",
            (result["track_attempt"], result["mean_cross"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.set(
        xlabel="Current attempt number within track",
        ylabel="P(cross | previous regain)",
        title="Crossing after regain by attempt number (mean ± SEM)",
        ylim=(-0.05, 1.05),
    )
    fig.tight_layout()


def plot_regain_transition_durations(events):
    """Plot durations from regain events to the next outcome attempt."""
    durations = get_regain_transition_durations(events)
    if durations.empty:
        print("No attempts follow a regain event.")
        return

    fig, axes = plt.subplots(1, len(OUTCOME_ORDER), figsize=(15, 4), sharex=True)
    for axis, outcome in zip(axes, OUTCOME_ORDER):
        values = durations.loc[durations["outcome"] == outcome, "elapsed_s"].to_numpy()
        transition = f"regain → {outcome}"
        axis.set(title=f"{transition}: n={len(values)}", xlabel="Time to next attempt (s)")
        if not len(values):
            axis.text(0.5, 0.5, "No transitions", ha="center", va="center",
                      transform=axis.transAxes)
            continue
        bins = min(20, max(1, int(np.ceil(np.sqrt(len(values))))))
        axis.hist(values, bins=bins, color=OUTCOME_COLOR[outcome], alpha=0.8)
        mean = values.mean()
        axis.axvline(mean, color="black", ls="--", lw=1, label=f"Mean: {mean:.2f} s")
        axis.legend(fontsize=9)
    axes[0].set_ylabel("Transition count")
    fig.suptitle("Timing from regain to the next attempt")
    fig.tight_layout(rect=(0, 0, 1, 0.93))


# %% Speed analysis
def get_post_entry_speed(events):
    """Return mean smooth speed during the first second after gap entry."""
    rows = []
    for _, event in events.iterrows():
        entry_index = int(event["entry_index"])
        time_s = event["time_s"]
        speed = event["speed_smooth"]
        start_time = time_s[entry_index]
        in_window = (time_s >= start_time) & (time_s < start_time + POST_SPEED_S)
        valid_speed = speed[in_window & np.isfinite(speed)]
        if len(valid_speed) < MIN_POST_SPEED_SAMPLES:
            continue
        rows.append(
            {
                "track_id": event["track_id"],
                "track_attempt": event["track_attempt"],
                "gap_order": event["gap_order"],
                "previous_outcome": event["previous_outcome"],
                "post_speed": np.mean(valid_speed),
            }
        )
    return pd.DataFrame(rows)


def plot_post_entry_speed(speed_events):
    """Plot post-entry speed by history category and attempt number."""
    if speed_events.empty:
        print("No events have enough post-entry speed samples.")
        return

    history = speed_events.dropna(subset=["previous_outcome"])
    track_history = history.groupby(["track_id", "previous_outcome"], as_index=False)["post_speed"].mean()
    fig, axis = plt.subplots(figsize=(7, 5))
    rng = np.random.default_rng(0)
    for index, outcome in enumerate(OUTCOME_ORDER):
        values = track_history.loc[track_history["previous_outcome"] == outcome, "post_speed"].to_numpy()
        if not len(values):
            continue
        axis.plot(index + rng.uniform(-0.12, 0.12, len(values)), values, "o",
                  color=OUTCOME_COLOR[outcome], alpha=0.45)
        error = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        axis.errorbar(index, np.mean(values), yerr=error, fmt="o", ms=10,
                      color="black", capsize=4, zorder=3)
    axis.set(
        xticks=range(len(OUTCOME_ORDER)), xticklabels=OUTCOME_ORDER,
        xlabel="Previous outcome", ylabel="Mean smooth speed, 0–1 s after entry",
        title="Post-entry speed by previous outcome"
    )
    fig.tight_layout()

    maximum_order = min(8, speed_events["track_attempt"].max())
    fig, axis = plt.subplots(figsize=(8, 5))
    ordered = speed_events.loc[speed_events["track_attempt"] <= maximum_order]
    for _, track_events in ordered.groupby("track_id"):
        track_events = track_events.sort_values("track_attempt")
        axis.plot(track_events["track_attempt"], track_events["post_speed"],
                  color="0.75", alpha=0.35)
    summary = ordered.groupby("track_attempt")["post_speed"].agg(["mean", "count", "std"])
    error = summary["std"] / np.sqrt(summary["count"])
    axis.errorbar(summary.index, summary["mean"], yerr=error, fmt="o-",
                  color="black", lw=3, capsize=4, label="Mean")
    axis.set(
        xlabel="Attempt number within track", ylabel="Mean smooth speed, 0–1 s after entry",
        title="Within-track post-entry speed"
    )
    axis.legend()
    fig.tight_layout()


# %% Sequence analysis
def make_second_order_transition_matrices(events):
    """Return future-row matrices conditioned on the outcome two attempts back."""
    ordered = events.sort_values(["track_id", "attempt_time_s"]).copy()
    ordered["conditioning_outcome"] = (
        ordered.groupby("track_id")["outcome"].shift(2)
    )
    ordered["current_outcome"] = ordered.groupby("track_id")["outcome"].shift(1)
    triplets = ordered.dropna(subset=["conditioning_outcome", "current_outcome"])
    matrices = {}
    counts = {}
    for conditioning_outcome in OUTCOME_ORDER:
        subset = triplets.loc[
            triplets["conditioning_outcome"] == conditioning_outcome
        ]
        counts[conditioning_outcome] = len(subset)
        matrices[conditioning_outcome] = pd.crosstab(
            subset["current_outcome"], subset["outcome"], normalize="index"
        ).T.reindex(index=OUTCOME_ORDER, columns=OUTCOME_ORDER, fill_value=0)
    return matrices, counts


def plot_second_order_transition_matrices(events):
    """Plot future-row matrices conditioned on the two-back outcome."""
    matrices, counts = make_second_order_transition_matrices(events)
    if not any(counts.values()):
        print("Too few attempts for second-order transition analysis.")
        return

    fig, axes = plt.subplots(1, len(OUTCOME_ORDER), figsize=(16, 5), sharey=True)
    for axis, conditioning_outcome in zip(axes, OUTCOME_ORDER):
        sns.heatmap(
            matrices[conditioning_outcome],
            vmin=0,
            vmax=1,
            cmap="Blues",
            annot=True,
            fmt=".2f",
            cbar=axis is axes[-1],
            ax=axis,
        )
        axis.set(
            xlabel="Current outcome",
            ylabel="Next outcome",
            title=(
                f"Two attempts back: {conditioning_outcome}\n"
                f"Triplets: n={counts[conditioning_outcome]}"
            ),
        )
    fig.suptitle("Second-order outcome transitions")
    fig.tight_layout(rect=(0, 0, 1, 0.93))


def plot_transition_model(events):
    """Plot raw and track-controlled outcome transition probabilities."""
    transition_events = events.dropna(subset=["previous_outcome"]).copy()
    if len(transition_events) < 10:
        print("Too few sequential events for transition analysis.")
        return

    raw = pd.crosstab(
        transition_events["outcome"], transition_events["previous_outcome"], normalize="columns"
    ).reindex(index=OUTCOME_ORDER, columns=OUTCOME_ORDER, fill_value=0)

    previous = pd.get_dummies(transition_events["previous_outcome"], prefix="previous")
    previous = previous.reindex(columns=[f"previous_{x}" for x in OUTCOME_ORDER], fill_value=0)
    gap = pd.get_dummies(transition_events["gap_order"], prefix="gap", drop_first=True)
    tracks = pd.get_dummies(transition_events["track_id"], prefix="track", drop_first=True)
    attempt = transition_events[["track_attempt"]].copy()
    attempt["track_attempt"] = (
        attempt["track_attempt"] - attempt["track_attempt"].mean()
    ) / (attempt["track_attempt"].std() + 1e-6)
    design = pd.concat([previous, gap, attempt, tracks], axis=1)
    target = transition_events["outcome"]

    if target.nunique() < len(OUTCOME_ORDER):
        print("Transition model needs all three outcome classes.")
        return

    model = LogisticRegression(C=1.0, max_iter=5000)
    model.fit(design, target)
    adjusted = pd.DataFrame(index=OUTCOME_ORDER, columns=OUTCOME_ORDER, dtype=float)
    previous_columns = [f"previous_{x}" for x in OUTCOME_ORDER]
    for prior in OUTCOME_ORDER:
        counterfactual = design.copy()
        counterfactual.loc[:, previous_columns] = 0
        counterfactual[f"previous_{prior}"] = 1
        probability = model.predict_proba(counterfactual).mean(axis=0)
        adjusted.loc[model.classes_, prior] = probability

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, matrix, title in [
        (axes[0], raw, "Observed transitions"),
        (axes[1], adjusted, "Track-controlled transitions"),
    ]:
        sns.heatmap(matrix, vmin=0, vmax=1, cmap="Blues", annot=True, fmt=".2f",
                    cbar=axis is axes[1], ax=axis)
        axis.set(xlabel="Previous outcome", ylabel="Current outcome", title=title)
    fig.tight_layout()


def make_early_late_transition_matrices(events):
    """Return early and late transition matrices from eligible tracks."""
    transition_events = events.loc[
        (events["track_attempt_count"] - 1 >= MIN_TRANSITIONS_FOR_PHASE)
        & events["previous_outcome"].notna()
    ].copy()
    if transition_events.empty:
        raise RuntimeError(
            "No tracks have more than two transitions for phase analysis."
        )

    transition_position = (
        transition_events["track_attempt"] - 1
    ) / (transition_events["track_attempt_count"] - 1)
    transition_events["phase"] = np.where(
        transition_position <= 0.5, "early", "late"
    )

    matrices = {}
    counts = {}
    for phase in ["early", "late"]:
        phase_events = transition_events.loc[transition_events["phase"] == phase]
        matrices[phase] = pd.crosstab(
            phase_events["outcome"], phase_events["previous_outcome"],
            normalize="columns",
        ).reindex(index=OUTCOME_ORDER, columns=OUTCOME_ORDER, fill_value=0)
        counts[phase] = len(phase_events)
    return matrices, counts


def plot_early_late_transition_matrices(events):
    """Plot transitions in the first and later portions of eligible tracks."""
    try:
        matrices, counts = make_early_late_transition_matrices(events)
    except RuntimeError as error:
        print(error)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, phase, title in [
        (axes[0], "early", "Early transitions"),
        (axes[1], "late", "Later transitions"),
    ]:
        sns.heatmap(
            matrices[phase], vmin=0, vmax=1, cmap="Blues", annot=True,
            fmt=".2f", cbar=axis is axes[1], ax=axis,
        )
        axis.set(
            xlabel="Previous outcome",
            ylabel="Current outcome",
            title=f"{title} (n={counts[phase]})",
        )
    fig.suptitle("Tracks with more than two transitions", y=1.02)
    fig.tight_layout()


def count_motifs(sequences, length):
    """Count overlapping outcome motifs of one length."""
    counts = Counter()
    for sequence in sequences:
        for start in range(len(sequence) - length + 1):
            counts[tuple(sequence[start:start + length])] += 1
    return counts


def get_motif_null_counts(sequences, motifs, length, null_type, rng):
    """Generate motif counts from a global or within-track label shuffle."""
    null_counts = np.zeros((MOTIF_PERMUTATIONS, len(motifs)), dtype=float)
    sequence_lengths = [len(sequence) for sequence in sequences]
    all_outcomes = np.concatenate([np.asarray(sequence) for sequence in sequences])

    for permutation in range(MOTIF_PERMUTATIONS):
        if null_type == "global":
            shuffled_outcomes = rng.permutation(all_outcomes)
            cut_points = np.cumsum(sequence_lengths)[:-1]
            shuffled = [part.tolist() for part in np.split(shuffled_outcomes, cut_points)]
        elif null_type == "within_track":
            shuffled = [rng.permutation(sequence).tolist() for sequence in sequences]
        else:
            raise ValueError("Unknown motif null type.")
        shuffled_counts = count_motifs(shuffled, length)
        null_counts[permutation] = [shuffled_counts[motif] for motif in motifs]
    return null_counts


def summarize_motif_null(observed_counts, motifs, null_counts):
    """Return motif enrichment statistics for one null distribution."""
    null_mean = null_counts.mean(axis=0)
    null_std = null_counts.std(axis=0)
    z_score = np.divide(
        observed_counts - null_mean, null_std,
        out=np.zeros_like(null_mean), where=null_std > 0
    )
    p_enriched = (1 + (null_counts >= observed_counts).sum(axis=0)) / (MOTIF_PERMUTATIONS + 1)
    return pd.DataFrame(
        {
            "motif": [" → ".join(motif) for motif in motifs],
            "count": observed_counts,
            "null_mean": null_mean,
            "z_score": z_score,
            "p_enriched": p_enriched,
        }
    )


def plot_motif_enrichment(events):
    """Compare motifs with global and within-track label-shuffle nulls."""
    sequences = [
        group.sort_values("attempt_time_s")["outcome"].to_list()
        for _, group in events.groupby("track_id")
    ]
    rng = np.random.default_rng(MOTIF_RANDOM_SEED)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for row, length in enumerate([2, 3]):
        motifs = list(product(OUTCOME_ORDER, repeat=length))
        observed = count_motifs(sequences, length)
        observed_counts = np.array([observed[motif] for motif in motifs])
        for column, (null_type, title, color) in enumerate([
            ("global", "Global label shuffle", "C1"),
            ("within_track", "Within-track label shuffle", "C0"),
        ]):
            null_counts = get_motif_null_counts(sequences, motifs, length, null_type, rng)
            results = summarize_motif_null(observed_counts, motifs, null_counts)
            results = results.loc[results["count"] > 0].nlargest(10, "z_score")
            axis = axes[row, column]
            axis.barh(results["motif"], results["z_score"], color=color)
            axis.axvline(0, color="black", lw=1)
            axis.set(xlabel="Enrichment z score", ylabel="Motif",
                     title=f"{length}-event motifs: {title}")
            axis.invert_yaxis()
            print(f"Top {length}-event motifs: {title}")
            print(results.to_string(index=False))
    fig.tight_layout()


def plot_event_paths(events):
    """Plot short paths after attempts to validate event labels."""
    frames = int(FRAME_RATE_HZ * OUTCOME_S)
    fig, axes = plt.subplots(1, len(OUTCOME_ORDER), figsize=(16, 5), sharey=True)
    for axis, outcome in zip(axes, OUTCOME_ORDER):
        subset = events.loc[events["outcome"] == outcome]
        for _, event in subset.iterrows():
            path = event["xy"][int(event["attempt_index"]):int(event["attempt_index"]) + frames]
            axis.plot(path[:, 0] - event["boundary_x_mm"], path[:, 1] - path[0, 1],
                      color=OUTCOME_COLOR[outcome], alpha=0.12)
        axis.axvline(0, color="black", ls="--", lw=1)
        axis.axvline(-DISTANCE_MM, color="black", ls=":", lw=1)
        axis.axvline(DISTANCE_MM, color="black", ls=":", lw=1)
        axis.set(xlabel="x relative to loss boundary (mm)", title=f"{outcome}: n={len(subset)}")
    axes[0].set_ylabel("y relative to attempt (mm)")
    fig.tight_layout()


# %% Track-based prediction
def evaluate_track_model(events):
    """Test prior-attempt effects and control for stable track effects."""
    model_events = events.dropna(subset=["previous_cross"]).copy()
    changing_tracks = model_events.groupby("track_id")["is_cross"].nunique()
    model_events = model_events.loc[
        model_events["track_id"].isin(changing_tracks[changing_tracks > 1].index)
    ].copy()
    if len(model_events) < 20 or model_events["track_id"].nunique() < 3:
        print("Too few mixed-outcome tracks for the within-track model.")
        return

    track_columns = pd.get_dummies(model_events["track_id"], prefix="track", drop_first=True)
    predictors = pd.DataFrame(
        {
            "previous_cross": model_events["previous_cross"],
            "attempt_number": model_events["track_attempt"],
            "gap_index": model_events["gap_id"],
        }, index=model_events.index
    )
    X_fixed_effect = pd.concat([predictors, track_columns], axis=1)
    y = model_events["is_cross"].to_numpy()
    groups = model_events["track_id"].to_numpy()

    # The fixed-effect model estimates within-track weights. It is not valid
    # for held-out tracks because their track intercepts are unknown.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fixed_effect)
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=5000).fit(X_scaled, y)
    weights = pd.Series(model.coef_[0][:len(predictors.columns)], index=predictors.columns)

    # This separate model measures transfer to new tracks without identity data.
    splitter = GroupShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
    scores = []
    for train, test in splitter.split(predictors, y, groups):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(predictors.iloc[train])
        X_test = scaler.transform(predictors.iloc[test])
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=5000)
        model.fit(X_train, y[train])
        probability = model.predict_proba(X_test)[:, 1]
        scores.append((balanced_accuracy_score(y[test], probability >= 0.5), roc_auc_score(y[test], probability)))

    print(f"Mixed-outcome tracks: {model_events['track_id'].nunique()}")
    print(f"Events in within-track model: {len(model_events)}")
    print("Within-track standardized weights:")
    print(weights.sort_values(ascending=False))
    if scores:
        scores = np.asarray(scores)
        print(f"New-track balanced accuracy: {scores[:, 0].mean():.3f} ± {scores[:, 0].std():.3f}")
        print(f"New-track AUC: {scores[:, 1].mean():.3f} ± {scores[:, 1].std():.3f}")


def run():
    """Run the folder-based gap-crossing analysis."""
    target_files = get_target_files(ROOT_DIR)
    print(f"Target files: {len(target_files)}")
    tracks = load_tracks(target_files)
    print(f"Valid tracks: {len(tracks)}")
    geometry = get_gap_geometry(tracks)
    print(f"Gap regions: {len(geometry)}")
    events = make_event_table(tracks, geometry)
    print(f"Repeated-attempt tracks: {events['track_id'].nunique()}")
    print(f"Valid attempts: {len(events)}")
    print(events.groupby("outcome").size().reindex(OUTCOME_ORDER, fill_value=0))

    plot_gap_geometry(tracks, geometry)
    plot_centerline_profiles(tracks, geometry)
    plot_event_summary(events)
    plot_within_track_summary(events)
    plot_cross_after_regain_by_attempt(events)
    plot_regain_transition_durations(events)
    speed_events = get_post_entry_speed(events)
    print(f"Events with post-entry speed: {len(speed_events)}")
    plot_post_entry_speed(speed_events)
    plot_transition_model(events)
    plot_second_order_transition_matrices(events)
    plot_motif_enrichment(events)
    plot_event_paths(events)
    evaluate_track_model(events)
    plt.show()


if __name__ == "__main__":
    run()
