"""Temporal gap-crossing analysis for periodic ribbon recordings.

An attempt starts at one pooled global signal loss. A track is eligible when
it had signal in the preceding PRE_SIGNAL_S seconds. The first qualified signal
return in the following OUTCOME_S seconds defines an upwind, downwind, or no
return outcome.
"""

from pathlib import Path
from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from optogui.io.data import load_exp_matrix


# %% Settings
ROOT_DIR = Path(
    r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\periodic_ribbon"
)
EXPERIMENT_TEXTS = ("freq0.25_dut0.5",)
FORBIDDEN_TEXTS = ()
MAX_FILES = None

FRAME_RATE_HZ = 60
MIN_TRACK_S = 10
PRE_SIGNAL_S = 2
OUTCOME_S = 3
DISTANCE_MM = 0
SIGNAL_THRESHOLD = 0
MAX_LOSS_POSITION_GAP_S = 1 / FRAME_RATE_HZ * 1.5
MOTIF_PERMUTATIONS = 1000
MOTIF_RANDOM_SEED = 0

OUTCOME_ORDER = ["upwind", "downwind", "no_return"]
OUTCOME_COLOR = {"upwind": "C0", "downwind": "C2", "no_return": "C3"}
OUTCOME_LABEL = {"upwind": "up", "downwind": "down", "no_return": "loss"}


def get_target_files(root_dir):
    """Return one supported experiment matrix from each selected recording."""
    candidates = {}
    for suffix, priority in (("exp_matrix.pklz", 0), ("exp_matrix.joblib", 1)):
        for path in root_dir.rglob(suffix):
            folder_text = str(path.parent)
            if not all(text in folder_text for text in EXPERIMENT_TEXTS):
                continue
            if any(text in folder_text for text in FORBIDDEN_TEXTS):
                continue
            previous = candidates.get(path.parent)
            if previous is None or priority > previous[0]:
                candidates[path.parent] = (priority, path)

    files = sorted((item[1] for item in candidates.values()), key=lambda path: str(path))
    return files if MAX_FILES is None else files[:MAX_FILES]


def load_tracks(target_files):
    """Load valid tracks from all selected recordings."""
    tracks = []
    min_frames = int(MIN_TRACK_S * FRAME_RATE_HZ)
    required = {
        "trjn", "headx_smooth", "heady_smooth", "signal", "t",
        "vx_smooth", "vy_smooth", "spd_smooth",
    }

    for file_index, file_path in enumerate(target_files):
        try:
            data = load_exp_matrix(file_path.parent)
            missing = required.difference(data)
            if missing:
                print(f"Skip file with missing data: {file_path} | {sorted(missing)}")
                continue
        except Exception as error:
            print(f"Skip unreadable file: {file_path} | {error}")
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
                or np.any(np.diff(time_s) <= 0)
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
                    "speed_smooth": speed_smooth,
                }
            )

    return tracks


def detect_global_timing(tracks):
    """Detect one global signal-loss timing definition from all selected tracks."""
    if not tracks:
        raise RuntimeError("No valid tracks. Check the settings and input data.")
    max_frame = max(
        np.rint(track["time_s"] * FRAME_RATE_HZ).astype(int).max()
        for track in tracks
    )
    observed_count = np.zeros(max_frame + 1, dtype=int)
    signal_count = np.zeros(max_frame + 1, dtype=int)
    for track in tracks:
        frame_index = np.rint(track["time_s"] * FRAME_RATE_HZ).astype(int)
        valid = np.isfinite(track["signal"])
        observed_count += np.bincount(frame_index[valid], minlength=max_frame + 1)
        signal_count += np.bincount(
            frame_index[valid & (track["signal"] > SIGNAL_THRESHOLD)],
            minlength=max_frame + 1,
        )

    signal_fraction = np.divide(
        signal_count,
        observed_count,
        out=np.zeros_like(signal_count, dtype=float),
        where=observed_count > 0,
    )
    global_signal = signal_count > 0
    global_time_s = np.arange(max_frame + 1) / FRAME_RATE_HZ
    loss_index = np.flatnonzero(global_signal[:-1] & ~global_signal[1:]) + 1
    return {
        "time_s": global_time_s,
        "signal_fraction": signal_fraction,
        "global_signal": global_signal,
        "loss_time_s": global_time_s[loss_index],
    }


# %% Event definitions
def _loss_position_index(track, loss_time_s):
    """Return the sample at or immediately before one global loss time."""
    index = np.searchsorted(track["time_s"], loss_time_s, side="right") - 1
    if index < 0 or loss_time_s - track["time_s"][index] > MAX_LOSS_POSITION_GAP_S:
        return None
    if not np.isfinite(track["xy"][index]).all():
        return None
    return index


def classify_outcome(track, loss_index, loss_time_s):
    """Return the first qualified return outcome after one global loss."""
    time_s = track["time_s"]
    signal = np.nan_to_num(track["signal"], nan=0.0)
    xy = track["xy"]
    future = np.flatnonzero(
        (time_s > loss_time_s)
        & (time_s <= loss_time_s + OUTCOME_S)
        & (signal > SIGNAL_THRESHOLD)
        & np.isfinite(xy).all(axis=1)
    )
    if not len(future):
        return "no_return", np.nan

    dx_mm = xy[future, 0] - xy[loss_index, 0]
    qualified = future[np.abs(dx_mm) >= DISTANCE_MM]
    if not len(qualified):
        return "no_return", np.nan
    outcome_index = int(qualified[0])
    if xy[outcome_index, 0] < xy[loss_index, 0]:
        return "upwind", outcome_index
    return "downwind", outcome_index


def find_attempts(tracks, timing):
    """Find track attempts at every pooled global signal loss."""
    rows = []
    for loss_number, loss_time_s in enumerate(timing["loss_time_s"], start=1):
        pre_start_s = loss_time_s - PRE_SIGNAL_S
        for track in tracks:
            time_s = track["time_s"]
            signal = np.nan_to_num(track["signal"], nan=0.0)
            had_signal = np.any(
                (time_s >= pre_start_s)
                & (time_s < loss_time_s)
                & (signal > SIGNAL_THRESHOLD)
            )
            if not had_signal:
                continue

            loss_index = _loss_position_index(track, loss_time_s)
            if loss_index is None:
                continue
            outcome, outcome_index = classify_outcome(track, loss_index, loss_time_s)
            rows.append(
                {
                    "track_id": track["track_id"],
                    "source_file": track["source_file"],
                    "loss_number": loss_number,
                    "loss_time_s": loss_time_s,
                    "pre_start_s": pre_start_s,
                    "loss_index": loss_index,
                    "loss_sample_time_s": time_s[loss_index],
                    "loss_x_mm": track["xy"][loss_index, 0],
                    "loss_y_mm": track["xy"][loss_index, 1],
                    "outcome": outcome,
                    "outcome_index": outcome_index,
                    "outcome_time_s": (
                        time_s[outcome_index] if np.isfinite(outcome_index) else np.nan
                    ),
                    "outcome_x_mm": (
                        track["xy"][outcome_index, 0]
                        if np.isfinite(outcome_index) else np.nan
                    ),
                    "return_latency_s": (
                        time_s[outcome_index] - loss_time_s
                        if np.isfinite(outcome_index) else np.nan
                    ),
                    "xy": track["xy"],
                    "time_s": time_s,
                    "signal": signal,
                }
            )
    return rows


def make_event_table(tracks, timing):
    """Create time-ordered temporal-gap events and within-track predictors."""
    rows = find_attempts(tracks, timing)
    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError("No valid attempts. Check the settings and input data.")

    events = events.sort_values(["track_id", "loss_time_s"]).reset_index(drop=True)
    events["track_attempt"] = events.groupby("track_id").cumcount() + 1
    events["track_attempt_count"] = events.groupby("track_id")["track_id"].transform("size")
    events["is_upwind"] = (events["outcome"] == "upwind").astype(int)
    events["previous_outcome"] = events.groupby("track_id")["outcome"].shift(1)
    return events


# %% Plots
def plot_time_signal(tracks, timing):
    """Overlay all selected tracks with one pooled timing definition."""
    fig, axis = plt.subplots(figsize=(16, 6))
    for track in tracks:
        axis.plot(
            track["time_s"], track["signal"], color="0.2", alpha=0.12,
            linewidth=0.4,
        )

    signal_values = np.concatenate([track["signal"] for track in tracks])
    finite_signal = signal_values[np.isfinite(signal_values)]
    display_level = max(255, float(finite_signal.max()) if len(finite_signal) else 0)
    axis.step(
        timing["time_s"], timing["signal_fraction"] * display_level,
        where="post", color="C1", linewidth=1.5, label="pooled signal fraction",
    )
    for loss_number, loss_time_s in enumerate(timing["loss_time_s"]):
        attempt_box = Rectangle(
            (loss_time_s - PRE_SIGNAL_S, 0), PRE_SIGNAL_S, display_level,
            fill=False, edgecolor="cyan", linewidth=1, alpha=0.5,
        )
        axis.add_patch(attempt_box)
        if loss_number == 0:
            axis.axvline(loss_time_s, color="C3", lw=1, label="global signal loss")
        else:
            axis.axvline(loss_time_s, color="C3", lw=1)

    axis.set(
        xlabel="Global time (s)", ylabel="Signal intensity",
        title="All selected track signals and pooled temporal attempt windows",
        ylim=(0, display_level * 1.05),
    )
    axis.legend(loc="upper right")
    fig.tight_layout()


def plot_event_summary(events):
    """Plot temporal attempt counts, outcomes, and track participation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    outcome_counts = events.groupby("outcome").size().reindex(OUTCOME_ORDER, fill_value=0)
    axes[0].bar(
        outcome_counts.index, outcome_counts.to_numpy(),
        color=[OUTCOME_COLOR[outcome] for outcome in OUTCOME_ORDER],
    )
    axes[0].set(xlabel="Outcome", ylabel="Attempt count", title="Temporal-gap outcomes")

    by_loss = (
        events.groupby(["loss_number", "outcome"]).size().unstack(fill_value=0)
        .reindex(columns=OUTCOME_ORDER, fill_value=0)
    )
    by_loss = by_loss.div(by_loss.sum(axis=1), axis=0)
    by_loss.plot(
        kind="bar", stacked=True, color=[OUTCOME_COLOR[outcome] for outcome in OUTCOME_ORDER],
        ax=axes[1], legend=False,
    )
    axes[1].set(
        xlabel="Global loss number", ylabel="Outcome fraction", ylim=(0, 1),
        title="Outcome by global signal loss",
    )

    counts = events[["track_id", "track_attempt_count"]].drop_duplicates()
    axes[2].hist(counts["track_attempt_count"], color="0.3")
    axes[2].set(xlabel="Attempts per track", ylabel="Track count", title="Track participation")
    fig.tight_layout()


def plot_event_paths(events):
    """Plot paths after global losses to validate the outcome labels."""
    fig, axes = plt.subplots(1, len(OUTCOME_ORDER), figsize=(16, 5), sharey=True)
    for axis, outcome in zip(axes, OUTCOME_ORDER):
        subset = events.loc[events["outcome"] == outcome]
        for _, event in subset.iterrows():
            time_s = event["time_s"]
            index = np.flatnonzero(
                (time_s >= event["loss_time_s"])
                & (time_s <= event["loss_time_s"] + OUTCOME_S)
            )
            if not len(index):
                continue
            axis.plot(
                time_s[index] - event["loss_time_s"],
                event["xy"][index, 0] - event["loss_x_mm"],
                color=OUTCOME_COLOR[outcome], alpha=0.1,
            )
        axis.axhline(0, color="black", ls="--", lw=1)
        axis.axhline(-DISTANCE_MM, color="black", ls=":", lw=1)
        axis.axhline(DISTANCE_MM, color="black", ls=":", lw=1)
        axis.set(
            xlabel="Time after global loss (s)",
            title=f"{outcome}: n={len(subset)}",
        )
    axes[0].set_ylabel("x relative to global loss (mm)")
    fig.tight_layout()


# %% Sequence analysis
def count_motifs(sequences, length):
    """Count overlapping outcome motifs of one length."""
    counts = Counter()
    for sequence in sequences:
        for start in range(len(sequence) - length + 1):
            counts[tuple(sequence[start:start + length])] += 1
    return counts


def get_motif_null_counts(sequences, motifs, length, null_type, rng):
    """Generate motif counts from global or within-track label shuffles."""
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
            raise ValueError(f"Unknown motif null type: {null_type}")
        shuffled_counts = count_motifs(shuffled, length)
        null_counts[permutation] = [shuffled_counts[motif] for motif in motifs]
    return null_counts


def summarize_motif_null(observed_counts, motifs, null_counts):
    """Return observed motif counts and enrichment relative to one null."""
    null_mean = null_counts.mean(axis=0)
    null_std = null_counts.std(axis=0)
    z_score = np.divide(
        observed_counts - null_mean,
        null_std,
        out=np.zeros_like(null_mean),
        where=null_std > 0,
    )
    p_enriched = (
        1 + (null_counts >= observed_counts).sum(axis=0)
    ) / (MOTIF_PERMUTATIONS + 1)
    return pd.DataFrame(
        {
            "motif": [" → ".join(OUTCOME_LABEL[item] for item in motif) for motif in motifs],
            "count": observed_counts,
            "null_mean": null_mean,
            "z_score": z_score,
            "p_enriched": p_enriched,
        }
    )


def plot_motif_enrichment(events):
    """Plot enriched two-event and three-event outcome sequences."""
    sequences = [
        group.sort_values("loss_time_s")["outcome"].to_list()
        for _, group in events.groupby("track_id")
        if len(group) >= 2
    ]
    if not sequences:
        print("Too few sequential events for motif analysis.")
        return

    rng = np.random.default_rng(MOTIF_RANDOM_SEED)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for row, length in enumerate((2, 3)):
        motifs = list(product(OUTCOME_ORDER, repeat=length))
        observed = count_motifs(sequences, length)
        observed_counts = np.array([observed[motif] for motif in motifs])
        for column, (null_type, title, color) in enumerate(
            (("global", "Global label shuffle", "C1"),
             ("within_track", "Within-track label shuffle", "C0"))
        ):
            null_counts = get_motif_null_counts(
                sequences, motifs, length, null_type, rng
            )
            results = summarize_motif_null(observed_counts, motifs, null_counts)
            results = results.loc[results["count"] > 0]
            if length == 3:
                low = results.nsmallest(6, "z_score")
                high = results.nlargest(6, "z_score")
                results = pd.concat([low, high]).drop_duplicates("motif")
                results = results.sort_values("z_score", ascending=False)
            else:
                results = results.nlargest(10, "z_score")
            axis = axes[row, column]
            axis.barh(results["motif"], results["z_score"], color=color)
            axis.axvline(0, color="black", lw=1)
            axis.set(
                xlabel="Enrichment z score", ylabel="Motif",
                title=f"{length}-event motifs: {title}",
            )
            axis.invert_yaxis()
    fig.tight_layout()


def run():
    """Run the temporal gap-crossing analysis for the selected recordings."""
    target_files = get_target_files(ROOT_DIR)
    print(f"Target files: {len(target_files)}")
    tracks = load_tracks(target_files)
    timing = detect_global_timing(tracks)
    print(f"Valid tracks: {len(tracks)}")
    print(f"Pooled global losses: {len(timing['loss_time_s'])}")
    print(f"Loss times (s): {timing['loss_time_s']}")
    plot_time_signal(tracks, timing)

    events = make_event_table(tracks, timing)
    print(f"Tracks with attempts: {events['track_id'].nunique()}")
    print(f"Valid attempts: {len(events)}")
    print(events.groupby("outcome").size().reindex(OUTCOME_ORDER, fill_value=0))
    plot_event_summary(events)
    plot_event_paths(events)
    plot_motif_enrichment(events)
    plt.show()


if __name__ == "__main__":
    run()
