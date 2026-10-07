"""Strain comparison of walking kinematics and signal responses.

Run load_db.py for each strain first, then set DATASET_DIRS to the strain
dataset folders. Each frame is in one state:
    in_signal:   signal present.
    post_signal: no signal, at most POST_SIGNAL_S after the last signal exit.
    baseline:    no signal, longer since the last exit or never in signal.

Every metric is computed per sampling unit: a day-vial session, or a block
of TRIALS_PER_BLOCK consecutive trials within a session (SAMPLE_UNIT). Either
way, confidence intervals and p values treat sessions as the independent
samples. Upwind is -x; theta_smooth of an upwind-moving fly is 180 deg.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from gap_crossing import run_io
    from gap_crossing import strain_compare as compare
except ModuleNotFoundError:
    import run_io
    import strain_compare as compare


# %% Dataset settings
# Strain label -> dataset folder written by load_db.py.
# DATASET_DIRS: dict[str, Path] = {}
DATASET_DIRS: dict[str, Path] = {
    "GMOCLKir_empty": Path(r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui\gap_crossing\datasets\2026-10-06_182613_gap_ribbon_GMOCLKir_empty"),
    "GMOCLKir_FC2": Path(r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui\gap_crossing\datasets\2026-10-06_182613_gap_ribbon_GMOCLKir_FC2"),
    "GMOCLKir_86861": Path(r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui\gap_crossing\datasets\2026-10-06_182613_gap_ribbon_GMOCLKir_86861"),
    "GMOCLKir_89256": Path(r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui\gap_crossing\datasets\2026-10-06_182613_gap_ribbon_GMOCLKir_89256"),
    "117_GMUCR": Path(r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui\gap_crossing\datasets\2026-10-06_182613_gap_ribbon_117_GMUCR"),
}
REFERENCE_LABEL = None  # None uses the first DATASET_DIRS label.

# %% Analysis settings
FRAME_RATE_HZ = 60
UPWIND_HEADING_DEG = 180.0
POST_SIGNAL_S = 2.0
STOP_SPEED_MM_S = 1.5
RECAPTURE_S = 2.0
PROFILE_PRE_S = 2.0
PROFILE_POST_S = 5.0
MIN_PRIOR_BLANK_S = 1.0
MIN_STATE_FRAMES = 600
MIN_SESSION_BOUTS = 5
MIN_SESSION_EVENTS = 5
BOOTSTRAP_SAMPLES = 10_000
PROFILE_BOOTSTRAP_SAMPLES = 2_000
PERMUTATION_COUNT = 10_000
RANDOM_SEED = 0

# %% Sampling unit
# "session": one dot per day-vial session, all trials pooled.
# "block": one dot per TRIALS_PER_BLOCK consecutive trials of a session. Blocks
# of one session are resampled within that session and keep their strain label
# together, so they show within-vial spread without adding independent samples.
SAMPLE_UNIT = "block" #"session"
TRIALS_PER_BLOCK = 5
MIN_BLOCK_TRIALS = 3  # A shorter last block joins the block before it (17 trials -> 5, 5, 7).
SAMPLE_UNITS = ("session", "block")

STATES = ["in_signal", "post_signal", "baseline"]
FRAME_METRIC_BINS = {
    "speed": ("Speed (mm/s)", np.linspace(0, 30, 61)),
    "upwind_velocity": ("Upwind velocity (mm/s)", np.linspace(-30, 30, 61)),
    "heading_from_upwind": ("Heading from upwind (deg)", np.linspace(-180, 180, 37)),
    "turn_rate": ("Turn rate (deg/s)", np.linspace(0, 500, 51)),
}
# Distribution rows drawn with a log y-axis (empty bins are not drawn).
LOG_Y_METRICS = ("speed", "upwind_velocity")
STATE_METRIC_LABELS = {
    "time_fraction": "Fraction of tracked time",
    "speed_median": "Median speed (mm/s)",
    "stop_fraction": f"Stop fraction (< {STOP_SPEED_MM_S} mm/s)",
    "upwind_velocity_mean": "Mean upwind velocity (mm/s)",
    "crosswind_speed_mean": "Mean crosswind speed (mm/s)",
    "upwind_alignment_mean": "Mean upwind alignment (cos)",
    "heading_resultant": "Heading concentration (resultant)",
    "turn_rate_median": "Median turn rate (deg/s)",
}
BOUT_METRIC_LABELS = {
    "signal_time_fraction": "Fraction of time in signal",
    "encounter_rate_per_min": "Signal entries per minute",
    "bout_duration_median": "Median bout duration (s)",
    "reencounter_interval_median": "Median re-encounter interval (s)",
    "recapture_probability": f"P(re-entry within {RECAPTURE_S:g} s)",
    "upwind_displacement_per_bout": "Upwind displacement per bout (mm)",
}
def make_frame_duration_bins(max_s, count=40):
    """Return log-spaced duration bin edges that fall halfway between frames.

    Every bin then contains at least one whole-frame duration.
    """
    frames = np.unique(np.round(np.logspace(0, np.log10(max_s * FRAME_RATE_HZ), count + 1)))
    return (np.r_[frames, frames[-1] + 1] - 0.5) / FRAME_RATE_HZ


BOUT_DURATION_BINS = make_frame_duration_bins(60)
INTERVAL_DURATION_BINS = make_frame_duration_bins(600)
UNTESTED_METRICS = ("bout_count",)
PROFILE_QUANTITIES = {
    "upwind_velocity": "Upwind velocity (mm/s)",
    "speed": "Speed (mm/s)",
    "turn_rate": "Turn rate (deg/s)",
    "upwind_alignment": "Upwind alignment (cos)",
    "signal_present": "P(in signal)",
}
EVENT_TYPES = ["onset", "offset"]


# %% Frame quantities
def wrap_degrees(angle):
    """Return angles in [-180, 180)."""
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


def label_states(in_signal, time_s, post_signal_s=POST_SIGNAL_S):
    """Return state codes: 0 in signal, 1 post-signal, 2 baseline."""
    in_signal = np.asarray(in_signal, dtype=bool)
    time_s = np.asarray(time_s, dtype=float)
    exits = np.zeros(len(in_signal), dtype=bool)
    exits[1:] = in_signal[:-1] & ~in_signal[1:]
    last_exit = np.maximum.accumulate(np.where(exits, np.arange(len(in_signal)), -1))
    since_exit = np.where(
        last_exit >= 0, time_s - time_s[np.clip(last_exit, 0, None)], np.inf
    )
    return np.where(in_signal, 0, np.where(since_exit <= post_signal_s, 1, 2))


def find_bouts(in_signal):
    """Return start and end (exclusive) indices of continuous in-signal runs."""
    padded = np.r_[False, np.asarray(in_signal, dtype=bool), False].astype(int)
    change = np.diff(padded)
    return np.flatnonzero(change == 1), np.flatnonzero(change == -1)


def fill_over_jumps(observed, jumps):
    """Return observed values with each jump run filled by the last valid value."""
    observed = np.asarray(observed, dtype=bool)
    last_valid = np.maximum.accumulate(np.where(~jumps, np.arange(len(jumps)), -1))
    return np.where(last_valid >= 0, observed[np.clip(last_valid, 0, None)], False)


def describe_track(track):
    """Return per-frame quantities for one track; jump frames become NaN.

    Signal is NaN on jump frames. in_signal carries the last observed signal
    state through each jump, so a jump does not create an exit and entry.
    """
    signal = np.asarray(track["signal"], dtype=float)
    jumps = np.asarray(track["jumps"], dtype=bool)
    observed = np.nan_to_num(signal, nan=0.0) > 0
    in_signal = fill_over_jumps(observed, jumps)
    time_s = np.asarray(track["time_s"], dtype=float)
    velocity = np.asarray(track["velocity"], dtype=float)
    valid = ~jumps
    heading = wrap_degrees(np.asarray(track["theta_smooth"], dtype=float) - UPWIND_HEADING_DEG)
    quantities = {
        "speed": np.asarray(track["speed_smooth"], dtype=float),
        "upwind_velocity": -velocity[:, 0],
        "crosswind_speed": np.abs(velocity[:, 1]),
        "heading_from_upwind": heading,
        "upwind_alignment": np.cos(np.radians(heading)),
        "turn_rate": np.abs(np.asarray(track["dtheta_smooth"], dtype=float)),
        "signal_present": observed.astype(float),
    }
    for values in quantities.values():
        values[~valid] = np.nan
    return {
        "time_s": time_s,
        "x": np.asarray(track["xy"], dtype=float)[:, 0],
        "in_signal": in_signal,
        "state": label_states(in_signal, time_s),
        "valid": valid,
        "jumps": jumps,
        **quantities,
    }


# %% Bouts and intervals
def make_track_bouts(frames, track_id):
    """Return in-signal bouts and the out-of-signal intervals after them.

    A bout edge next to the track end or a jump is censored. An interval stops
    at the first jump frame and is then censored.
    """
    starts, ends = find_bouts(frames["in_signal"])
    jumps = frames["jumps"]
    frame_count = len(frames["in_signal"])
    bouts = [
        {
            "track_id": track_id,
            "bout_index": index,
            "start_time_s": frames["time_s"][start],
            "duration_s": (end - start) / FRAME_RATE_HZ,
            "left_censored": start == 0 or bool(jumps[start - 1]),
            "right_censored": end == frame_count or bool(jumps[end - 1]),
            "has_jump": bool(frames["jumps"][start:end].any()),
            "upwind_displacement_mm": frames["x"][start] - frames["x"][end - 1],
        }
        for index, (start, end) in enumerate(zip(starts, ends))
    ]
    intervals = []
    for index, end in enumerate(ends):
        if end == frame_count or jumps[end - 1]:
            continue
        next_start = starts[index + 1] if index + 1 < len(starts) else frame_count
        jump_frames = np.flatnonzero(jumps[end:next_start])
        stop = end + jump_frames[0] if len(jump_frames) else next_start
        intervals.append({
            "track_id": track_id,
            "after_bout_index": index,
            "duration_s": (stop - end) / FRAME_RATE_HZ,
            "censored": index + 1 >= len(starts) or len(jump_frames) > 0,
        })
    return bouts, intervals


# %% Event-triggered profiles
def get_profile_offsets():
    """Return frame offsets and times of the profile window."""
    offsets = np.arange(
        -int(round(PROFILE_PRE_S * FRAME_RATE_HZ)),
        int(round(PROFILE_POST_S * FRAME_RATE_HZ)) + 1,
    )
    return offsets, offsets / FRAME_RATE_HZ


def find_profile_events(in_signal, jumps=None):
    """Return onset and offset frame indices used for triggered profiles.

    Onsets need MIN_PRIOR_BLANK_S without signal or jumps before them. Events
    whose timing is uncertain because of a jump are left out.
    """
    jumps = np.zeros(len(in_signal), dtype=bool) if jumps is None else np.asarray(jumps, dtype=bool)
    starts, ends = find_bouts(in_signal)
    blank_frames = int(round(MIN_PRIOR_BLANK_S * FRAME_RATE_HZ))
    previous_ends = np.r_[0, ends[:-1]]
    jump_count = np.r_[0, np.cumsum(jumps)]
    clean_blank = jump_count[starts] - jump_count[np.clip(starts - blank_frames, 0, None)] == 0
    onsets = starts[(starts > 0) & (starts - previous_ends >= blank_frames) & clean_blank]
    offsets = ends[(ends < len(in_signal)) & ~jumps[np.clip(ends - 1, 0, None)]]
    return {"onset": onsets, "offset": offsets}


def gather_windows(values, events, offsets):
    """Return an events x offsets array; samples outside the track are NaN."""
    values = np.asarray(values, dtype=float)
    index = np.asarray(events)[:, None] + offsets[None, :]
    inside = (index >= 0) & (index < len(values))
    windows = np.full(index.shape, np.nan)
    windows[inside] = values[index[inside]]
    return windows


# %% Session analysis
def _state_metrics(frames, state_code):
    """Return scalar metrics for one state in one session."""
    mask = frames["valid"] & (frames["state"] == state_code)
    frame_count = int(mask.sum())
    metrics = {"time_fraction": frame_count / max(int(frames["valid"].sum()), 1)}
    if frame_count < MIN_STATE_FRAMES:
        return frame_count, {**metrics, **{name: np.nan for name in STATE_METRIC_LABELS if name != "time_fraction"}}

    def finite(name):
        values = frames[name][mask]
        return values[np.isfinite(values)]

    speed = finite("speed")
    heading = finite("heading_from_upwind")
    turn_rate = finite("turn_rate")
    metrics.update({
        "speed_median": np.median(speed) if len(speed) else np.nan,
        "stop_fraction": np.mean(speed < STOP_SPEED_MM_S) if len(speed) else np.nan,
        "upwind_velocity_mean": np.mean(finite("upwind_velocity")),
        "crosswind_speed_mean": np.mean(finite("crosswind_speed")),
        "upwind_alignment_mean": np.mean(finite("upwind_alignment")) if len(heading) else np.nan,
        "heading_resultant": (
            np.abs(np.mean(np.exp(1j * np.radians(heading)))) if len(heading) else np.nan
        ),
        "turn_rate_median": np.median(turn_rate) if len(turn_rate) else np.nan,
    })
    return frame_count, metrics


def _bout_metrics(frames, bouts, intervals, track_minutes):
    """Return scalar bout and encounter metrics for one session."""
    complete = bouts.loc[~bouts["left_censored"] & ~bouts["right_censored"] & ~bouts["has_jump"]]
    closed = intervals.loc[~intervals["censored"]] if len(intervals) else intervals
    entries = int((~bouts["left_censored"]).sum()) if len(bouts) else 0
    metrics = {
        "signal_time_fraction": float(np.mean(frames["in_signal"][frames["valid"]])),
        "encounter_rate_per_min": entries / track_minutes if track_minutes > 0 else np.nan,
        "bout_count": len(bouts),
    }
    enough = len(complete) >= MIN_SESSION_BOUTS
    metrics["bout_duration_median"] = complete["duration_s"].median() if enough else np.nan
    metrics["upwind_displacement_per_bout"] = (
        complete["upwind_displacement_mm"].mean() if enough else np.nan
    )
    metrics["reencounter_interval_median"] = (
        closed["duration_s"].median() if len(closed) >= MIN_SESSION_BOUTS else np.nan
    )
    if len(intervals):
        recaptured = ~intervals["censored"] & (intervals["duration_s"] <= RECAPTURE_S)
        known = recaptured | (intervals["duration_s"] > RECAPTURE_S)
        metrics["recapture_probability"] = (
            recaptured[known].mean() if known.sum() >= MIN_SESSION_BOUTS else np.nan
        )
    else:
        metrics["recapture_probability"] = np.nan
    return metrics


UNIT_COLUMNS = [*compare.SESSION_KEYS, "session_id", "unit_id", "block", "block_trials"]


def analyze_session(session, tracks):
    """Return all tables for one sampling unit (a session or a trial block)."""
    session_values = {key: session[key] for key in UNIT_COLUMNS}
    offsets, profile_times = get_profile_offsets()
    frame_parts = []
    bout_rows = []
    interval_rows = []
    profile_sums = {
        (event, name): np.zeros(len(offsets)) for event in EVENT_TYPES for name in PROFILE_QUANTITIES
    }
    profile_counts = {key: np.zeros(len(offsets)) for key in profile_sums}
    event_counts = dict.fromkeys(EVENT_TYPES, 0)

    for track in tracks:
        frames = describe_track(track)
        frame_parts.append(frames)
        bouts, intervals = make_track_bouts(frames, track["track_id"])
        bout_rows.extend(bouts)
        interval_rows.extend(intervals)
        for event, indices in find_profile_events(frames["in_signal"], frames["jumps"]).items():
            if not len(indices):
                continue
            event_counts[event] += len(indices)
            for name in PROFILE_QUANTITIES:
                windows = gather_windows(frames[name], indices, offsets)
                profile_sums[(event, name)] += np.nansum(windows, axis=0)
                profile_counts[(event, name)] += np.isfinite(windows).sum(axis=0)

    frames = {
        name: np.concatenate([part[name] for part in frame_parts])
        for name in frame_parts[0]
    }
    track_minutes = sum(int(part["valid"].sum()) for part in frame_parts) / FRAME_RATE_HZ / 60

    state_rows = []
    histogram_rows = []
    for state_code, state in enumerate(STATES):
        frame_count, metrics = _state_metrics(frames, state_code)
        state_rows.extend(
            {**session_values, "state": state, "metric": name, "value": value,
             "frame_count": frame_count, "track_count": len(tracks)}
            for name, value in metrics.items()
        )
        if frame_count < MIN_STATE_FRAMES:
            continue
        mask = frames["valid"] & (frames["state"] == state_code)
        for name, (_, bins) in FRAME_METRIC_BINS.items():
            fractions = compare.session_histogram(frames[name][mask], bins)
            centers = (bins[:-1] + bins[1:]) / 2
            histogram_rows.extend(
                {**session_values, "state": state, "metric": name,
                 "bin_center": center, "fraction": fraction}
                for center, fraction in zip(centers, fractions)
            )

    bouts = pd.DataFrame(bout_rows, columns=[
        "track_id", "bout_index", "start_time_s", "duration_s", "left_censored",
        "right_censored", "has_jump", "upwind_displacement_mm",
    ])
    intervals = pd.DataFrame(interval_rows, columns=[
        "track_id", "after_bout_index", "duration_s", "censored",
    ])
    bout_metrics = _bout_metrics(frames, bouts, intervals, track_minutes)
    bout_metric_rows = [
        {**session_values, "metric": name, "value": value, "track_count": len(tracks)}
        for name, value in bout_metrics.items()
    ]
    duration_rows = []
    complete = bouts.loc[~bouts["left_censored"] & ~bouts["right_censored"] & ~bouts["has_jump"]]
    closed = intervals.loc[~intervals["censored"]]
    for kind, values, bins in [
        ("bout_duration", complete["duration_s"], BOUT_DURATION_BINS),
        ("reencounter_interval", closed["duration_s"], INTERVAL_DURATION_BINS),
    ]:
        if len(values) < MIN_SESSION_BOUTS:
            continue
        fractions = compare.session_histogram(values, bins)
        centers = np.sqrt(bins[:-1] * bins[1:])
        duration_rows.extend(
            {**session_values, "metric": kind, "bin_center": center, "fraction": fraction}
            for center, fraction in zip(centers, fractions)
        )

    profile_rows = []
    for (event, name), sums in profile_sums.items():
        if event_counts[event] < MIN_SESSION_EVENTS:
            continue
        counts = profile_counts[(event, name)]
        means = np.divide(sums, counts, out=np.full(len(sums), np.nan), where=counts > 0)
        profile_rows.extend(
            {**session_values, "event": event, "quantity": name, "time_s": time,
             "value": value, "event_count": event_counts[event]}
            for time, value in zip(profile_times, means)
        )

    return {
        "state_metrics": pd.DataFrame(state_rows),
        "state_histograms": pd.DataFrame(histogram_rows),
        "bouts": bouts.assign(**session_values),
        "intervals": intervals.assign(**session_values),
        "bout_metrics": pd.DataFrame(bout_metric_rows),
        "duration_histograms": pd.DataFrame(duration_rows),
        "profiles": pd.DataFrame(profile_rows),
    }


def get_trials_per_block():
    """Return TRIALS_PER_BLOCK in block mode and None in session mode."""
    if SAMPLE_UNIT not in SAMPLE_UNITS:
        raise ValueError(f"SAMPLE_UNIT must be one of {SAMPLE_UNITS}, not {SAMPLE_UNIT!r}.")
    return TRIALS_PER_BLOCK if SAMPLE_UNIT == "block" else None


def get_unit_label():
    """Return the plural name of one plotted dot."""
    return "sessions" if SAMPLE_UNIT == "session" else f"{TRIALS_PER_BLOCK}-trial blocks"


def analyze_units(tracks, recordings):
    """Return tables for every sampling unit, concatenated."""
    units = compare.group_tracks_by_unit(
        tracks, recordings, get_trials_per_block(), MIN_BLOCK_TRIALS
    )
    parts = {}
    for unit_id, (unit, unit_tracks) in units.items():
        print(f"{unit_id}: {len(unit_tracks)} tracks, {unit['block_trials']} trials")
        for name, table in analyze_session(unit, unit_tracks).items():
            parts.setdefault(name, []).append(table)
    return {name: pd.concat(tables, ignore_index=True) for name, tables in parts.items()}


def summarize(tables, rng):
    """Return strain summaries of the session tables."""
    return {
        "state_metrics": compare.summarize_by_strain(
            tables["state_metrics"], "value", ["state", "metric"], rng, BOOTSTRAP_SAMPLES,
            "session_id",
        ),
        "state_histograms": compare.summarize_by_strain(
            tables["state_histograms"], "fraction", ["state", "metric", "bin_center"],
            rng, PROFILE_BOOTSTRAP_SAMPLES, "session_id",
        ),
        "bout_metrics": compare.summarize_by_strain(
            tables["bout_metrics"], "value", ["metric"], rng, BOOTSTRAP_SAMPLES, "session_id",
        ),
        "duration_histograms": compare.summarize_by_strain(
            tables["duration_histograms"], "fraction", ["metric", "bin_center"],
            rng, PROFILE_BOOTSTRAP_SAMPLES, "session_id",
        ),
        "profiles": compare.summarize_by_strain(
            tables["profiles"], "value", ["event", "quantity", "time_s"],
            rng, PROFILE_BOOTSTRAP_SAMPLES, "session_id",
        ),
    }


def compare_strains(tables, reference, rng):
    """Return reference comparisons for the scalar session metrics."""
    state = compare.compare_to_reference(
        tables["state_metrics"], "value", reference, ["state", "metric"], rng, PERMUTATION_COUNT,
        "session_id",
    )
    bout_metrics = tables["bout_metrics"]
    bout = compare.compare_to_reference(
        bout_metrics.loc[~bout_metrics["metric"].isin(UNTESTED_METRICS)],
        "value", reference, ["metric"], rng, PERMUTATION_COUNT, "session_id",
    )
    comparisons = pd.concat(
        [state.assign(table="state_metrics"), bout.assign(table="bout_metrics")],
        ignore_index=True,
    )
    comparisons["q_value"] = compare.benjamini_hochberg(comparisons["p_value"])
    return comparisons


def make_variance_components(tables):
    """Return between- and within-session spread of the scalar unit metrics."""
    state = compare.variance_components(tables["state_metrics"], "value", ["state", "metric"])
    bout = compare.variance_components(tables["bout_metrics"], "value", ["metric"])
    return pd.concat(
        [state.assign(table="state_metrics"), bout.assign(table="bout_metrics")],
        ignore_index=True,
    )


# %% Plots
def plot_state_distributions(summary, order, colors):
    """Plot session-averaged distributions by state and strain."""
    fig, axes = plt.subplots(len(FRAME_METRIC_BINS), len(STATES), figsize=(15, 13), squeeze=False)
    for row, (metric, (label, _)) in enumerate(FRAME_METRIC_BINS.items()):
        for column, state in enumerate(STATES):
            axis = axes[row, column]
            subset = summary.loc[(summary["metric"] == metric) & (summary["state"] == state)]
            if metric in LOG_Y_METRICS:
                axis.set_yscale("log", nonpositive="mask")
            compare.plot_strain_curves(axis, subset, "bin_center", order, colors)
            axis.set(xlabel=label, ylabel="Fraction of frames" if column == 0 else None)
            if row == 0:
                axis.set_title(state.replace("_", " "))
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        f"Kinematic distributions by signal state (mean of {get_unit_label()}, 95% CI); "
        "upwind velocity > 0 is toward the source"
    )
    fig.tight_layout()
    return fig


def plot_state_metrics(session_metrics, summary, order, colors):
    """Plot scalar state metrics with session points and strain CIs."""
    metrics = list(STATE_METRIC_LABELS)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        compare.plot_strain_points(
            axis,
            session_metrics.loc[session_metrics["metric"] == metric],
            summary.loc[summary["metric"] == metric],
            "value", order, colors, x_column="state", x_order=STATES,
        )
        axis.set_xticklabels([state.replace("_", "\n") for state in STATES])
        axis.set(ylabel=STATE_METRIC_LABELS[metric])
        if metric == "upwind_velocity_mean":
            axis.axhline(0, color="0.6", lw=1)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Metrics by signal state (dots: {get_unit_label()}; bars: 95% CI over sessions)")
    fig.tight_layout()
    return fig


def plot_bout_metrics(session_metrics, summary, duration_summary, order, colors):
    """Plot bout and interval distributions and encounter metrics."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
    for axis, kind, label in [
        (axes[0, 0], "bout_duration", "In-signal bout duration (s)"),
        (axes[0, 1], "reencounter_interval", "Exit to next entry (s)"),
    ]:
        compare.plot_strain_curves(
            axis, duration_summary.loc[duration_summary["metric"] == kind],
            "bin_center", order, colors,
        )
        axis.set(xscale="log", xlabel=label, ylabel="Fraction of bouts")
    axes[0, 0].legend(fontsize=9)
    for axis, metric in zip(list(axes.flat)[2:], BOUT_METRIC_LABELS):
        compare.plot_strain_points(
            axis,
            session_metrics.loc[session_metrics["metric"] == metric],
            summary.loc[summary["metric"] == metric],
            "value", order, colors,
        )
        axis.set(ylabel=BOUT_METRIC_LABELS[metric])
    fig.suptitle(f"Signal bouts and encounters (dots: {get_unit_label()}; bars: 95% CI over sessions)")
    fig.tight_layout()
    return fig


def plot_profiles(summary, event, order, colors):
    """Plot onset- or offset-triggered profiles by strain."""
    fig, axes = plt.subplots(1, len(PROFILE_QUANTITIES), figsize=(22, 4.5), squeeze=False)
    for axis, (quantity, label) in zip(axes[0], PROFILE_QUANTITIES.items()):
        subset = summary.loc[(summary["event"] == event) & (summary["quantity"] == quantity)]
        compare.plot_strain_curves(axis, subset, "time_s", order, colors)
        axis.axvline(0, color="0.5", lw=1, ls="--")
        axis.set(xlabel=f"Time from signal {event} (s)", ylabel=label)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Signal {event} response (mean of {get_unit_label()}, 95% CI)")
    fig.tight_layout()
    return fig


# %% Run
def run():
    """Compare kinematics and signal responses across strain datasets."""
    settings_modules = [compare, sys.modules[__name__]]
    with run_io.analysis_run(__file__, DATASET_DIRS, settings_modules) as run_info:
        rng = np.random.default_rng(RANDOM_SEED)
        order = compare.get_strain_order(run_info.datasets, REFERENCE_LABEL)
        colors = compare.get_strain_colors(order)
        print(f"Strains: {order} (reference: {order[0]})")

        print(f"Sampling unit: {get_unit_label()}")
        tables = analyze_units(run_info.tracks, run_info.recordings)
        counts = tables["bout_metrics"].groupby("dataset_label").agg(
            sessions=("session_id", "nunique"), units=("unit_id", "nunique")
        )
        print("Sessions and plotted units per strain:")
        print(counts.reindex(order).to_string())
        summaries = summarize(tables, rng)
        comparisons = compare_strains(tables, order[0], rng)

        for name, table in tables.items():
            run_info.save_table(
                f"{SAMPLE_UNIT}_{name}" if name not in ("bouts", "intervals") else name, table
            )
        if SAMPLE_UNIT == "block":
            variance = make_variance_components(tables)
            run_info.save_table("variance_components", variance)
            print("Share of variance between sessions (1 = sessions differ, 0 = blocks differ):")
            print(variance.groupby(["table", "state", "metric"], dropna=False)["between_fraction"]
                  .median().round(2).to_string())
        for name, table in summaries.items():
            run_info.save_table(f"strain_{name}", table)
        run_info.save_table("strain_comparisons", comparisons)
        if not comparisons.empty:
            print("Differences from the reference strain (permutation of whole sessions; "
                  "q = Benjamini-Hochberg over all tests in this table):")
            print(comparisons[["table", "state", "metric", "dataset_label", "difference",
                               "p_value", "q_value", "session_count", "reference_session_count"]]
                  .to_string(index=False))

        plot_state_distributions(summaries["state_histograms"], order, colors)
        plot_state_metrics(tables["state_metrics"], summaries["state_metrics"], order, colors)
        plot_bout_metrics(
            tables["bout_metrics"], summaries["bout_metrics"],
            summaries["duration_histograms"], order, colors,
        )
        for event in EVENT_TYPES:
            plot_profiles(summaries["profiles"], event, order, colors)
        run_info.show()


if __name__ == "__main__":
    run()
