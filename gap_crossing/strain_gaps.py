"""Strain comparison of gap-crossing attempts, outcomes, and transitions.

Run load_db.py for each strain first, then set DATASET_DIRS to the strain
dataset folders. Gap geometry is detected once from all strains' tracks, since
all recordings share one camera mounting. Attempts and outcomes follow
gap_cross_track.py; tracks with fewer than MIN_ATTEMPTS_PER_TRACK attempts
there are left out.

Every metric is computed per sampling unit: a day-vial session, or a block of
TRIALS_PER_BLOCK consecutive trials within a session (SAMPLE_UNIT):
    attempts_per_track:      attempts per attempting fly track.
    outcome_fraction:        fraction of attempts that cross, regain, or abort.
    interval_mean:           mean decision interval, the time between
                             consecutive attempts of one track, over all
                             intervals and per transition.
    transition_probability:  P(current outcome | previous outcome).
Either way, confidence intervals and p values treat sessions as the
independent samples.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from gap_crossing import gap_cross_track as analysis
    from gap_crossing import run_io
    from gap_crossing import strain_compare as compare
except ModuleNotFoundError:
    import gap_cross_track as analysis
    import run_io
    import strain_compare as compare


# %% Dataset settings
# Strain label -> dataset folder written by load_db.py.
DATASET_DIRS: dict[str, Path] = {
    "GMOCLKir_empty": Path(r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui\gap_crossing\datasets\2026-10-06_182613_gap_ribbon_GMOCLKir_empty"),
    "GMOCLKir_FC2": Path(r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui\gap_crossing\datasets\2026-10-06_182613_gap_ribbon_GMOCLKir_FC2"),
    "GMOCLKir_86861": Path(r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui\gap_crossing\datasets\2026-10-06_182613_gap_ribbon_GMOCLKir_86861"),
    "GMOCLKir_89256": Path(r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui\gap_crossing\datasets\2026-10-06_182613_gap_ribbon_GMOCLKir_89256"),
    "117_GMUCR": Path(r"C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\projects\optogui\gap_crossing\datasets\2026-10-06_182613_gap_ribbon_117_GMUCR"),
}
REFERENCE_LABEL = None  # None uses the first DATASET_DIRS label.

# %% Analysis settings
# Units with fewer attempts, intervals, or transitions from the conditioning
# outcome get NaN for that metric.
MIN_UNIT_ATTEMPTS = 5
MIN_UNIT_INTERVALS = 5
MIN_UNIT_TRANSITIONS = 5
BOOTSTRAP_SAMPLES = 10_000
PERMUTATION_COUNT = 10_000
RANDOM_SEED = 0

# %% Sampling unit
# "session": one dot per day-vial session, all trials pooled.
# "block": one dot per TRIALS_PER_BLOCK consecutive trials of a session. Blocks
# of one session are resampled within that session and keep their strain label
# together, so they show within-vial spread without adding independent samples.
SAMPLE_UNIT = "session"
TRIALS_PER_BLOCK = 5
MIN_BLOCK_TRIALS = 3  # Blocks with fewer trials, e.g. a short last block, are left out.
SAMPLE_UNITS = ("session", "block")

# (previous outcome, current outcome) pairs to compare.
TRANSITIONS = [("regain", "regain"), ("regain", "cross"), ("cross", "cross")]
TRANSITION_LABELS = [f"{previous}->{current}" for previous, current in TRANSITIONS]
METRIC_CATEGORIES = {
    "attempts_per_track": ["all"],
    "outcome_fraction": analysis.OUTCOME_ORDER,
    "interval_mean": ["all", *TRANSITION_LABELS],
    "transition_probability": TRANSITION_LABELS,
}
METRIC_LABELS = {
    "attempts_per_track": "Attempts per attempting track",
    "outcome_fraction": "Fraction of attempts",
    "interval_mean": "Mean decision interval (s)",
    "transition_probability": "P(current | previous)",
}
UNIT_COLUMNS = [*compare.SESSION_KEYS, "session_id", "unit_id", "block", "block_trials"]
METRIC_COLUMNS = [*UNIT_COLUMNS, "metric", "category", "value", "count",
                  "attempt_count", "track_count"]


# %% Events
def add_unit_columns(events, recordings, trials_per_block=None, min_block_trials=1):
    """Return events with sampling-unit columns; events in dropped blocks are left out."""
    lookup = compare.make_unit_lookup(recordings, trials_per_block, min_block_trials)
    units = events["source_file"].map(lambda source_file: lookup[source_file])
    kept = units.notna()
    unit_values = pd.DataFrame(units[kept].tolist(), index=events.index[kept])[UNIT_COLUMNS]
    return events.loc[kept].drop(columns=UNIT_COLUMNS, errors="ignore").join(unit_values)


def make_transition_table(events):
    """Return one row per pair of consecutive attempts within a track.

    interval_s is the decision interval: the time from the previous attempt to
    the current one. Non-positive or non-finite intervals become NaN.
    """
    ordered = events.sort_values(["track_id", "attempt_time_s"], kind="stable")
    grouped = ordered.groupby("track_id", sort=False)
    interval_s = ordered["attempt_time_s"] - grouped["attempt_time_s"].shift(1)
    transitions = ordered.assign(
        previous_outcome=grouped["outcome"].shift(1),
        interval_s=interval_s.where(np.isfinite(interval_s) & (interval_s > 0)),
    )
    transitions = transitions.loc[transitions["previous_outcome"].notna()].copy()
    transitions["transition"] = transitions["previous_outcome"] + "->" + transitions["outcome"]
    return transitions


# %% Unit analysis
def analyze_unit(unit_events):
    """Return metric rows (metric, category, value, count) for one sampling unit.

    count is the number of attempts, intervals, or conditioning transitions
    behind each value.
    """
    transitions = make_transition_table(unit_events)
    attempt_count = len(unit_events)
    track_count = unit_events["parent_track_id"].nunique()
    rows = [("attempts_per_track", "all", attempt_count / track_count, track_count)]

    for outcome in analysis.OUTCOME_ORDER:
        fraction = (unit_events["outcome"] == outcome).mean()
        rows.append((
            "outcome_fraction", outcome,
            fraction if attempt_count >= MIN_UNIT_ATTEMPTS else np.nan, attempt_count,
        ))

    intervals = transitions.dropna(subset=["interval_s"])
    for category in METRIC_CATEGORIES["interval_mean"]:
        values = (
            intervals["interval_s"] if category == "all"
            else intervals.loc[intervals["transition"] == category, "interval_s"]
        )
        rows.append((
            "interval_mean", category,
            values.mean() if len(values) >= MIN_UNIT_INTERVALS else np.nan, len(values),
        ))

    for (previous, current), category in zip(TRANSITIONS, TRANSITION_LABELS):
        following = transitions.loc[transitions["previous_outcome"] == previous, "outcome"]
        rows.append((
            "transition_probability", category,
            (following == current).mean() if len(following) >= MIN_UNIT_TRANSITIONS else np.nan,
            len(following),
        ))

    unit_values = unit_events.iloc[0][UNIT_COLUMNS].to_dict()
    return pd.DataFrame(
        [
            {**unit_values, "metric": metric, "category": category, "value": value,
             "count": count, "attempt_count": attempt_count, "track_count": track_count}
            for metric, category, value, count in rows
        ],
        columns=METRIC_COLUMNS,
    )


def get_trials_per_block():
    """Return TRIALS_PER_BLOCK in block mode and None in session mode."""
    if SAMPLE_UNIT not in SAMPLE_UNITS:
        raise ValueError(f"SAMPLE_UNIT must be one of {SAMPLE_UNITS}, not {SAMPLE_UNIT!r}.")
    return TRIALS_PER_BLOCK if SAMPLE_UNIT == "block" else None


def get_unit_label():
    """Return the plural name of one plotted dot."""
    return "sessions" if SAMPLE_UNIT == "session" else f"{TRIALS_PER_BLOCK}-trial blocks"


def analyze_units(events, recordings):
    """Return unit-labeled events, transitions, and per-unit metrics."""
    events = add_unit_columns(events, recordings, get_trials_per_block(), MIN_BLOCK_TRIALS)
    metrics = []
    transitions = []
    for _, unit_events in events.groupby("unit_id", sort=True):
        metrics.append(analyze_unit(unit_events))
        transitions.append(make_transition_table(unit_events))
    return {
        "events": events,
        "transitions": pd.concat(transitions, ignore_index=True),
        "unit_metrics": pd.concat(metrics, ignore_index=True),
    }


def summarize(unit_metrics, rng):
    """Return the strain mean and 95% CI of every metric and category."""
    return compare.summarize_by_strain(
        unit_metrics, "value", ["metric", "category"], rng, BOOTSTRAP_SAMPLES, "session_id",
    )


def compare_strains(unit_metrics, reference, rng):
    """Return reference comparisons with Benjamini-Hochberg q values over all tests."""
    comparisons = compare.compare_to_reference(
        unit_metrics, "value", reference, ["metric", "category"], rng, PERMUTATION_COUNT,
        "session_id",
    )
    if not comparisons.empty:
        comparisons["q_value"] = compare.benjamini_hochberg(comparisons["p_value"])
    return comparisons


# %% Plots
def format_category(category):
    """Return a plot label for an outcome or transition category."""
    return category.replace("->", "\n→ ")


def plot_gap_metrics(unit_metrics, summary, order, colors):
    """Plot attempts, outcomes, decision intervals, and transitions by strain."""
    widths = [1, 3, 4, 3]
    fig, axes = plt.subplots(
        1, len(METRIC_CATEGORIES), figsize=(20, 5.5), gridspec_kw={"width_ratios": widths},
    )
    for axis, (metric, categories) in zip(axes, METRIC_CATEGORIES.items()):
        compare.plot_strain_points(
            axis,
            unit_metrics.loc[unit_metrics["metric"] == metric],
            summary.loc[summary["metric"] == metric],
            "value", order, colors, x_column="category", x_order=categories,
        )
        axis.set_xticklabels([format_category(category) for category in categories])
        axis.set(ylabel=METRIC_LABELS[metric])
        if metric in ("outcome_fraction", "transition_probability"):
            axis.set_ylim(-0.03, 1.03)
    axes[0].set_xticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(order), fontsize=9, frameon=False)
    fig.suptitle(
        f"Gap crossing by strain (dots: {get_unit_label()}; bars: 95% CI over sessions)"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# %% Run
def run():
    """Compare gap-crossing behavior across strain datasets."""
    settings_modules = [analysis, compare, sys.modules[__name__]]
    with run_io.analysis_run(__file__, DATASET_DIRS, settings_modules) as run_info:
        rng = np.random.default_rng(RANDOM_SEED)
        order = compare.get_strain_order(run_info.datasets, REFERENCE_LABEL)
        colors = compare.get_strain_colors(order)
        print(f"Strains: {order} (reference: {order[0]})")

        geometry = analysis.get_gap_geometry(run_info.tracks)
        print(f"Gap regions (all strains pooled): {len(geometry)}")
        events = analysis.make_event_table(run_info.tracks, geometry)

        print(f"Sampling unit: {get_unit_label()}")
        tables = analyze_units(events, run_info.recordings)
        unit_metrics = tables["unit_metrics"]
        counts = (
            unit_metrics.loc[unit_metrics["metric"] == "attempts_per_track"]
            .groupby("dataset_label")
            .agg(sessions=("session_id", "nunique"), units=("unit_id", "nunique"),
                 tracks=("track_count", "sum"), attempts=("attempt_count", "sum"))
        )
        print("Sessions, plotted units, attempting tracks, and attempts per strain:")
        print(counts.reindex(order).to_string())
        summary = summarize(unit_metrics, rng)
        comparisons = compare_strains(unit_metrics, order[0], rng)

        run_info.save_table("gap_geometry", geometry)
        run_info.save_table("events", tables["events"])
        run_info.save_table("transitions", tables["transitions"])
        run_info.save_table(f"{SAMPLE_UNIT}_gap_metrics", unit_metrics)
        if SAMPLE_UNIT == "block":
            variance = compare.variance_components(unit_metrics, "value", ["metric", "category"])
            run_info.save_table("variance_components", variance)
            print("Share of variance between sessions (1 = sessions differ, 0 = blocks differ):")
            print(variance.groupby(["metric", "category"])["between_fraction"]
                  .median().round(2).to_string())
        run_info.save_table("strain_gap_metrics", summary)
        run_info.save_table("strain_comparisons", comparisons)
        print("Strain means (95% CI over sessions):")
        print(summary.to_string(index=False))
        if not comparisons.empty:
            print("Differences from the reference strain (permutation of whole sessions; "
                  "q = Benjamini-Hochberg over all tests in this table):")
            print(comparisons[["metric", "category", "dataset_label", "difference",
                               "p_value", "q_value", "session_count", "reference_session_count"]]
                  .to_string(index=False))

        analysis.plot_gap_geometry(run_info.tracks, geometry)
        plot_gap_metrics(unit_metrics, summary, order, colors)
        run_info.show()


if __name__ == "__main__":
    run()
