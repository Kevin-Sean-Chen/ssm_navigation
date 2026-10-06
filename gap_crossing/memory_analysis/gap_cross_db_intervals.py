"""Test serial correlation between gap-crossing decision intervals."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gap_crossing import gap_cross_db as pooled
from gap_crossing import gap_cross_track as analysis


INTERVAL_COLUMNS = ["track_id", "interval_number", "interval_s"]
PAIR_COLUMNS = ["track_id", "previous_interval_s", "next_interval_s"]
TRACK_CORRELATION_COLUMNS = [
    "track_id",
    "pair_count",
    "observed_rho",
    "null_median_rho",
    "excess_rho",
]
PERMUTATION_COUNT = 1000
RANDOM_SEED = 0
MIN_TRACK_PAIR_COUNT = 10


@dataclass(frozen=True)
class IntervalCorrelation:
    """Observed lag-1 correlation and its within-track shuffled null."""

    observed_rho: float
    null_rho: np.ndarray
    null_median_rho: float
    excess_rho: float
    p_value: float


def make_interval_table(events: pd.DataFrame) -> pd.DataFrame:
    """Return positive consecutive attempt intervals within each track."""
    required = {"track_id", "attempt_time_s"}
    missing = required.difference(events.columns)
    if missing:
        raise RuntimeError(f"Events lack interval fields: {sorted(missing)}")

    rows = []
    ordered = events.sort_values(["track_id", "attempt_time_s"], kind="stable")
    for track_id, track_events in ordered.groupby("track_id", sort=False):
        times_s = track_events["attempt_time_s"].to_numpy(dtype=float)
        if not np.isfinite(times_s).all():
            continue
        intervals_s = np.diff(times_s)
        for interval_number, interval_s in enumerate(intervals_s, start=1):
            if np.isfinite(interval_s) and interval_s > 0:
                rows.append(
                    {
                        "track_id": track_id,
                        "interval_number": interval_number,
                        "interval_s": interval_s,
                    }
                )
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS)


def make_adjacent_pair_table(intervals: pd.DataFrame) -> pd.DataFrame:
    """Return adjacent interval pairs within each track."""
    required = set(INTERVAL_COLUMNS)
    missing = required.difference(intervals.columns)
    if missing:
        raise RuntimeError(f"Intervals lack pair fields: {sorted(missing)}")

    rows = []
    ordered = intervals.sort_values(["track_id", "interval_number"], kind="stable")
    for track_id, track_intervals in ordered.groupby("track_id", sort=False):
        numbers = track_intervals["interval_number"].to_numpy(dtype=int)
        values = track_intervals["interval_s"].to_numpy(dtype=float)
        for index in range(len(track_intervals) - 1):
            if numbers[index + 1] == numbers[index] + 1:
                rows.append(
                    {
                        "track_id": track_id,
                        "previous_interval_s": values[index],
                        "next_interval_s": values[index + 1],
                    }
                )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def shuffle_intervals_within_tracks(
    intervals: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Return interval values shuffled within each track."""
    required = set(INTERVAL_COLUMNS)
    missing = required.difference(intervals.columns)
    if missing:
        raise RuntimeError(f"Intervals lack shuffle fields: {sorted(missing)}")

    shuffled = intervals.copy()
    for _, track_intervals in intervals.groupby("track_id", sort=False):
        shuffled.loc[track_intervals.index, "interval_s"] = rng.permutation(
            track_intervals["interval_s"].to_numpy(dtype=float)
        )
    return shuffled


def _get_lag_correlation(intervals: pd.DataFrame) -> float:
    """Return pooled lag-1 Spearman correlation for interval rows."""
    pairs = make_adjacent_pair_table(intervals)
    if len(pairs) < 3:
        raise RuntimeError("Correlation needs at least three adjacent interval pairs.")
    if pairs["previous_interval_s"].nunique() < 2:
        raise RuntimeError("Previous intervals are constant.")
    if pairs["next_interval_s"].nunique() < 2:
        raise RuntimeError("Next intervals are constant.")
    rho = float(
        spearmanr(
            pairs["previous_interval_s"], pairs["next_interval_s"]
        ).statistic
    )
    if not np.isfinite(rho):
        raise RuntimeError("Spearman correlation is not finite.")
    return rho


def analyze_interval_correlation(
    intervals: pd.DataFrame,
    permutation_count: int = 1000,
    random_seed: int = 0,
) -> IntervalCorrelation:
    """Compare observed lag-1 correlation with within-track shuffles."""
    if permutation_count < 1:
        raise ValueError("Permutation count must be positive.")

    observed_rho = _get_lag_correlation(intervals)
    rng = np.random.default_rng(random_seed)
    null_values = []
    for _ in range(10 * permutation_count):
        shuffled = shuffle_intervals_within_tracks(intervals, rng)
        try:
            null_values.append(_get_lag_correlation(shuffled))
        except RuntimeError:
            continue
        if len(null_values) == permutation_count:
            break
    if len(null_values) < permutation_count:
        raise RuntimeError("Could not obtain the requested finite shuffled correlations.")

    null_rho = np.asarray(null_values, dtype=float)
    null_median_rho = float(np.median(null_rho))
    excess_rho = observed_rho - null_median_rho
    extreme_count = np.count_nonzero(
        np.abs(null_rho - null_median_rho) >= abs(excess_rho)
    )
    p_value = float((1 + extreme_count) / (permutation_count + 1))
    return IntervalCorrelation(
        observed_rho=observed_rho,
        null_rho=null_rho,
        null_median_rho=null_median_rho,
        excess_rho=excess_rho,
        p_value=p_value,
    )


def analyze_track_correlations(
    intervals: pd.DataFrame,
    min_pair_count: int = 10,
    permutation_count: int = 1000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Return lag-1 correlation estimates for eligible tracks."""
    if min_pair_count < 1:
        raise ValueError("Minimum pair count must be positive.")
    if permutation_count < 1:
        raise ValueError("Permutation count must be positive.")

    required = set(INTERVAL_COLUMNS)
    missing = required.difference(intervals.columns)
    if missing:
        raise RuntimeError(f"Intervals lack track fields: {sorted(missing)}")

    rows = []
    rng = np.random.default_rng(random_seed)
    ordered = intervals.sort_values(["track_id", "interval_number"], kind="stable")
    for track_id, track_intervals in ordered.groupby("track_id", sort=False):
        numbers = track_intervals["interval_number"].to_numpy(dtype=int)
        values = track_intervals["interval_s"].to_numpy(dtype=float)
        adjacent = np.diff(numbers) == 1
        pair_count = int(np.count_nonzero(adjacent))
        if pair_count < min_pair_count:
            continue

        previous = values[:-1][adjacent]
        following = values[1:][adjacent]
        if np.unique(previous).size < 2 or np.unique(following).size < 2:
            continue
        observed_rho = float(spearmanr(previous, following).statistic)
        if not np.isfinite(observed_rho):
            continue

        null_values = []
        for _ in range(10 * permutation_count):
            shuffled = rng.permutation(values)
            rho = float(
                spearmanr(
                    shuffled[:-1][adjacent], shuffled[1:][adjacent]
                ).statistic
            )
            if np.isfinite(rho):
                null_values.append(rho)
            if len(null_values) == permutation_count:
                break
        if len(null_values) < permutation_count:
            continue

        null_median_rho = float(np.median(null_values))
        rows.append(
            {
                "track_id": track_id,
                "pair_count": pair_count,
                "observed_rho": observed_rho,
                "null_median_rho": null_median_rho,
                "excess_rho": observed_rho - null_median_rho,
            }
        )
    return pd.DataFrame(rows, columns=TRACK_CORRELATION_COLUMNS)


def plot_interval_correlation(
    intervals: pd.DataFrame,
    result: IntervalCorrelation,
    track_results: pd.DataFrame,
):
    """Plot pooled and individual-track interval correlations."""
    pairs = make_adjacent_pair_table(intervals)
    figure, axes = analysis.plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(
        pairs["previous_interval_s"],
        pairs["next_interval_s"],
        color="C0",
        alpha=0.25,
    )
    lower = min(pairs["previous_interval_s"].min(), pairs["next_interval_s"].min())
    upper = max(pairs["previous_interval_s"].max(), pairs["next_interval_s"].max())
    axes[0].plot([lower, upper], [lower, upper], color="black", lw=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Previous decision interval (s)",
        ylabel="Next decision interval (s)",
        title=f"Adjacent intervals: n={len(pairs)}, Spearman rho={result.observed_rho:.3f}",
    )

    axes[1].hist(result.null_rho, bins=30, color="0.75", edgecolor="white")
    axes[1].axvline(
        result.null_median_rho,
        color="black",
        linestyle="--",
        label=f"Shuffled median: {result.null_median_rho:.3f}",
    )
    axes[1].axvline(
        result.observed_rho,
        color="C3",
        label=f"Observed: {result.observed_rho:.3f}",
    )
    axes[1].set(
        xlabel="Spearman rho",
        ylabel="Permutation count",
        title=f"Excess rho={result.excess_rho:.3f}, p={result.p_value:.4f}",
    )
    axes[1].legend()

    axes[2].axhline(0, color="black", lw=1)
    if track_results.empty:
        axes[2].text(
            0.5,
            0.5,
            "No eligible tracks",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        track_title = "Individual tracks: n=0"
    else:
        axes[2].scatter(
            track_results["pair_count"],
            track_results["excess_rho"],
            color="C0",
            alpha=0.7,
        )
        median_excess = track_results["excess_rho"].median()
        positive_fraction = track_results["excess_rho"].gt(0).mean()
        track_title = (
            f"Individual tracks: n={len(track_results)}, "
            f"median excess={median_excess:.3f}, positive={positive_fraction:.1%}"
        )
    axes[2].set(
        xlabel="Adjacent interval pairs per track",
        ylabel="Excess Spearman rho",
        title=track_title,
    )
    figure.suptitle("Decision-interval serial correlation")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def run() -> None:
    """Query recordings and plot decision-interval serial correlation."""
    experiments = pooled.select_experiments()
    print(f"Database records: {len(experiments)}")
    loaded_recordings, failed_experiments = pooled.load_recordings(experiments)
    print(f"Loaded recordings: {len(loaded_recordings)}")
    print(f"Failed recordings: {len(failed_experiments)}")
    if not failed_experiments.empty:
        print(failed_experiments.to_string(index=False))

    tracks = pooled.make_tracks(loaded_recordings)
    print(f"Valid tracks: {len(tracks)}")
    if not tracks:
        raise RuntimeError("No valid tracks. Check QUERY_FILTERS and matrix fields.")

    geometry = analysis.get_gap_geometry(tracks)
    events = analysis.make_event_table(tracks, geometry)
    intervals = make_interval_table(events)
    pairs = make_adjacent_pair_table(intervals)
    print(f"Decision intervals: {len(intervals)}")
    print(f"Adjacent interval pairs: {len(pairs)}")

    result = analyze_interval_correlation(
        intervals,
        permutation_count=PERMUTATION_COUNT,
        random_seed=RANDOM_SEED,
    )
    track_results = analyze_track_correlations(
        intervals,
        min_pair_count=MIN_TRACK_PAIR_COUNT,
        permutation_count=PERMUTATION_COUNT,
        random_seed=RANDOM_SEED,
    )
    print(f"Observed Spearman rho: {result.observed_rho:.6f}")
    print(f"Shuffled median rho: {result.null_median_rho:.6f}")
    print(f"Excess rho: {result.excess_rho:.6f}")
    print(f"Finite permutations: {len(result.null_rho)}")
    print(f"Permutation p value: {result.p_value:.6f}")
    print(
        f"Tracks with valid estimates (at least {MIN_TRACK_PAIR_COUNT} pairs): "
        f"{len(track_results)}"
    )
    if not track_results.empty:
        print(f"Median track excess rho: {track_results['excess_rho'].median():.6f}")
        print(
            "Tracks with positive excess rho: "
            f"{track_results['excess_rho'].gt(0).mean():.1%}"
        )
    plot_interval_correlation(intervals, result, track_results)
    analysis.plt.show()


if __name__ == "__main__":
    run()
