"""Shared session statistics and plots for strain comparisons.

A session is one strain (dataset_label) on one day with one experimenter and
vial. A unit is either a whole session or a block of consecutive trials within
a session. Metrics are computed per unit first. Strains are then compared with
bootstrap intervals and label-permutation tests that treat sessions as the
independent samples: blocks of one session are resampled within that session
and keep their strain label together.
"""

import numpy as np
import pandas as pd


SESSION_FIELDS = [
    "recording_year",
    "recording_month",
    "recording_day",
    "recording_experimenter",
    "recording_vial",
]
SESSION_KEYS = ["dataset_label", *SESSION_FIELDS]
BOOTSTRAP_SAMPLES = 10_000
PERMUTATION_COUNT = 10_000
SUMMARY_COLUMNS = ["mean", "ci_low", "ci_high", "session_count", "unit_count"]


# %% Sessions and strains
def make_session_id(row) -> str:
    """Return a readable session name."""
    return (
        f"{row['dataset_label']} | {int(row['recording_year']):04d}-"
        f"{int(row['recording_month']):02d}-{int(row['recording_day']):02d} | "
        f"{row['recording_experimenter']} | vial {row['recording_vial']}"
    )


def make_session_lookup(recordings: pd.DataFrame) -> dict:
    """Return source_file -> session values (dataset_label and session fields)."""
    missing = set(SESSION_KEYS).difference(recordings.columns)
    if missing:
        raise ValueError(f"Recordings lack session columns: {sorted(missing)}")
    lookup = {}
    for row in recordings[["source_file", *SESSION_KEYS]].to_dict("records"):
        session = {key: row[key] for key in SESSION_KEYS}
        session["session_id"] = make_session_id(session)
        lookup[row["source_file"]] = session
    return lookup


def assign_trial_blocks(trial_count, trials_per_block, min_block_trials=1) -> np.ndarray:
    """Return the block number (from 1) of each ranked trial.

    Every trials_per_block consecutive trials form one block. A last block with
    fewer than min_block_trials trials joins the block before it, so no trial
    is dropped: 17 trials in blocks of 5 with min_block_trials 3 give 5, 5, 7.
    """
    blocks = np.arange(trial_count) // trials_per_block + 1
    remainder = trial_count % trials_per_block
    if remainder and remainder < min_block_trials and trial_count > trials_per_block:
        blocks[blocks == blocks.max()] -= 1
    return blocks


def make_unit_lookup(recordings: pd.DataFrame, trials_per_block=None, min_block_trials=1) -> dict:
    """Return source_file -> unit values (session keys, block, block_trials, unit_id).

    With trials_per_block None, each session is one unit. Otherwise trials are
    ranked within each session by trial number and grouped into blocks by
    assign_trial_blocks (block 1 holds the earliest trials).
    """
    sessions = make_session_lookup(recordings)
    if trials_per_block is not None and (int(trials_per_block) != trials_per_block or trials_per_block < 1):
        raise ValueError("trials_per_block must be a positive integer or None.")
    trial_values = (
        recordings["recording_trial"] if "recording_trial" in recordings
        else pd.Series(np.nan, index=recordings.index)
    )
    trials = pd.DataFrame({
        "source_file": recordings["source_file"],
        "trial_number": pd.to_numeric(trial_values, errors="coerce"),
    })
    if trials_per_block is not None and trials["trial_number"].isna().any():
        missing = trials.loc[trials["trial_number"].isna(), "source_file"].tolist()[:5]
        raise ValueError(f"Trial blocks need numeric recording_trial values; missing for {missing}.")
    trials["session_id"] = trials["source_file"].map(lambda name: sessions[name]["session_id"])

    lookup = {}
    for session_id, session_trials in trials.groupby("session_id", sort=False):
        ordered = sorted(session_trials["trial_number"].dropna().unique())
        if trials_per_block is None:
            block_of = {trial: 1 for trial in ordered}
        else:
            block_of = dict(zip(ordered, assign_trial_blocks(len(ordered), trials_per_block, min_block_trials)))
        block_sizes = pd.Series(block_of).value_counts().to_dict()
        for row in session_trials.itertuples(index=False):
            session = sessions[row.source_file]
            if trials_per_block is None:
                lookup[row.source_file] = {
                    **session, "block": 1, "block_trials": len(ordered), "unit_id": session_id,
                }
                continue
            block = int(block_of[row.trial_number])
            lookup[row.source_file] = {
                **session,
                "block": block,
                "block_trials": int(block_sizes[block]),
                "unit_id": f"{session_id} | trial block {block}",
            }
    return lookup


def group_tracks_by_unit(tracks, recordings, trials_per_block=None, min_block_trials=1) -> dict:
    """Return unit_id -> (unit values, tracks)."""
    lookup = make_unit_lookup(recordings, trials_per_block, min_block_trials)
    units = {}
    for track in tracks:
        unit = lookup[track["source_file"]]
        units.setdefault(unit["unit_id"], (unit, []))[1].append(track)
    return dict(sorted(units.items()))


def group_tracks_by_session(tracks: list[dict], recordings: pd.DataFrame) -> dict:
    """Return session_id -> (session values, tracks)."""
    return group_tracks_by_unit(tracks, recordings)


def get_strain_order(labels, reference=None) -> list:
    """Return strain labels with the reference first, others in given order."""
    order = list(dict.fromkeys(labels))
    if reference is not None:
        if reference not in order:
            raise ValueError(f"Reference strain {reference!r} is not in {order}.")
        order.remove(reference)
        order.insert(0, reference)
    return order


def get_strain_colors(order) -> dict:
    """Return one fixed color per strain."""
    return {label: f"C{index % 10}" for index, label in enumerate(order)}


# %% Statistics
def bootstrap_mean(values, rng, samples=BOOTSTRAP_SAMPLES) -> tuple[float, float, float]:
    """Return the mean and a bootstrap 95% interval of finite session values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return values[0], np.nan, np.nan
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return values.mean(), low, high


def bootstrap_clustered_mean(values, clusters, rng, samples=BOOTSTRAP_SAMPLES):
    """Return the unit mean and a two-level bootstrap 95% interval.

    Each draw resamples clusters (sessions) with replacement, then units
    (blocks) within each drawn cluster. With one unit per cluster this is the
    ordinary bootstrap. Fewer than two clusters give no interval.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    finite = np.isfinite(values)
    values, clusters = values[finite], clusters[finite]
    if not len(values):
        return np.nan, np.nan, np.nan
    _, codes = np.unique(clusters, return_inverse=True)
    sizes = np.bincount(codes)
    if len(sizes) < 2:
        return values.mean(), np.nan, np.nan
    table = np.full((len(sizes), sizes.max()), np.nan)
    order = np.argsort(codes, kind="stable")
    position = np.arange(len(values)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    table[codes[order], position] = values[order]

    chosen = rng.integers(0, len(sizes), size=(samples, len(sizes)))
    chosen_sizes = sizes[chosen]
    within = np.floor(
        rng.random((samples, len(sizes), sizes.max())) * chosen_sizes[..., None]
    ).astype(int)
    used = np.arange(sizes.max())[None, None, :] < chosen_sizes[..., None]
    drawn = np.where(used, table[chosen[..., None], within], 0.0)
    draws = drawn.sum(axis=(1, 2)) / used.sum(axis=(1, 2))
    low, high = np.quantile(draws, [0.025, 0.975])
    return values.mean(), low, high


def summarize_by_strain(
    session_values: pd.DataFrame,
    value_column: str,
    group_columns=(),
    rng=None,
    samples=BOOTSTRAP_SAMPLES,
    cluster_column=None,
) -> pd.DataFrame:
    """Return the strain mean, bootstrap CI, and session and unit counts per group.

    With cluster_column (e.g. "session_id"), rows are units nested in clusters
    and the interval comes from the two-level bootstrap.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    keys = [*group_columns, "dataset_label"]
    rows = []
    for key, group in session_values.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        values = group[value_column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if cluster_column is None:
            mean, low, high = bootstrap_mean(values, rng, samples)
            session_count = int(finite.sum())
        else:
            clusters = group[cluster_column].to_numpy()
            mean, low, high = bootstrap_clustered_mean(values, clusters, rng, samples)
            session_count = int(len(np.unique(clusters[finite])))
        rows.append({
            **dict(zip(keys, key)),
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
            "session_count": session_count,
            "unit_count": int(finite.sum()),
        })
    return pd.DataFrame(rows, columns=[*keys, *SUMMARY_COLUMNS])


def permutation_difference(reference_values, other_values, rng, count=PERMUTATION_COUNT):
    """Return other minus reference mean and a two-sided permutation p value."""
    reference_values = np.asarray(reference_values, dtype=float)
    other_values = np.asarray(other_values, dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]
    other_values = other_values[np.isfinite(other_values)]
    if not len(reference_values) or not len(other_values):
        return np.nan, np.nan
    observed = other_values.mean() - reference_values.mean()
    pooled = np.concatenate([reference_values, other_values])
    shuffled = np.array([rng.permutation(pooled) for _ in range(count)])
    null = shuffled[:, len(reference_values):].mean(axis=1) - shuffled[:, :len(reference_values)].mean(axis=1)
    p_value = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (count + 1)
    return observed, p_value


def permutation_difference_clustered(
    reference_values, reference_clusters, other_values, other_clusters, rng,
    count=PERMUTATION_COUNT,
):
    """Return other minus reference unit mean and a p value from shuffling whole clusters.

    Strain labels are permuted across clusters (sessions); all units of a
    cluster keep the same label. With one unit per cluster this matches
    permutation_difference.
    """
    values = np.r_[
        np.asarray(reference_values, dtype=float), np.asarray(other_values, dtype=float)
    ]
    clusters = np.array(
        [f"r:{name}" for name in reference_clusters] + [f"o:{name}" for name in other_clusters]
    )
    finite = np.isfinite(values)
    values, clusters = values[finite], clusters[finite]
    if not len(values):
        return np.nan, np.nan
    names, codes = np.unique(clusters, return_inverse=True)
    is_other = np.char.startswith(names.astype(str), "o:")
    if not is_other.any() or is_other.all():
        return np.nan, np.nan
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes).astype(float)
    observed = (
        sums[is_other].sum() / counts[is_other].sum()
        - sums[~is_other].sum() / counts[~is_other].sum()
    )
    chosen = np.argsort(rng.random((count, len(names))), axis=1)[:, :is_other.sum()]
    other_sum, other_count = sums[chosen].sum(axis=1), counts[chosen].sum(axis=1)
    null = other_sum / other_count - (sums.sum() - other_sum) / (counts.sum() - other_count)
    p_value = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (count + 1)
    return observed, p_value


def compare_to_reference(
    session_values: pd.DataFrame,
    value_column: str,
    reference: str,
    group_columns=(),
    rng=None,
    count=PERMUTATION_COUNT,
    cluster_column=None,
) -> pd.DataFrame:
    """Return each strain's difference from the reference with a p value.

    With cluster_column, whole clusters (sessions) are permuted between strains.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    groups = (
        session_values.groupby(list(group_columns), sort=False)
        if group_columns else [((), session_values)]
    )

    def count_sessions(rows):
        finite = np.isfinite(rows[value_column].to_numpy(dtype=float))
        if cluster_column is None:
            return int(finite.sum())
        return int(rows.loc[finite, cluster_column].nunique())

    rows = []
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        reference_rows = group.loc[group["dataset_label"] == reference]
        for label in group["dataset_label"].unique():
            if label == reference:
                continue
            other_rows = group.loc[group["dataset_label"] == label]
            if cluster_column is None:
                difference, p_value = permutation_difference(
                    reference_rows[value_column], other_rows[value_column], rng, count
                )
            else:
                difference, p_value = permutation_difference_clustered(
                    reference_rows[value_column], reference_rows[cluster_column],
                    other_rows[value_column], other_rows[cluster_column], rng, count,
                )
            rows.append({
                **dict(zip(group_columns, key)),
                "value_column": value_column,
                "dataset_label": label,
                "reference_label": reference,
                "difference": difference,
                "p_value": p_value,
                "session_count": count_sessions(other_rows),
                "reference_session_count": count_sessions(reference_rows),
            })
    return pd.DataFrame(rows)


def variance_components(unit_values, value_column, group_columns=(), cluster_column="session_id"):
    """Return between-session and within-session spread of unit values per strain.

    between_session_sd is the SD of session means; within_session_sd is the
    root mean within-session variance over sessions with at least two units.
    between_fraction near 1 means sessions differ more than blocks within a
    session, so more sessions help more than more trials per session.
    """
    keys = [*group_columns, "dataset_label"]
    rows = []
    for key, group in unit_values.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        group = group.loc[np.isfinite(group[value_column].to_numpy(dtype=float))]
        by_session = group.groupby(cluster_column)[value_column]
        means = by_session.mean()
        within = by_session.var(ddof=1).dropna()
        between_var = means.var(ddof=1) if len(means) > 1 else np.nan
        within_var = within.mean() if len(within) else np.nan
        total = between_var + within_var
        rows.append({
            **dict(zip(keys, key)),
            "session_count": len(means),
            "unit_count": len(group),
            "between_session_sd": np.sqrt(between_var),
            "within_session_sd": np.sqrt(within_var),
            "between_fraction": (
                between_var / total if np.isfinite(total) and total > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values) -> np.ndarray:
    """Return Benjamini-Hochberg q values; NaN p values stay NaN."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(len(p_values), np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if not len(finite):
        return q_values
    order = finite[np.argsort(p_values[finite])]
    ranked = p_values[order] * len(finite) / np.arange(1, len(finite) + 1)
    q_values[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return q_values


def session_histogram(values, bins) -> np.ndarray:
    """Return the fraction of finite values in each bin (sums to 1 when any fall in range)."""
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values[np.isfinite(values)], bins=bins)
    total = counts.sum()
    return counts / total if total else np.full(len(counts), np.nan)


# %% Plots
def plot_strain_points(axis, session_values, summary, value_column, order, colors,
                       x_column=None, x_order=None, width=0.7):
    """Plot session dots and strain means with 95% CIs.

    With x_column, strains are offset within each x category (e.g. state).
    """
    x_order = [None] if x_column is None else list(x_order)
    offsets = np.linspace(-width / 2, width / 2, len(order) + 2)[1:-1]
    for x_index, x_value in enumerate(x_order):
        for label, offset in zip(order, offsets):
            position = x_index + offset
            sessions = session_values.loc[session_values["dataset_label"] == label]
            strain = summary.loc[summary["dataset_label"] == label]
            if x_column is not None:
                sessions = sessions.loc[sessions[x_column] == x_value]
                strain = strain.loc[strain[x_column] == x_value]
            axis.scatter(
                np.full(len(sessions), position), sessions[value_column],
                color=colors[label], alpha=0.35, s=18, linewidths=0,
            )
            if strain.empty:
                continue
            row = strain.iloc[0]
            error = np.array([[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]])
            axis.errorbar(
                position, row["mean"],
                yerr=None if np.isnan(error).any() else error,
                fmt="o", color=colors[label], ms=7, capsize=3,
                label=label if x_index == 0 else None,
            )
    axis.set_xticks(range(len(x_order)))
    axis.set_xticklabels([] if x_column is None else x_order)


def plot_strain_curves(axis, summary, x_column, order, colors):
    """Plot strain mean curves with 95% CI bands.

    On a log y-axis, values at or below 0 become gaps in the line and band.
    """
    log_y = axis.get_yscale() == "log"
    for label in order:
        strain = summary.loc[summary["dataset_label"] == label].sort_values(x_column)
        if strain.empty:
            continue
        mean, low, high = (
            strain[column].to_numpy(dtype=float) for column in ("mean", "ci_low", "ci_high")
        )
        if log_y:
            mean, low, high = (np.where(values > 0, values, np.nan) for values in (mean, low, high))
        axis.plot(strain[x_column], mean, color=colors[label], label=label)
        axis.fill_between(
            strain[x_column], low, high, color=colors[label], alpha=0.2, linewidth=0,
        )
