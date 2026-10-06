"""Shared session statistics and plots for strain comparisons.

A session is one strain (dataset_label) on one day with one experimenter and
vial. Metrics are computed per session first. Strains are then compared
across sessions with bootstrap intervals and label-permutation tests.
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
SUMMARY_COLUMNS = ["mean", "ci_low", "ci_high", "session_count"]


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


def group_tracks_by_session(tracks: list[dict], recordings: pd.DataFrame) -> dict:
    """Return session_id -> (session values, tracks)."""
    lookup = make_session_lookup(recordings)
    sessions = {}
    for track in tracks:
        session = lookup[track["source_file"]]
        sessions.setdefault(session["session_id"], (session, []))[1].append(track)
    return dict(sorted(sessions.items()))


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


def summarize_by_strain(
    session_values: pd.DataFrame,
    value_column: str,
    group_columns=(),
    rng=None,
    samples=BOOTSTRAP_SAMPLES,
) -> pd.DataFrame:
    """Return the strain mean, bootstrap CI, and session count per group."""
    rng = np.random.default_rng(0) if rng is None else rng
    keys = [*group_columns, "dataset_label"]
    rows = []
    for key, group in session_values.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        values = group[value_column].to_numpy(dtype=float)
        mean, low, high = bootstrap_mean(values, rng, samples)
        rows.append({
            **dict(zip(keys, key)),
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
            "session_count": int(np.isfinite(values).sum()),
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


def compare_to_reference(
    session_values: pd.DataFrame,
    value_column: str,
    reference: str,
    group_columns=(),
    rng=None,
    count=PERMUTATION_COUNT,
) -> pd.DataFrame:
    """Return each strain's difference from the reference with a p value."""
    rng = np.random.default_rng(0) if rng is None else rng
    groups = (
        session_values.groupby(list(group_columns), sort=False)
        if group_columns else [((), session_values)]
    )
    rows = []
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        reference_values = group.loc[group["dataset_label"] == reference, value_column]
        for label in group["dataset_label"].unique():
            if label == reference:
                continue
            other_values = group.loc[group["dataset_label"] == label, value_column]
            difference, p_value = permutation_difference(
                reference_values, other_values, rng, count
            )
            rows.append({
                **dict(zip(group_columns, key)),
                "value_column": value_column,
                "dataset_label": label,
                "reference_label": reference,
                "difference": difference,
                "p_value": p_value,
                "session_count": int(np.isfinite(other_values.to_numpy(dtype=float)).sum()),
                "reference_session_count": int(
                    np.isfinite(reference_values.to_numpy(dtype=float)).sum()
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
    """Plot strain mean curves with 95% CI bands."""
    for label in order:
        strain = summary.loc[summary["dataset_label"] == label].sort_values(x_column)
        if strain.empty:
            continue
        axis.plot(strain[x_column], strain["mean"], color=colors[label], label=label)
        axis.fill_between(
            strain[x_column], strain["ci_low"], strain["ci_high"],
            color=colors[label], alpha=0.2, linewidth=0,
        )
