"""Gap-crossing outcome variability across day-vial sessions.

This script uses the same database query as gap_cross_db. Each unique day and
vial is one sample. Error bars are bootstrap 95% confidence intervals across
these sessions.
"""

from itertools import product

import numpy as np
import pandas as pd

try:
    from gap_crossing import gap_cross_db as pooled
    from gap_crossing import gap_cross_track as analysis
except ModuleNotFoundError:
    import gap_cross_db as pooled
    import gap_cross_track as analysis


SESSION_FIELDS = [
    "recording_year",
    "recording_month",
    "recording_day",
    "recording_experimenter",
    "recording_vial",
]
BOOTSTRAP_SAMPLES = 10_000
MOTIF_LENGTHS = [2, 3]
MOTIFS_PER_TAIL = 6
TRIAL_WINDOW_SIZE = 3


def get_session_trial_table(events):
    """Return numeric trial order and count for each vial-date session."""
    required = {*SESSION_FIELDS, "recording_trial"}
    missing = required.difference(events.columns)
    if missing:
        raise RuntimeError(f"Events lack trial fields: {sorted(missing)}")

    trials = events[[*SESSION_FIELDS, "recording_trial"]].drop_duplicates().copy()
    trials["trial_number"] = pd.to_numeric(trials["recording_trial"], errors="coerce")
    if trials["trial_number"].isna().any():
        raise RuntimeError("Recording trials must be numeric.")
    trials = trials.sort_values([*SESSION_FIELDS, "trial_number"])
    trials["trial_rank"] = trials.groupby(SESSION_FIELDS).cumcount() + 1
    trials["trial_count"] = trials.groupby(SESSION_FIELDS)["trial_number"].transform("size")
    return trials


def make_trial_regain_cross_summary(events):
    """Return session-mean P(cross | previous regain) by trial number."""
    trials = get_session_trial_table(events)
    available = (
        trials.groupby("trial_number").size()
        .rename("available_session_count")
        .reset_index()
    )
    transition_events = events.loc[events["previous_outcome"] == "regain"].copy()
    transition_events["trial_number"] = pd.to_numeric(
        transition_events["recording_trial"], errors="coerce"
    )
    session_rates = (
        transition_events.groupby([*SESSION_FIELDS, "trial_number"], as_index=False)
        .agg(cross_probability=("is_cross", "mean"))
    )
    summary = (
        session_rates.groupby("trial_number")["cross_probability"]
        .agg(mean_cross="mean", session_count="size", std_cross="std")
        .reset_index()
    )
    summary = available.merge(summary, on="trial_number", how="left")
    summary["session_count"] = summary["session_count"].fillna(0).astype(int)
    summary["std_cross"] = summary["std_cross"].fillna(0.0)
    summary["sem_cross"] = np.divide(
        summary["std_cross"],
        np.sqrt(summary["session_count"]),
        out=np.full(len(summary), np.nan),
        where=summary["session_count"] > 0,
    )
    return summary.rename(columns={"trial_number": "recording_trial"})


def make_first_last_trial_transition_matrices(events, trial_window=TRIAL_WINDOW_SIZE):
    """Return session-mean matrices for non-overlapping trial windows."""
    trials = get_session_trial_table(events)
    trials = trials.loc[trials["trial_count"] >= 2 * trial_window].copy()
    trials["phase"] = np.where(
        trials["trial_rank"] <= trial_window,
        "first",
        np.where(trials["trial_rank"] > trials["trial_count"] - trial_window, "last", None),
    )
    trial_phases = trials.dropna(subset=["phase"])
    session_counts = {
        phase: int(
            trial_phases.loc[trial_phases["phase"] == phase, SESSION_FIELDS]
            .drop_duplicates()
            .shape[0]
        )
        for phase in ["first", "last"]
    }
    transition_events = events.loc[events["previous_outcome"].notna()].copy()
    transition_events["trial_number"] = pd.to_numeric(
        transition_events["recording_trial"], errors="coerce"
    )
    transition_events = transition_events.merge(
        trial_phases[[*SESSION_FIELDS, "trial_number", "phase"]],
        on=[*SESSION_FIELDS, "trial_number"],
        how="inner",
    )
    matrices = {
        phase: pd.DataFrame(
            np.nan, index=analysis.OUTCOME_ORDER, columns=analysis.OUTCOME_ORDER
        )
        for phase in ["first", "last"]
    }
    if transition_events.empty:
        return matrices, session_counts

    count_columns = [*SESSION_FIELDS, "phase", "previous_outcome", "outcome"]
    counts = transition_events.groupby(count_columns).size().unstack("outcome", fill_value=0)
    counts = counts.reindex(columns=analysis.OUTCOME_ORDER, fill_value=0)
    rates = counts.div(counts.sum(axis=1), axis=0)
    for phase in ["first", "last"]:
        if phase not in rates.index.get_level_values("phase"):
            continue
        phase_rates = rates.xs(phase, level="phase", drop_level=False)
        mean_rates = phase_rates.groupby("previous_outcome").mean()
        matrices[phase] = mean_rates.T.reindex(
            index=analysis.OUTCOME_ORDER, columns=analysis.OUTCOME_ORDER
        )
    return matrices, session_counts


def plot_trial_regain_cross_summary(events):
    """Plot session-mean P(cross | previous regain) by trial number."""
    summary = make_trial_regain_cross_summary(events)
    valid = summary.loc[summary["session_count"] > 0]
    if valid.empty:
        print("No regain-following transitions by trial.")
        return

    fig, axis = analysis.plt.subplots(figsize=(9, 5))
    axis.errorbar(
        valid["recording_trial"],
        valid["mean_cross"],
        yerr=valid["sem_cross"],
        fmt="o-",
        color=analysis.OUTCOME_COLOR["cross"],
        capsize=4,
    )
    for _, result in valid.iterrows():
        axis.annotate(
            f"n={result['session_count']}/{result['available_session_count']}",
            (result["recording_trial"], result["mean_cross"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.set(
        xlabel="Recorded trial number",
        ylabel="P(cross | previous regain)",
        ylim=(-0.05, 1.05),
        title="Crossing after regain across trials (session mean ± SEM)",
    )
    fig.tight_layout()
    print("Crossing after regain by recorded trial:")
    print(summary.to_string(index=False))


def plot_first_last_trial_transition_matrices(events):
    """Plot session-mean matrices for first and last trial windows."""
    matrices, session_counts = make_first_last_trial_transition_matrices(events)
    fig, axes = analysis.plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, phase in zip(axes, ["first", "last"]):
        matrix = matrices[phase]
        if matrix.notna().any().any():
            labels = matrix.map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )
            analysis.sns.heatmap(
                matrix,
                vmin=0,
                vmax=1,
                cmap="Blues",
                annot=labels,
                fmt="",
                mask=matrix.isna(),
                cbar=axis is axes[1],
                ax=axis,
            )
        else:
            axis.text(0.5, 0.5, "No transitions", ha="center", va="center")
        axis.set(
            xlabel="Previous outcome",
            ylabel="Current outcome",
            title=(
                f"{phase.title()} {TRIAL_WINDOW_SIZE} trials "
                f"(eligible sessions: n={session_counts[phase]})"
            ),
        )
    fig.suptitle("Transition matrices across trial windows")
    fig.tight_layout(rect=(0, 0, 1, 0.93))


def make_session_summary(events):
    """Return outcome fractions for each day-vial session and gap order."""
    missing = set(SESSION_FIELDS).difference(events.columns)
    if missing:
        raise RuntimeError(f"Events lack session fields: {sorted(missing)}")

    summary = (
        events.groupby([*SESSION_FIELDS, "gap_order"], as_index=False)
        .agg(
            attempts=("outcome", "size"),
            cross_fraction=("is_cross", "mean"),
            regain_fraction=("is_regain", "mean"),
            abort_fraction=("is_abort", "mean"),
        )
    )
    summary["session_id"] = (
        summary["recording_year"].astype(str)
        + "-"
        + summary["recording_month"].astype(str).str.zfill(2)
        + "-"
        + summary["recording_day"].astype(str).str.zfill(2)
        + " / "
        + summary["recording_experimenter"].astype(str)
        + " / vial "
        + summary["recording_vial"].astype(str)
    )
    return summary


def bootstrap_mean_interval(values, rng):
    """Return the mean and a bootstrap 95% interval for session values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return values[0], values[0], values[0]

    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    interval = np.quantile(samples.mean(axis=1), [0.025, 0.975])
    return values.mean(), interval[0], interval[1]


def get_motif_results(events, length, null_type, rng):
    """Return motif z scores for one event table and shuffle null."""
    sequences = [
        group.sort_values("attempt_time_s")["outcome"].to_list()
        for _, group in events.groupby("track_id")
    ]
    motifs = list(product(analysis.OUTCOME_ORDER, repeat=length))
    observed = analysis.count_motifs(sequences, length)
    observed_counts = np.array([observed[motif] for motif in motifs])
    null_counts = analysis.get_motif_null_counts(sequences, motifs, length, null_type, rng)
    results = analysis.summarize_motif_null(observed_counts, motifs, null_counts)
    null_std = null_counts.std(axis=0)
    results.loc[null_std == 0, "z_score"] = np.nan
    return results


def make_session_motif_scores(events, length, null_type, rng):
    """Return motif z scores for each independent day-vial session."""
    rows = []
    for session_values, session_events in events.groupby(SESSION_FIELDS):
        year, month, day, experimenter, vial = session_values
        results = get_motif_results(session_events, length, null_type, rng)
        results["session_id"] = (
            f"{int(year):04d}-{int(month):02d}-{int(day):02d} / {experimenter} / vial {vial}"
        )
        rows.append(results)
    return pd.concat(rows, ignore_index=True)


def select_motif_tails(results):
    """Return the highest and lowest pooled motifs without duplicates."""
    candidates = results.dropna(subset=["z_score"])
    preferred = candidates.nlargest(MOTIFS_PER_TAIL, "z_score")
    dispreferred = candidates.nsmallest(MOTIFS_PER_TAIL, "z_score")
    return pd.concat([preferred, dispreferred]).drop_duplicates("motif")


def plot_session_motif_scores(events):
    """Plot motif z-score uncertainty across independent day-vial sessions."""
    fig, axes = analysis.plt.subplots(2, 2, figsize=(16, 12))
    null_specs = [
        ("global", "Global label shuffle", "C1"),
        ("within_track", "Within-track label shuffle", "C0"),
    ]

    for row, length in enumerate(MOTIF_LENGTHS):
        for column, (null_type, title, color) in enumerate(null_specs):
            rng = np.random.default_rng(
                analysis.MOTIF_RANDOM_SEED + 10 * length + column
            )
            pooled_results = get_motif_results(events, length, null_type, rng)
            motifs = select_motif_tails(pooled_results)["motif"].to_list()
            session_scores = make_session_motif_scores(events, length, null_type, rng)
            axis = axes[row, column]
            summary_rows = []

            for motif in motifs:
                values = session_scores.loc[
                    session_scores["motif"] == motif, "z_score"
                ].dropna().to_numpy()
                if not len(values):
                    continue
                mean, low, high = bootstrap_mean_interval(values, rng)
                summary_rows.append({
                    "motif": motif,
                    "sessions": len(values),
                    "mean_z_score": mean,
                    "interval_low": low,
                    "interval_high": high,
                    "values": values,
                })

            if not summary_rows:
                axis.set_axis_off()
                print(f"No valid motif z scores: {length}-event motifs, {title}")
                continue
            summary = pd.DataFrame(summary_rows).sort_values(
                "mean_z_score", ascending=False
            ).reset_index(drop=True)
            for index, result in summary.iterrows():
                values = result["values"]
                jitter = rng.uniform(-0.12, 0.12, len(values))
                axis.plot(values, index + jitter, "o", color=color, alpha=0.4)
                axis.errorbar(
                    result["mean_z_score"], index,
                    xerr=[[
                        result["mean_z_score"] - result["interval_low"]
                    ], [
                        result["interval_high"] - result["mean_z_score"]
                    ]],
                    fmt="o", color="black", capsize=4, zorder=3,
                )

            axis.axvline(0, color="black", lw=1)
            axis.set(
                yticks=range(len(summary)),
                yticklabels=summary["motif"],
                xlabel="Session mean z score with 95% bootstrap interval",
                title=(
                    f"{length}-event motifs: {title}\n"
                    f"Top and bottom {MOTIFS_PER_TAIL} pooled z scores"
                ),
            )
            axis.invert_yaxis()
            print(f"{length}-event motifs across day-vial sessions: {title}")
            print(summary.drop(columns="values").to_string(index=False))

    fig.suptitle("Independent sample: one day-vial session", y=0.995)
    fig.tight_layout()


def run():
    """Query, load, and summarize the selected recordings by day-vial session."""
    experiments = pooled.select_experiments()
    print(f"Database records: {len(experiments)}")
    loaded_recordings, failed_experiments = pooled.load_recordings(experiments)
    print(f"Loaded recordings: {len(loaded_recordings)}")
    print(f"Failed recordings: {len(failed_experiments)}")
    if not failed_experiments.empty:
        print(failed_experiments.to_string(index=False))
    tracks = pooled.make_tracks(loaded_recordings)
    if not tracks:
        raise RuntimeError("No valid tracks. Check QUERY_FILTERS and matrix fields.")

    geometry = analysis.get_gap_geometry(tracks)
    events = analysis.make_event_table(tracks, geometry)
    events = events.merge(pooled.make_recording_metadata(loaded_recordings), on="source_file")
    summary = make_session_summary(events)
    print(f"Day-vial sessions: {summary['session_id'].nunique()}")

    analysis.plot_gap_geometry(tracks, geometry)
    plot_trial_regain_cross_summary(events)
    plot_first_last_trial_transition_matrices(events)
    plot_session_motif_scores(events)
    analysis.plt.show()


if __name__ == "__main__":
    run()
