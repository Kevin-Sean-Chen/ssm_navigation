"""Outcome-history entropy across independent vial-date sessions."""

from collections import Counter

import numpy as np
import pandas as pd

from gap_crossing import gap_cross_db as pooled
from gap_crossing import gap_cross_track as analysis


SESSION_FIELDS = [
    "recording_year",
    "recording_month",
    "recording_day",
    "recording_experimenter",
    "recording_vial",
]
MAX_HISTORY_LENGTH = 4


def get_conditional_entropy(sequences, history_length):
    """Return next-outcome entropy in bits for one history length."""
    context_counts = Counter()
    joint_counts = Counter()
    for sequence in sequences:
        for index in range(history_length, len(sequence)):
            context = tuple(sequence[index - history_length:index])
            outcome = sequence[index]
            context_counts[context] += 1
            joint_counts[(context, outcome)] += 1

    total_count = sum(joint_counts.values())
    if not total_count:
        return np.nan
    entropy = 0.0
    for (context, _), count in joint_counts.items():
        probability = count / total_count
        conditional_probability = count / context_counts[context]
        entropy -= probability * np.log2(conditional_probability)
    return entropy


def make_session_entropy_summary(events):
    """Return entropy estimates for sessions with five-attempt tracks."""
    required = {*SESSION_FIELDS, "track_id", "attempt_time_s", "outcome"}
    missing = required.difference(events.columns)
    if missing:
        raise RuntimeError(f"Events lack entropy fields: {sorted(missing)}")

    rows = []
    entropy_columns = [f"entropy_{order}_bits" for order in range(MAX_HISTORY_LENGTH + 1)]
    for session_values, session_events in events.groupby(SESSION_FIELDS):
        sequences = [
            track_events.sort_values("attempt_time_s")["outcome"].to_list()
            for _, track_events in session_events.groupby("track_id")
        ]
        sequences = [
            sequence for sequence in sequences if len(sequence) > MAX_HISTORY_LENGTH
        ]
        entropy_values = [
            get_conditional_entropy(sequences, history_length=order)
            for order in range(MAX_HISTORY_LENGTH + 1)
        ]
        if not np.isfinite(entropy_values).all():
            continue
        rows.append({
            **dict(zip(SESSION_FIELDS, session_values)),
            **dict(zip(entropy_columns, entropy_values)),
        })
    return pd.DataFrame(rows, columns=[*SESSION_FIELDS, *entropy_columns])


def make_entropy_decay_summary(session_entropy):
    """Return mean and SEM entropy across vial-date sessions."""
    rows = []
    for order in range(MAX_HISTORY_LENGTH + 1):
        values = session_entropy[f"entropy_{order}_bits"].to_numpy(dtype=float)
        rows.append(
            {
                "history_length": order,
                "mean_entropy_bits": values.mean() if len(values) else np.nan,
                "std_entropy_bits": values.std(ddof=1) if len(values) > 1 else 0.0,
                "sem_entropy_bits": (
                    values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                ),
                "session_count": len(values),
            }
        )
    return pd.DataFrame(rows)


def make_entropy_gain_summary(session_entropy):
    """Return mean and SEM information gain from each added outcome."""
    rows = []
    for order in range(1, MAX_HISTORY_LENGTH + 1):
        values = (
            session_entropy[f"entropy_{order - 1}_bits"]
            - session_entropy[f"entropy_{order}_bits"]
        ).to_numpy(dtype=float)
        rows.append(
            {
                "history_length": order,
                "mean_gain_bits": values.mean() if len(values) else np.nan,
                "std_gain_bits": values.std(ddof=1) if len(values) > 1 else 0.0,
                "sem_gain_bits": (
                    values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                ),
                "session_count": len(values),
            }
        )
    return pd.DataFrame(rows)


def plot_session_entropy_decay(events):
    """Plot entropy decay and information gain across vial-date sessions."""
    session_entropy = make_session_entropy_summary(events)
    if session_entropy.empty:
        print("No vial-date sessions have a five-attempt track.")
        return

    decay = make_entropy_decay_summary(session_entropy)
    gain = make_entropy_gain_summary(session_entropy)
    entropy_columns = [f"entropy_{order}_bits" for order in range(MAX_HISTORY_LENGTH + 1)]
    fig, axes = analysis.plt.subplots(1, 2, figsize=(14, 5))

    for _, result in session_entropy.iterrows():
        axes[0].plot(
            range(MAX_HISTORY_LENGTH + 1), result[entropy_columns], "o-",
            color="0.7", alpha=0.45,
        )
        gains = [
            result[f"entropy_{order - 1}_bits"] - result[f"entropy_{order}_bits"]
            for order in range(1, MAX_HISTORY_LENGTH + 1)
        ]
        axes[1].plot(range(1, MAX_HISTORY_LENGTH + 1), gains, "o-", color="0.7", alpha=0.45)

    axes[0].errorbar(
        decay["history_length"], decay["mean_entropy_bits"],
        yerr=decay["sem_entropy_bits"], fmt="o-", color="black", capsize=4,
    )
    for _, result in decay.iterrows():
        axes[0].annotate(
            f"n={int(result['session_count'])}",
            (result["history_length"], result["mean_entropy_bits"]),
            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9,
        )
    axes[0].set(
        xlabel="Number of preceding outcomes", ylabel="Conditional entropy (bits)",
        xticks=range(MAX_HISTORY_LENGTH + 1), title="Outcome uncertainty by history",
    )

    axes[1].errorbar(
        gain["history_length"], gain["mean_gain_bits"],
        yerr=gain["sem_gain_bits"], fmt="o-", color="black", capsize=4,
    )
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(
        xlabel="Added preceding outcome", ylabel="Entropy reduction (bits)",
        xticks=range(1, MAX_HISTORY_LENGTH + 1),
        title="Information from additional history",
    )
    fig.suptitle("Independent sample: one vial-date session")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    print("Entropy decay by history length:")
    print(decay.to_string(index=False))
    print("Information gain from added history:")
    print(gain.to_string(index=False))


def run():
    """Query recordings and plot vial-date outcome entropy."""
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
    print(f"Day-vial sessions: {events.groupby(SESSION_FIELDS).ngroups}")
    plot_session_entropy_decay(events)
    analysis.plt.show()


if __name__ == "__main__":
    run()
