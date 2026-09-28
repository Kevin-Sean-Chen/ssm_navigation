"""Outcome-history entropy across independent vial-date sessions."""

from collections import Counter
from dataclasses import dataclass

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
MARKOV_NULL_SIMULATIONS = 1000
MARKOV_NULL_RANDOM_SEED = 0


@dataclass
class SecondOrderMarkovNullComparison:
    """Observed second-order differences and their Markov-null control."""

    session_summary: pd.DataFrame
    residual_matrices: dict
    triplet_counts: dict


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


def get_first_order_transition_probabilities(sequences):
    """Return fitted first-order outcome probabilities from sequences."""
    outcome_index = {outcome: index for index, outcome in enumerate(analysis.OUTCOME_ORDER)}
    counts = np.zeros((len(analysis.OUTCOME_ORDER), len(analysis.OUTCOME_ORDER)))
    for sequence in sequences:
        for current, next_outcome in zip(sequence[:-1], sequence[1:]):
            counts[outcome_index[current], outcome_index[next_outcome]] += 1

    fallback = counts.sum(axis=0)
    fallback /= fallback.sum()
    probabilities = np.empty_like(counts)
    for index, row in enumerate(counts):
        row_total = row.sum()
        probabilities[index] = row / row_total if row_total else fallback
    return probabilities


def simulate_first_order_markov_sequences(sequences, rng):
    """Return tracks with matched starts and lengths from a Markov null."""
    probabilities = get_first_order_transition_probabilities(sequences)
    outcome_index = {outcome: index for index, outcome in enumerate(analysis.OUTCOME_ORDER)}
    simulated_sequences = []
    for sequence in sequences:
        simulated = [sequence[0]]
        for _ in sequence[1:]:
            current_index = outcome_index[simulated[-1]]
            simulated.append(rng.choice(analysis.OUTCOME_ORDER, p=probabilities[current_index]))
        simulated_sequences.append(simulated)
    return simulated_sequences


def get_second_order_transition_probabilities(sequences):
    """Return next-by-current matrices for each outcome two attempts back."""
    outcome_index = {outcome: index for index, outcome in enumerate(analysis.OUTCOME_ORDER)}
    counts = np.zeros((len(analysis.OUTCOME_ORDER), len(analysis.OUTCOME_ORDER), len(analysis.OUTCOME_ORDER)))
    for sequence in sequences:
        for conditioning, current, next_outcome in zip(sequence[:-2], sequence[1:-1], sequence[2:]):
            counts[
                outcome_index[conditioning],
                outcome_index[current],
                outcome_index[next_outcome],
            ] += 1

    probabilities = np.zeros_like(counts)
    for conditioning_index, current_index in np.ndindex(counts.shape[:2]):
        count = counts[conditioning_index, current_index].sum()
        if count:
            probabilities[conditioning_index, current_index] = (
                counts[conditioning_index, current_index] / count
            )
    return probabilities.transpose(0, 2, 1), counts


def get_second_order_total_variation(sequences):
    """Return weighted second-order distance from first-order transitions."""
    second_order, counts = get_second_order_transition_probabilities(sequences)
    total_count = counts.sum()
    if not total_count:
        return np.nan
    first_order = get_first_order_transition_probabilities(sequences).T
    distance = 0.0
    for conditioning_index, current_index in np.ndindex(counts.shape[:2]):
        count = counts[conditioning_index, current_index].sum()
        if count:
            distance += (count / total_count) * 0.5 * np.abs(
                second_order[conditioning_index, :, current_index]
                - first_order[:, current_index]
            ).sum()
    return distance


def get_session_sequences(session_events, minimum_length):
    """Return time-ordered outcome sequences with a required track length."""
    sequences = [
        track_events.sort_values("attempt_time_s")["outcome"].to_list()
        for _, track_events in session_events.groupby("track_id")
    ]
    return [sequence for sequence in sequences if len(sequence) >= minimum_length]


def get_second_order_matrix_frames(probabilities):
    """Return labeled second-order probability matrices."""
    return {
        conditioning_outcome: pd.DataFrame(
            probabilities[index], index=analysis.OUTCOME_ORDER, columns=analysis.OUTCOME_ORDER
        )
        for index, conditioning_outcome in enumerate(analysis.OUTCOME_ORDER)
    }


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


def make_markov_null_entropy_summary(
    events,
    simulation_count=MARKOV_NULL_SIMULATIONS,
    random_seed=MARKOV_NULL_RANDOM_SEED,
):
    """Compare observed entropy with matched first-order Markov tracks."""
    if simulation_count < 1:
        raise ValueError("Markov-null simulation count must be positive.")
    required = {*SESSION_FIELDS, "track_id", "attempt_time_s", "outcome"}
    missing = required.difference(events.columns)
    if missing:
        raise RuntimeError(f"Events lack entropy fields: {sorted(missing)}")

    rng = np.random.default_rng(random_seed)
    rows = []
    for session_values, session_events in events.groupby(SESSION_FIELDS):
        sequences = [
            track_events.sort_values("attempt_time_s")["outcome"].to_list()
            for _, track_events in session_events.groupby("track_id")
        ]
        sequences = [sequence for sequence in sequences if len(sequence) > MAX_HISTORY_LENGTH]
        if not sequences:
            continue
        observed_entropy = np.asarray(
            [
                get_conditional_entropy(sequences, history_length=order)
                for order in range(MAX_HISTORY_LENGTH + 1)
            ]
        )
        simulated_entropy = []
        for _ in range(simulation_count):
            simulated_sequences = simulate_first_order_markov_sequences(sequences, rng)
            simulated_entropy.append(
                [
                    get_conditional_entropy(simulated_sequences, history_length=order)
                    for order in range(MAX_HISTORY_LENGTH + 1)
                ]
            )
        simulated_entropy = np.asarray(simulated_entropy)
        for order in range(MAX_HISTORY_LENGTH + 1):
            observed_gain = 0.0 if order == 0 else observed_entropy[order - 1] - observed_entropy[order]
            null_gain = (
                np.zeros(simulation_count)
                if order == 0
                else simulated_entropy[:, order - 1] - simulated_entropy[:, order]
            )
            rows.append(
                {
                    **dict(zip(SESSION_FIELDS, session_values)),
                    "history_length": order,
                    "observed_entropy_bits": observed_entropy[order],
                    "null_mean_entropy_bits": simulated_entropy[:, order].mean(),
                    "null_std_entropy_bits": (
                        simulated_entropy[:, order].std(ddof=1)
                        if simulation_count > 1
                        else 0.0
                    ),
                    "observed_information_gain_bits": observed_gain,
                    "null_mean_information_gain_bits": null_gain.mean(),
                    "null_std_information_gain_bits": (
                        null_gain.std(ddof=1) if simulation_count > 1 else 0.0
                    ),
                    "excess_information_bits": observed_gain - null_gain.mean(),
                }
            )
    return pd.DataFrame(rows)


def make_markov_null_comparison_summary(null_entropy):
    """Return session mean and SEM for observed and null entropy summaries."""
    return (
        null_entropy.groupby("history_length", sort=True)
        .agg(
            session_count=("excess_information_bits", "size"),
            mean_observed_entropy_bits=("observed_entropy_bits", "mean"),
            sem_observed_entropy_bits=(
                "observed_entropy_bits",
                lambda values: values.std(ddof=1) / np.sqrt(len(values))
                if len(values) > 1
                else 0.0,
            ),
            mean_null_entropy_bits=("null_mean_entropy_bits", "mean"),
            sem_null_entropy_bits=(
                "null_mean_entropy_bits",
                lambda values: values.std(ddof=1) / np.sqrt(len(values))
                if len(values) > 1
                else 0.0,
            ),
            mean_excess_information_bits=("excess_information_bits", "mean"),
            sem_excess_information_bits=(
                "excess_information_bits",
                lambda values: values.std(ddof=1) / np.sqrt(len(values))
                if len(values) > 1
                else 0.0,
            ),
        )
        .reset_index()
    )


def plot_markov_null_comparison(
    events,
    simulation_count=MARKOV_NULL_SIMULATIONS,
    random_seed=MARKOV_NULL_RANDOM_SEED,
):
    """Plot observed entropy and excess information over a Markov null."""
    null_entropy = make_markov_null_entropy_summary(
        events, simulation_count=simulation_count, random_seed=random_seed
    )
    if null_entropy.empty:
        print("No vial-date sessions have a five-attempt track.")
        return
    summary = make_markov_null_comparison_summary(null_entropy)
    figure, axes = analysis.plt.subplots(1, 2, figsize=(14, 5))
    axes[0].errorbar(
        summary["history_length"], summary["mean_observed_entropy_bits"],
        yerr=summary["sem_observed_entropy_bits"], fmt="o-", color="black",
        capsize=4, label="Observed",
    )
    axes[0].errorbar(
        summary["history_length"], summary["mean_null_entropy_bits"],
        yerr=summary["sem_null_entropy_bits"], fmt="o--", color="0.45",
        capsize=4, label="First-order Markov null",
    )
    axes[0].set(
        xlabel="Number of preceding outcomes", ylabel="Conditional entropy (bits)",
        xticks=range(MAX_HISTORY_LENGTH + 1),
        title="Observed entropy and Markov control",
    )
    axes[0].legend()

    information = null_entropy.loc[null_entropy["history_length"] > 0]
    for _, session_values in information.groupby(SESSION_FIELDS, sort=False):
        axes[1].plot(
            session_values["history_length"], session_values["excess_information_bits"],
            "o-", color="0.7", alpha=0.45,
        )
    information_summary = summary.loc[summary["history_length"] > 0]
    axes[1].errorbar(
        information_summary["history_length"],
        information_summary["mean_excess_information_bits"],
        yerr=information_summary["sem_excess_information_bits"],
        fmt="o-", color="black", capsize=4,
    )
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(
        xlabel="Added preceding outcome", ylabel="Information beyond Markov null (bits)",
        xticks=range(1, MAX_HISTORY_LENGTH + 1),
        title="Excess history information",
    )
    figure.suptitle("Control: matched track starts and lengths")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    print("Observed entropy versus first-order Markov null:")
    print(summary.to_string(index=False))
    return figure


def make_second_order_markov_null_comparison(
    events,
    simulation_count=MARKOV_NULL_SIMULATIONS,
    random_seed=MARKOV_NULL_RANDOM_SEED,
):
    """Compare two-back matrices with matched first-order Markov tracks."""
    if simulation_count < 1:
        raise ValueError("Markov-null simulation count must be positive.")
    required = {*SESSION_FIELDS, "track_id", "attempt_time_s", "outcome"}
    missing = required.difference(events.columns)
    if missing:
        raise RuntimeError(f"Events lack entropy fields: {sorted(missing)}")

    session_sequences = []
    for session_values, session_events in events.groupby(SESSION_FIELDS):
        sequences = get_session_sequences(session_events, minimum_length=3)
        if sequences:
            session_sequences.append((session_values, sequences))
    if not session_sequences:
        return SecondOrderMarkovNullComparison(pd.DataFrame(), {}, {})

    pooled_sequences = [
        sequence for _, sequences in session_sequences for sequence in sequences
    ]
    observed_probabilities, observed_counts = get_second_order_transition_probabilities(
        pooled_sequences
    )
    observed_distances = [
        get_second_order_total_variation(sequences)
        for _, sequences in session_sequences
    ]
    null_distances = [[] for _ in session_sequences]
    null_probabilities = []
    rng = np.random.default_rng(random_seed)
    for _ in range(simulation_count):
        simulated_pooled_sequences = []
        for index, (_, sequences) in enumerate(session_sequences):
            simulated_sequences = simulate_first_order_markov_sequences(sequences, rng)
            simulated_pooled_sequences.extend(simulated_sequences)
            null_distances[index].append(
                get_second_order_total_variation(simulated_sequences)
            )
        probabilities, _ = get_second_order_transition_probabilities(
            simulated_pooled_sequences
        )
        null_probabilities.append(probabilities)
    null_mean_probabilities = np.mean(null_probabilities, axis=0)

    rows = []
    for (session_values, _), observed_distance, simulated_distances in zip(
        session_sequences, observed_distances, null_distances
    ):
        simulated_distances = np.asarray(simulated_distances)
        rows.append(
            {
                **dict(zip(SESSION_FIELDS, session_values)),
                "observed_total_variation": observed_distance,
                "null_mean_total_variation": simulated_distances.mean(),
                "null_std_total_variation": (
                    simulated_distances.std(ddof=1) if simulation_count > 1 else 0.0
                ),
                "excess_total_variation": observed_distance - simulated_distances.mean(),
            }
        )
    triplet_counts = {
        outcome: int(observed_counts[index].sum())
        for index, outcome in enumerate(analysis.OUTCOME_ORDER)
    }
    residual_matrices = get_second_order_matrix_frames(
        observed_probabilities - null_mean_probabilities
    )
    return SecondOrderMarkovNullComparison(
        pd.DataFrame(rows), residual_matrices, triplet_counts
    )


def plot_second_order_markov_null_comparison(
    events,
    simulation_count=MARKOV_NULL_SIMULATIONS,
    random_seed=MARKOV_NULL_RANDOM_SEED,
):
    """Plot observed two-back transition changes beyond a Markov null."""
    comparison = make_second_order_markov_null_comparison(
        events, simulation_count=simulation_count, random_seed=random_seed
    )
    if comparison.session_summary.empty:
        print("Too few attempts for second-order Markov control.")
        return
    maximum = max(
        0.05,
        max(
            np.abs(matrix.to_numpy()).max()
            for matrix in comparison.residual_matrices.values()
        ),
    )
    figure, axes = analysis.plt.subplots(1, len(analysis.OUTCOME_ORDER), figsize=(16, 5))
    heatmap = None
    for axis, conditioning_outcome in zip(axes, analysis.OUTCOME_ORDER):
        heatmap = analysis.sns.heatmap(
            comparison.residual_matrices[conditioning_outcome],
            vmin=-maximum,
            vmax=maximum,
            center=0,
            cmap="RdBu_r",
            annot=True,
            fmt=".2f",
            cbar=False,
            ax=axis,
        )
        axis.set(
            xlabel="Current outcome",
            ylabel="Next outcome",
            title=(
                f"Two attempts back: {conditioning_outcome}\n"
                f"Triplets: n={comparison.triplet_counts[conditioning_outcome]}"
            ),
        )
    figure.colorbar(heatmap.collections[0], ax=axes, label="Observed − Markov-null probability")
    figure.suptitle("Second-order transition structure beyond first-order Markov control")
    figure.subplots_adjust(top=0.78, wspace=0.28)
    summary = comparison.session_summary
    mean_excess = summary["excess_total_variation"].mean()
    sem_excess = (
        summary["excess_total_variation"].std(ddof=1) / np.sqrt(len(summary))
        if len(summary) > 1
        else 0.0
    )
    print(
        "Second-order total-variation excess over Markov null: "
        f"{mean_excess:.4f} ± {sem_excess:.4f} across {len(summary)} sessions"
    )
    return figure


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
    plot_markov_null_comparison(events)
    plot_second_order_markov_null_comparison(events)
    analysis.plt.show()


if __name__ == "__main__":
    run()
