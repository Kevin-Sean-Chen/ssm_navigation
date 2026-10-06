"""Compare fixed, Markov, and crossing-history gap-crossing models."""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gap_crossing import gap_cross_db as pooled
from gap_crossing import gap_cross_track as analysis
from gap_crossing import run_io
from gap_crossing.memory_analysis.gap_cross_db_entropy import SESSION_FIELDS


# Label -> dataset folder written by load_db.py.
DATASET_DIRS: dict[str, Path] = {}


CURRENT_FEATURE_OUTCOMES = ["regain", "abort"]
MODEL_ORDER = ["fixed", "markov", "markov_cross_fraction"]
MODEL_LABEL = {
    "fixed": "Fixed outcome",
    "markov": "Markov",
    "markov_cross_fraction": "Markov + prior crossing fraction",
}
TRANSITION_COLUMNS = [
    *SESSION_FIELDS,
    "track_id",
    "current_outcome",
    "next_outcome",
    "prior_cross_fraction",
]


@dataclass
class FittedTransitionModel:
    """One fitted transition model and its training covariate scale."""

    model_name: str
    classifier: LogisticRegression
    covariate_mean: float | None = None
    covariate_std: float | None = None


def make_transition_table(events):
    """Return consecutive outcomes with their prior crossing fraction."""
    required = {*SESSION_FIELDS, "track_id", "attempt_time_s", "outcome"}
    missing = required.difference(events.columns)
    if missing:
        raise RuntimeError(f"Events lack history-model fields: {sorted(missing)}")

    rows = []
    ordered = events.sort_values(["track_id", "attempt_time_s"], kind="stable")
    for track_id, track_events in ordered.groupby("track_id", sort=False):
        track_events = track_events.reset_index(drop=True)
        outcomes = track_events["outcome"].to_numpy()
        times_s = track_events["attempt_time_s"].to_numpy(dtype=float)
        prior_cross_fraction = np.cumsum(outcomes == "cross") / np.arange(
            1, len(outcomes) + 1
        )
        for index in range(len(track_events) - 1):
            elapsed_s = times_s[index + 1] - times_s[index]
            if not np.isfinite(elapsed_s) or elapsed_s <= 0:
                continue
            event = track_events.iloc[index]
            rows.append(
                {
                    **{field: event[field] for field in SESSION_FIELDS},
                    "track_id": track_id,
                    "current_outcome": outcomes[index],
                    "next_outcome": outcomes[index + 1],
                    "prior_cross_fraction": prior_cross_fraction[index],
                }
            )
    return pd.DataFrame(rows, columns=TRANSITION_COLUMNS)


def make_model_features(
    transitions, model_name, covariate_mean=None, covariate_std=None
):
    """Return predictors for one fixed, Markov, or history model."""
    if model_name == "fixed":
        return pd.DataFrame({"constant": np.zeros(len(transitions))})
    current = pd.get_dummies(transitions["current_outcome"], dtype=float)
    current = current.reindex(columns=CURRENT_FEATURE_OUTCOMES, fill_value=0.0)
    current.columns = [f"current_{outcome}" for outcome in current.columns]
    if model_name == "markov":
        return current
    if model_name == "markov_cross_fraction":
        if covariate_mean is None or covariate_std is None:
            raise ValueError(
                "Cross-fraction features need a training covariate scale."
            )
        cross_fraction = (
            transitions["prior_cross_fraction"] - covariate_mean
        ) / covariate_std
        return pd.concat(
            [current, cross_fraction.rename("prior_cross_fraction")], axis=1
        )
    raise ValueError(f"Unknown outcome model: {model_name}")


def fit_transition_model(transitions, model_name):
    """Fit one fixed, Markov, or crossing-history transition model."""
    target = transitions["next_outcome"]
    if set(target.unique()) != set(analysis.OUTCOME_ORDER):
        raise RuntimeError("Training transitions must contain all three outcomes.")
    covariate_mean = covariate_std = None
    if model_name == "markov_cross_fraction":
        covariate_mean = float(transitions["prior_cross_fraction"].mean())
        covariate_std = float(
            transitions["prior_cross_fraction"].std(ddof=0)
        )
        if covariate_std == 0:
            covariate_std = 1.0
    features = make_model_features(
        transitions, model_name, covariate_mean, covariate_std
    )
    classifier = LogisticRegression(C=1.0, max_iter=5000, random_state=0)
    classifier.fit(features, target)
    return FittedTransitionModel(
        model_name, classifier, covariate_mean, covariate_std
    )


def predict_probabilities(fitted_model, transitions):
    """Return probabilities with the training covariate scale."""
    features = make_model_features(
        transitions,
        fitted_model.model_name,
        fitted_model.covariate_mean,
        fitted_model.covariate_std,
    )
    return fitted_model.classifier.predict_proba(features)


def separate_alpha_beta(fitted_model):
    """Return reference-coded baseline and history log-odds parameters."""
    if fitted_model.model_name != "markov_cross_fraction":
        raise ValueError(
            "Alpha and beta need a fitted Markov cross-fraction model."
        )
    baseline = pd.DataFrame(
        {
            "current_outcome": analysis.OUTCOME_ORDER,
            "prior_cross_fraction": fitted_model.covariate_mean,
        }
    )
    features = make_model_features(
        baseline,
        fitted_model.model_name,
        fitted_model.covariate_mean,
        fitted_model.covariate_std,
    )
    logits = pd.DataFrame(
        fitted_model.classifier.decision_function(features),
        index=analysis.OUTCOME_ORDER,
        columns=fitted_model.classifier.classes_,
    )
    alpha = logits.reindex(columns=analysis.OUTCOME_ORDER).T
    alpha = alpha.subtract(alpha.loc["cross"], axis="columns")

    history_index = features.columns.get_loc("prior_cross_fraction")
    beta = pd.Series(
        fitted_model.classifier.coef_[:, history_index],
        index=fitted_model.classifier.classes_,
    ).reindex(analysis.OUTCOME_ORDER)
    beta = beta - beta.loc["cross"]
    return alpha, beta


def fit_parameter_tables(events):
    """Return fixed, Markov, and history-model fitted parameters."""
    transitions = make_transition_table(events)
    fitted_fixed = fit_transition_model(transitions, "fixed")
    fitted_markov = fit_transition_model(transitions, "markov")
    fitted_history = fit_transition_model(transitions, "markov_cross_fraction")
    grid = pd.DataFrame(
        {
            "current_outcome": analysis.OUTCOME_ORDER,
            "prior_cross_fraction": fitted_history.covariate_mean,
        }
    )
    probabilities = predict_probabilities(fitted_markov, grid)
    fixed_probability = pd.Series(
        predict_probabilities(fitted_fixed, grid.iloc[:1])[0],
        index=fitted_fixed.classifier.classes_,
    ).reindex(analysis.OUTCOME_ORDER)
    markov_matrix = pd.DataFrame(
        probabilities,
        index=analysis.OUTCOME_ORDER,
        columns=fitted_markov.classifier.classes_,
    ).reindex(columns=analysis.OUTCOME_ORDER).T
    alpha, beta = separate_alpha_beta(fitted_history)
    return fixed_probability, markov_matrix, alpha, beta


def get_nll_bits(fitted_model, transitions):
    """Return mean negative log likelihood in bits."""
    probabilities = predict_probabilities(fitted_model, transitions)
    class_index = {
        outcome: index
        for index, outcome in enumerate(fitted_model.classifier.classes_)
    }
    target_index = transitions["next_outcome"].map(class_index).to_numpy()
    selected = probabilities[np.arange(len(transitions)), target_index]
    return float(-np.mean(np.log2(np.clip(selected, 1e-12, 1.0))))


def get_session_mask(events, session_values):
    """Return rows for one vial-day session."""
    mask = np.ones(len(events), dtype=bool)
    for field, value in zip(SESSION_FIELDS, session_values):
        mask &= events[field].eq(value)
    return mask


def evaluate_leave_one_session_out(events):
    """Return held-out fixed, Markov, and history scores for each session."""
    session_values_list = list(events.groupby(SESSION_FIELDS, sort=True).groups)
    if len(session_values_list) < 2:
        raise RuntimeError("Drift comparison needs at least two vial-day sessions.")

    rows = []
    for session_values in session_values_list:
        test_mask = get_session_mask(events, session_values)
        train_transitions = make_transition_table(events.loc[~test_mask])
        test_transitions = make_transition_table(events.loc[test_mask])
        if test_transitions.empty:
            continue
        for model_name in MODEL_ORDER:
            fitted_model = fit_transition_model(train_transitions, model_name)
            rows.append(
                {
                    **dict(zip(SESSION_FIELDS, session_values)),
                    "model": model_name,
                    "transition_count": len(test_transitions),
                    "test_nll_bits": get_nll_bits(
                        fitted_model, test_transitions
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_information_gain(scores):
    """Return held-out information gain over fixed and Markov baselines."""
    fixed = scores.loc[
        scores["model"] == "fixed",
        [*SESSION_FIELDS, "test_nll_bits"],
    ].rename(columns={"test_nll_bits": "fixed_nll_bits"})
    baseline = scores.loc[
        scores["model"] == "markov",
        [*SESSION_FIELDS, "test_nll_bits"],
    ].rename(columns={"test_nll_bits": "markov_nll_bits"})
    results = scores.merge(fixed, on=SESSION_FIELDS, how="inner")
    results = results.merge(baseline, on=SESSION_FIELDS, how="inner")
    results["gain_over_fixed_bits"] = (
        results["fixed_nll_bits"] - results["test_nll_bits"]
    )
    results["gain_over_markov_bits"] = (
        results["markov_nll_bits"] - results["test_nll_bits"]
    )
    return results


def plot_model_comparison(events=None, raw_scores=None):
    """Plot held-out gains for fixed, Markov, and history models."""
    if raw_scores is None:
        if events is None:
            raise ValueError("Model comparison needs events or raw scores.")
        raw_scores = evaluate_leave_one_session_out(events)
    scores = add_information_gain(raw_scores)
    summary = (
        scores.groupby("model", sort=False)
        .agg(
            mean_nll_bits=("test_nll_bits", "mean"),
            session_count=("test_nll_bits", "size"),
            mean_gain_over_fixed_bits=("gain_over_fixed_bits", "mean"),
            std_gain_over_fixed_bits=("gain_over_fixed_bits", "std"),
            mean_gain_over_markov_bits=("gain_over_markov_bits", "mean"),
        )
        .reindex(MODEL_ORDER)
    )
    summary["sem_gain_over_fixed_bits"] = summary[
        "std_gain_over_fixed_bits"
    ] / np.sqrt(summary["session_count"])

    figure, axis = analysis.plt.subplots(figsize=(7, 5))
    for _, session_scores in scores.groupby(SESSION_FIELDS, sort=False):
        session_scores = session_scores.set_index("model").reindex(MODEL_ORDER)
        axis.plot(
            range(len(MODEL_ORDER)),
            session_scores["gain_over_fixed_bits"],
            color="0.75",
            alpha=0.7,
        )
    axis.errorbar(
        range(len(MODEL_ORDER)),
        summary["mean_gain_over_fixed_bits"],
        yerr=summary["sem_gain_over_fixed_bits"].fillna(0.0),
        color="black",
        marker="o",
        linewidth=2,
        capsize=4,
    )
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set(
        xticks=range(len(MODEL_ORDER)),
        xticklabels=[MODEL_LABEL[name] for name in MODEL_ORDER],
        ylabel="Held-out gain over fixed (bits / transition)",
        title="Does cumulative crossing history improve prediction?",
    )
    figure.tight_layout()
    print("Held-out fixed, Markov, and crossing-history comparison:")
    print(summary.to_string())
    return figure


def plot_fitted_parameters(events):
    """Plot fitted fixed, Markov, and separated history parameters."""
    fixed_probability, markov_matrix, alpha, beta = fit_parameter_tables(events)
    figure, axes = analysis.plt.subplots(1, 4, figsize=(22, 5))

    axes[0].bar(
        fixed_probability.index,
        fixed_probability,
        color=[
            analysis.OUTCOME_COLOR[outcome]
            for outcome in fixed_probability.index
        ],
    )
    axes[0].set(
        xlabel="Outcome",
        ylabel="Probability",
        ylim=(0, 1),
        title="Fixed outcome probabilities",
    )

    analysis.sns.heatmap(
        markov_matrix,
        vmin=0,
        vmax=1,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=axes[1],
    )
    axes[1].set(
        xlabel="Current outcome",
        ylabel="Next outcome",
        title="Stationary Markov probabilities",
    )

    alpha_limit = max(float(np.abs(alpha.to_numpy()).max()), 1e-6)
    analysis.sns.heatmap(
        alpha,
        vmin=-alpha_limit,
        vmax=alpha_limit,
        center=0,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=axes[2],
    )
    axes[2].set(
        xlabel="Current outcome",
        ylabel="Next outcome",
        title="Crossing-history baseline α",
    )

    axes[3].bar(
        beta.index,
        beta,
        color=[analysis.OUTCOME_COLOR[outcome] for outcome in beta.index],
    )
    axes[3].axhline(0, color="black", linewidth=1)
    axes[3].set(
        xlabel="Next outcome",
        ylabel="Log-odds change per crossing-fraction SD",
        title="Crossing-history β",
    )
    figure.suptitle("Alpha and beta use cross as the reference outcome")
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    print("Fitted fixed outcome probabilities:")
    print(fixed_probability.to_string())
    print("Fitted stationary Markov probability matrix:")
    print(markov_matrix.to_string())
    print("Crossing-history baseline alpha log-odds relative to cross:")
    print(alpha.to_string())
    print("Crossing-history beta log-odds relative to cross:")
    print(beta.to_string())
    return figure


def plot_cross_fraction_probabilities(events):
    """Plot next-outcome probabilities across prior crossing fraction."""
    transitions = make_transition_table(events)
    fitted_model = fit_transition_model(transitions, "markov_cross_fraction")
    minimum = float(transitions["prior_cross_fraction"].min())
    maximum = float(transitions["prior_cross_fraction"].max())
    cross_fractions = np.linspace(minimum, maximum, 100)

    figure, axes = analysis.plt.subplots(
        1, len(analysis.OUTCOME_ORDER), figsize=(16, 5), sharey=True
    )
    for axis, current_outcome in zip(axes, analysis.OUTCOME_ORDER):
        grid = pd.DataFrame(
            {
                "current_outcome": current_outcome,
                "prior_cross_fraction": cross_fractions,
            }
        )
        probabilities = predict_probabilities(fitted_model, grid)
        for class_index, next_outcome in enumerate(
            fitted_model.classifier.classes_
        ):
            axis.plot(
                cross_fractions,
                probabilities[:, class_index],
                color=analysis.OUTCOME_COLOR[next_outcome],
                label=next_outcome,
            )
        axis.set(
            xlabel="Cumulative prior crossing fraction",
            ylabel="Predicted next-outcome probability",
            ylim=(0, 1),
            title=f"Current outcome: {current_outcome}",
        )
    axes[0].legend(title="Next outcome")
    figure.suptitle("Descriptive full-data crossing-history fit")
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    _, contrasts = separate_alpha_beta(fitted_model)
    print("Crossing-history log-odds contrasts relative to cross:")
    print(contrasts.to_string())
    return figure


def run():
    """Load saved datasets and compare fixed, Markov, and history models."""
    settings_modules = [analysis, sys.modules[__name__]]
    with run_io.analysis_run(__file__, DATASET_DIRS, settings_modules) as run_info:
        tracks = run_info.tracks
        print(f"Valid tracks: {len(tracks)}")
        if not tracks:
            raise RuntimeError("No valid tracks in DATASET_DIRS.")

        _, events = pooled.make_events(tracks, run_info.recordings)
        run_info.save_table("events", events)
        print(f"Day-vial sessions: {events.groupby(SESSION_FIELDS).ngroups}")
        raw_scores = evaluate_leave_one_session_out(events)
        run_info.save_table("model_scores", raw_scores)
        plot_model_comparison(raw_scores=raw_scores)
        plot_fitted_parameters(events)
        plot_cross_fraction_probabilities(events)
        run_info.show()


if __name__ == "__main__":
    run()
