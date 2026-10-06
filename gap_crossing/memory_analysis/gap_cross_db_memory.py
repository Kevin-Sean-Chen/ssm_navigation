"""Compare fixed, Markov, and attempt-memory gap-crossing outcome models."""

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


DEFAULT_MEMORY_RETENTION = 0.5
MEMORY_RETENTION_GRID = tuple(np.round(np.arange(0.0, 1.0, 0.1), 2))
ASYMMETRIC_RETENTION_GRID = (0.0, 0.3, 0.6, 0.9)
MEMORY_FEATURE_OUTCOMES = analysis.OUTCOME_ORDER
MODEL_ORDER = ["fixed", "markov", "markov_memory"]
DESCRIPTION_MODEL_ORDER = [
    *MODEL_ORDER,
    "markov_asymmetric_memory",
    "markov_second_order",
]
MODEL_LABEL = {
    "fixed": "Fixed outcome",
    "markov": "Markov",
    "markov_memory": "Markov + fitted outcome memory",
    "markov_asymmetric_memory": "Markov + asymmetric outcome memory",
    "markov_second_order": "Second-order Markov",
}
CURRENT_FEATURE_OUTCOMES = ["regain", "abort"]
TRANSITION_COLUMNS = [
    *SESSION_FIELDS,
    "track_id",
    "previous_outcome",
    "current_outcome",
    "next_outcome",
    "elapsed_s",
    "log_elapsed_s",
    "memory_cross",
    "memory_regain",
    "memory_abort",
]


@dataclass
class FittedTransitionModel:
    """One fitted next-outcome model and its training feature scale."""

    model_name: str
    classifier: LogisticRegression
    time_mean: float | None = None
    time_std: float | None = None


@dataclass
class MemoryRetentionSelection:
    """One inner-CV retention selection and its held-out scores."""

    retention: float
    scores: pd.DataFrame


@dataclass
class AsymmetricRetentionSelection:
    """Coordinate-wise inner-CV selection of one decay per outcome."""

    retentions: dict[str, float]
    scores: pd.DataFrame


def make_transition_table(events, memory_retention=DEFAULT_MEMORY_RETENTION):
    """Return valid transitions with an attempt-indexed outcome trace."""
    if isinstance(memory_retention, dict):
        retention = np.array(
            [memory_retention.get(outcome, np.nan) for outcome in MEMORY_FEATURE_OUTCOMES],
            dtype=float,
        )
    else:
        retention = np.full(len(MEMORY_FEATURE_OUTCOMES), memory_retention, dtype=float)
    if not np.isfinite(retention).all() or ((retention < 0) | (retention >= 1)).any():
        raise ValueError("Memory retention must be in [0, 1).")
    required = {*SESSION_FIELDS, "track_id", "attempt_time_s", "outcome"}
    missing = required.difference(events.columns)
    if missing:
        raise RuntimeError(f"Events lack time-model fields: {sorted(missing)}")

    rows = []
    ordered = events.sort_values(["track_id", "attempt_time_s"], kind="stable")
    for track_id, track_events in ordered.groupby("track_id", sort=False):
        track_events = track_events.reset_index(drop=True)
        outcomes = track_events["outcome"].to_numpy()
        times_s = track_events["attempt_time_s"].to_numpy(dtype=float)
        memory = np.zeros(len(MEMORY_FEATURE_OUTCOMES), dtype=float)
        for index in range(len(track_events) - 1):
            elapsed_s = times_s[index + 1] - times_s[index]
            if np.isfinite(elapsed_s) and elapsed_s > 0:
                event = track_events.iloc[index]
                rows.append(
                    {
                        **{field: event[field] for field in SESSION_FIELDS},
                        "track_id": track_id,
                        "previous_outcome": outcomes[index - 1] if index else None,
                        "current_outcome": outcomes[index],
                        "next_outcome": outcomes[index + 1],
                        "elapsed_s": elapsed_s,
                        "log_elapsed_s": np.log1p(elapsed_s),
                        **{
                            f"memory_{outcome}": memory[outcome_index]
                            for outcome_index, outcome in enumerate(MEMORY_FEATURE_OUTCOMES)
                        },
                    }
                )
            current_memory = np.zeros(len(MEMORY_FEATURE_OUTCOMES), dtype=float)
            current_memory[MEMORY_FEATURE_OUTCOMES.index(outcomes[index])] = 1.0
            memory = retention * memory + (1 - retention) * current_memory
    return pd.DataFrame(rows, columns=TRANSITION_COLUMNS)


def get_time_scale(transitions):
    """Return a stable training-only scale for log elapsed time."""
    time_mean = float(transitions["log_elapsed_s"].mean())
    time_std = float(transitions["log_elapsed_s"].std(ddof=0))
    return time_mean, time_std if time_std > 0 else 1.0


def make_model_features(transitions, model_name, time_mean=None, time_std=None):
    """Return predictors for one nested next-outcome model."""
    if model_name == "fixed":
        return pd.DataFrame({"constant": np.zeros(len(transitions))})

    current = pd.get_dummies(transitions["current_outcome"], dtype=float)
    current = current.reindex(columns=CURRENT_FEATURE_OUTCOMES, fill_value=0.0)
    current.columns = [f"current_{outcome}" for outcome in current.columns]
    if model_name == "markov":
        return current
    if model_name in {"markov_memory", "markov_asymmetric_memory"}:
        memory = transitions[
            [f"memory_{outcome}" for outcome in MEMORY_FEATURE_OUTCOMES]
        ].copy()
        return pd.concat([current, memory], axis=1)
    if model_name == "markov_second_order":
        previous = pd.get_dummies(transitions["previous_outcome"], dtype=float)
        previous = previous.reindex(columns=CURRENT_FEATURE_OUTCOMES, fill_value=0.0)
        previous.columns = [f"previous_{outcome}" for outcome in previous.columns]
        interactions = pd.DataFrame(index=transitions.index)
        for current_column in current.columns:
            for previous_column in previous.columns:
                interactions[f"{current_column}__{previous_column}"] = (
                    current[current_column] * previous[previous_column]
                )
        return pd.concat([current, previous, interactions], axis=1)
    if model_name == "markov_time":
        if time_mean is None or time_std is None:
            raise ValueError("Markov-time features need a training time scale.")
        time_feature = (transitions["log_elapsed_s"] - time_mean) / time_std
        return pd.concat([current, time_feature.rename("log_elapsed_s")], axis=1)
    raise ValueError(f"Unknown outcome model: {model_name}")


def fit_transition_model(transitions, model_name):
    """Fit one multinomial next-outcome model from training transitions."""
    target = transitions["next_outcome"]
    if set(target.unique()) != set(analysis.OUTCOME_ORDER):
        raise RuntimeError("Training transitions must contain all three outcomes.")
    time_mean = time_std = None
    if model_name == "markov_time":
        time_mean, time_std = get_time_scale(transitions)
    features = make_model_features(transitions, model_name, time_mean, time_std)
    classifier = LogisticRegression(C=1.0, max_iter=5000, random_state=0)
    classifier.fit(features, target)
    return FittedTransitionModel(model_name, classifier, time_mean, time_std)


def predict_probabilities(fitted_model, transitions):
    """Return next-outcome probabilities using training-only feature scaling."""
    features = make_model_features(
        transitions,
        fitted_model.model_name,
        fitted_model.time_mean,
        fitted_model.time_std,
    )
    return fitted_model.classifier.predict_proba(features)


def get_nll_bits(fitted_model, transitions):
    """Return mean negative log likelihood in bits for one transition table."""
    probabilities = predict_probabilities(fitted_model, transitions)
    class_index = {
        outcome: index for index, outcome in enumerate(fitted_model.classifier.classes_)
    }
    target_index = transitions["next_outcome"].map(class_index).to_numpy()
    selected = probabilities[np.arange(len(transitions)), target_index]
    return float(-np.mean(np.log2(np.clip(selected, 1e-12, 1.0))))


def get_session_mask(events, session_values):
    """Return rows that belong to one vial-date session."""
    mask = np.ones(len(events), dtype=bool)
    for field, value in zip(SESSION_FIELDS, session_values):
        mask &= events[field].eq(value)
    return mask


def select_memory_retention(events, retentions=MEMORY_RETENTION_GRID):
    """Select memory retention by inner held-out vial-date score."""
    retentions = tuple(float(retention) for retention in retentions)
    if not retentions:
        raise ValueError("Memory retention selection needs at least one value.")
    if any(retention < 0 or retention >= 1 for retention in retentions):
        raise ValueError("Memory retentions must be in [0, 1).")

    session_values_list = list(events.groupby(SESSION_FIELDS, sort=True).groups)
    if len(session_values_list) < 2:
        raise RuntimeError("Memory retention fitting needs at least two vial-date sessions.")

    rows = []
    for retention in retentions:
        for session_values in session_values_list:
            test_mask = get_session_mask(events, session_values)
            train_transitions = make_transition_table(
                events.loc[~test_mask], memory_retention=retention
            )
            test_transitions = make_transition_table(
                events.loc[test_mask], memory_retention=retention
            )
            if test_transitions.empty:
                continue
            fitted_model = fit_transition_model(train_transitions, "markov_memory")
            rows.append(
                {
                    **dict(zip(SESSION_FIELDS, session_values)),
                    "memory_retention": retention,
                    "test_nll_bits": get_nll_bits(fitted_model, test_transitions),
                }
            )
    scores = pd.DataFrame(rows)
    if scores.empty:
        raise RuntimeError("No valid inner held-out transitions for memory fitting.")
    mean_scores = scores.groupby("memory_retention", sort=True)["test_nll_bits"].mean()
    retention = float(mean_scores.index[np.argmin(mean_scores.to_numpy())])
    return MemoryRetentionSelection(retention=retention, scores=scores)


def select_asymmetric_memory_retentions(events):
    """Select cross, regain, and abort decays by coordinate-wise inner CV."""
    retentions = {outcome: DEFAULT_MEMORY_RETENTION for outcome in MEMORY_FEATURE_OUTCOMES}
    all_scores = []
    session_values_list = list(events.groupby(SESSION_FIELDS, sort=True).groups)
    if len(session_values_list) < 2:
        raise RuntimeError("Memory retention fitting needs at least two vial-date sessions.")
    for outcome in MEMORY_FEATURE_OUTCOMES:
        candidate_scores = []
        for retention in ASYMMETRIC_RETENTION_GRID:
            candidate_retentions = {**retentions, outcome: retention}
            for session_values in session_values_list:
                test_mask = get_session_mask(events, session_values)
                train_transitions = make_transition_table(
                    events.loc[~test_mask], memory_retention=candidate_retentions
                )
                test_transitions = make_transition_table(
                    events.loc[test_mask], memory_retention=candidate_retentions
                )
                if test_transitions.empty:
                    continue
                fitted_model = fit_transition_model(train_transitions, "markov_memory")
                candidate_scores.append(
                    {
                        **dict(zip(SESSION_FIELDS, session_values)),
                        "outcome": outcome,
                        "candidate_retention": retention,
                        "test_nll_bits": get_nll_bits(fitted_model, test_transitions),
                    }
                )
        outcome_scores = pd.DataFrame(candidate_scores)
        mean_scores = outcome_scores.groupby("candidate_retention")["test_nll_bits"].mean()
        retentions[outcome] = float(mean_scores.index[np.argmin(mean_scores.to_numpy())])
        all_scores.append(outcome_scores)
    return AsymmetricRetentionSelection(
        retentions=retentions,
        scores=pd.concat(all_scores, ignore_index=True),
    )


def evaluate_leave_one_session_out(events):
    """Return held-out next-outcome scores for each vial-date session."""
    session_values_list = list(events.groupby(SESSION_FIELDS, sort=True).groups)
    if len(session_values_list) < 2:
        raise RuntimeError("Time-model comparison needs at least two vial-date sessions.")

    rows = []
    for session_values in session_values_list:
        test_mask = get_session_mask(events, session_values)
        train_events = events.loc[~test_mask]
        test_events = events.loc[test_mask]
        train_transitions = make_transition_table(train_events)
        test_transitions = make_transition_table(test_events)
        if test_transitions.empty:
            continue
        selected_retention = None
        for model_name in DESCRIPTION_MODEL_ORDER:
            memory_retention = np.nan
            if model_name == "markov_memory":
                selection = select_memory_retention(train_events)
                selected_retention = selection.retention
                memory_retention = selected_retention
                train_transitions = make_transition_table(
                    train_events, memory_retention=memory_retention
                )
                test_transitions = make_transition_table(
                    test_events, memory_retention=memory_retention
                )
            if model_name == "markov_asymmetric_memory":
                selection = select_asymmetric_memory_retentions(train_events)
                memory_retention = selection.retentions
                train_transitions = make_transition_table(
                    train_events, memory_retention=memory_retention
                )
                test_transitions = make_transition_table(
                    test_events, memory_retention=memory_retention
                )
            if model_name == "markov_second_order":
                train_model_transitions = train_transitions.dropna(
                    subset=["previous_outcome"]
                )
                test_model_transitions = test_transitions.dropna(
                    subset=["previous_outcome"]
                )
            else:
                train_model_transitions = train_transitions
                test_model_transitions = test_transitions
            fitted_model = fit_transition_model(train_model_transitions, model_name)
            score_sets = [("two_back", test_model_transitions)]
            if model_name in MODEL_ORDER:
                score_sets.insert(0, ("all", test_transitions))
            for sample, score_transitions in score_sets:
                rows.append(
                    {
                        **dict(zip(SESSION_FIELDS, session_values)),
                        "model": model_name,
                        "sample": sample,
                        "transition_count": len(score_transitions),
                        "test_nll_bits": get_nll_bits(fitted_model, score_transitions),
                        "memory_retention": memory_retention,
                    }
                )
    return pd.DataFrame(rows)


def add_information_gain(scores):
    """Return gains over fixed and Markov held-out baselines."""
    scores = scores.copy()
    if "sample" not in scores:
        scores["sample"] = "all"
    merge_fields = [*SESSION_FIELDS, "sample"]
    fixed = scores.loc[
        scores["model"] == "fixed", [*merge_fields, "test_nll_bits"]
    ].rename(columns={"test_nll_bits": "fixed_nll_bits"})
    markov = scores.loc[
        scores["model"] == "markov", [*merge_fields, "test_nll_bits"]
    ].rename(columns={"test_nll_bits": "markov_nll_bits"})
    results = scores.merge(fixed, on=merge_fields, how="inner")
    results = results.merge(markov, on=merge_fields, how="inner")
    results["gain_over_fixed_bits"] = (
        results["fixed_nll_bits"] - results["test_nll_bits"]
    )
    results["gain_over_markov_bits"] = (
        results["markov_nll_bits"] - results["test_nll_bits"]
    )
    return results


def make_model_comparison_summary(scores, model_order=MODEL_ORDER):
    """Return session-mean held-out score and information-gain summaries."""
    scores = add_information_gain(scores)
    summary = (
        scores.groupby("model", sort=False)
        .agg(
            session_count=("test_nll_bits", "size"),
            mean_nll_bits=("test_nll_bits", "mean"),
            std_nll_bits=("test_nll_bits", "std"),
            mean_gain_over_fixed_bits=("gain_over_fixed_bits", "mean"),
            std_gain_over_fixed_bits=("gain_over_fixed_bits", "std"),
            mean_gain_over_markov_bits=("gain_over_markov_bits", "mean"),
            std_gain_over_markov_bits=("gain_over_markov_bits", "std"),
        )
        .reindex(model_order)
        .reset_index()
    )
    for prefix in ["nll", "gain_over_fixed", "gain_over_markov"]:
        summary[f"sem_{prefix}_bits"] = np.divide(
            summary[f"std_{prefix}_bits"],
            np.sqrt(summary["session_count"]),
            out=np.zeros(len(summary), dtype=float),
            where=summary["session_count"] > 1,
        )
    return summary


def plot_memory_model_comparison(events=None, raw_scores=None):
    """Plot held-out gains for fixed, Markov, and outcome-memory models."""
    if raw_scores is None:
        raw_scores = evaluate_leave_one_session_out(events)
    if raw_scores.empty:
        raise RuntimeError("No valid held-out time-model transitions.")
    raw_scores = raw_scores.loc[raw_scores["sample"] == "all"]
    summary = make_model_comparison_summary(raw_scores)
    scores = add_information_gain(raw_scores)
    figure, axis = analysis.plt.subplots(figsize=(8, 5))
    for _, session_scores in scores.groupby(SESSION_FIELDS, sort=False):
        session_scores = session_scores.set_index("model").reindex(MODEL_ORDER)
        axis.plot(
            range(len(MODEL_ORDER)), session_scores["gain_over_fixed_bits"],
            "o-", color="0.75", alpha=0.45,
        )
    axis.errorbar(
        range(len(MODEL_ORDER)), summary["mean_gain_over_fixed_bits"],
        yerr=summary["sem_gain_over_fixed_bits"],
        fmt="o-", color="black", capsize=4,
    )
    axis.axhline(0, color="black", lw=1)
    axis.set(
        xticks=range(len(MODEL_ORDER)),
        xticklabels=[MODEL_LABEL[name] for name in MODEL_ORDER],
        ylabel="Held-out information gain (bits / transition)",
        title="Does fitted outcome memory improve prediction?",
    )
    figure.tight_layout()
    selected_retentions = raw_scores.loc[
        raw_scores["model"] == "markov_memory", "memory_retention"
    ]
    print("Nested-CV selected retention per outer held-out session:")
    print(selected_retentions.to_string(index=False))
    print("Held-out fixed, Markov, and fitted outcome-memory comparison:")
    print(summary.to_string(index=False))
    return figure


def plot_memory_description_comparison(events=None, raw_scores=None):
    """Compare short two-back history with the fitted outcome-memory trace."""
    if raw_scores is None:
        raw_scores = evaluate_leave_one_session_out(events)
    raw_scores = raw_scores.loc[raw_scores["sample"] == "two_back"]
    summary = make_model_comparison_summary(raw_scores, DESCRIPTION_MODEL_ORDER)
    scores = add_information_gain(raw_scores)
    figure, axis = analysis.plt.subplots(figsize=(10, 5))
    for _, session_scores in scores.groupby(SESSION_FIELDS, sort=False):
        session_scores = session_scores.set_index("model").reindex(DESCRIPTION_MODEL_ORDER)
        axis.plot(
            range(len(DESCRIPTION_MODEL_ORDER)),
            session_scores["gain_over_fixed_bits"],
            "o-", color="0.75", alpha=0.45,
        )
    axis.errorbar(
        range(len(DESCRIPTION_MODEL_ORDER)),
        summary["mean_gain_over_fixed_bits"],
        yerr=summary["sem_gain_over_fixed_bits"],
        fmt="o-", color="black", capsize=4,
    )
    axis.axhline(0, color="black", lw=1)
    axis.set(
        xticks=range(len(DESCRIPTION_MODEL_ORDER)),
        xticklabels=[MODEL_LABEL[name] for name in DESCRIPTION_MODEL_ORDER],
        ylabel="Held-out information gain (bits / transition)",
        title="Does distributed outcome memory beat two-back history?",
    )
    figure.tight_layout()
    print("Two-back held-out model comparison:")
    print(summary.to_string(index=False))
    return figure


def get_time_coefficient_contrasts(fitted_model):
    """Return time-coefficient log-odds contrasts versus next cross."""
    classes = list(fitted_model.classifier.classes_)
    coefficient_index = list(
        make_model_features(
            pd.DataFrame(
                {
                    "current_outcome": ["cross"],
                    "log_elapsed_s": [fitted_model.time_mean],
                }
            ),
            "markov_time",
            fitted_model.time_mean,
            fitted_model.time_std,
        ).columns
    )
    coefficients = pd.DataFrame(
        fitted_model.classifier.coef_, index=classes, columns=coefficient_index
    )
    return coefficients["log_elapsed_s"] - coefficients.loc["cross", "log_elapsed_s"]


def plot_time_effect(events):
    """Plot descriptive full-data next-outcome probabilities across elapsed time."""
    transitions = make_transition_table(events)
    if transitions.empty:
        raise RuntimeError("No valid time-model transitions.")
    fitted_model = fit_transition_model(transitions, "markov_time")
    lower, upper = np.quantile(transitions["elapsed_s"], [0.05, 0.95])
    if lower == upper:
        lower = max(0.01, lower - 0.5)
        upper = upper + 0.5
    elapsed_s = np.linspace(lower, upper, 100)
    figure, axes = analysis.plt.subplots(
        1, len(analysis.OUTCOME_ORDER), figsize=(15, 4), sharey=True
    )
    for axis, current_outcome in zip(axes, analysis.OUTCOME_ORDER):
        prediction_rows = pd.DataFrame(
            {
                "current_outcome": current_outcome,
                "elapsed_s": elapsed_s,
                "log_elapsed_s": np.log1p(elapsed_s),
            }
        )
        probabilities = predict_probabilities(fitted_model, prediction_rows)
        for class_index, next_outcome in enumerate(fitted_model.classifier.classes_):
            axis.plot(
                elapsed_s,
                probabilities[:, class_index],
                label=next_outcome,
                color=analysis.OUTCOME_COLOR[next_outcome],
            )
        axis.set(
            title=f"Current outcome: {current_outcome}",
            xlabel="Time to next attempt (s)",
            ylim=(0, 1),
        )
    axes[0].set_ylabel("Predicted next-outcome probability")
    axes[-1].legend(title="Next outcome")
    figure.suptitle("Descriptive full-data fit; not held-out performance")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    contrasts = get_time_coefficient_contrasts(fitted_model)
    print("Time coefficient contrasts versus next cross")
    print("Per training SD of log(1 + elapsed time in s):")
    print(contrasts.to_string())
    return figure


def run():
    """Load saved datasets and compare fixed, Markov, and outcome-memory models."""
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
        plot_memory_model_comparison(raw_scores=raw_scores)
        plot_memory_description_comparison(raw_scores=raw_scores)
        run_info.show()


if __name__ == "__main__":
    run()
