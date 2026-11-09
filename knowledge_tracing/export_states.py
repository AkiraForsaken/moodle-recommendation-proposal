#!/usr/bin/env python3
"""Export validated learner-by-KC estimates from the selected KT model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import load_selected_model, score_akt_questions, sha256_file
from kt_common import PROJECT_ROOT, load_data_config

DEFAULT_INPUT = PROJECT_ROOT / "data/raw/english_learning_events.csv"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "knowledge_tracing/artifacts/knowledge_states"
REQUIRED_EVENT_COLUMNS = {
    "learner_id",
    "event_index",
    "item_id",
    "kc_id",
    "kc_dimension",
    "event_type",
    "timestamp",
    "correct",
}


def normalize_as_of(as_of: str | datetime | pd.Timestamp | None) -> str | None:
    if as_of is None:
        return None
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must include a timezone, for example 2024-01-01T00:00:00Z")
    return cutoff.tz_convert("UTC").isoformat()


def load_answer_events(
    path: Path, as_of: str | datetime | pd.Timestamp | None = None
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Answer event CSV not found: {path}")
    events = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={
            "learner_id": str,
            "item_id": str,
            "kc_id": str,
            "kc_dimension": str,
            "event_type": str,
        },
    )
    events.columns = [column.strip() for column in events.columns]
    missing = REQUIRED_EVENT_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"Answer event CSV is missing columns: {sorted(missing)}")
    events = events[events["event_type"].str.strip().eq("answer")].copy()
    events["correct"] = pd.to_numeric(events["correct"], errors="coerce")
    events = events[events["correct"].isin([0, 1])].copy()
    events["correct"] = events["correct"].astype(int)
    events["event_index"] = pd.to_numeric(events["event_index"], errors="raise").astype(
        int
    )
    events["timestamp"] = pd.to_numeric(events["timestamp"], errors="raise").astype(
        "int64"
    )
    normalized_cutoff = normalize_as_of(as_of)
    if normalized_cutoff is not None:
        cutoff = pd.Timestamp(normalized_cutoff)
        event_times = pd.to_datetime(events["timestamp"], unit="s", utc=True)
        events = events[event_times.le(cutoff)].copy()
    if events.duplicated(["learner_id", "event_index"]).any():
        raise ValueError("Duplicate (learner_id, event_index) answer events found")
    return events.sort_values(["learner_id", "event_index"], kind="mergesort")


def choose_representative_questions(
    question_map: pd.DataFrame, max_questions_per_kc: int
) -> pd.DataFrame:
    """Select sorted, evenly spaced question IDs per KC deterministically."""
    if max_questions_per_kc < 0:
        raise ValueError("max_questions_per_kc must be non-negative")
    selected = []
    for _, group in question_map.sort_values("questions").groupby("kc_id", sort=True):
        if max_questions_per_kc and len(group) > max_questions_per_kc:
            positions = np.linspace(0, len(group) - 1, max_questions_per_kc, dtype=int)
            group = group.iloc[positions]
        selected.append(group)
    return pd.concat(selected, ignore_index=True)


def validate_mappings(
    question_map: pd.DataFrame,
    concept_map: pd.DataFrame,
    data_config: dict,
) -> None:
    """Fail if Milestone 1 mappings no longer match the trained vocabulary."""
    required_question_columns = {
        "item_id",
        "questions",
        "kc_id",
        "concepts",
        "kc_dimension",
    }
    required_concept_columns = {"kc_id", "concepts", "kc_dimension"}
    if not required_question_columns <= set(question_map.columns):
        raise ValueError("question_map.csv is missing required mapping columns")
    if not required_concept_columns <= set(concept_map.columns):
        raise ValueError("concept_map.csv is missing required mapping columns")
    if question_map.empty or concept_map.empty:
        raise ValueError("Question and concept mapping files must not be empty")
    if (
        question_map[["item_id", "kc_id", "kc_dimension"]].isna().any().any()
        or concept_map[["kc_id", "kc_dimension"]].isna().any().any()
    ):
        raise ValueError("Question and concept mappings must not contain missing IDs")
    if (
        question_map["item_id"].duplicated().any()
        or question_map["questions"].duplicated().any()
        or concept_map["kc_id"].duplicated().any()
        or concept_map["concepts"].duplicated().any()
    ):
        raise ValueError("Question and concept mapping IDs and indices must be unique")

    for frame, column, label in (
        (question_map, "questions", "question"),
        (concept_map, "concepts", "concept"),
    ):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any():
            raise ValueError(f"{label.title()} mapping indices must not be missing")
        integer_values = values.astype("int64")
        if not values.eq(integer_values).all():
            raise ValueError(f"{label.title()} mapping indices must be integers")
        if not np.array_equal(
            np.sort(integer_values.to_numpy()), np.arange(len(frame))
        ):
            raise ValueError(f"{label.title()} mapping indices must be contiguous from 0")

    concept_ids = set(concept_map["kc_id"])
    question_kc_ids = set(question_map["kc_id"])
    if question_kc_ids != concept_ids:
        raise ValueError("Every mapped KC must have questions and a concept index")
    concept_indices = concept_map.set_index("kc_id")["concepts"]
    dimensions = concept_map.set_index("kc_id")["kc_dimension"]
    if not question_map["concepts"].eq(
        question_map["kc_id"].map(concept_indices)
    ).all():
        raise ValueError("Question-to-concept indices differ from concept_map.csv")
    if not question_map["kc_dimension"].eq(
        question_map["kc_id"].map(dimensions)
    ).all():
        raise ValueError("Question KC dimensions differ from concept_map.csv")
    if int(data_config.get("num_q", -1)) != len(question_map):
        raise ValueError("question_map.csv size differs from the trained model config")
    if int(data_config.get("num_c", -1)) != len(concept_map):
        raise ValueError("concept_map.csv size differs from the trained model config")


def make_skill_state_frame(
    learner_ids: list[str],
    learner_splits: pd.Series,
    concept_map: pd.DataFrame,
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    model_metadata: dict,
    inference_timestamp: str,
    as_of: str | None,
) -> pd.DataFrame:
    ordered_events = events.sort_values(
        ["learner_id", "event_index"], kind="mergesort"
    )
    evidence = ordered_events.groupby(["learner_id", "kc_id"], as_index=False).agg(
        attempt_count=("correct", "size"),
        correct_count=("correct", "sum"),
        last_attempt_epoch=("timestamp", "last"),
    )
    evidence["observed_accuracy"] = (
        evidence["correct_count"] / evidence["attempt_count"]
    )
    evidence["last_attempt_at"] = pd.to_datetime(
        evidence["last_attempt_epoch"], unit="s", utc=True
    ).map(lambda value: value.isoformat())

    pairs = pd.MultiIndex.from_product(
        [learner_ids, concept_map["kc_id"].tolist()],
        names=["learner_id", "kc_id"],
    ).to_frame(index=False)
    states = pairs.merge(
        concept_map[["kc_id", "kc_dimension"]],
        on="kc_id",
        how="left",
        validate="many_to_one",
    )
    states["split"] = states["learner_id"].map(learner_splits)
    if states["split"].isna().any():
        raise ValueError("Every learner must have a split assignment")
    states = states.merge(
        evidence[
            [
                "learner_id",
                "kc_id",
                "attempt_count",
                "observed_accuracy",
                "last_attempt_at",
            ]
        ],
        on=["learner_id", "kc_id"],
        how="left",
        validate="one_to_one",
    )
    states = states.merge(
        predictions,
        on=["learner_id", "kc_id"],
        how="left",
        validate="one_to_one",
    )
    states["attempt_count"] = states["attempt_count"].fillna(0).astype(int)
    states["evidence_count"] = states["attempt_count"]
    states["evidence_tier"] = pd.cut(
        states["attempt_count"],
        bins=[-1, 0, 2, 5, np.inf],
        labels=["none", "low", "moderate", "high"],
    ).astype(str)
    states["state_status"] = np.where(
        states["attempt_count"].gt(0), "estimated", "cold_start"
    )
    states["model_name"] = model_metadata["model_name"]
    states["model_version"] = model_metadata["model_version"]
    states["checkpoint_id"] = model_metadata["checkpoint_id"]
    states["checkpoint_sha256"] = model_metadata["checkpoint_sha256"]
    states["inference_timestamp"] = inference_timestamp
    states["as_of"] = as_of
    states["aggregation_method"] = model_metadata["aggregation_method"]
    states["last_attempt_at"] = states["last_attempt_at"].where(
        states["attempt_count"].gt(0), None
    )
    return states[
        [
            "learner_id",
            "split",
            "kc_id",
            "kc_dimension",
            "attempt_count",
            "observed_accuracy",
            "predicted_correctness",
            "predicted_correctness_10q",
            "last_attempt_at",
            "model_name",
            "model_version",
            "checkpoint_id",
            "checkpoint_sha256",
            "inference_timestamp",
            "as_of",
            "evidence_count",
            "evidence_tier",
            "prediction_question_count",
            "sensitivity_question_count",
            "aggregation_method",
            "state_status",
        ]
    ]


def validate_states(
    states: pd.DataFrame,
    events: pd.DataFrame,
    learner_ids: list[str],
    concept_map: pd.DataFrame,
) -> list[str]:
    issues = []
    expected_pairs = len(learner_ids) * len(concept_map)
    if (
        len(states) != expected_pairs
        or states.duplicated(["learner_id", "kc_id"]).any()
    ):
        issues.append("Learner-KC coverage is incomplete or contains duplicate pairs")
    if set(states["learner_id"]) != set(learner_ids):
        issues.append("Output does not preserve all original learner IDs")
    if states["split"].isna().any():
        issues.append("Output is missing learner split assignments")
    if set(states["kc_id"]) != set(concept_map["kc_id"]):
        issues.append("Output does not preserve all KC IDs from concept_map.csv")
    dimension_map = concept_map.set_index("kc_id")["kc_dimension"]
    if not states["kc_dimension"].eq(states["kc_id"].map(dimension_map)).all():
        issues.append("KC dimensions do not match concept_map.csv")
    expected_counts = events.groupby(["learner_id", "kc_id"]).size()
    actual_counts = states.set_index(["learner_id", "kc_id"])["attempt_count"]
    aligned_counts = expected_counts.reindex(actual_counts.index, fill_value=0).astype(
        int
    )
    if not aligned_counts.equals(actual_counts.astype(int)):
        issues.append("Attempt counts do not match source answer events")
    expected_accuracies = events.groupby(["learner_id", "kc_id"])["correct"].mean()
    aligned_accuracies = expected_accuracies.reindex(actual_counts.index)
    actual_accuracies = states.set_index(["learner_id", "kc_id"])[
        "observed_accuracy"
    ]
    if not np.allclose(
        aligned_accuracies.to_numpy(dtype=float),
        actual_accuracies.to_numpy(dtype=float),
        equal_nan=True,
    ):
        issues.append("Observed accuracy does not match source answer events")
    ordered_events = events.sort_values(
        ["learner_id", "event_index"], kind="mergesort"
    )
    expected_last_attempts = ordered_events.groupby(["learner_id", "kc_id"])[
        "timestamp"
    ].last()
    expected_last_attempts = expected_last_attempts.reindex(actual_counts.index)
    expected_last_attempts = pd.to_datetime(
        expected_last_attempts, unit="s", utc=True
    ).map(lambda value: None if pd.isna(value) else value.isoformat())
    actual_last_attempts = states.set_index(["learner_id", "kc_id"])[
        "last_attempt_at"
    ]
    if not np.array_equal(
        expected_last_attempts.fillna("").astype(str).to_numpy(),
        actual_last_attempts.fillna("").astype(str).to_numpy(),
    ):
        issues.append("Last-attempt timestamps do not match source answer events")
    observed = states["observed_accuracy"].dropna()
    predicted = states["predicted_correctness"]
    if not observed.between(0, 1).all():
        issues.append("Observed accuracy is outside [0, 1]")
    if predicted.isna().any() or not predicted.between(0, 1).all():
        issues.append("Predicted correctness is missing or outside [0, 1]")
    sensitivity_predictions = states["predicted_correctness_10q"]
    if sensitivity_predictions.isna().any() or not sensitivity_predictions.between(
        0, 1
    ).all():
        issues.append("10-question sensitivity predictions are missing or outside [0, 1]")
    if not states["evidence_count"].eq(states["attempt_count"]).all():
        issues.append("Evidence counts do not match attempt counts")
    expected_tiers = pd.cut(
        states["attempt_count"],
        bins=[-1, 0, 2, 5, np.inf],
        labels=["none", "low", "moderate", "high"],
    ).astype(str)
    if not expected_tiers.equals(states["evidence_tier"]):
        issues.append("Evidence tiers do not match attempt counts")
    cold = states["state_status"].eq("cold_start")
    expected_cold = states["attempt_count"].eq(0)
    if not cold.equals(expected_cold):
        issues.append("State status does not match the available KC evidence")
    if states.loc[cold, "observed_accuracy"].notna().any():
        issues.append("Cold-start states have an observed accuracy")
    if states.loc[~cold, "observed_accuracy"].isna().any():
        issues.append("Evidence-backed states are missing observed accuracy")
    if states["last_attempt_at"].isna().ne(expected_cold).any():
        issues.append("Last-attempt timestamps do not match the available KC evidence")
    provenance = [
        "model_name",
        "model_version",
        "checkpoint_id",
        "checkpoint_sha256",
        "inference_timestamp",
        "aggregation_method",
    ]
    if states[provenance].isna().any().any():
        issues.append("Model provenance is incomplete")
    if (
        states["prediction_question_count"].isna().any()
        or states["prediction_question_count"].lt(1).any()
    ):
        issues.append("At least one state has no question-level predictions")
    if (
        states["sensitivity_question_count"].isna().any()
        or states["sensitivity_question_count"].lt(1).any()
    ):
        issues.append("At least one state has no sensitivity-question predictions")
    return issues


def write_validation_report(
    path: Path, states: pd.DataFrame, events: pd.DataFrame, issues: list[str]
) -> None:
    evidence_states = states[states["observed_accuracy"].notna()].copy()
    evidence_states["prediction_minus_observed"] = (
        evidence_states["predicted_correctness"] - evidence_states["observed_accuracy"]
    )
    sensitivity_difference = (
        states["predicted_correctness"] - states["predicted_correctness_10q"]
    ).abs()
    dimensions = states.groupby("kc_dimension").agg(
        state_count=("predicted_correctness", "size"),
        predicted_mean=("predicted_correctness", "mean"),
        predicted_std=("predicted_correctness", "std"),
        observed_mean=("observed_accuracy", "mean"),
    )
    dimension_differences = evidence_states.groupby("kc_dimension")[
        "prediction_minus_observed"
    ].mean()
    dimensions["mean_prediction_minus_observed"] = dimensions.index.map(
        dimension_differences
    )
    split_summary = states.groupby("split").agg(
        learners=("learner_id", "nunique"),
        learner_kcs=("kc_id", "size"),
        single_attempt_states=("attempt_count", lambda values: int(values.eq(1).sum())),
    ).join(
        evidence_states.groupby("split").agg(
            evidence_states=("observed_accuracy", "size"),
            mean_predicted=("predicted_correctness", "mean"),
            mean_observed=("observed_accuracy", "mean"),
            descriptive_difference=("prediction_minus_observed", "mean"),
        ),
        how="left",
    )
    evidence_stats = states["attempt_count"].describe()
    prediction_stats = states["predicted_correctness"].describe()
    status_counts = states["state_status"].value_counts().to_dict()
    tier_counts = states["evidence_tier"].value_counts().to_dict()
    expected_pairs = states["learner_id"].nunique() * states["kc_id"].nunique()
    single_attempt_count = int(states["attempt_count"].eq(1).sum())
    estimated_count = int(states["state_status"].eq("estimated").sum())
    single_attempt_fraction = (
        single_attempt_count / estimated_count if estimated_count else 0.0
    )
    extreme_prediction_count = int(
        (
            states["predicted_correctness"].le(0.05)
            | states["predicted_correctness"].ge(0.95)
        ).sum()
    )
    report = [
        "# Knowledge-State Extraction Validation",
        "",
        f"- Validation status: {'PASS' if not issues else 'FAIL'}",
        f"- Learners: {states['learner_id'].nunique():,}",
        f"- Knowledge Components: {states['kc_id'].nunique():,}",
        f"- Learner-KC records: {len(states):,}",
        f"- Learner-KC coverage: {len(states) / expected_pairs:.1%}",
        f"- Source answer events: {len(events):,}",
        f"- Snapshot cutoff (UTC): {(states['as_of'].iloc[0] or 'not specified') if 'as_of' in states and len(states) else 'not specified'}",
        f"- State status counts: {status_counts}",
        f"- Evidence tier counts: {tier_counts}",
        f"- Estimated states with one direct KC attempt: {single_attempt_count:,} ({single_attempt_fraction:.1%})",
        f"- Predictions at or beyond 0.05/0.95: {extreme_prediction_count:,}",
        "- Predicted correctness is an estimated probability of answering a mapped question correctly; it is not a mastery probability.",
        "",
        "## Structural Checks",
        "",
    ]
    report.extend(
        [f"- FAIL: {issue}" for issue in issues] or ["- All structural checks passed."]
    )
    report.extend(
        [
            "",
            "## Distributions",
            "",
            "Attempt count summary:",
            "",
            "```text",
            evidence_stats.to_string(),
            "```",
            "",
            "Predicted correctness summary:",
            "",
            "```text",
            prediction_stats.to_string(),
            "```",
            "",
            "Observed accuracy summary (direct evidence only):",
            "",
            "```text",
            states["observed_accuracy"].dropna().describe().to_string(),
            "```",
            "",
            "Prediction minus observed historical KC accuracy, descriptive comparison only:",
            "",
            "```text",
            evidence_states["prediction_minus_observed"].describe().to_string(),
            "```",
            "",
            "## Split-Level Descriptive Comparison",
            "",
            "The following compares full-map model estimates with each learner's historical KC accuracy through the cutoff. It is not a calibration/error estimate: train learners contributed to model fitting, and the compared quantities average over different questions.",
            "",
            "| Split | Learners | Learner-KC states | Evidence-backed learner-KCs | Mean predicted | Mean observed | Mean predicted minus observed | Single-attempt learner-KCs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {split} | {int(row['learners'])} | {int(row['learner_kcs'])} | "
                f"{int(row['evidence_states']) if pd.notna(row['evidence_states']) else 0} | "
                f"{row['mean_predicted']:.4f} | {row['mean_observed']:.4f} | "
                f"{row['descriptive_difference']:.4f} | {int(row['single_attempt_states'])} |"
                for split, row in split_summary.iterrows()
            ],
            "",
            "## Dimension Summary",
            "",
            "| Dimension | States | Mean predicted | Predicted std. dev. | Mean observed | Mean prediction minus observed |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {dimension} | {int(row['state_count'])} | {row['predicted_mean']:.4f} | "
                f"{row['predicted_std']:.4f} | {row['observed_mean']:.4f} | "
                f"{row['mean_prediction_minus_observed']:.4f} |"
                for dimension, row in dimensions.iterrows()
            ],
            "",
            "## All-Question vs 10-Question Sensitivity",
            "",
            f"- Mean absolute difference across all learner-KCs: {sensitivity_difference.mean():.4f}",
            f"- 95th percentile absolute difference: {sensitivity_difference.quantile(0.95):.4f}",
            "- Primary predictions average over every mapped question in the KC. The deterministic 10-question estimate is retained separately as a sensitivity check from the same model scores.",
            "",
            "## Interpretation and Limitations",
            "",
            "- Historical observed accuracy is reported separately from the model prediction.",
            "- Cold-start rows have no direct KC attempts and a null observed accuracy; their predicted correctness can still use the learner's history in other KCs, or the model's learned prior when history is empty.",
            "- Evidence tiers are none (0 attempts), low (1-2), moderate (3-5), and high (6 or more); consumers should treat low-evidence estimates cautiously.",
            "- The checkpoint was selected by validation AUC only. Held-out test metrics are descriptive and were not used for model selection.",
            "",
        ]
    )
    path.write_text("\n".join(report))


def export_states(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = OUTPUT_DIR,
    max_questions_per_kc: int = 10,
    batch_size: int = 128,
    device: str | None = None,
    as_of: str | datetime | pd.Timestamp | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    normalized_as_of = normalize_as_of(as_of)

    def progress(message: str) -> None:
        if show_progress:
            print(f"[knowledge-state] {message}", flush=True)

    progress(f"Loading answer events from {input_path}")
    events = load_answer_events(input_path, normalized_as_of)
    progress(
        f"Loaded {len(events):,} answer events"
        + (
            f" at or before {normalized_as_of}"
            if normalized_as_of is not None
            else " (no as_of cutoff)"
        )
    )
    question_map = pd.read_csv(
        PROCESSED_DIR / "question_map.csv", dtype={"item_id": str, "kc_id": str}
    )
    concept_map = pd.read_csv(
        PROCESSED_DIR / "concept_map.csv", dtype={"kc_id": str, "kc_dimension": str}
    )
    assignments = pd.read_csv(
        PROCESSED_DIR / "split_assignments.csv", dtype={"learner_id": str}
    )
    data_config = load_data_config()
    validate_mappings(question_map, concept_map, data_config)
    if "learner_id" not in assignments or assignments["learner_id"].isna().any():
        raise ValueError("split_assignments.csv must contain learner IDs")
    if assignments["learner_id"].duplicated().any():
        raise ValueError("split_assignments.csv contains duplicate learner IDs")
    if "split" not in assignments or assignments["split"].isna().any():
        raise ValueError("split_assignments.csv must contain split for every learner")
    learner_ids = sorted(assignments["learner_id"].dropna().unique().tolist())
    if not learner_ids:
        raise ValueError("split_assignments.csv contains no learners")
    if not set(events["learner_id"]) <= set(learner_ids):
        raise ValueError(
            "Source answer events contain learners missing from split_assignments.csv"
        )
    learner_splits = assignments.set_index("learner_id")["split"]
    progress(
        f"Validated maps: {len(question_map):,} questions, "
        f"{len(concept_map):,} KCs, {len(learner_ids):,} assigned learners"
    )

    question_lookup = question_map.set_index("item_id")["questions"]
    concept_lookup = concept_map.set_index("kc_id")["concepts"]
    unmapped_items = set(events["item_id"]) - set(question_lookup.index)
    unmapped_concepts = set(events["kc_id"]) - set(concept_lookup.index)
    if unmapped_items or unmapped_concepts:
        raise ValueError(
            "Source answer events contain IDs absent from Milestone 1 maps"
        )
    events["question_idx"] = events["item_id"].map(question_lookup).astype(int)
    events["concept_idx"] = events["kc_id"].map(concept_lookup).astype(int)
    mapped_items = question_map.set_index("item_id")["kc_id"]
    if not events["kc_id"].eq(events["item_id"].map(mapped_items)).all():
        raise ValueError("Source item-to-KC assignments differ from question_map.csv")
    mapped_dimensions = concept_map.set_index("kc_id")["kc_dimension"]
    if not events["kc_dimension"].eq(events["kc_id"].map(mapped_dimensions)).all():
        raise ValueError("Source KC dimensions differ from concept_map.csv")

    sensitivity_questions = choose_representative_questions(
        question_map, max_questions_per_kc
    )
    sensitivity_question_ids = set(sensitivity_questions["questions"].astype(int))
    all_questions = question_map.sort_values("questions").reset_index(drop=True)
    candidate_question_ids = all_questions["questions"].astype(int).tolist()
    candidate_concept_ids = all_questions["concepts"].astype(int).tolist()
    sensitivity_mask = all_questions["questions"].isin(sensitivity_question_ids)
    progress(
        f"Loading selected AKT checkpoint; scoring all "
        f"{len(all_questions):,} mapped questions per learner"
    )
    model, target_device, selection, checkpoint_sha256 = load_selected_model(device)
    inference_timestamp = datetime.now(timezone.utc).isoformat()
    predictions = []

    histories = {
        learner_id: history
        for learner_id, history in events.groupby("learner_id", sort=True)
    }
    empty_history = events.iloc[0:0]
    progress(
        f"Scoring {len(learner_ids):,} learners in batches of {batch_size} "
        f"on {target_device}"
    )
    progress_interval = max(1, int(np.ceil(len(learner_ids) / 20)))
    for learner_index, learner_id in enumerate(learner_ids, start=1):
        history = histories.get(learner_id, empty_history)
        history = history.sort_values("event_index", kind="mergesort")
        scores = score_akt_questions(
            model=model,
            device=target_device,
            concepts=history["concept_idx"].tolist(),
            responses=history["correct"].tolist(),
            history_questions=history["question_idx"].tolist(),
            target_questions=candidate_question_ids,
            target_concepts=candidate_concept_ids,
            maxlen=int(data_config["maxlen"]),
            batch_size=batch_size,
        )
        per_question = all_questions[["kc_id"]].copy()
        per_question["predicted_correctness"] = scores
        per_question["sensitivity_correctness"] = np.where(
            sensitivity_mask.to_numpy(), scores, np.nan
        )
        grouped = per_question.groupby("kc_id").agg(
            predicted_correctness=("predicted_correctness", "mean"),
            prediction_question_count=("predicted_correctness", "size"),
            predicted_correctness_10q=("sensitivity_correctness", "mean"),
            sensitivity_question_count=("sensitivity_correctness", "count"),
        )
        for kc_id, row in grouped.iterrows():
            predictions.append(
                {
                    "learner_id": learner_id,
                    "kc_id": kc_id,
                    "predicted_correctness": float(row["predicted_correctness"]),
                    "prediction_question_count": int(row["prediction_question_count"]),
                    "predicted_correctness_10q": float(
                        row["predicted_correctness_10q"]
                    ),
                    "sensitivity_question_count": int(
                        row["sensitivity_question_count"]
                    ),
                }
            )
        if (
            learner_index % progress_interval == 0
            or learner_index == len(learner_ids)
        ):
            progress(
                f"Scored {learner_index:,}/{len(learner_ids):,} learners "
                f"({learner_index / len(learner_ids):.0%})"
            )

    model_metadata = {
        **selection,
        "checkpoint_sha256": checkpoint_sha256,
        "aggregation_method": "arithmetic_mean_of_all_mapped_question_probabilities",
    }
    states = make_skill_state_frame(
        learner_ids,
        learner_splits,
        concept_map,
        events,
        pd.DataFrame(predictions),
        model_metadata,
        inference_timestamp,
        normalized_as_of,
    )
    issues = validate_states(states, events, learner_ids, concept_map)
    expected_prediction_counts = all_questions.groupby("kc_id").size()
    expected_prediction_counts = states["kc_id"].map(expected_prediction_counts)
    if not states["prediction_question_count"].eq(expected_prediction_counts).all():
        issues.append(
            "Prediction-question counts do not match complete question-map coverage"
        )
    expected_sensitivity_counts = sensitivity_questions.groupby("kc_id").size()
    expected_sensitivity_counts = states["kc_id"].map(expected_sensitivity_counts)
    if not states["sensitivity_question_count"].eq(expected_sensitivity_counts).all():
        issues.append("Sensitivity-question counts do not match selected coverage")
    if issues:
        raise ValueError("Knowledge-state validation failed: " + "; ".join(issues))

    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = output_dir / "learner_skill_state.csv"
    manifest_path = output_dir / "inference_manifest.json"
    report_path = output_dir / "knowledge_state_validation_report.md"
    mapping_hashes = {
        "question_map": sha256_file(PROCESSED_DIR / "question_map.csv"),
        "concept_map": sha256_file(PROCESSED_DIR / "concept_map.csv"),
    }
    progress("Validation passed; writing learner states and manifest")
    states.to_csv(states_path, index=False)
    write_validation_report(report_path, states, events, issues)
    manifest = {
        "inference_timestamp": inference_timestamp,
        "as_of": normalized_as_of,
        "input_events": str(input_path.resolve()),
        "input_events_sha256": sha256_file(input_path),
        "n_answer_events": int(len(events)),
        "n_learners": len(learner_ids),
        "n_knowledge_components": int(len(concept_map)),
        "n_questions_scored_per_learner": int(len(all_questions)),
        "n_sensitivity_questions_per_kc": int(max_questions_per_kc),
        "n_sensitivity_questions": int(len(sensitivity_questions)),
        "aggregation_method": model_metadata["aggregation_method"],
        "data_config": data_config,
        "mapping_sha256": mapping_hashes,
        "model": selection,
        "training_environment": {
            "pykt_toolkit_version": selection.get("pykt_toolkit_version"),
            "torch_version": selection.get("torch_version"),
        },
        "runtime_environment": selection.get("runtime_versions", {}),
        "checkpoint_sha256": checkpoint_sha256,
        "device": target_device,
        "validation_status": "pass",
        "outputs": {
            "learner_skill_state": str(states_path),
            "validation_report": str(report_path),
            "inference_manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    progress(f"Wrote {states_path} ({len(states):,} learner-KC rows)")
    progress(f"Wrote {manifest_path}")
    progress(f"Wrote {report_path}")
    progress("Knowledge-state export complete")
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--questions-per-kc",
        type=int,
        default=10,
        help="questions per KC for the 10-question sensitivity estimate",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="inclusive, timezone-aware ISO 8601 cutoff, e.g. 2024-01-01T00:00:00Z",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--quiet", action="store_true", help="disable progress output")
    args = parser.parse_args()
    try:
        export_states(
            input_path=args.input,
            output_dir=args.output_dir,
            max_questions_per_kc=args.questions_per_kc,
            batch_size=args.batch_size,
            device=args.device,
            as_of=args.as_of,
            show_progress=not args.quiet,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
