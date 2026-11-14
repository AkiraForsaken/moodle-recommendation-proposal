"""The knowledge-state contract shared by every estimator.

One row per learner and Knowledge Component. ``learner_model`` reads this file
and nothing else from the knowledge engine, so an estimator can be replaced
without touching the learner model.

Meaning of the two estimate columns:

    predicted_correctness  probability that the learner answers the next
                           question on this KC correctly. Every estimator
                           produces it, which makes phases comparable and is
                           what the promotion gate evaluates.
    p_mastery              BKT's latent probability that the KC is known.
                           Null for estimators that have no such quantity.

Neither is evidence by itself: ``attempt_count`` / ``observed_accuracy`` hold
the direct evidence, and ``state_status = cold_start`` marks a row that is a
prior only (no answer on that KC yet).
"""

import pandas as pd

KNOWLEDGE_STATE_COLUMNS = [
    "learner_id",
    "kc_id",
    "kc_dimension",
    "attempt_count",
    "evidence_count",
    "evidence_tier",
    "observed_accuracy",
    "predicted_correctness",
    "p_mastery",
    "last_attempt_at",
    "state_status",
    "model_name",
    "model_version",
    "estimator_phase",
    "checkpoint_id",
    "checkpoint_sha256",
    "aggregation_method",
    "prediction_question_count",
    "inference_timestamp",
    "as_of",
    "mapping_version",
    "source_population",
]
STATE_ESTIMATED = "estimated"
STATE_COLD_START = "cold_start"
VALID_STATE_STATUSES = {STATE_ESTIMATED, STATE_COLD_START}


def validate_states(states):
    """Raise ValueError if ``states`` breaks the contract; return it otherwise."""
    missing = [c for c in KNOWLEDGE_STATE_COLUMNS if c not in states.columns]
    if missing:
        raise ValueError(f"Knowledge states are missing columns: {missing}")
    if states[["learner_id", "kc_id", "kc_dimension"]].isna().any().any():
        raise ValueError("Knowledge states need learner_id, kc_id and kc_dimension")
    if states.duplicated(["learner_id", "kc_id"]).any():
        raise ValueError("Knowledge states contain duplicate learner/KC rows")
    unknown = set(states["state_status"].dropna()) - VALID_STATE_STATUSES
    if unknown or states["state_status"].isna().any():
        raise ValueError(
            f"Knowledge states have invalid state_status: {sorted(unknown)}"
        )
    for column in ("observed_accuracy", "predicted_correctness", "p_mastery"):
        values = pd.to_numeric(states[column], errors="coerce")
        if (values.notna() & ~values.between(0, 1)).any():
            raise ValueError(f"Knowledge-state column {column!r} is outside [0, 1]")
    attempts = pd.to_numeric(states["attempt_count"], errors="coerce")
    cold = states["state_status"].eq(STATE_COLD_START)
    if (
        attempts.isna().any()
        or (attempts[cold] != 0).any()
        or (attempts[~cold] <= 0).any()
    ):
        raise ValueError(
            "state_status must be cold_start exactly when attempt_count is 0"
        )
    if states.loc[cold, "observed_accuracy"].notna().any():
        raise ValueError("A cold-start row cannot carry observed accuracy")
    if states["predicted_correctness"].isna().any():
        raise ValueError("Every learner/KC row needs a predicted_correctness")
    return states[KNOWLEDGE_STATE_COLUMNS]
