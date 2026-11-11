"""Explainable evidence- and prediction-based weak-area screening.

Rules (first match wins), applied per learner-KC row:

1. >= MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN direct attempts and observed accuracy
   below LOW_OBSERVED_ACCURACY            -> evidence_based_concern
2. >= MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN direct attempts, accuracy not low
                                          -> no_current_concern
3. fewer attempts and predicted_correctness below LOW_PREDICTED_CORRECTNESS
                                          -> prediction_based_concern
4. otherwise                              -> insufficient_evidence

A high prediction is never promoted to a "mastery" label: with limited direct
evidence the row stays insufficient_evidence. Cold-start rows can therefore only
be prediction_based_concern or insufficient_evidence, never evidence-based.
"""

import pandas as pd

from .config import (
    LOW_OBSERVED_ACCURACY,
    LOW_PREDICTED_CORRECTNESS,
    MAPPING_VERSION,
    MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN,
)
from .schemas import WEAK_AREA_COLUMNS

CONCERN_STATUSES = (
    "evidence_based_concern",
    "prediction_based_concern",
    "insufficient_evidence",
    "no_current_concern",
)


def _classify(attempts, observed, predicted):
    low_pred = f"{LOW_PREDICTED_CORRECTNESS:.2f}"
    if attempts >= MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN and observed is not None:
        if observed < LOW_OBSERVED_ACCURACY:
            reason = (
                f"Observed accuracy {observed:.3f} is below {LOW_OBSERVED_ACCURACY:.2f} "
                f"across {attempts} direct attempts"
            )
            if predicted is not None and predicted < LOW_PREDICTED_CORRECTNESS:
                reason += f"; predicted correctness {predicted:.3f} is also below {low_pred}"
            return "evidence_based_concern", reason
        return (
            "no_current_concern",
            f"Observed accuracy {observed:.3f} is at or above {LOW_OBSERVED_ACCURACY:.2f} "
            f"across {attempts} direct attempts",
        )

    basis = (
        "no direct attempts (cold-start)"
        if attempts == 0
        else f"only {attempts} direct attempt(s), fewer than {MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN}"
    )
    if predicted is not None and predicted < LOW_PREDICTED_CORRECTNESS:
        return (
            "prediction_based_concern",
            f"Predicted correctness {predicted:.3f} is below {low_pred}; {basis}. "
            "Model prediction only: not verified difficulty and not a mastery estimate",
        )
    if predicted is None:
        return "insufficient_evidence", f"{basis[0].upper()}{basis[1:]} and no prediction available"
    return (
        "insufficient_evidence",
        f"{basis[0].upper()}{basis[1:]}; predicted correctness {predicted:.3f} is not below "
        f"{low_pred}, so no prediction-based concern is raised",
    )


def identify_weak_areas(states, generated_at, mapping_version=MAPPING_VERSION):
    """Classify KC-level concerns without interpreting predictions as mastery."""
    rows = []
    for state in states.itertuples(index=False):
        attempts = int(getattr(state, "attempt_count", 0) or 0)
        evidence = int(getattr(state, "evidence_count", 0) or 0)
        observed = getattr(state, "observed_accuracy", None)
        predicted = getattr(state, "predicted_correctness", None)
        observed = None if observed is None or pd.isna(observed) else float(observed)
        predicted = None if predicted is None or pd.isna(predicted) else float(predicted)
        concern_status, reason = _classify(attempts, observed, predicted)
        rows.append(
            {
                "learner_id": state.learner_id,
                "entity_type": "kc",
                "entity_id": state.kc_id,
                "entity_name": state.kc_id,
                "concern_status": concern_status,
                "observed_accuracy": observed,
                "predicted_correctness": predicted,
                "evidence_count": evidence,
                "reason": reason,
                "mapping_version": mapping_version,
                "generated_at": generated_at,
            }
        )
    return pd.DataFrame(rows, columns=WEAK_AREA_COLUMNS)
