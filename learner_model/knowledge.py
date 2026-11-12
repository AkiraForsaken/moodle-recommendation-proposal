"""Load and summarize learner-by-KC knowledge states (the knowledge-engine contract).

The learner model does not know which estimator produced the rows: Phase 1
BKT and the Phase 3 deep model write the same columns.
"""

from pathlib import Path

import pandas as pd

REQUIRED_KNOWLEDGE_COLUMNS = {
    "learner_id",
    "kc_id",
    "kc_dimension",
    "attempt_count",
    "observed_accuracy",
    "predicted_correctness",
    "last_attempt_at",
    "model_name",
    "model_version",
    "estimator_phase",
    "inference_timestamp",
    "evidence_count",
    "state_status",
    "source_population",
}
# Present for some estimators only (null otherwise); added as null if absent.
OPTIONAL_KNOWLEDGE_COLUMNS = (
    "p_mastery",
    "checkpoint_id",
    "checkpoint_sha256",
    "aggregation_method",
    "prediction_question_count",
    "mapping_version",
)


def _utc_timestamp(value):
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="s", utc=True)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return pd.to_datetime(int(value), unit="s", utc=True)
    return pd.to_datetime(value, utc=True)


def _source_snapshot(states):
    """Latest event time the source states may reflect (None if indeterminable)."""
    if "as_of" in states.columns:
        declared = pd.to_datetime(states["as_of"], errors="coerce", utc=True).dropna()
        if not declared.empty:
            if declared.nunique() > 1:
                raise ValueError("Knowledge states mix several as_of cutoffs")
            return declared.iloc[0]
    attempts = pd.to_datetime(states["last_attempt_at"], errors="coerce", utc=True).dropna()
    return attempts.max() if not attempts.empty else None


def load_knowledge_states(path, as_of=None):
    """Load source rows without changing their probability semantics or provenance.

    With a cutoff, the knowledge snapshot must not post-date it (otherwise the
    estimates leak future events and a ValueError is raised). Cold-start rows
    (no attempts, hence no timestamp) are retained. Rows that report attempts
    but have no timestamp cannot be placed in time and are excluded; counts are
    exposed in ``DataFrame.attrs``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge-state file does not exist: {path}")
    states = pd.read_csv(path, dtype={"learner_id": "string", "kc_id": "string"})
    missing = REQUIRED_KNOWLEDGE_COLUMNS.difference(states.columns)
    if missing:
        raise ValueError(
            f"Knowledge-state file is missing required columns: {sorted(missing)}"
        )
    if states[["learner_id", "kc_id", "kc_dimension"]].isna().any().any():
        raise ValueError(
            "Knowledge-state learner, KC, and dimension fields cannot be empty"
        )
    if states.duplicated(["learner_id", "kc_id"]).any():
        raise ValueError(
            "Knowledge-state file contains duplicate learner_id/kc_id rows"
        )
    if states["source_population"].nunique(dropna=False) > 1:
        raise ValueError("Knowledge-state file mixes several source populations")
    for column in OPTIONAL_KNOWLEDGE_COLUMNS:
        if column not in states:
            states[column] = pd.NA
    states["kc_dimension"] = states["kc_dimension"].astype("string").str.strip().str.lower()

    source_learner_ids = states["learner_id"].dropna().astype(str).unique().tolist()
    dropped_after_cutoff = 0
    dropped_without_timestamp = 0
    knowledge_snapshot = _source_snapshot(states)
    if as_of is not None:
        cutoff = _utc_timestamp(as_of)
        if pd.isna(cutoff):
            raise ValueError(f"Invalid as_of cutoff: {as_of!r}")

        # Leakage guard. The estimates were computed from the answers the
        # knowledge engine was allowed to see. Those are bounded by its
        # cutoff (the `as_of` column) or, for a full-history run, by the
        # latest last_attempt_at. If that snapshot lies after the requested
        # cutoff, the estimates embed future events and are invalid.
        # (The wall-clock `inference_timestamp` is provenance only; comparing it
        # to a data-time cutoff would reject every valid re-run.)
        if knowledge_snapshot is not None and knowledge_snapshot > cutoff:
            raise ValueError(
                f"Knowledge states reflect events up to {knowledge_snapshot}, which is "
                f"after the requested cutoff {cutoff}. Predictions computed from later "
                "events would leak the future; re-run `python -m knowledge_engine "
                "estimate --as-of` at or before the cutoff."
            )

        attempts = pd.to_datetime(states["last_attempt_at"], errors="coerce", utc=True)
        attempt_counts = pd.to_numeric(states["attempt_count"], errors="coerce").fillna(0)
        after = attempts.notna() & ~attempts.le(cutoff)
        # A cold-start row (no direct attempts) has no timestamp by definition;
        # it is a valid state at any cutoff and must not be silently dropped.
        # Only rows claiming attempts but lacking a timestamp cannot be placed.
        untimed = attempts.isna() & attempt_counts.gt(0)
        dropped_after_cutoff = int(after.sum())
        dropped_without_timestamp = int(untimed.sum())
        states = states.loc[~after & ~untimed].copy()

    for column in ("observed_accuracy", "predicted_correctness", "p_mastery"):
        numeric = pd.to_numeric(states[column], errors="coerce")
        invalid = numeric.notna() & ~numeric.between(0, 1)
        if invalid.any():
            raise ValueError(
                f"Knowledge-state column {column!r} contains values outside [0, 1]"
            )
        states[column] = numeric

    states.attrs["dropped_after_cutoff"] = dropped_after_cutoff
    states.attrs["dropped_without_timestamp"] = dropped_without_timestamp
    states.attrs["source_learner_ids"] = source_learner_ids
    populations = states["source_population"].dropna().unique()
    states.attrs["source_population"] = str(populations[0]) if len(populations) else None
    states.attrs["as_of"] = None if as_of is None else cutoff.isoformat()
    states.attrs["knowledge_snapshot_as_of"] = (
        None if knowledge_snapshot is None else knowledge_snapshot.isoformat()
    )
    return states.reset_index(drop=True)


def summarize_dimensions(states):
    """Return one traceable aggregate row per learner and KC dimension."""
    if states.empty:
        return pd.DataFrame(
            columns=[
                "learner_id",
                "kc_dimension",
                "kc_count",
                "estimated_kc_count",
                "cold_start_kc_count",
                "mean_predicted_correctness",
                "mean_observed_accuracy",
                "total_attempt_count",
                "knowledge_snapshot_at",
            ]
        )

    aggregate_fields = {
        "kc_count": ("kc_id", "nunique"),
        "estimated_kc_count": (
            "state_status",
            lambda values: values.eq("estimated").sum(),
        ),
        "cold_start_kc_count": (
            "state_status",
            lambda values: values.eq("cold_start").sum(),
        ),
        "mean_predicted_correctness": ("predicted_correctness", "mean"),
        "mean_observed_accuracy": ("observed_accuracy", "mean"),
        "total_attempt_count": ("attempt_count", "sum"),
    }
    if "inference_timestamp" in states.columns:
        aggregate_fields["knowledge_snapshot_at"] = ("inference_timestamp", "first")
    else:
        aggregate_fields["knowledge_snapshot_at"] = ("last_attempt_at", "max")
    result = (
        states.groupby(["learner_id", "kc_dimension"], dropna=False)
        .agg(**aggregate_fields)
        .reset_index()
    )
    return result

