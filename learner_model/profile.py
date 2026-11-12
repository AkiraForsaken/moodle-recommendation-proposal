"""Assemble the top-level learner profile: one row per learner."""

import pandas as pd

from .config import PROFILE_SCHEMA_VERSION
from .schemas import PROFILE_COLUMNS

# Components whose own table already carries a per-learner status column. The
# profile reports that status rather than flattening it to "available".
COMPONENT_STATUS_COLUMNS = {
    "behavior": "behavior_summary_status",
    "engagement": "engagement_status",
    "history": "history_status",
}
# Statuses that count toward profile_completeness. "unavailable" and
# "no_observations" do not.
USABLE_STATUSES = {"available", "partial", "cold_start_only"}
# Knowledge-derived components need an estimator run; without one they are
# "unavailable". Collector-derived components exist for every enrolled learner:
# a learner with no rows simply has "no_observations" yet.
KNOWLEDGE_COMPONENTS = {"knowledge", "skills", "weak_areas"}


def _status_by_learner(component, frame):
    """Map learner_id -> component status for learners present in ``frame``."""
    if frame.empty or "learner_id" not in frame:
        return {}
    ids = frame["learner_id"].astype(str)
    column = COMPONENT_STATUS_COLUMNS.get(component)
    if column and column in frame:
        status = frame[column].fillna("unavailable").astype(str)
        return dict(zip(ids, status))
    if component == "knowledge" and "state_status" in frame:
        has_estimate = frame["state_status"].eq("estimated").groupby(ids).any()
        return {
            learner: "available" if flag else "cold_start_only"
            for learner, flag in has_estimate.items()
        }
    # Skills and weak areas built only from cold-start priors carry no
    # individual evidence either, and say so.
    evidence_column = {
        "skills": "estimated_kc_count",
        "weak_areas": "evidence_count",
    }.get(component)
    if evidence_column and evidence_column in frame:
        has_evidence = (
            pd.to_numeric(frame[evidence_column], errors="coerce")
            .fillna(0)
            .gt(0)
            .groupby(ids)
            .any()
        )
        return {
            learner: "available" if flag else "cold_start_only"
            for learner, flag in has_evidence.items()
        }
    return {learner: "available" for learner in ids.unique()}


def _knowledge_provenance(knowledge):
    """learner_id -> (estimator phase, model name) from the KC states."""
    if knowledge.empty or "learner_id" not in knowledge:
        return {}
    first = knowledge.assign(
        learner_id=knowledge["learner_id"].astype(str)
    ).drop_duplicates("learner_id")
    result = {}
    for row in first.itertuples(index=False):
        phase = getattr(row, "estimator_phase", None)
        result[row.learner_id] = (
            None if phase is None or pd.isna(phase) else int(phase),
            getattr(row, "model_name", None),
        )
    return result


def build_profiles(
    population,
    learner_ids,
    component_frames,
    generated_at,
    as_of=None,
    latest_timestamps=None,
):
    """Build one profile per learner of a single source population.

    Every enrolled learner gets a profile, including one with no activity and
    no knowledge estimate. Each ``*_status`` reports the component's real
    status (available, partial, cold_start_only, no_observations,
    unavailable), not just its presence. ``knowledge_phase`` /
    ``knowledge_model`` say which estimator produced the knowledge component,
    so a consumer can tell a cold-start prior from a personalised estimate.
    """
    latest_timestamps = latest_timestamps or {}
    component_statuses = {
        component: _status_by_learner(component, frame)
        for component, frame in component_frames.items()
    }
    provenance = _knowledge_provenance(
        component_frames.get("knowledge", pd.DataFrame())
    )
    rows = []
    for learner_id in sorted({str(value) for value in learner_ids}):
        snapshot = str(as_of or latest_timestamps.get(learner_id) or "latest")
        statuses = {
            component: by_learner.get(
                learner_id,
                (
                    "unavailable"
                    if component in KNOWLEDGE_COMPONENTS
                    else "no_observations"
                ),
            )
            for component, by_learner in component_statuses.items()
        }
        usable = [c for c, status in statuses.items() if status in USABLE_STATUSES]
        phase, model = provenance.get(learner_id, (None, None))
        rows.append(
            {
                "learner_id": learner_id,
                "profile_id": f"{population}:{learner_id}:{snapshot}",
                "profile_version": PROFILE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "as_of": snapshot,
                "source_population": population,
                "knowledge_status": statuses.get("knowledge", "unavailable"),
                "knowledge_phase": phase,
                "knowledge_model": model,
                "skills_status": statuses.get("skills", "unavailable"),
                "weak_areas_status": statuses.get("weak_areas", "unavailable"),
                "behavior_status": statuses.get("behavior", "no_observations"),
                "engagement_status": statuses.get("engagement", "no_observations"),
                "history_status": statuses.get("history", "no_observations"),
                "profile_completeness": (
                    ",".join(usable) if usable else "no_components_available"
                ),
            }
        )
    profiles = pd.DataFrame(rows, columns=PROFILE_COLUMNS)
    profiles["knowledge_phase"] = profiles["knowledge_phase"].astype("Int64")
    if profiles.duplicated(["source_population", "learner_id", "as_of"]).any():
        raise ValueError("Duplicate learner profiles found for a snapshot")
    return profiles
