"""Assemble population-isolated top-level learner profiles."""

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
    return {learner: "available" for learner in ids.unique()}


def build_profiles(
    population_learners,
    component_frames,
    generated_at,
    as_of=None,
    latest_timestamps=None,
):
    """Build one profile per source population and learner.

    Component outputs are never joined across populations; the population is
    included in the stable profile key to prevent coincident IDs colliding.
    Each ``*_status`` reports the component's real status (available, partial,
    cold_start_only, no_observations, unavailable), not just its presence.
    """
    rows = []
    latest_timestamps = latest_timestamps or {}
    for population, learner_ids in population_learners.items():
        component_statuses = {}
        for component, frame in component_frames.items():
            if not frame.empty and "source_population" in frame:
                frame = frame[frame["source_population"] == population]
            component_statuses[component] = _status_by_learner(component, frame)
        for learner_id in sorted({str(value) for value in learner_ids}):
            learner_as_of = as_of.get(population) if isinstance(as_of, dict) else as_of
            if learner_as_of is None:
                learner_as_of = (
                    latest_timestamps.get(population, {}).get(learner_id) or "latest"
                )
            statuses = {
                component: by_learner.get(learner_id, "unavailable")
                for component, by_learner in component_statuses.items()
            }
            usable = [c for c, status in statuses.items() if status in USABLE_STATUSES]
            profile_snapshot = str(learner_as_of)
            rows.append(
                {
                    "learner_id": learner_id,
                    "profile_id": f"{population}:{learner_id}:{profile_snapshot}",
                    "profile_version": PROFILE_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "as_of": profile_snapshot,
                    "source_population": population,
                    "knowledge_status": statuses.get("knowledge", "unavailable"),
                    "skills_status": statuses.get("skills", "unavailable"),
                    "weak_areas_status": statuses.get("weak_areas", "unavailable"),
                    "behavior_status": statuses.get("behavior", "unavailable"),
                    "engagement_status": statuses.get("engagement", "unavailable"),
                    "history_status": statuses.get("history", "unavailable"),
                    "profile_completeness": ",".join(usable)
                    if usable
                    else "no_components_available",
                }
            )
    profiles = pd.DataFrame(rows, columns=PROFILE_COLUMNS)
    if not profiles.empty and profiles.duplicated(
        ["source_population", "learner_id", "as_of"]
    ).any():
        raise ValueError("Duplicate learner profiles found for a population snapshot")
    return profiles
