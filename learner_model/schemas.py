"""Column contracts for the learner-model outputs."""

PROFILE_COLUMNS = [
    "learner_id",
    "profile_id",
    "profile_version",
    "generated_at",
    "as_of",
    "source_population",
    "knowledge_status",
    "knowledge_phase",
    "knowledge_model",
    "skills_status",
    "weak_areas_status",
    "behavior_status",
    "engagement_status",
    "history_status",
    "profile_completeness",
]

SKILL_SUMMARY_COLUMNS = [
    "learner_id",
    "skill_id",
    "skill_name",
    "kc_dimension",
    "kc_count",
    "estimated_kc_count",
    "cold_start_kc_count",
    "mean_predicted_correctness",
    "mean_observed_accuracy",
    "total_attempt_count",
    "mapping_version",
    "state_status",
]

WEAK_AREA_COLUMNS = [
    "learner_id",
    "entity_type",
    "entity_id",
    "entity_name",
    "concern_status",
    "observed_accuracy",
    "predicted_correctness",
    "evidence_count",
    "reason",
    "mapping_version",
    "generated_at",
]

KC_SKILL_MAP_COLUMNS = [
    "kc_id",
    "kc_name",
    "kc_dimension",
    "skill_id",
    "skill_name",
    "topic",
    "mapping_status",
    "mapping_source",
    "mapping_version",
]
