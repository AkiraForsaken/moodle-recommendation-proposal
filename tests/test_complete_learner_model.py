import json

import pandas as pd
import pytest

from learner_model.builder import (
    ENGLISH_POPULATION,
    MOODLE_POPULATION,
    build_complete_learner_model,
    normalize_as_of,
)
from learner_model.engagement import build_engagement
from learner_model.history import build_history
from learner_model.knowledge import load_knowledge_states
from learner_model.skills import build_taxonomy, summarize_skills
from learner_model.weak_areas import identify_weak_areas
from tests.demo_data import write_collector_csvs


def _knowledge_rows():
    return pd.DataFrame(
        [
            {
                "learner_id": "L0001",
                "split": "train",
                "kc_id": "KC001",
                "kc_dimension": "grammar",
                "attempt_count": 4,
                "observed_accuracy": 0.25,
                "predicted_correctness": 0.50,
                "last_attempt_at": "2023-10-14T09:30:14+00:00",
                "model_name": "akt",
                "model_version": "test-version",
                "checkpoint_id": "test-checkpoint",
                "checkpoint_sha256": "abc123",
                "aggregation_method": "test-mean",
                "evidence_count": 4,
                "prediction_question_count": 5,
                "state_status": "estimated",
                "inference_timestamp": "2026-11-08T08:40:57+00:00",
            },
            {
                "learner_id": "L0001",
                "split": "train",
                "kc_id": "KC002",
                "kc_dimension": "grammar",
                "attempt_count": 0,
                "observed_accuracy": None,
                "predicted_correctness": 0.45,
                "last_attempt_at": None,
                "model_name": "akt",
                "model_version": "test-version",
                "checkpoint_id": "test-checkpoint",
                "checkpoint_sha256": "abc123",
                "aggregation_method": "test-mean",
                "evidence_count": 0,
                "prediction_question_count": 5,
                "state_status": "cold_start",
                "inference_timestamp": "2026-11-08T08:40:57+00:00",
            },
        ]
    )


def test_knowledge_cutoff_retains_cold_start_and_inclusive_boundary(tmp_path):
    source = tmp_path / "states.csv"
    states = _knowledge_rows()
    states.loc[0, "last_attempt_at"] = "2024-01-01T00:00:00+00:00"
    states.to_csv(source, index=False)

    loaded = load_knowledge_states(source, as_of="2024-01-01T00:00:00+00:00")

    # Cold-start KC002 has no attempts, hence no timestamp, and must survive.
    assert loaded["kc_id"].tolist() == ["KC001", "KC002"]
    assert loaded.attrs["dropped_after_cutoff"] == 0
    assert loaded.attrs["dropped_without_timestamp"] == 0
    assert loaded.loc[1, "observed_accuracy"] != loaded.loc[1, "observed_accuracy"]  # NaN


def test_knowledge_cutoff_rejects_states_computed_from_later_events(tmp_path):
    source = tmp_path / "states.csv"
    _knowledge_rows().to_csv(source, index=False)  # last attempt 2023-10-14
    with pytest.raises(ValueError, match="leak the future"):
        load_knowledge_states(source, as_of="2023-10-01T00:00:00+00:00")


def test_knowledge_cutoff_uses_m3_snapshot_not_wall_clock_inference_time(tmp_path):
    """Regression: a valid M3 re-run (as_of column set) was rejected because the
    wall-clock inference_timestamp (2026) post-dates any data-time cutoff."""
    source = tmp_path / "states.csv"
    states = _knowledge_rows()
    states["as_of"] = "2023-12-01T00:00:00+00:00"
    states.to_csv(source, index=False)

    loaded = load_knowledge_states(source, as_of="2023-12-01T00:00:00+00:00")
    assert len(loaded) == 2
    assert loaded.attrs["knowledge_snapshot_as_of"] == "2023-12-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="leak the future"):
        load_knowledge_states(source, as_of="2023-11-30T00:00:00+00:00")


def test_knowledge_cutoff_excludes_rows_with_attempts_but_no_timestamp(tmp_path):
    source = tmp_path / "states.csv"
    states = _knowledge_rows()
    states.loc[0, "last_attempt_at"] = None
    states.to_csv(source, index=False)
    loaded = load_knowledge_states(source, as_of="2024-01-01T00:00:00+00:00")
    assert loaded["kc_id"].tolist() == ["KC002"]
    assert loaded.attrs["dropped_without_timestamp"] == 1


def test_draft_taxonomy_and_weak_area_preserve_review_and_probability_semantics():
    concept_map = pd.DataFrame(
        {
            "kc_id": ["KC001", "KC002"],
            "kc_dimension": ["grammar", "grammar"],
            "concepts": [0, 1],
        }
    )
    taxonomy, mapping = build_taxonomy(concept_map)
    states = _knowledge_rows().iloc[[0]].copy()
    skills = summarize_skills(states, mapping)
    weak = identify_weak_areas(states, generated_at="2026-11-09T00:00:00+00:00")

    assert taxonomy["taxonomy_status"].eq("needs_review").all()
    assert mapping["topic"].isna().all()
    assert mapping["mapping_status"].eq("needs_review").all()
    assert skills.iloc[0]["state_status"] == "needs_review"
    assert weak.iloc[0]["concern_status"] == "evidence_based_concern"
    assert weak.iloc[0]["predicted_correctness"] == pytest.approx(0.5)
    assert "not a mastery estimate" not in weak.iloc[0]["reason"]


def test_complete_build_keeps_english_and_moodle_populations_separate(tmp_path):
    data_dir = write_collector_csvs(tmp_path / "data")
    states_path = tmp_path / "knowledge_states.csv"
    concept_map_path = tmp_path / "concept_map.csv"
    output_dir = tmp_path / "out"
    _knowledge_rows().to_csv(states_path, index=False)
    pd.DataFrame(
        {
            "kc_id": ["KC001", "KC002"],
            "concepts": [0, 1],
            "kc_dimension": ["grammar", "grammar"],
        }
    ).to_csv(concept_map_path, index=False)

    outputs = build_complete_learner_model(
        data_dir=data_dir,
        output_dir=output_dir,
        knowledge_state_file=states_path,
        concept_map_file=concept_map_path,
        mappings_dir=tmp_path / "mappings",
    )

    profiles = outputs["learner_profile"]
    english = profiles[profiles["source_population"] == ENGLISH_POPULATION]
    moodle = profiles[profiles["source_population"] == MOODLE_POPULATION]
    assert english["learner_id"].tolist() == ["L0001"]
    assert len(moodle) == 11
    assert english.iloc[0]["knowledge_status"] == "available"
    assert english.iloc[0]["behavior_status"] == "unavailable"
    assert moodle["knowledge_status"].eq("unavailable").all()
    assert moodle["behavior_status"].eq("available").all()
    # Completion tracking is absent, so engagement must stay "partial" in the
    # profile rather than being flattened to "available".
    assert moodle["engagement_status"].eq("partial").all()
    assert "learner_topic_state" not in outputs
    assert outputs["learner_kc_state"]["source_population"].eq(
        ENGLISH_POPULATION
    ).all()
    assert outputs["learner_engagement"]["completed_activity_count"].isna().all()
    assert outputs["learner_engagement"]["topic_completion_rate"].isna().all()
    assert outputs["learner_history"]["total_completed_activities"].isna().all()

    topic_mapping = pd.read_csv(tmp_path / "mappings" / "kc_topic_map.csv")
    manifest = json.loads((output_dir / "build_manifest.json").read_text())
    assert topic_mapping["topic"].isna().all()
    assert manifest["mapping"]["kc_topic_mapping_status"] == "needs_review"
    assert "predicted_correctness" in manifest["knowledge_semantics"]
    assert (output_dir / "validation_report.md").is_file()


def test_complete_build_attaches_moodle_kt_and_roster_cold_start_profiles(tmp_path):
    data_dir = write_collector_csvs(tmp_path / "data")
    roster = pd.DataFrame(
        {
            "learner_id": ["99"],
            "course_id": [2],
            "enrol_start_unix": [0],
            "enrol_end_unix": [0],
        }
    )
    roster.to_csv(data_dir / "learner_roster.csv", index=False)
    states_path = tmp_path / "english_states.csv"
    _knowledge_rows().to_csv(states_path, index=False)
    moodle_states_path = tmp_path / "moodle_states.csv"
    moodle_states = _knowledge_rows().iloc[[1]].copy()
    moodle_states["learner_id"] = "99"
    moodle_states["mapping_version"] = "moodle-map-1"
    moodle_states.to_csv(moodle_states_path, index=False)
    concept_map_path = tmp_path / "concept_map.csv"
    pd.DataFrame(
        {
            "kc_id": ["KC001", "KC002"],
            "concepts": [0, 1],
            "kc_dimension": ["grammar", "grammar"],
        }
    ).to_csv(concept_map_path, index=False)

    outputs = build_complete_learner_model(
        data_dir=data_dir,
        output_dir=tmp_path / "out",
        knowledge_state_file=states_path,
        moodle_knowledge_state_file=moodle_states_path,
        concept_map_file=concept_map_path,
        mappings_dir=tmp_path / "mappings",
    )

    moodle_profile = outputs["learner_profile"].query(
        "source_population == @MOODLE_POPULATION and learner_id == '99'"
    ).iloc[0]
    assert moodle_profile["knowledge_status"] == "cold_start_only"
    assert moodle_profile["behavior_status"] == "unavailable"
    moodle_kc = outputs["learner_kc_state"].query(
        "source_population == @MOODLE_POPULATION and learner_id == '99'"
    )
    assert moodle_kc["state_status"].eq("cold_start").all()
    assert moodle_kc["observed_accuracy"].isna().all()
    assert moodle_kc["predicted_correctness"].notna().all()
    manifest = json.loads((tmp_path / "out" / "build_manifest.json").read_text())
    assert manifest["moodle_knowledge_provenance"]["mapping_versions"] == [
        "moodle-map-1"
    ]


def test_engagement_summarizes_collector_rows_without_legacy_topic_state():
    events = pd.DataFrame(
        [
            {
                "learner_id": "1",
                "timestamp_unix": 100,
                "event_category": "content_view",
                "is_learning_event": 1,
                "session_id": "s1",
                "seconds_since_prev_event": 100,
            },
            {
                "learner_id": "1",
                "timestamp_unix": 200,
                "event_category": "question_interaction",
                "is_learning_event": 1,
                "session_id": "s1",
                "seconds_since_prev_event": 60,
            },
            {
                "learner_id": "1",
                "timestamp_unix": 2300,
                "event_category": "quiz_attempt",
                "is_learning_event": 1,
                "session_id": "s1",
                "seconds_since_prev_event": 2100,
            },
        ]
    )
    quizzes = pd.DataFrame(
        [{"learner_id": "1", "finish_time_unix": 240, "duration_seconds": 45}]
    )
    engagement = build_engagement(
        events,
        quizzes,
        pd.DataFrame(),
        pd.DataFrame(columns=["course_module_id", "completion_tracking"]),
        generated_at="2026-11-09T00:00:00+00:00",
        completion_tracking_available=False,
    )

    row = engagement.iloc[0]
    assert row["total_activity_views"] == 1
    assert row["estimated_learning_time_seconds"] == 205



# ---------------------------------------------------------------- weak areas
@pytest.mark.parametrize(
    "attempts, observed, predicted, expected",
    [
        (4, 0.25, 0.90, "evidence_based_concern"),   # evidence wins over high prediction
        (4, 0.90, 0.20, "no_current_concern"),       # good evidence overrides low prediction
        (0, None, 0.30, "prediction_based_concern"),  # cold-start, low prediction
        (0, None, 0.95, "insufficient_evidence"),     # cold-start, high prediction != mastery
        (2, 0.0, 0.90, "insufficient_evidence"),      # 0/2 correct is not "mastered"
        (2, 0.0, 0.30, "prediction_based_concern"),
        (0, None, 0.70, "insufficient_evidence"),
    ],
)
def test_weak_area_rules(attempts, observed, predicted, expected):
    states = pd.DataFrame(
        [
            {
                "learner_id": "L1",
                "kc_id": "KC1",
                "attempt_count": attempts,
                "evidence_count": attempts,
                "observed_accuracy": observed,
                "predicted_correctness": predicted,
            }
        ]
    )
    row = identify_weak_areas(states, "2026-11-09T00:00:00+00:00").iloc[0]
    assert row["concern_status"] == expected
    assert row["reason"]
    if attempts == 0:
        assert row["concern_status"] != "evidence_based_concern"


# ------------------------------------------------------------------- mapping
def _mapping_frame(version="1.0.0-reviewed", status="mapped"):
    return pd.DataFrame(
        {
            "kc_id": ["KC001", "KC002"],
            "kc_dimension": ["grammar", "grammar"],
            "skill_id": ["skill:tenses", "skill:tenses"],
            "skill_name": ["Verb tenses", "Verb tenses"],
            "topic": ["Tenses", None],
            "mapping_status": [status, status],
            "mapping_source": ["supervisor review", "supervisor review"],
            "mapping_version": [version, version],
        }
    )


def _build(tmp_path, **kwargs):
    data_dir = write_collector_csvs(tmp_path / "data")
    states_path = tmp_path / "knowledge_states.csv"
    concept_map_path = tmp_path / "concept_map.csv"
    _knowledge_rows().to_csv(states_path, index=False)
    pd.DataFrame(
        {"kc_id": ["KC001", "KC002"], "concepts": [0, 1],
         "kc_dimension": ["grammar", "grammar"]}
    ).to_csv(concept_map_path, index=False)
    return build_complete_learner_model(
        data_dir=data_dir,
        output_dir=tmp_path / "out",
        knowledge_state_file=states_path,
        concept_map_file=concept_map_path,
        mappings_dir=tmp_path / "mappings",
        **kwargs,
    )


def test_reviewed_mapping_is_loaded_and_never_overwritten(tmp_path):
    (tmp_path / "mappings").mkdir()
    mapping_path = tmp_path / "mappings" / "kc_topic_map.csv"
    _mapping_frame().to_csv(mapping_path, index=False)
    pd.DataFrame(
        {"taxonomy_status": ["reviewed"]}
    ).to_csv(tmp_path / "mappings" / "skill_taxonomy.csv", index=False)
    before = mapping_path.read_bytes()

    outputs = _build(tmp_path)

    skills = outputs["learner_skill_summary"]
    assert skills["skill_id"].tolist() == ["skill:tenses"]
    assert skills["mapping_version"].eq("1.0.0-reviewed").all()
    assert skills["state_status"].eq("mapped").all()
    assert mapping_path.read_bytes() == before
    manifest = json.loads((tmp_path / "out" / "build_manifest.json").read_text())
    assert manifest["mapping"]["version"] == "1.0.0-reviewed"
    assert manifest["mapping"]["origin"] == "existing_file"
    assert manifest["mapping"]["skill_taxonomy_status"] == "reviewed"
    assert outputs["weak_areas"]["mapping_version"].eq("1.0.0-reviewed").all()


def test_invalid_reviewed_mapping_is_rejected(tmp_path):
    (tmp_path / "mappings").mkdir()
    bad = _mapping_frame(status="reviewed-ish")
    bad.to_csv(tmp_path / "mappings" / "kc_topic_map.csv", index=False)
    with pytest.raises(ValueError, match="invalid mapping_status"):
        _build(tmp_path)


def test_unmapped_kcs_get_no_skill_summary(tmp_path):
    (tmp_path / "mappings").mkdir()
    mapping = _mapping_frame()
    mapping.loc[1, ["skill_id", "skill_name", "mapping_status"]] = [None, None, "unmapped"]
    mapping.to_csv(tmp_path / "mappings" / "kc_topic_map.csv", index=False)
    outputs = _build(tmp_path)
    assert outputs["learner_skill_summary"]["kc_count"].tolist() == [1]
    assert len(outputs["learner_kc_state"]) == 2  # KC rows are never dropped


# ------------------------------------------------------- profile / snapshots
def test_profile_reports_cold_start_only_knowledge(tmp_path):
    data_dir = write_collector_csvs(tmp_path / "data")
    states_path = tmp_path / "knowledge_states.csv"
    states = _knowledge_rows().iloc[[1]]  # only the cold-start KC
    states.to_csv(states_path, index=False)
    pd.DataFrame(
        {"kc_id": ["KC001", "KC002"], "concepts": [0, 1],
         "kc_dimension": ["grammar", "grammar"]}
    ).to_csv(tmp_path / "cm.csv", index=False)
    outputs = build_complete_learner_model(
        data_dir=data_dir, output_dir=tmp_path / "out",
        knowledge_state_file=states_path, concept_map_file=tmp_path / "cm.csv",
        mappings_dir=tmp_path / "mappings",
    )
    english = outputs["learner_profile"].query("source_population == @ENGLISH_POPULATION")
    assert english.iloc[0]["knowledge_status"] == "cold_start_only"


def test_rebuild_with_pinned_timestamp_is_byte_identical(tmp_path):
    stamp = "2026-11-10T00:00:00+00:00"
    _build(tmp_path / "a", generated_at=stamp)
    _build(tmp_path / "b", generated_at=stamp)
    for name in ("learner_profile", "learner_kc_state", "weak_areas",
                 "learner_engagement", "learner_history", "learner_behavior"):
        assert (tmp_path / "a" / "out" / f"{name}.csv").read_bytes() == (
            tmp_path / "b" / "out" / f"{name}.csv"
        ).read_bytes()


def test_build_with_cutoff_keeps_cold_start_and_normalizes_as_of(tmp_path):
    outputs = _build(tmp_path, as_of="2026-12-01T00:00:00Z")
    kc = outputs["learner_kc_state"]
    assert set(kc["state_status"]) == {"estimated", "cold_start"}
    assert outputs["learner_profile"]["as_of"].eq("2026-12-01T00:00:00+00:00").all()


def test_naive_as_of_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_as_of("2026-12-01T00:00:00")
    assert normalize_as_of("1790000000")[1] == 1790000000


def test_history_uses_unix_only_completion_times():
    events = pd.DataFrame()
    quizzes = pd.DataFrame()
    completions = pd.DataFrame(
        {"learner_id": [1], "course_module_id": [10], "completion_state": [1],
         "time_modified_unix": [1_790_000_000]}
    )
    activities = pd.DataFrame({"course_module_id": [10], "topic": ["T"]})
    row = build_history(events, quizzes, completions, activities, "g").iloc[0]
    assert row["first_activity_at"] is not None
    assert row["latest_activity_type"] == "completion"
    assert row["historical_session_count"] is None  # unknown, not 0


def test_per_population_cutoffs(tmp_path):
    outputs = _build(
        tmp_path,
        knowledge_as_of="2023-12-01T00:00:00Z",
        moodle_as_of="2027-01-01T00:00:00Z",
    )
    profiles = outputs["learner_profile"]
    english = profiles[profiles["source_population"] == ENGLISH_POPULATION]
    moodle = profiles[profiles["source_population"] == MOODLE_POPULATION]
    assert english["as_of"].eq("2023-12-01T00:00:00+00:00").all()
    assert moodle["as_of"].eq("2027-01-01T00:00:00+00:00").all()
    assert len(outputs["learner_kc_state"]) == 2
    assert not outputs["learner_history"].empty
