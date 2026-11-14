import json

import pandas as pd
import pytest

from learner_model.builder import (
    MOODLE_POPULATION,
    SIMULATED_POPULATION,
    build_complete_learner_model,
    normalize_as_of,
)
from learner_model.engagement import build_engagement
from learner_model.history import build_history
from learner_model.knowledge import load_knowledge_states
from learner_model.skills import build_kc_skill_map, summarize_skills
from learner_model.weak_areas import identify_weak_areas
from tests.demo_data import write_collector_csvs


def _knowledge_rows(population=MOODLE_POPULATION):
    """Two KC states of one learner, in the knowledge-engine contract."""
    shared = {
        "learner_id": "L0001",
        "kc_dimension": "grammar",
        "model_name": "bkt_item_level",
        "model_version": "test-version",
        "estimator_phase": 1,
        "checkpoint_id": None,
        "checkpoint_sha256": None,
        "aggregation_method": "test-mean",
        "prediction_question_count": None,
        "inference_timestamp": "2026-11-08T08:40:57+00:00",
        "mapping_version": "map-1",
        "source_population": population,
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "kc_id": "KC001",
                "attempt_count": 4,
                "evidence_count": 4,
                "observed_accuracy": 0.25,
                "predicted_correctness": 0.50,
                "p_mastery": 0.40,
                "last_attempt_at": "2023-10-14T09:30:14+00:00",
                "state_status": "estimated",
            },
            {
                **shared,
                "kc_id": "KC002",
                "attempt_count": 0,
                "evidence_count": 0,
                "observed_accuracy": None,
                "predicted_correctness": 0.45,
                "p_mastery": 0.30,
                "last_attempt_at": None,
                "state_status": "cold_start",
            },
        ]
    )


def _course_files(data_dir):
    """The KC catalog and topic links the collector exports for a course."""
    pd.DataFrame(
        {
            "kc_id": ["KC001", "KC002"],
            "kc_name": ["Present simple -s", "Negation"],
            "kc_dimension": ["Grammar", "grammar"],
            "learning_objective": ["Apply -s/-es.", "Form negatives."],
        }
    ).to_csv(data_dir / "kc_catalog.csv", index=False)
    pd.DataFrame(
        {"kc_id": ["KC001"], "topic": ["Unit 1"], "mapping_version": ["map-1"]}
    ).to_csv(data_dir / "kc_topic_map.csv", index=False)


def _build(tmp_path, states=None, with_course=True, **kwargs):
    data_dir = write_collector_csvs(tmp_path / "data")
    if with_course:
        _course_files(data_dir)
    states_path = tmp_path / "learner_kc_state.csv"
    (_knowledge_rows() if states is None else states).to_csv(states_path, index=False)
    return build_complete_learner_model(
        data_dir=data_dir,
        output_dir=tmp_path / "out",
        knowledge_state_file=states_path,
        **kwargs,
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


def test_knowledge_cutoff_uses_snapshot_not_wall_clock_inference_time(tmp_path):
    """Regression: a valid re-run (as_of column set) was rejected because the
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


def test_skills_group_kcs_by_dimension_within_topic():
    catalog = pd.DataFrame(
        {
            "kc_id": ["KC001", "KC002", "KC003"],
            "kc_name": ["A", "B", "C"],
            "kc_dimension": ["grammar", "grammar", "vocabulary"],
        }
    )
    topics = pd.DataFrame({"kc_id": ["KC001", "KC001", "KC003"], "topic": ["Unit 1", "Unit 2", "Unit 1"]})

    mapping = build_kc_skill_map(catalog, topics, "map-1")

    # A KC used in two topics belongs to two skills; one without a topic falls
    # back to its dimension.
    assert mapping[["kc_id", "skill_id"]].values.tolist() == [
        ["KC001", "grammar|Unit 1"],
        ["KC001", "grammar|Unit 2"],
        ["KC002", "dimension:grammar"],
        ["KC003", "vocabulary|Unit 1"],
    ]
    assert mapping.loc[0, "skill_name"] == "Grammar: Unit 1"
    assert mapping["mapping_status"].eq("mapped").all()

    skills = summarize_skills(_knowledge_rows(), mapping, "map-1")
    assert skills["skill_id"].tolist() == ["dimension:grammar", "grammar|Unit 1", "grammar|Unit 2"]
    assert skills["kc_count"].tolist() == [1, 1, 1]
    assert skills["mapping_version"].eq("map-1").all()


def test_profiles_build_without_any_knowledge_states(tmp_path):
    data_dir = write_collector_csvs(tmp_path / "data")

    outputs = build_complete_learner_model(
        data_dir=data_dir,
        output_dir=tmp_path / "out",
        knowledge_state_file=tmp_path / "no_such_file.csv",
    )

    profiles = outputs["learner_profile"]
    assert len(profiles) == 11
    assert profiles["source_population"].eq(MOODLE_POPULATION).all()
    assert profiles["knowledge_status"].eq("unavailable").all()
    assert profiles["knowledge_phase"].isna().all()
    assert profiles["behavior_status"].eq("available").all()
    # Completion tracking is absent, so engagement must stay "partial" in the
    # profile rather than being flattened to "available".
    assert profiles["engagement_status"].eq("partial").all()
    assert outputs["learner_kc_state"].empty and outputs["weak_areas"].empty
    assert outputs["learner_engagement"]["completed_activity_count"].isna().all()
    assert outputs["learner_history"]["total_completed_activities"].isna().all()
    manifest = json.loads((tmp_path / "out" / "build_manifest.json").read_text())
    assert manifest["knowledge_provenance"] is None
    assert manifest["components"] == ["behavior", "engagement", "history"]
    assert (tmp_path / "out" / "validation_report.md").is_file()


def test_build_joins_knowledge_states_to_the_course_catalog(tmp_path):
    outputs = _build(tmp_path)

    profile = outputs["learner_profile"].query("learner_id == 'L0001'").iloc[0]
    assert profile["knowledge_status"] == "available"
    assert profile["knowledge_phase"] == 1
    assert profile["knowledge_model"] == "bkt_item_level"
    # L0001 has knowledge states but no Moodle activity rows.
    assert profile["behavior_status"] == "no_observations"
    kc = outputs["learner_kc_state"]
    assert kc["kc_name"].tolist() == ["Present simple -s", "Negation"]
    assert kc["p_mastery"].tolist() == [0.40, 0.30]
    weak = outputs["weak_areas"].set_index("entity_id")
    assert weak.loc["KC001", "entity_name"] == "Present simple -s"
    assert weak.loc["KC001", "concern_status"] == "evidence_based_concern"
    assert weak["mapping_version"].eq("map-1").all()
    skills = outputs["learner_skill_summary"]
    assert skills["skill_id"].tolist() == ["dimension:grammar", "grammar|Unit 1"]
    manifest = json.loads((tmp_path / "out" / "build_manifest.json").read_text())
    assert manifest["knowledge_provenance"]["estimator_phases"] == [1]
    assert manifest["mapping"]["version"] == "map-1"
    assert "kc_catalog" in manifest["inputs"]
    skill_map = pd.read_csv(tmp_path / "out" / "kc_skill_map.csv")
    assert skill_map["mapping_status"].eq("mapped").all()


def test_enrolled_learner_without_activity_gets_a_cold_start_profile(tmp_path):
    data_dir = write_collector_csvs(tmp_path / "data")
    _course_files(data_dir)
    pd.DataFrame(
        {"learner_id": ["99"], "course_id": [2], "enrol_start_unix": [0], "enrol_end_unix": [0]}
    ).to_csv(data_dir / "learner_roster.csv", index=False)
    states = pd.concat([_knowledge_rows().iloc[[1]]] * 2, ignore_index=True)
    states["learner_id"] = "99"
    states["kc_id"] = ["KC001", "KC002"]
    states_path = tmp_path / "learner_kc_state.csv"
    states.to_csv(states_path, index=False)

    outputs = build_complete_learner_model(
        data_dir=data_dir, output_dir=tmp_path / "out", knowledge_state_file=states_path
    )

    profile = outputs["learner_profile"].query("learner_id == '99'").iloc[0]
    # Priors only: nothing about this learner is individual evidence yet.
    assert profile["knowledge_status"] == "cold_start_only"
    assert profile["skills_status"] == "cold_start_only"
    assert profile["weak_areas_status"] == "cold_start_only"
    assert profile["behavior_status"] == "no_observations"
    kc = outputs["learner_kc_state"].query("learner_id == '99'")
    assert kc["state_status"].eq("cold_start").all()
    assert kc["observed_accuracy"].isna().all()
    assert kc["predicted_correctness"].notna().all()
    weak = outputs["weak_areas"].query("learner_id == '99'")
    assert weak["concern_status"].eq("insufficient_evidence").all()


def test_simulated_learners_are_labelled_and_never_joined_to_moodle_data(tmp_path):
    data_dir = write_collector_csvs(tmp_path / "data")
    states_path = tmp_path / "learner_kc_state.csv"
    _knowledge_rows(SIMULATED_POPULATION).to_csv(states_path, index=False)

    with pytest.raises(ValueError, match="refusing to combine"):
        build_complete_learner_model(
            data_dir=data_dir, output_dir=tmp_path / "out", knowledge_state_file=states_path
        )

    (data_dir / "simulation_manifest.json").write_text("{}", encoding="utf-8")
    outputs = build_complete_learner_model(
        data_dir=data_dir, output_dir=tmp_path / "out", knowledge_state_file=states_path
    )
    for frame in outputs.values():
        assert frame["source_population"].eq(SIMULATED_POPULATION).all()


def test_kcs_missing_from_the_catalog_stay_visible_but_join_no_skill(tmp_path):
    outputs = _build(tmp_path, with_course=False)

    assert outputs["learner_skill_summary"].empty
    assert len(outputs["learner_kc_state"]) == 2  # KC rows are never dropped
    skill_map = pd.read_csv(tmp_path / "out" / "kc_skill_map.csv")
    assert skill_map["mapping_status"].eq("unmapped").all()


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


def test_phase_one_cold_start_prior_is_not_flagged_as_a_concern():
    """The fixed prior is shared by every new learner; it says nothing about one."""
    state = {
        "learner_id": "L1",
        "kc_id": "KC1",
        "attempt_count": 0,
        "evidence_count": 0,
        "observed_accuracy": None,
        "predicted_correctness": 0.445,
    }
    prior = identify_weak_areas(pd.DataFrame([{**state, "estimator_phase": 1}]), "g").iloc[0]
    trained = identify_weak_areas(pd.DataFrame([{**state, "estimator_phase": 3}]), "g").iloc[0]

    assert prior["concern_status"] == "insufficient_evidence"
    assert "shared" in prior["reason"]
    assert trained["concern_status"] == "prediction_based_concern"


# ------------------------------------------------------- profile / snapshots
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


def test_revisit_means_returning_to_a_topic_in_a_later_session():
    from learner_model.behavior import build_behavior

    def event(learner, cmid, session, minute, category="content_view"):
        return {
            "learner_id": learner,
            "course_module_id": cmid,
            "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=minute),
            "event_category": category,
            "is_learning_event": 1,
            "session_id": session,
        }

    activities = pd.DataFrame(
        {"course_module_id": [1, 2, 3, 4], "topic": ["Unit 1", "Unit 1", "Unit 1", "Unit 2"]}
    )
    events = pd.DataFrame(
        [
            # Learner A: overview, lesson and quiz of Unit 1 in one visit -> no revisit.
            event("A", 1, "a1", 0), event("A", 2, "a1", 2), event("A", 3, "a1", 5, "quiz_attempt"),
            # Learner B: Unit 1 in two sessions (revisit), Unit 2 once.
            event("B", 2, "b1", 0), event("B", 4, "b1", 3), event("B", 3, "b2", 3000, "quiz_attempt"),
        ]
    )
    behavior = build_behavior(events, activities, "2026-01-02T00:00:00+00:00").set_index("learner_id")
    assert behavior.loc["A", "revisit_rate"] == 0.0
    assert behavior.loc["B", "revisit_rate"] == 0.5


def test_course_progress_counts_every_trackable_activity():
    events = pd.DataFrame(
        [{"learner_id": "1", "timestamp_unix": 100, "event_category": "content_view",
          "is_learning_event": 1, "session_id": "s1", "seconds_since_prev_event": 0}]
    )
    activities = pd.DataFrame({"course_module_id": [1, 2, 3, 4, 5], "completion_tracking": [2, 2, 2, 2, 0]})
    completions = pd.DataFrame(
        {"learner_id": ["1", "1"], "course_module_id": [1, 2], "completion_state": [2, 3]}
    )
    row = build_engagement(
        events, pd.DataFrame(), completions, activities,
        generated_at="2026-11-09T00:00:00+00:00",
    ).iloc[0]
    # complete_pass counts, complete_fail does not; 4 activities track completion.
    assert row["completed_activity_count"] == 1
    assert row["topic_completion_rate"] == 0.5
    assert row["course_progress"] == 0.25
