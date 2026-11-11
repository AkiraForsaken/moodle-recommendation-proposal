import sys
from contextlib import nullcontext
import json

import numpy as np
import pandas as pd
import pytest

import knowledge_tracing.export_states as export_module
import knowledge_tracing.inference as inference_module
from knowledge_tracing.export_states import (
    choose_representative_questions,
    load_answer_events,
    load_training_question_ids,
    make_skill_state_frame,
    normalize_as_of,
    training_supported_questions,
    validate_mappings,
    validate_states,
)


def test_representative_questions_are_evenly_spaced_and_deterministic():
    question_map = pd.DataFrame(
        {
            "item_id": [f"Q{i}" for i in range(5)],
            "questions": [0, 1, 2, 3, 4],
            "kc_id": ["KC1"] * 5,
            "concepts": [0] * 5,
            "kc_dimension": ["grammar"] * 5,
        }
    )

    selected = choose_representative_questions(question_map, 3)

    assert selected["questions"].tolist() == [0, 2, 4]
    pd.testing.assert_frame_equal(
        selected, choose_representative_questions(question_map, 3)
    )


def test_training_question_loader_ignores_validation_questions_and_padding(tmp_path):
    train_sequences = tmp_path / "train_valid_sequences.csv"
    pd.DataFrame(
        {
            "fold": [0, 1],
            "questions": ["1,3,-1,-1", "2,3,-1,-1"],
        }
    ).to_csv(train_sequences, index=False)

    question_ids = load_training_question_ids(
        {"dpath": str(tmp_path), "train_valid_file": train_sequences.name}
    )

    assert question_ids == {1, 3}


def test_moodle_inference_uses_only_training_supported_questions():
    question_map = pd.DataFrame(
        {
            "item_id": ["Q1", "Q2", "Q3"],
            "questions": [0, 1, 2],
            "kc_id": ["KC1", "KC1", "KC2"],
            "concepts": [0, 0, 1],
            "kc_dimension": ["grammar", "grammar", "reading"],
        }
    )
    concept_map = pd.DataFrame(
        {
            "kc_id": ["KC1", "KC2"],
            "concepts": [0, 1],
            "kc_dimension": ["grammar", "reading"],
        }
    )

    supported = training_supported_questions(question_map, concept_map, {0, 2})

    assert supported["item_id"].tolist() == ["Q1", "Q3"]
    with pytest.raises(ValueError, match="no mapped question observed"):
        training_supported_questions(question_map, concept_map, {0})


def test_answer_loader_filters_nonanswers_and_preserves_event_order(tmp_path):
    source = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "learner_id": "L1",
                "event_index": 2,
                "item_id": "Q2",
                "kc_id": "KC1",
                "kc_dimension": "grammar",
                "event_type": "answer",
                "timestamp": 200,
                "correct": 1,
            },
            {
                "learner_id": "L1",
                "event_index": 1,
                "item_id": "Q1",
                "kc_id": "KC1",
                "kc_dimension": "grammar",
                "event_type": "answer",
                "timestamp": 100,
                "correct": 0,
            },
            {
                "learner_id": "L1",
                "event_index": 3,
                "item_id": "Q3",
                "kc_id": "KC1",
                "kc_dimension": "grammar",
                "event_type": "answer_change",
                "timestamp": 300,
                "correct": None,
            },
        ]
    ).to_csv(source, index=False)

    events = load_answer_events(source)

    assert events["event_index"].tolist() == [1, 2]
    assert events["correct"].tolist() == [0, 1]


def test_answer_loader_applies_inclusive_timezone_aware_as_of_cutoff(tmp_path):
    source = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "learner_id": "L1",
                "event_index": index,
                "item_id": f"Q{index}",
                "kc_id": "KC1",
                "kc_dimension": "grammar",
                "event_type": "answer",
                "timestamp": timestamp,
                "correct": 1,
            }
            for index, timestamp in enumerate([100, 200, 300], start=1)
        ]
    ).to_csv(source, index=False)

    events = load_answer_events(source, "1970-01-01T00:03:20Z")

    assert events["event_index"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="timezone"):
        normalize_as_of("2026-11-08T00:00:00")


def test_skill_states_keep_observed_evidence_separate_from_cold_start():
    concept_map = pd.DataFrame(
        {
            "kc_id": ["KC1", "KC2"],
            "concepts": [0, 1],
            "kc_dimension": ["grammar", "reading"],
        }
    )
    events = pd.DataFrame(
        {
            "learner_id": ["L1", "L1"],
            "kc_id": ["KC1", "KC1"],
            "event_index": [1, 2],
            "correct": [1, 0],
            "timestamp": [200, 100],
        }
    )
    predictions = pd.DataFrame(
        {
            "learner_id": ["L1", "L1"],
            "kc_id": ["KC1", "KC2"],
            "predicted_correctness": [0.65, 0.4],
            "predicted_correctness_10q": [0.6, 0.45],
            "prediction_question_count": [3, 2],
            "sensitivity_question_count": [2, 1],
        }
    )
    metadata = {
        "model_name": "akt",
        "model_version": "akt-seed42",
        "checkpoint_id": "akt-seed42-qid-model",
        "checkpoint_sha256": "abc123",
        "aggregation_method": "arithmetic_mean",
    }

    states = make_skill_state_frame(
        ["L1"],
        pd.Series({"L1": "test"}),
        concept_map,
        events,
        predictions,
        metadata,
        "2026-11-07T00:00:00+00:00",
        "2026-11-07T00:00:00+00:00",
    ).set_index("kc_id")

    assert states.loc["KC1", "attempt_count"] == 2
    assert states.loc["KC1", "evidence_count"] == 2
    assert states.loc["KC1", "observed_accuracy"] == pytest.approx(0.5)
    assert states.loc["KC1", "last_attempt_at"] == "1970-01-01T00:01:40+00:00"
    assert states.loc["KC1", "state_status"] == "estimated"
    assert states.loc["KC1", "evidence_tier"] == "low"
    assert states.loc["KC1", "split"] == "test"
    assert states.loc["KC2", "attempt_count"] == 0
    assert pd.isna(states.loc["KC2", "observed_accuracy"])
    assert pd.isna(states.loc["KC2", "last_attempt_at"])
    assert states.loc["KC2", "state_status"] == "cold_start"
    assert states.loc["KC2", "evidence_tier"] == "none"
    assert states.loc["KC2", "predicted_correctness"] == pytest.approx(0.4)


def test_validation_detects_incorrect_attempt_counts():
    concept_map = pd.DataFrame(
        {"kc_id": ["KC1"], "concepts": [0], "kc_dimension": ["grammar"]}
    )
    events = pd.DataFrame(
        {
            "learner_id": ["L1"],
            "kc_id": ["KC1"],
            "event_index": [1],
            "correct": [1],
            "timestamp": [100],
        }
    )
    predictions = pd.DataFrame(
        {
            "learner_id": ["L1"],
            "kc_id": ["KC1"],
            "predicted_correctness": [0.8],
            "predicted_correctness_10q": [0.75],
            "prediction_question_count": [1],
            "sensitivity_question_count": [1],
        }
    )
    metadata = {
        "model_name": "akt",
        "model_version": "akt-seed42",
        "checkpoint_id": "akt-seed42-qid-model",
        "checkpoint_sha256": "abc123",
        "aggregation_method": "arithmetic_mean",
    }
    states = make_skill_state_frame(
        ["L1"],
        pd.Series({"L1": "test"}),
        concept_map,
        events,
        predictions,
        metadata,
        "2026-11-07T00:00:00+00:00",
        None,
    )
    assert validate_states(states, events, ["L1"], concept_map) == []
    states.loc[0, "attempt_count"] = 0

    issues = validate_states(states, events, ["L1"], concept_map)

    assert "Attempt counts do not match source answer events" in issues


def test_mapping_validation_checks_training_dimensions_and_indices():
    question_map = pd.DataFrame(
        {
            "item_id": ["Q1", "Q2"],
            "questions": [0, 1],
            "kc_id": ["KC1", "KC2"],
            "concepts": [0, 1],
            "kc_dimension": ["grammar", "reading"],
        }
    )
    concept_map = pd.DataFrame(
        {
            "kc_id": ["KC1", "KC2"],
            "concepts": [0, 1],
            "kc_dimension": ["grammar", "reading"],
        }
    )

    validate_mappings(question_map, concept_map, {"num_q": 2, "num_c": 2})

    with pytest.raises(ValueError, match="contiguous"):
        validate_mappings(
            question_map.assign(questions=[1, 2]),
            concept_map,
            {"num_q": 2, "num_c": 2},
        )


def test_validation_detects_observed_accuracy_that_disagrees_with_events():
    concept_map = pd.DataFrame(
        {"kc_id": ["KC1"], "concepts": [0], "kc_dimension": ["grammar"]}
    )
    events = pd.DataFrame(
        {
            "learner_id": ["L1", "L1"],
            "kc_id": ["KC1", "KC1"],
            "event_index": [1, 2],
            "correct": [1, 0],
            "timestamp": [100, 200],
        }
    )
    predictions = pd.DataFrame(
        {
            "learner_id": ["L1"],
            "kc_id": ["KC1"],
            "predicted_correctness": [0.6],
            "predicted_correctness_10q": [0.61],
            "prediction_question_count": [1],
            "sensitivity_question_count": [1],
        }
    )
    metadata = {
        "model_name": "akt",
        "model_version": "akt-seed42",
        "checkpoint_id": "akt-seed42-qid-model",
        "checkpoint_sha256": "abc123",
        "aggregation_method": "arithmetic_mean",
    }
    states = make_skill_state_frame(
        ["L1"],
        pd.Series({"L1": "test"}),
        concept_map,
        events,
        predictions,
        metadata,
        "2026-11-07T00:00:00+00:00",
        None,
    )
    states.loc[0, "observed_accuracy"] = 1.0

    issues = validate_states(states, events, ["L1"], concept_map)

    assert "Observed accuracy does not match source answer events" in issues


def test_export_states_writes_validated_dataset_and_manifest(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame(
        {
            "item_id": ["Q1", "Q2", "Q3"],
            "questions": [0, 1, 2],
            "kc_id": ["KC1", "KC1", "KC2"],
            "concepts": [0, 0, 1],
            "kc_dimension": ["grammar", "grammar", "reading"],
        }
    ).to_csv(processed_dir / "question_map.csv", index=False)
    pd.DataFrame(
        {
            "kc_id": ["KC1", "KC2"],
            "concepts": [0, 1],
            "kc_dimension": ["grammar", "reading"],
        }
    ).to_csv(processed_dir / "concept_map.csv", index=False)
    pd.DataFrame(
        {"learner_id": ["L1", "L2"], "split": ["test", "train"]}
    ).to_csv(
        processed_dir / "split_assignments.csv", index=False
    )
    roster_path = tmp_path / "roster.csv"
    pd.DataFrame(
        {
            "learner_id": ["L3"],
            "enrol_start_unix": [0],
            "enrol_end_unix": [0],
        }
    ).to_csv(roster_path, index=False)
    input_path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "learner_id": "L1",
                "event_index": 1,
                "item_id": "Q1",
                "kc_id": "KC1",
                "kc_dimension": "grammar",
                "event_type": "answer",
                "timestamp": 100,
                "correct": 1,
            },
            {
                "learner_id": "L1",
                "event_index": 2,
                "item_id": "Q2",
                "kc_id": "KC1",
                "kc_dimension": "grammar",
                "event_type": "answer",
                "timestamp": 200,
                "correct": 0,
            },
        ]
    ).to_csv(input_path, index=False)

    monkeypatch.setattr(export_module, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(
        export_module,
        "load_data_config",
        lambda _dataset="english_kt": {
            "num_q": 3,
            "num_c": 2,
            "maxlen": 200,
        },
    )
    monkeypatch.setattr(
        export_module,
        "load_selected_model",
        lambda *_: (
            object(),
            "cpu",
            {
                "model_name": "akt",
                "model_version": "akt-seed42",
                "checkpoint_id": "akt-seed42-qid-model",
                "aggregation_method": "arithmetic_mean_of_all_mapped_question_probabilities",
            },
            "abc123",
        ),
    )
    monkeypatch.setattr(
        export_module,
        "score_akt_questions",
        lambda **_: [0.5, 0.3, 0.4],
    )

    states = export_module.export_states(
        input_path=input_path,
        output_dir=tmp_path / "outputs",
        max_questions_per_kc=0,
        as_of="1970-01-01T00:03:20Z",
        show_progress=False,
        learner_roster_path=roster_path,
    )

    assert len(states) == 6
    assert (
        states.set_index(["learner_id", "kc_id"])
        .loc[("L1", "KC2"), "state_status"]
        == "cold_start"
    )
    cold_learner = states[states["learner_id"].eq("L2")]
    assert cold_learner["state_status"].eq("cold_start").all()
    assert cold_learner["observed_accuracy"].isna().all()
    assert cold_learner["predicted_correctness"].notna().all()
    assert cold_learner["split"].eq("train").all()
    roster_learner = states[states["learner_id"].eq("L3")]
    assert roster_learner["split"].eq("inference_only").all()
    assert roster_learner["state_status"].eq("cold_start").all()
    assert roster_learner["predicted_correctness"].notna().all()
    assert (tmp_path / "outputs/learner_skill_state.csv").is_file()
    assert (tmp_path / "outputs/inference_manifest.json").is_file()
    manifest = json.loads(
        (tmp_path / "outputs/inference_manifest.json").read_text()
    )
    assert set(manifest["mapping_sha256"]) == {"question_map", "concept_map"}
    assert manifest["as_of"] == "1970-01-01T00:03:20+00:00"
    assert manifest["n_questions_scored_per_learner"] == 3
    report = (tmp_path / "outputs/knowledge_state_validation_report.md").read_text()
    assert "- Validation status: PASS" in report
    assert "Split-Level Descriptive Comparison" in report
    assert "All-Question vs 10-Question Sensitivity" in report


def test_akt_question_scoring_supports_empty_history(monkeypatch):
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            _, column = key
            return FakeTensor([row[column] for row in self.values])

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

        def numpy(self):
            return np.asarray(self.values)

    class FakeTorch:
        long = object()

        @staticmethod
        def tensor(values, **_):
            return FakeTensor(values)

        @staticmethod
        def inference_mode():
            return nullcontext()

    class FakeModel:
        def __call__(self, concepts, responses, questions):
            assert concepts.values == [[3]]
            assert responses.values == [[1]]
            assert questions.values == [[10]]
            return FakeTensor([[0.7]]), None

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    predictions = inference_module.score_akt_questions(
        model=FakeModel(),
        device="cpu",
        concepts=[],
        responses=[],
        history_questions=[],
        target_questions=[10],
        target_concepts=[3],
    )

    np.testing.assert_array_equal(predictions, [0.7])


def test_next_interaction_scorer_batches_equal_history_lengths(monkeypatch):
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            _, column = key
            return FakeTensor([row[column] for row in self.values])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.values)

    class FakeTorch:
        long = object()

        @staticmethod
        def tensor(values, **_):
            return FakeTensor(values)

        @staticmethod
        def inference_mode():
            return nullcontext()

    class FakeModel:
        def __call__(self, concepts, responses, questions):
            assert all(len(row) == len(concepts.values[0]) for row in concepts.values)
            return FakeTensor([[row[-1] / 100] for row in questions.values]), None

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    predictions = inference_module.score_akt_next_interactions(
        model=FakeModel(),
        device="cpu",
        history_concepts=[[], [1], [1]],
        history_responses=[[], [0], [1]],
        history_questions=[[], [11], [12]],
        target_questions=[20, 30, 40],
        target_concepts=[2, 3, 4],
        batch_size=2,
    )

    np.testing.assert_array_equal(predictions, [0.2, 0.3, 0.4])
