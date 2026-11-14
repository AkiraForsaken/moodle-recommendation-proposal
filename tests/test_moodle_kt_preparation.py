import json
import sys

import pandas as pd
import pytest

import knowledge_tracing.init_pykt_dataset as bridge_module
import knowledge_tracing.prepare_data as prepare_module
from knowledge_tracing.kt_common import load_data_config
from knowledge_tracing.prepare_data import assign_splits, build_maps


def _events():
    return pd.DataFrame(
        [
            {
                "learner_id": "3",
                "item_id": "moodle-qbe:10",
                "kc_id": "KC001",
                "kc_dimension": "grammar",
                "correct": 1,
                "timestamp": 100,
                "response_time_s": 12,
            }
        ]
    )


def _crosswalk():
    return pd.DataFrame(
        [
            {
                "question_bank_entry_id": "10",
                "kc_id": "KC001",
                "kc_dimension": "grammar",
                "mapping_version": "1.0",
                "mapping_source": "reviewed",
            },
            {
                "question_bank_entry_id": "11",
                "kc_id": "KC002",
                "kc_dimension": "reading",
                "mapping_version": "1.0",
                "mapping_source": "reviewed",
            },
        ]
    )


def test_moodle_maps_include_crosswalk_items_not_yet_answered():
    question_map, concept_map = build_maps(_events(), _crosswalk())

    assert question_map["item_id"].tolist() == ["moodle-qbe:10", "moodle-qbe:11"]
    assert question_map["kc_id"].tolist() == ["KC001", "KC002"]
    assert concept_map["kc_dimension"].tolist() == ["grammar", "reading"]


def test_moodle_map_rejects_events_disagreeing_with_curated_crosswalk():
    with pytest.raises(ValueError, match="differ from the curated crosswalk"):
        build_maps(_events().assign(kc_id="KC999"), _crosswalk())


def test_moodle_splits_keep_three_learners_in_distinct_folds():
    sequences = pd.DataFrame(
        {"learner_id": ["3", "5", "8"], "seq_len": [10, 4, 2]}
    )

    split = assign_splits(sequences, seed=3)

    assert set(split["split"]) == {"train", "valid", "test"}
    assert split["learner_id"].is_unique
    assert split["fold"].isin([0, 1, 2]).all()


def test_moodle_splits_reject_cohorts_too_small_for_holdout_evaluation():
    with pytest.raises(ValueError, match="At least 3 learners"):
        assign_splits(pd.DataFrame({"learner_id": ["3", "5"]}), seed=3)


def test_moodle_prepare_and_pykt_bridge_use_isolated_vocabulary(tmp_path, monkeypatch):
    event_path = tmp_path / "question_answer_events.csv"
    events = []
    for learner_id, answers in (("3", (1, 0)), ("5", (0, 1)), ("8", (1, 1))):
        for index, (item, kc, dimension, correct) in enumerate(
            (
                ("moodle-qbe:10", "KC001", "grammar", answers[0]),
                ("moodle-qbe:11", "KC002", "reading", answers[1]),
            ),
            start=1,
        ):
            events.append(
                {
                    "learner_id": learner_id,
                    "event_index": index,
                    "item_id": item,
                    "kc_id": kc,
                    "kc_dimension": dimension,
                    "event_type": "answer",
                    "timestamp": 100 * index,
                    "response_time_s": 12,
                    "correct": correct,
                }
            )
    pd.DataFrame(events).to_csv(event_path, index=False)
    mapping_path = tmp_path / "question_kc_map.csv"
    _crosswalk().to_csv(mapping_path, index=False)
    prepared = tmp_path / "processed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data",
            "--dataset-name",
            "moodle_kt",
            "--input",
            str(event_path),
            "--question-mapping",
            str(mapping_path),
            "--output-dir",
            str(prepared),
            "--seed",
            "4",
        ],
    )

    prepare_module.main()
    preparation_manifest = json.loads(
        (prepared / "prepare_data_manifest.json").read_text()
    )
    assert preparation_manifest["mapping_coverage"]["mapping_version"] == "1.0"
    assert preparation_manifest["mapping_coverage"]["mapped_questions"] == 2
    assert preparation_manifest["mapping_coverage"]["mapped_questions_without_train_exposure"] == 0

    bridge_output = tmp_path / "pykt_moodle"
    configs_dir = tmp_path / "configs"
    monkeypatch.setattr(bridge_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(bridge_module, "CONFIG_DIR", configs_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "init_pykt_dataset",
            "--dataset-name",
            "moodle_kt",
            "--input",
            str(prepared / "pykt_formatted_sequences.csv"),
            "--output-dir",
            str(bridge_output),
        ],
    )

    bridge_module.main()

    config = json.loads((configs_dir / "data_config.json").read_text())
    assert set(config) == {"moodle_kt"}
    assert config["moodle_kt"]["num_q"] == 2
    assert config["moodle_kt"]["num_c"] == 2
    assert (bridge_output / "train_valid_sequences.csv").is_file()
    assert (bridge_output / "test_sequences.csv").is_file()
    monkeypatch.setattr(prepare_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("knowledge_tracing.kt_common.CONFIG_DIR", configs_dir)
    monkeypatch.setattr("knowledge_tracing.kt_common.PROJECT_ROOT", tmp_path)
    loaded = load_data_config("moodle_kt")
    assert loaded["mapping_dir"] == str(prepared.resolve())
    assert loaded["dpath"] == str(bridge_output.resolve())
