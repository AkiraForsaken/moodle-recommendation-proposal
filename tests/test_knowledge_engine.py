import json

import pandas as pd
import pytest

from knowledge_engine import bkt, config, deep, phase, simulator
from knowledge_engine.estimate import choose_estimator, estimate
from knowledge_engine.schemas import KNOWLEDGE_STATE_COLUMNS, validate_states
from collector.collect_question_answers import ANSWER_EVENT_COLUMNS, ROSTER_COLUMNS


def _course_dir(path, events=(), roster=("7", "8")):
    """A minimal collector export: catalog, roster and (optionally) answers."""
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "kc_id": ["KC-G1", "KC-V1"],
            "kc_name": ["Present simple", "Routine words"],
            "kc_dimension": ["Grammar", "vocabulary"],
            "learning_objective": ["", ""],
        }
    ).to_csv(path / "kc_catalog.csv", index=False)
    pd.DataFrame(
        {"learner_id": list(roster), "course_id": 2, "enrol_start_unix": 0, "enrol_end_unix": 0}
    ).to_csv(path / "learner_roster.csv", index=False)
    if events:
        rows = [
            {
                "learner_id": learner,
                "event_index": index,
                "item_id": f"moodle-qbe:{index}",
                "kc_id": kc,
                "kc_dimension": "grammar" if kc == "KC-G1" else "vocabulary",
                "event_type": "answer",
                "timestamp": timestamp,
                "correct": correct,
                "mapping_version": "map-1",
            }
            for learner, index, kc, timestamp, correct in events
        ]
        pd.DataFrame(rows).to_csv(path / "question_answer_events.csv", index=False)
    return path


# ------------------------------------------------------------------------ BKT
def test_bkt_step_matches_hand_computed_values():
    params = bkt.BKTParams(p_init=0.3, p_learn=0.1, p_slip=0.1, p_guess=0.25)

    predicted, posterior, nxt = bkt.bkt_step(0.3, True, params)
    assert predicted == pytest.approx(0.3 * 0.9 + 0.7 * 0.25)
    assert posterior == pytest.approx(0.27 / 0.445)
    assert nxt == pytest.approx(posterior + (1 - posterior) * 0.1)

    _, wrong_posterior, _ = bkt.bkt_step(0.3, False, params)
    assert wrong_posterior == pytest.approx(0.03 / 0.555)
    assert wrong_posterior < 0.3 < posterior


def test_bkt_prediction_never_uses_the_answer_it_predicts():
    first = bkt.next_answer_predictions(["a", "a", "b"], [1, 1, 0])
    flipped = bkt.next_answer_predictions(["a", "a", "b"], [1, 0, 1])

    prior = bkt.predicted_correct(bkt.DEFAULT_PARAMS.p_init)
    assert first[0] == pytest.approx(prior)  # nothing seen yet on KC a
    assert first[2] == pytest.approx(prior)  # KC b is independent of KC a
    assert first[1] > prior  # one right answer on KC a raised its estimate
    assert first[:2] == flipped[:2]  # changing answer 2 cannot change prediction 2


def test_invalid_bkt_parameters_are_rejected():
    with pytest.raises(ValueError, match="p_slip \\+ p_guess"):
        bkt.BKTParams(p_slip=0.6, p_guess=0.5)


def test_estimate_gives_every_enrolled_learner_a_row_per_kc(tmp_path):
    data_dir = _course_dir(
        tmp_path / "data",
        events=[("7", 1, "KC-G1", 100, 1), ("7", 2, "KC-G1", 200, 1), ("7", 3, "KC-V1", 300, 0)],
    )

    states = estimate(data_dir, tmp_path / "states", active_model_file=tmp_path / "none.json")

    assert list(states.columns) == KNOWLEDGE_STATE_COLUMNS
    assert len(states) == 4  # 2 learners x 2 KCs
    assert states["source_population"].eq(config.MOODLE_POPULATION).all()
    assert states["estimator_phase"].eq(1).all()
    assert states["mapping_version"].eq("map-1").all()
    by_key = states.set_index(["learner_id", "kc_id"])
    grammar = by_key.loc[("7", "KC-G1")]
    assert grammar["attempt_count"] == 2
    assert grammar["observed_accuracy"] == 1.0
    assert grammar["p_mastery"] > bkt.DEFAULT_PARAMS.p_init
    assert grammar["kc_dimension"] == "grammar"  # normalised from the catalog's "Grammar"
    assert by_key.loc[("7", "KC-V1"), "p_mastery"] < bkt.DEFAULT_PARAMS.p_init
    # Learner 8 is enrolled but has answered nothing: priors, flagged as such.
    new_learner = by_key.loc["8"]
    assert new_learner["state_status"].eq("cold_start").all()
    assert new_learner["attempt_count"].eq(0).all()
    assert new_learner["observed_accuracy"].isna().all()
    assert new_learner["p_mastery"].eq(bkt.DEFAULT_PARAMS.p_init).all()
    manifest = json.loads((tmp_path / "states" / config.MANIFEST_FILE_NAME).read_text())
    assert manifest["estimator"] == "bkt"
    assert manifest["state_status_counts"] == {"cold_start": 2, "estimated": 2}


def test_estimate_works_for_a_course_with_no_answers_at_all(tmp_path):
    states = estimate(
        _course_dir(tmp_path / "data"), tmp_path / "states", active_model_file=tmp_path / "none.json"
    )
    assert len(states) == 4
    assert states["state_status"].eq("cold_start").all()


def test_estimate_cutoff_is_inclusive_and_ignores_later_answers(tmp_path):
    data_dir = _course_dir(
        tmp_path / "data",
        events=[("7", 1, "KC-G1", 100, 1), ("7", 2, "KC-G1", 200, 0)],
    )
    states = estimate(
        data_dir, tmp_path / "states", as_of=100, active_model_file=tmp_path / "none.json"
    ).set_index(["learner_id", "kc_id"])

    assert states.loc[("7", "KC-G1"), "attempt_count"] == 1
    assert states.loc[("7", "KC-G1"), "observed_accuracy"] == 1.0
    assert states["as_of"].eq("1970-01-01T00:01:40+00:00").all()


def test_answers_on_a_kc_outside_the_catalog_are_rejected(tmp_path):
    data_dir = _course_dir(tmp_path / "data", events=[("7", 1, "KC-ZZ", 100, 1)])
    with pytest.raises(ValueError, match="absent from the KC catalog"):
        estimate(data_dir, tmp_path / "states", active_model_file=tmp_path / "none.json")


def test_contract_rejects_cold_start_rows_with_evidence(tmp_path):
    states = estimate(
        _course_dir(tmp_path / "data"), tmp_path / "states", active_model_file=tmp_path / "none.json"
    )
    states.loc[0, "observed_accuracy"] = 0.0
    with pytest.raises(ValueError, match="cold-start row"):
        validate_states(states)


# --------------------------------------------------------------- phase / gate
def test_readiness_reports_each_unmet_threshold():
    catalog = pd.DataFrame({"kc_id": ["K1", "K2"], "kc_dimension": ["g", "g"]})
    events = pd.DataFrame({"learner_id": ["a", "a", "b"], "kc_id": ["K1", "K1", "K1"]})
    limits = {
        "min_learners_with_answers": 2,
        "min_answer_events": 3,
        "min_answers_per_kc": 2,
        "min_kc_coverage": 1.0,
    }

    result = phase.readiness(events, catalog, limits)
    assert not result["ready"]
    assert result["counts"]["kc_coverage"] == 0.5  # K2 has no answers
    assert len(result["unmet"]) == 1 and "KCs" in result["unmet"][0]

    assert phase.readiness(events, catalog, {**limits, "min_kc_coverage": 0.5})["ready"]
    empty = phase.readiness(events.iloc[0:0], catalog, limits)
    assert not empty["ready"] and len(empty["unmet"]) == 3


def test_gate_keeps_bkt_unless_the_deep_model_is_clearly_better():
    assert phase.decide_promotion(0.70, 0.705, margin=0.01)[0] is False
    assert phase.decide_promotion(0.70, 0.69, margin=0.01)[0] is False
    assert phase.decide_promotion(0.70, None)[0] is False
    promoted, reason = phase.decide_promotion(0.70, 0.72, margin=0.01)
    assert promoted is True and "0.7200" in reason


def test_training_is_not_started_when_data_is_not_ready(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("no training step may run before the data is ready")

    monkeypatch.setattr(phase, "_run", fail)
    data_dir = _course_dir(tmp_path / "data", events=[("7", 1, "KC-G1", 100, 1)])

    record = phase.train(data_dir, active_model_file=tmp_path / "active.json")

    assert record["phase"] == 1
    assert record["estimator"] == bkt.MODEL_NAME
    assert "not enough answer data" in record["reason"]
    assert phase.load_active_model(tmp_path / "active.json")["phase"] == 1


def test_promoted_model_is_only_used_for_the_data_it_was_trained_on():
    promoted = {
        "phase": 3,
        "training_population": config.SIMULATED_POPULATION,
        "mapping_version": "map-1",
        "reason": "passed the gate",
    }
    assert choose_estimator(promoted, config.SIMULATED_POPULATION, "map-1") == (
        "deep",
        "passed the gate",
    )
    # Never apply a model trained on simulated learners to real ones.
    assert choose_estimator(promoted, config.MOODLE_POPULATION, "map-1")[0] == "bkt"
    # A changed question bank means the model must be retrained first.
    assert choose_estimator(promoted, config.SIMULATED_POPULATION, "map-2")[0] == "bkt"
    assert choose_estimator({"phase": 2, "reason": "gate failed"}, "x", "y") == ("bkt", "gate failed")
    assert choose_estimator(phase.load_active_model("/no/such/file.json"), "x", "y")[0] == "bkt"


def test_deep_export_is_renamed_to_the_same_contract():
    deep_export = pd.DataFrame(
        {
            "learner_id": ["7", "7"],
            "split": ["test", "test"],
            "kc_id": ["KC-G1", "KC-V1"],
            "kc_dimension": ["grammar", "vocabulary"],
            "attempt_count": [3, 0],
            "observed_accuracy": [2 / 3, None],
            "predicted_correctness": [0.7, 0.5],
            "predicted_correctness_10q": [0.7, 0.5],
            "last_attempt_at": ["2026-01-01T00:00:00+00:00", None],
            "model_name": "akt",
            "model_version": "moodle_kt-akt-seed42",
            "checkpoint_id": "moodle_kt-akt-seed42",
            "checkpoint_sha256": "abc",
            "inference_timestamp": "2026-01-02T00:00:00+00:00",
            "as_of": None,
            "evidence_count": [3, 0],
            "evidence_tier": ["moderate", "none"],
            "prediction_question_count": [10, 10],
            "sensitivity_question_count": [10, 10],
            "aggregation_method": "mean",
            "state_status": ["estimated", "cold_start"],
        }
    )

    states = deep.to_contract(deep_export, "map-1", config.MOODLE_POPULATION)

    assert list(states.columns) == KNOWLEDGE_STATE_COLUMNS
    assert states["estimator_phase"].eq(3).all()
    assert states["p_mastery"].isna().all()  # the deep model has no such quantity


# ------------------------------------------------------------------ simulator
def test_simulator_writes_collector_schemas_and_is_reproducible(tmp_path):
    first = simulator.simulate(tmp_path / "a", n_learners=40, seed=3)
    assert 0.5 < first["ceiling_auc"] <= 1.0
    simulator.simulate(tmp_path / "b", n_learners=40, seed=3)
    simulator.simulate(tmp_path / "c", n_learners=40, seed=4)

    events = pd.read_csv(tmp_path / "a" / "question_answer_events.csv")
    roster = pd.read_csv(tmp_path / "a" / "learner_roster.csv")
    assert list(events.columns) == ANSWER_EVENT_COLUMNS
    assert list(roster.columns) == ROSTER_COLUMNS
    assert len(roster) == 40 and first["simulated"] is True
    assert events["learner_id"].str.startswith("sim-").all()
    assert events["correct"].isin([0, 1]).all()
    # Partial credit is never counted as correct, as in the collector.
    assert events.loc[events["raw_fraction"].lt(1.0), "correct"].eq(0).all()
    assert not events.duplicated(["learner_id", "event_index"]).any()
    assert events.groupby("learner_id")["timestamp"].is_monotonic_increasing.all()
    same = (tmp_path / "b" / "question_answer_events.csv").read_bytes()
    assert (tmp_path / "a" / "question_answer_events.csv").read_bytes() == same
    assert (tmp_path / "c" / "question_answer_events.csv").read_bytes() != same


def test_simulated_data_is_labelled_and_never_overwrites_collector_data(tmp_path):
    simulator.simulate(tmp_path / "sim", n_learners=20, seed=1)
    states = estimate(tmp_path / "sim", tmp_path / "states", active_model_file=tmp_path / "none.json")
    assert states["source_population"].eq(config.SIMULATED_POPULATION).all()

    real = _course_dir(tmp_path / "real", events=[("7", 1, "KC-G1", 100, 1)])
    with pytest.raises(ValueError, match="refusing to overwrite"):
        simulator.simulate(real, n_learners=5)
    with pytest.raises(ValueError, match="must differ"):
        simulator.simulate(tmp_path / "sim", course_dir=tmp_path / "sim")


def test_simulator_can_reuse_a_real_course_catalog(tmp_path):
    course = _course_dir(tmp_path / "course")
    pd.DataFrame(
        {
            "question_bank_entry_id": ["10", "11", "20"],
            "kc_id": ["KC-G1", "KC-G1", "KC-V1"],
            "kc_dimension": ["grammar", "grammar", "vocabulary"],
            "mapping_version": "qcat-abc",
            "mapping_source": "moodle_question_category",
        }
    ).to_csv(course / "moodle_question_kc_map.csv", index=False)

    simulator.simulate(tmp_path / "sim", course_dir=course, n_learners=15, seed=2)

    events = pd.read_csv(tmp_path / "sim" / "question_answer_events.csv")
    assert set(events["kc_id"]) <= {"KC-G1", "KC-V1"}
    assert set(events["item_id"]) <= {"moodle-qbe:10", "moodle-qbe:11", "moodle-qbe:20"}
    assert events["mapping_version"].eq("qcat-abc").all()
