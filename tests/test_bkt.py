"""BKT producer: maths, leakage guarantees, cold start, and the heatmap/API contract."""

import json

import numpy as np
import pandas as pd
import pytest

from api.main import build_heatmap
from learner_model.bkt import (
    DEFAULT_PARAMS,
    BKTParams,
    bkt_step,
    bkt_trace,
    compute_bkt_state,
)
from learner_model.build_model import COLUMNS_ORDERED, STATE_COLUMNS
from tests.test_as_of import ACTIVITIES, BASE, EVENTS, QUIZZES, build, make_events, make_quizzes, row


# ---------------------------------------------------------------- the maths
def test_one_step_matches_hand_computation():
    p = BKTParams(p_init=0.3, p_learn=0.3, p_slip=0.1, p_guess=0.1)
    predicted, posterior, nxt = bkt_step(0.3, True, p)
    assert predicted == pytest.approx(0.3 * 0.9 + 0.7 * 0.1)  # 0.34
    assert posterior == pytest.approx(0.27 / 0.34)
    assert nxt == pytest.approx(posterior + (1 - posterior) * 0.3)
    _, post_fail, _ = bkt_step(0.3, False, p)
    assert post_fail == pytest.approx(0.03 / 0.66)


def test_pass_raises_and_fail_lowers_the_estimate():
    p0 = DEFAULT_PARAMS.p_init
    assert bkt_step(p0, True, DEFAULT_PARAMS)[1] > p0
    assert bkt_step(p0, False, DEFAULT_PARAMS)[1] < p0


def test_order_matters_fail_then_pass_differs_from_pass_then_fail():
    fp = bkt_trace([0, 1])[-1]["p_mastery"]
    pf = bkt_trace([1, 0])[-1]["p_mastery"]
    assert fp > pf  # recent success counts for more than an old one


def test_prediction_is_made_before_the_current_outcome_is_applied():
    # Same history, different CURRENT outcome -> identical prediction for that step.
    a = bkt_trace([1, 0, 1])[2]
    b = bkt_trace([1, 0, 0])[2]
    assert a["predicted_pass"] == b["predicted_pass"]
    assert a["p_known_before"] == b["p_known_before"]
    assert a["p_mastery"] != b["p_mastery"]  # ...but the state after it differs


def test_p_mastery_always_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(200):
        seq = rng.integers(0, 2, size=rng.integers(1, 40))
        p = BKTParams(
            p_init=rng.uniform(0, 1),
            p_learn=rng.uniform(0, 1),
            p_slip=rng.uniform(0.01, 0.45),
            p_guess=rng.uniform(0.01, 0.45),
            p_forget=rng.uniform(0, 0.5),
        )
        assert all(0.0 <= r["p_mastery"] <= 1.0 for r in bkt_trace(seq, p))


@pytest.mark.parametrize(
    "bad",
    [dict(p_slip=0.0), dict(p_guess=1.0), dict(p_init=1.5), dict(p_forget=1.0),
     dict(p_slip=0.6, p_guess=0.5)],
)
def test_invalid_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        BKTParams(**bad)


# ---------------------------------------------------------------- learner-model integration
def test_state_columns_are_appended_and_old_columns_keep_their_positions():
    assert COLUMNS_ORDERED[-len(STATE_COLUMNS):] == STATE_COLUMNS
    assert COLUMNS_ORDERED[:16][-2:] == ["mastery_level", "last_activity"]
    m = build(None)
    assert list(m.columns) == COLUMNS_ORDERED


def test_mastery_level_is_unchanged_and_p_mastery_follows_the_attempt_history():
    m = build(None)
    # learner 1, T1: fail (0.4) then pass (0.9)
    r = row(m, 1, "T1")
    assert r.mastery_level == "high" and r.evidence_count == 2
    assert r.state_status == "estimated" and 0.8 <= r.p_mastery <= 1.0
    # learner 1, T2: single failing attempt (0.7): score band says "medium", BKT says low
    r2 = row(m, 1, "T2")
    assert r2.mastery_level == "medium" and r2.p_mastery < 0.6


def test_provenance_columns():
    r = row(build(None), 1, "T1")
    assert (r.model_name, r.model_version, r.source_dataset) == (
        "bkt_attempt_level", "0.1.0-fixed-params", "moodle_quiz_attempts")
    assert pd.isna(r.generated_at)  # the pure function stays deterministic


def test_cold_start_is_explicit_not_zero():
    # Before any quiz is finished the learner has events but no evidence.
    r = row(build(BASE + 200), 1, "T1")
    assert pd.isna(r.p_mastery)  # NOT 0.0, NOT p_init
    assert r.evidence_count == 0 and r.state_status == "cold_start"
    assert r.mastery_level == "unknown"


def test_topic_with_events_but_no_quiz_is_cold_start():
    # learner 2 viewed T2 content but never took the T2 quiz
    r = row(build(None), 2, "T2")
    assert pd.isna(r.p_mastery) and r.state_status == "cold_start" and r.evidence_count == 0


# ---------------------------------------------------------------- temporal integrity
def test_p_mastery_before_and_after_the_retry():
    before = row(build(BASE + 500), 1, "T1")  # only the failed attempt is known
    after = row(build(BASE + 700), 1, "T1")
    assert before.p_mastery < 0.6 < 0.8 < after.p_mastery
    assert (before.evidence_count, after.evidence_count) == (1, 2)


def test_snapshot_state_equals_state_from_physically_truncated_data():
    for cut in (BASE + 500, BASE + 700, BASE + 3300, BASE + 10_000):
        ev = EVENTS[EVENTS.timestamp_unix < cut]
        qz = QUIZZES[QUIZZES.finish_time_unix < cut]
        pd.testing.assert_frame_equal(build(cut), build(None, ev, qz), check_dtype=False)


def test_future_attempts_cannot_change_p_mastery():
    cut = BASE + 1000
    future_q = make_quizzes([(1, 1, 3, 0.0, 5000), (1, 2, 2, 1.0, 5100), (3, 1, 1, 0.1, 6100)])
    future_e = make_events([(1, 2, "content_view", 4990), (3, 4, "content_view", 6000)])
    noisy_q = pd.concat([QUIZZES, future_q], ignore_index=True)
    noisy_e = pd.concat([EVENTS, future_e], ignore_index=True)
    pd.testing.assert_frame_equal(build(cut), build(cut, noisy_e, noisy_q), check_dtype=False)


def test_attempt_exactly_at_the_cutoff_is_hidden():
    # learner 1's passing retry finishes at BASE+650: visible at +651, hidden at +650
    assert row(build(BASE + 651), 1, "T1").evidence_count == 2
    assert row(build(BASE + 650), 1, "T1").evidence_count == 1


def test_csv_row_order_does_not_change_p_mastery():
    shuffled = QUIZZES.sample(frac=1, random_state=3)
    pd.testing.assert_frame_equal(build(None), build(None, quizzes=shuffled), check_dtype=False)


def test_time_order_across_quizzes_of_one_topic_beats_quiz_id_order():
    # Two quizzes in T1: quiz 9 is taken FIRST (fail), quiz 1 LATER (pass).
    acts = pd.concat(
        [ACTIVITIES, ACTIVITIES.iloc[[1]].assign(course_module_id=50, activity_id=9, activity_name="Extra quiz")],
        ignore_index=True,
    )
    q = make_quizzes([(1, 9, 1, 0.2, 100), (1, 1, 1, 0.9, 200)])
    state = compute_bkt_state(q, acts[acts.topic == "T1"])
    expected = bkt_trace([0, 1])[-1]["p_mastery"]
    assert state.p_mastery.iloc[0] == pytest.approx(expected, abs=1e-6)


def test_quiz_without_a_topic_and_attempts_without_a_score_are_dropped():
    q = make_quizzes([(1, 1, 1, 0.9, 100), (1, 77, 1, 0.9, 110), (1, 1, 2, np.nan, 120)])
    state = compute_bkt_state(q, ACTIVITIES)
    assert len(state) == 1 and state.evidence_count.iloc[0] == 1


def test_real_export_style_quiz_frame_with_its_own_topic_column():
    q = make_quizzes([(1, 1, 1, 0.9, 100)]).assign(topic="stale name", activity_id=999)
    state = compute_bkt_state(q, ACTIVITIES)
    assert state.topic.tolist() == ["T1"]  # catalog wins over the export's copy


# ---------------------------------------------------------------- heatmap uses p_mastery
def _cell(model_rows):
    body = build_heatmap(pd.DataFrame(model_rows))
    return body, {(r.learner_id, t): c for r in body.learners for t, c in r.topics.items()}


def test_heatmap_prefers_p_mastery_and_bins_at_existing_thresholds():
    rows = [
        dict(learner_id=1, topic="A", mastery_level="high", p_mastery=0.35),   # BKT says low
        dict(learner_id=1, topic="B", mastery_level="low", p_mastery=0.65),    # medium
        dict(learner_id=1, topic="C", mastery_level="low", p_mastery=0.80),    # boundary -> high
        dict(learner_id=1, topic="D", mastery_level="medium", p_mastery=np.nan),  # label fallback
        dict(learner_id=1, topic="E", mastery_level="unknown", p_mastery=np.nan),
    ]
    body, c = _cell(rows)
    assert [c[(1, t)].mastery_level for t in "ABCDE"] == ["low", "medium", "high", "medium", "unknown"]
    assert c[(1, "A")].p_mastery == 0.35 and c[(1, "D")].p_mastery is None
    assert body.summary == {"high": 1, "medium": 2, "low": 1, "unknown": 1}


def test_heatmap_still_works_without_a_p_mastery_column():
    _, c = _cell([dict(learner_id=1, topic="A", mastery_level="medium")])
    assert c[(1, "A")].mastery_level == "medium" and c[(1, "A")].p_mastery is None


def test_api_heatmap_exposes_p_mastery_as_strict_json(client):
    assert client.post("/run-pipeline").status_code == 200
    raw = client.get("/analytics/heatmap").text
    body = json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    cells = [c for row in body["learners"] for c in row["topics"].values()]
    assert any(c["p_mastery"] is None for c in cells)  # unscored topics: null, not 0
    assert all(0.0 <= c["p_mastery"] <= 1.0 for c in cells if c["p_mastery"] is not None)
    assert sum(body["summary"].values()) == len(cells)


def test_recommender_consumes_p_mastery_end_to_end(client, data_dir):
    client.post("/run-pipeline")
    lm = pd.read_csv(data_dir / "learner_model.csv")
    assert lm["p_mastery"].notna().any() and {"mastery_level", "p_mastery"} <= set(lm.columns)
    body = client.get("/recommend/2?k=3").json()
    assert body["focus"]["band"] == "low"  # learner 2 failed 'Conditions' twice
