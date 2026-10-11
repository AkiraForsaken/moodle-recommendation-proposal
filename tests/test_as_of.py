"""Temporal snapshot tests: the learner model and recommender data must never see the future."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.build_model import COLUMNS_ORDERED, build_learner_model_from_frames
from model.temporal import resolve_cutoff, truncate_frames
from recommender.data import RecommenderData
from recommender.engine import RecommendationEngine

UTC7 = 7 * 3600
BASE = 1_789_000_000  # arbitrary unix origin


def local(unix):  # what the collector writes: local (UTC+7) wall-clock string
    return (pd.Timestamp(unix + UTC7, unit="s")).strftime("%Y-%m-%d %H:%M:%S")


ACTIVITIES = pd.DataFrame(
    [
        # cmid, type, activity_id, name, topic, sequence, completion_tracking
        (4, "resource", 2, "Python Variables", "T1", 1, 0),
        (2, "quiz", 1, "Variable quiz", "T1", 2, 0),
        (7, "lesson", 1, "Conditions lesson", "T2", 3, 0),
        (6, "quiz", 2, "Cond quiz", "T2", 4, 0),
    ],
    columns=[
        "course_module_id",
        "activity_type",
        "activity_id",
        "activity_name",
        "topic",
        "sequence",
        "completion_tracking",
    ],
)
ACTIVITIES["section_id"] = 1
ACTIVITIES["section_number"] = [1, 1, 2, 2]
ACTIVITIES["visible"] = 1


def make_events(rows):
    """rows: (learner, cmid, category, offset_seconds). Sessions/deltas computed backwards-only,
    exactly like collector/sessions.py, so truncation behaves as in production."""
    out, prev, sess, lid_prev = [], None, 0, None
    for i, (lid, cmid, cat, off) in enumerate(sorted(rows, key=lambda r: (r[0], r[3]))):
        t = BASE + off
        if lid != lid_prev:
            sess, delta = 1, np.nan
        else:
            delta = t - prev
            if delta > 1800:
                sess += 1
        out.append(
            dict(
                log_id=i + 1,
                learner_id=lid,
                course_module_id=cmid,
                event_category=cat,
                is_learning_event=1,
                timestamp=local(t),
                timestamp_unix=t,
                seconds_since_prev_event=delta,
                session_id=f"u{lid}-s{sess:03d}",
            )
        )
        prev, lid_prev = t, lid
    return pd.DataFrame(out)


def make_quizzes(rows):
    """rows: (learner, quiz_id, attempt, score, finish_offset)"""
    return pd.DataFrame(
        [
            dict(
                learner_id=l,
                quiz_id=q,
                quiz_name=f"quiz{q}",
                attempt=a,
                state="finished",
                normalized_score=s,
                duration_seconds=60,
                start_time_unix=BASE + f - 60,
                finish_time_unix=BASE + f,
                finish_time=local(BASE + f),
            )
            for l, q, a, s, f in rows
        ]
    )


# Learner 1: fails Variables quiz, retries, passes; then starts T2.   Learner 2: strong.
EVENTS = make_events(
    [
        (1, 4, "content_view", 0),
        (1, 2, "content_view", 60),
        (1, 2, "content_view", 400),
        (1, 2, "content_view", 700),
        (1, 7, "content_view", 3000),
        (1, 6, "content_view", 3100),
        (2, 4, "content_view", 100),
        (2, 2, "content_view", 200),
        (2, 7, "content_view", 600),
    ]
)
QUIZZES = make_quizzes(
    [
        (1, 1, 1, 0.4, 300),  # fail  @ +300
        (1, 1, 2, 0.9, 650),  # pass  @ +650
        (1, 2, 1, 0.7, 3200),  # T2 quiz @ +3200
        (2, 1, 1, 1.0, 250),
    ]
)


def build(as_of=None, events=EVENTS, quizzes=QUIZZES):
    return build_learner_model_from_frames(
        ACTIVITIES, events, quizzes, None, as_of=as_of
    )


def row(model, learner, topic):
    r = model[(model.learner_id == learner) & (model.topic == topic)]
    return r.iloc[0] if len(r) else None


# ---------------------------------------------------------------- cutoff parsing
def test_cutoff_accepts_unix_string_and_local_time():
    assert resolve_cutoff(None) is None
    assert resolve_cutoff(123) == 123 and resolve_cutoff("123") == 123
    t = BASE + 400
    assert resolve_cutoff(local(t), EVENTS) == t  # UTC+7 offset inferred from the data
    assert resolve_cutoff({1: t, 2: local(t)}, EVENTS) == {1: t, 2: t}


def test_missing_time_column_fails_loudly():
    with pytest.raises(ValueError, match="finish_time_unix"):
        truncate_frames(
            EVENTS, QUIZZES.drop(columns="finish_time_unix"), None, BASE + 10
        )


# ---------------------------------------------------------------- the leakage guarantees
def test_cut_is_strictly_before_the_cutoff():
    assert build(BASE + 0).empty  # event AT T is hidden
    assert len(build(BASE + 1)) == 1  # one second later it is visible


def test_snapshot_equals_model_built_on_physically_truncated_data():
    for cut in (BASE + 500, BASE + 2000, BASE + 3150, BASE + 10_000):
        ev = EVENTS[EVENTS.timestamp_unix < cut]
        qz = QUIZZES[QUIZZES.finish_time_unix < cut]
        pd.testing.assert_frame_equal(
            build(cut), build(None, ev, qz), check_dtype=False
        )


def test_future_rows_cannot_change_the_snapshot():
    cut = BASE + 1000
    future_events = make_events(
        [
            (1, 7, "content_view", 5000),
            (1, 6, "content_view", 5050),
            (3, 4, "content_view", 6000),
        ]
    )  # incl. a brand-new learner
    future_quiz = make_quizzes([(1, 2, 2, 1.0, 5100), (3, 1, 1, 0.1, 6100)])
    noisy_ev = pd.concat([EVENTS, future_events], ignore_index=True)
    noisy_qz = pd.concat([QUIZZES, future_quiz], ignore_index=True)
    pd.testing.assert_frame_equal(
        build(cut), build(cut, noisy_ev, noisy_qz), check_dtype=False
    )


def test_no_cutoff_and_far_future_cutoff_match():
    pd.testing.assert_frame_equal(build(None), build(BASE + 10**9))


# ---------------------------------------------------------------- the model evolves
def test_mastery_and_trend_evolve_with_the_cutoff():
    before_retry = row(build(BASE + 500), 1, "T1")  # only the failed attempt is known
    after_retry = row(build(BASE + 700), 1, "T1")
    assert before_retry.mastery_level == "low" and before_retry.best_score == 0.4
    assert before_retry.quiz_attempts == 1 and before_retry.score_trend == 0.0
    assert after_retry.mastery_level == "high" and after_retry.best_score == 0.9
    assert after_retry.quiz_attempts == 2 and after_retry.score_trend == pytest.approx(
        0.5
    )
    assert after_retry.latest_score == 0.9


def test_before_any_quiz_mastery_is_unknown_and_before_any_event_model_is_empty():
    early = build(BASE + 200)
    assert row(early, 1, "T1").mastery_level == "unknown"
    empty = build(BASE - 1)
    assert empty.empty and list(empty.columns) == COLUMNS_ORDERED


def test_latest_score_does_not_depend_on_csv_row_order():
    shuffled = QUIZZES.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(
        build(None), build(None, quizzes=shuffled), check_dtype=False
    )


# ---------------------------------------------------------------- per-learner cutoffs
def test_per_learner_cutoffs_and_exclusion_of_unlisted_learners():
    m = build({1: BASE + 500})  # learner 2 not listed
    assert set(m.learner_id) == {1}
    assert row(m, 1, "T1").mastery_level == "low"
    m2 = build({1: BASE + 500, 2: BASE + 10_000})
    assert (
        row(m2, 1, "T1").mastery_level == "low"
        and row(m2, 2, "T1").mastery_level == "high"
    )


# ---------------------------------------------------------------- recommender integration
def snapshot(cut):
    return RecommenderData.from_moodle_frames(
        ACTIVITIES, EVENTS, QUIZZES, None, as_of=cut
    )


def test_recommender_interactions_and_state_are_cut_at_the_same_instant():
    early, late = snapshot(BASE + 500), snapshot(BASE + 10_000)
    assert early.seen(1) == {4, 2} and late.seen(1) == {4, 2, 7, 6}
    assert early.state(1).topics["T1"].band == "low"
    assert late.state(1).topics["T1"].band == "high"
    assert (early.interactions.last_time < pd.Timestamp(local(BASE + 500))).all()
    # best_score of the quiz as known at the cutoff, not the final one
    q = early.interactions.query("learner_id == 1 and course_module_id == 2").iloc[0]
    assert q.best_score == 0.4


def test_recommendation_changes_as_the_learner_progresses():
    first = RecommendationEngine(snapshot(BASE + 500)).recommend(
        1, k=1, method="rule_based"
    )[0]
    later = RecommendationEngine(snapshot(BASE + 700)).recommend(
        1, k=1, method="rule_based"
    )[0]
    assert first.topic == "T1" and first.activity_type in (
        "resource",
        "page",
        "lesson",
    )  # low -> review
    assert later.topic == "T2"  # mastered -> advance


# ---------------------------------------------------------------- regression on the real export
DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.skipif(
    not (DATA / "learner_model.csv").exists(),
    reason="real collector export not present",
)
def test_no_cutoff_reproduces_the_existing_learner_model_csv():
    from learner_model.build_model import load_collector_frames

    built = build_learner_model_from_frames(*load_collector_frames(DATA))
    shared = pd.read_csv(DATA / "learner_model.csv")
    reread = pd.read_csv(pd.io.common.StringIO(built.to_csv(index=False)))
    pd.testing.assert_frame_equal(reread, shared)
