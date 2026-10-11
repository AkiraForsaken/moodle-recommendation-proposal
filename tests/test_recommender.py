import numpy as np
import pandas as pd
import pytest

from recommender.data import RecommenderData
from recommender.engine import RecommendationEngine
from recommender.base import OUTPUT_COLUMNS, to_frame
from recommender.rule_based import RuleBasedRecommender
from recommender.popularity import PopularityRecommender
from recommender.content_based import ContentBasedRecommender
from recommender.collaborative import CollaborativeRecommender

T1, T2, T3, T4 = "Variables and Data Types", "Conditions", "Loops", "Functions"

# Same shape as the real Moodle activity_catalog.csv (cmid -> topic/type)
CATALOG = pd.DataFrame(
    [
        (1, 0, "resource", 1, "Syllabus", "Topic 0", 1, 1),
        (3, 0, "forum", 1, "Announcements", "Topic 0", 2, 1),
        (16, 0, "qbank", 1, "", "Topic 0", 3, 0),
        (2, 1, "quiz", 1, "Python variable quiz", T1, 4, 1),
        (4, 1, "resource", 2, "Python Variables", T1, 5, 1),
        (7, 1, "lesson", 1, "Python Variables Lesson", T1, 6, 1),
        (5, 2, "resource", 3, "Python If-statements", T2, 7, 1),
        (6, 2, "quiz", 2, "Python cond quiz", T2, 8, 1),
        (12, 2, "lesson", 3, "Python conditions lesson", T2, 9, 1),
        (9, 3, "page", 1, "Python loops", T3, 10, 1),
        (13, 3, "lesson", 4, "Python loops lesson", T3, 11, 1),
        (15, 3, "quiz", 5, "Python loops quiz", T3, 12, 1),
        (17, 4, "lesson", 5, "Python functions lesson", T4, 13, 1),
        (18, 4, "quiz", 6, "Python functions quiz", T4, 14, 1),
    ],
    columns=[
        "course_module_id",
        "section_number",
        "activity_type",
        "activity_id",
        "activity_name",
        "topic",
        "sequence",
        "visible",
    ],
)


def lm_row(lid, topic, level, score=0.9, trend=0.0, minutes=100, last="2026-09-20"):
    return dict(
        learner_id=lid,
        topic=topic,
        mastery_level=level,
        best_score=score,
        score_trend=trend,
        lesson_time_estimate=minutes,
        last_activity=last,
        sequence_position=0,
    )


def inter(lid, cmid, score=None, views=3):
    return dict(
        learner_id=lid,
        course_module_id=cmid,
        first_time="2026-09-20",
        last_time="2026-09-20",
        n_views=views,
        n_events=views,
        n_sessions=1,
        best_score=score,
    )


def make_data(lm_rows, inter_rows=()):
    return RecommenderData(
        CATALOG, pd.DataFrame(lm_rows), pd.DataFrame(list(inter_rows))
    )


def top(recs, n=3):
    return [r.course_module_id for r in recs[:n]]


# ---------------------------------------------------------------- data layer
def test_non_learning_content_is_never_recommendable():
    d = make_data([lm_row(1, T1, "low")])
    assert set(d.catalog["course_module_id"]).isdisjoint(
        {1, 3, 16}
    )  # syllabus, forum, qbank
    assert d.topic_order == [T1, T2, T3, T4]


def test_missing_columns_fail_loudly():
    with pytest.raises(ValueError, match="learner_model is missing"):
        RecommenderData(CATALOG, pd.DataFrame({"mastery_level": ["low"]}))
    with pytest.raises(ValueError, match="mastery_level"):
        RecommenderData(CATALOG, pd.DataFrame({"learner_id": [1], "topic": [T1]}))


def test_probability_overrides_mastery_label():
    row = lm_row(1, T1, "high")
    row["p_mastery"] = 0.35  # BKT says otherwise
    d = make_data([row])
    assert d.state(1).topics[T1].band == "low"


# ---------------------------------------------------------------- rule-based
@pytest.mark.parametrize(
    "level, expected_first_role",
    [
        ("low", "learn"),
        ("medium", "practice"),
        ("unknown", "learn"),
    ],
)
def test_rule_based_routes_by_mastery_within_the_focus_topic(
    level, expected_first_role
):
    # T1 mastered so that T2 is the frontier (an untouched T1 would be the focus otherwise)
    d = make_data([lm_row(1, T1, "high"), lm_row(1, T2, level)])
    recs = RuleBasedRecommender().fit(d).recommend(1, k=5)
    assert recs[0].topic == T2
    role = d.catalog.set_index("course_module_id").loc[recs[0].course_module_id, "role"]
    assert role == expected_first_role


def test_rule_based_regression_old_sequence_bug():
    """Old code: low mastery -> [] ; high mastery in T1 -> a resource from T1 itself."""
    low = RuleBasedRecommender().fit(make_data([lm_row(1, T1, "low")])).recommend(1)
    assert low and low[0].activity_type in ("resource", "lesson")  # was: empty
    high = RuleBasedRecommender().fit(make_data([lm_row(1, T1, "high")])).recommend(1)
    assert high[0].topic == T2  # was: same topic


def test_rule_based_frontier_goes_back_to_an_unmastered_gap():
    d = make_data(
        [
            lm_row(1, T1, "low", last="2026-09-01"),
            lm_row(1, T2, "high", last="2026-09-02"),
            lm_row(1, T3, "high", last="2026-09-03"),
        ]
    )
    assert (
        RuleBasedRecommender("frontier").fit(d).recommend(1)[0].topic == T1
    )  # fix the gap first
    assert (
        RuleBasedRecommender("latest").fit(d).recommend(1)[0].topic == T4
    )  # follow the learner forward


def test_rule_based_cold_start_and_completion():
    d = make_data(
        [
            lm_row(2, T1, "high"),
            lm_row(2, T2, "high"),
            lm_row(2, T3, "high"),
            lm_row(2, T4, "high"),
        ]
    )
    rb = RuleBasedRecommender().fit(d)
    assert rb.recommend(2) == []  # course complete
    assert rb.recommend(999)[0].topic == T1  # unknown learner -> start of course
    assert rb.recommend(999)[0].activity_type in ("resource", "page", "lesson")


def test_rule_based_retry_is_allowed_for_medium_mastery():
    d = make_data([lm_row(1, T1, "medium")], [inter(1, 2, score=0.7)])
    recs = RuleBasedRecommender().fit(d).recommend(1)
    assert recs[0].course_module_id == 2 and recs[0].is_review  # retake the quiz


# ---------------------------------------------------------------- popularity
def test_popularity_counts_distinct_learners_not_raw_views():
    inters = [inter(1, 9, views=500), inter(2, 13), inter(3, 13), inter(4, 13)]
    d = make_data([lm_row(i, T1, "high") for i in range(1, 5)], inters)
    recs = PopularityRecommender().fit(d).recommend(99, k=2)
    assert recs[0].course_module_id == 13  # 3 learners beats 1 heavy user
    views = PopularityRecommender(metric="views").fit(d).recommend(99, k=1)
    assert views[0].course_module_id == 9


# ---------------------------------------------------------------- content-based
def test_content_based_ranks_focus_topic_first_and_demotes_seen():
    d = make_data(
        [lm_row(1, T1, "high"), lm_row(1, T2, "high"), lm_row(1, T3, "low")],
        [inter(1, 9)],
    )  # focus = Loops; saw the loops page
    recs = ContentBasedRecommender().fit(d).recommend(1, k=3)
    assert {r.topic for r in recs[:2]} == {T3}
    assert recs[0].is_review is False and not any(r.is_review for r in recs[:2])


def test_content_based_exclude_policy():
    d = make_data(
        [lm_row(1, T1, "high"), lm_row(1, T2, "high"), lm_row(1, T3, "low")],
        [inter(1, 9)],
    )
    recs = ContentBasedRecommender(seen_policy="exclude").fit(d).recommend(1, k=20)
    assert 9 not in top(recs, 20)


# ---------------------------------------------------------------- collaborative
def cohort():
    lm, it = [], []
    # learners 1,2: strong, fast; learners 3,4: weak, slow. Learner 0 resembles 1,2.
    for lid, score, mins in [
        (0, 0.9, 100),
        (1, 0.9, 110),
        (2, 0.85, 90),
        (3, 0.3, 900),
        (4, 0.35, 950),
    ]:
        lm += [
            lm_row(lid, T1, "high" if score > 0.8 else "low", score, 0.0, mins),
            lm_row(lid, T2, "high" if score > 0.8 else "low", score, 0.0, mins),
        ]
    it += [inter(0, 2, 0.9), inter(0, 4)]  # learner 0 has done T1 only
    it += [
        inter(1, 2, 0.9),
        inter(1, 12, 0.9),
        inter(1, 17),
    ]  # strong peers succeeded on 12, 17
    it += [inter(2, 2, 0.85), inter(2, 12), inter(2, 17)]
    it += [inter(3, 2, 0.3), inter(3, 5, 0.2)]  # weak peers failed / drifted
    it += [inter(4, 2, 0.35), inter(4, 5)]
    return make_data(lm, it)


def test_collaborative_follows_similar_learners_successes():
    recs = CollaborativeRecommender(n_neighbors=2).fit(cohort()).recommend(0, k=3)
    assert recs[0].course_module_id in (12, 17)
    assert recs[0].score > 0


def test_collaborative_ignores_failed_attempts_of_neighbours():
    d = cohort()
    cf = CollaborativeRecommender(n_neighbors=4).fit(d)
    ids = {r.course_module_id: r.score for r in cf.recommend(0, k=20)}
    assert ids[2] > 0 and ids.get(12, 0) > ids.get(
        5, 0
    )  # strong peers' quiz 12 >> weak peers' resource 5


def test_collaborative_cold_start_returns_nothing():
    assert CollaborativeRecommender().fit(cohort()).recommend(999) == []


# ---------------------------------------------------------------- engine contract
def test_engine_output_schema_ranks_and_no_duplicates():
    eng = RecommendationEngine(cohort())
    for method in eng.methods:
        recs = eng.recommend(0, k=5, method=method)
        assert [r.rank for r in recs] == list(range(1, len(recs) + 1))
        assert len({r.course_module_id for r in recs}) == len(recs)
        assert len(recs) <= 5
    frame = eng.recommend_all(0, k=3)
    assert list(frame.columns) == OUTPUT_COLUMNS


def test_engine_flags_fallback_instead_of_hiding_it():
    eng = RecommendationEngine(cohort())
    recs = eng.recommend(999, k=3, method="collaborative")  # no profile -> fallback
    assert recs and all(r.via_fallback for r in recs)
    assert "fallback" in recs[0].reason


def test_engine_rejects_unknown_method():
    with pytest.raises(KeyError):
        RecommendationEngine(cohort()).recommend(0, method="magic")


def test_fallback_rows_are_attributed_to_the_requested_method():
    recs = RecommendationEngine(cohort()).recommend(999, k=3, method="collaborative")
    assert {r.method for r in recs} == {"collaborative"}
