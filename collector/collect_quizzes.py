"""Step 3 - finished, non-preview quiz attempts with normalized score and duration."""

from .config import MASTERY_THRESHOLD, t
from .db import unix_to_local, write_csv


def collect_quiz_attempts(cursor, course_id, catalog_rows, user_filter):
    topic_by_quiz = {
        row["activity_id"]: row["topic"]
        for row in catalog_rows
        if row["activity_type"] == "quiz"
    }
    seq_by_quiz = {
        row["activity_id"]: row["sequence"]
        for row in catalog_rows
        if row["activity_type"] == "quiz"
    }

    query = f"""
        SELECT
            qa.userid, q.id, q.name, qa.attempt, qa.state,
            qa.sumgrades, q.sumgrades, qa.timestart, qa.timefinish
        FROM {t('quiz_attempts')} qa
        JOIN {t('quiz')} q ON q.id = qa.quiz
        WHERE q.course = ?
          AND qa.preview = 0
          AND qa.state = 'finished'
        ORDER BY qa.userid, qa.timestart, qa.id
    """
    cursor.execute(query, (course_id,))
    rows = []
    for row in cursor.fetchall():
        (
            learner_id,
            quiz_id,
            quiz_name,
            attempt,
            state,
            score,
            max_score,
            timestart,
            timefinish,
        ) = row

        if user_filter is not None and learner_id not in user_filter:
            continue

        max_value = float(max_score or 0)
        score_value = float(score or 0)
        normalized = "" if max_value == 0 else round(score_value / max_value, 4)
        mastery = 0 if normalized == "" else int(normalized >= MASTERY_THRESHOLD)
        duration = ""
        if timestart and timefinish:
            duration = int(timefinish - timestart)

        rows.append(
            {
                "learner_id": learner_id,
                "quiz_id": quiz_id,
                "quiz_name": quiz_name,
                "topic": topic_by_quiz.get(quiz_id, ""),
                "sequence_position": seq_by_quiz.get(quiz_id, ""),
                "attempt": attempt,
                "state": state,
                "score": score_value,
                "max_score": max_value,
                "normalized_score": normalized,
                "mastery_threshold_reached": mastery,
                "start_time": unix_to_local(timestart),
                "finish_time": unix_to_local(timefinish),
                "duration_seconds": duration,
                "start_time_unix": timestart,
                "finish_time_unix": timefinish,
            }
        )

    write_csv(
        "quiz_attempts.csv",
        rows,
        [
            "learner_id",
            "quiz_id",
            "quiz_name",
            "topic",
            "sequence_position",
            "attempt",
            "state",
            "score",
            "max_score",
            "normalized_score",
            "mastery_threshold_reached",
            "start_time",
            "finish_time",
            "duration_seconds",
            "start_time_unix",
            "finish_time_unix",
        ],
    )
    return rows
