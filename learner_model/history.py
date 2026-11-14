"""Compact chronological and aggregate Moodle learning-history summaries."""

import pandas as pd


HISTORY_COLUMNS = [
    "learner_id",
    "first_activity_at",
    "last_activity_at",
    "total_learning_events",
    "total_quiz_attempts",
    "total_completed_activities",
    "distinct_topics_accessed",
    "historical_learning_days",
    "historical_session_count",
    "latest_activity_type",
    "history_status",
    "generated_at",
]


def build_history(events, quizzes, completions, activities, generated_at):
    learner_ids = set()
    for frame in (events, quizzes, completions):
        if not frame.empty and "learner_id" in frame:
            learner_ids.update(frame["learner_id"].dropna().astype(str))
    if not learner_ids:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    topic_lookup = (
        activities[["course_module_id", "topic"]].drop_duplicates("course_module_id")
        if {"course_module_id", "topic"}.issubset(activities.columns)
        else pd.DataFrame(columns=["course_module_id", "topic"])
    )
    mapped_events = events.copy()
    if not mapped_events.empty and "course_module_id" in mapped_events:
        mapped_events = mapped_events.merge(
            topic_lookup, on="course_module_id", how="left"
        )
    if not mapped_events.empty:
        mapped_events["_timestamp_utc"] = (
            pd.to_datetime(
                pd.to_numeric(mapped_events["timestamp_unix"], errors="coerce"),
                unit="s",
                errors="coerce",
                utc=True,
            )
            if "timestamp_unix" in mapped_events
            else pd.to_datetime(mapped_events["timestamp"], errors="coerce", utc=True)
        )
    rows = []
    for learner_id in sorted(learner_ids):
        event_group = mapped_events[
            mapped_events["learner_id"].astype(str).eq(learner_id)
        ] if not mapped_events.empty else mapped_events
        quiz_group = quizzes[
            quizzes["learner_id"].astype(str).eq(learner_id)
        ] if not quizzes.empty else quizzes
        completion_group = completions[
            completions["learner_id"].astype(str).eq(learner_id)
        ] if not completions.empty else completions

        timeline = []
        if not event_group.empty:
            event_times = event_group["_timestamp_utc"]
            for idx, ts in event_times.items():
                if pd.notna(ts):
                    timeline.append(
                        (ts, str(event_group.loc[idx].get("event_category", "event")))
                    )
        if not quiz_group.empty and (
            "finish_time_unix" in quiz_group or "finish_time" in quiz_group
        ):
            quiz_times = (
                pd.to_datetime(
                    pd.to_numeric(quiz_group["finish_time_unix"], errors="coerce"),
                    unit="s",
                    errors="coerce",
                    utc=True,
                )
                if "finish_time_unix" in quiz_group
                else pd.to_datetime(
                    quiz_group["finish_time"], errors="coerce", utc=True
                )
            )
            for ts in quiz_times.dropna():
                timeline.append((ts, "quiz_attempt"))
        if not completion_group.empty and (
            "time_modified_unix" in completion_group
            or "time_modified" in completion_group
        ):
            completion_times = (
                pd.to_datetime(
                    pd.to_numeric(
                        completion_group["time_modified_unix"], errors="coerce"
                    ),
                    unit="s",
                    errors="coerce",
                    utc=True,
                )
                if "time_modified_unix" in completion_group
                else pd.to_datetime(
                    completion_group["time_modified"], errors="coerce", utc=True
                )
            )
            for ts in completion_times.dropna():
                timeline.append((ts, "completion"))
        timeline.sort(key=lambda item: item[0])

        event_category = (
            event_group["event_category"].astype("string")
            if not event_group.empty and "event_category" in event_group
            else pd.Series(dtype="string")
        )
        if "is_learning_event" in event_group:
            learning_count = int(
                pd.to_numeric(event_group["is_learning_event"], errors="coerce")
                .eq(1)
                .sum()
            )
        else:
            learning_count = int(
                event_category.isin(
                    [
                        "content_view",
                        "lesson_interaction",
                        "question_interaction",
                        "quiz_attempt",
                    ]
                ).sum()
            )
        categories = set(event_category.dropna().tolist())
        # None (unknown) when the collector did not record sessions at all,
        # rather than a misleading 0.
        if "session_id" not in mapped_events.columns:
            session_count = None
        else:
            session_count = (
                int(event_group["session_id"].nunique()) if not event_group.empty else 0
            )
        dates = {item[0].date() for item in timeline}
        latest_type = timeline[-1][1] if timeline else None
        if latest_type is None and not completion_group.empty:
            latest_type = "completion"
        completed_count = None
        if (
            not completion_group.empty
            and {"completion_state", "course_module_id"}.issubset(
                completion_group.columns
            )
        ):
            completed_count = int(
                completion_group.loc[
                    pd.to_numeric(
                        completion_group["completion_state"], errors="coerce"
                    ).isin([1, 2]),
                    "course_module_id",
                ].nunique()
            )
        rows.append(
            {
                "learner_id": learner_id,
                "first_activity_at": timeline[0][0].isoformat() if timeline else None,
                "last_activity_at": timeline[-1][0].isoformat() if timeline else None,
                "total_learning_events": learning_count,
                "total_quiz_attempts": len(quiz_group),
                "total_completed_activities": completed_count,
                "distinct_topics_accessed": (
                    int(event_group["topic"].nunique())
                    if not event_group.empty and "topic" in event_group
                    else 0
                ),
                "historical_learning_days": len(dates),
                "historical_session_count": session_count,
                "latest_activity_type": latest_type,
                "history_status": (
                    "available"
                    if categories or len(quiz_group) or not completion_group.empty
                    else "no_observations"
                ),
                "generated_at": generated_at,
            }
        )
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)

