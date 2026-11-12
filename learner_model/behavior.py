"""Moodle learning-behavior summaries derived from collector events."""

import pandas as pd

BEHAVIOR_COLUMNS = [
    "learner_id",
    "observation_start",
    "observation_end",
    "session_count",
    "average_session_interval_seconds",
    "average_session_duration_seconds",
    "learning_activity_count",
    "practice_activity_count",
    "review_activity_count",
    "distinct_topics_accessed",
    "revisit_rate",
    "behavior_summary_status",
    "generated_at",
]


def _event_timestamps(events):
    if "timestamp_unix" in events.columns:
        unix = pd.to_numeric(events["timestamp_unix"], errors="coerce")
        return pd.to_datetime(unix, unit="s", errors="coerce", utc=True)
    return pd.to_datetime(events["timestamp"], errors="coerce", utc=True)


def _session_ids(events):
    if "session_id" in events.columns:
        return events["session_id"].astype("string")
    ordered = events.sort_values(["learner_id", "timestamp"])
    timestamps = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=True)
    gaps = timestamps.groupby(ordered["learner_id"]).diff().dt.total_seconds()
    starts = gaps.isna() | gaps.gt(1800)
    numbers = starts.groupby(ordered["learner_id"]).cumsum().astype(int)
    return pd.Series(
        ordered["learner_id"].astype(str) + "-session-" + numbers.astype(str),
        index=ordered.index,
    ).reindex(events.index)


def build_behavior(events, activities, generated_at):
    if events.empty:
        return pd.DataFrame(columns=BEHAVIOR_COLUMNS)
    required = {"learner_id", "timestamp", "event_category"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(
            f"Learning events are missing behavior fields: {sorted(missing)}"
        )

    frame = events.copy()
    frame["timestamp"] = _event_timestamps(frame)
    frame = frame.dropna(subset=["timestamp"])
    if frame.empty:
        return pd.DataFrame(columns=BEHAVIOR_COLUMNS)
    frame["_session_id"] = _session_ids(frame)
    if "course_module_id" in frame and "course_module_id" in activities:
        lookup = activities[["course_module_id", "topic"]].drop_duplicates(
            "course_module_id"
        )
        frame = frame.merge(lookup, on="course_module_id", how="left")
    else:
        frame["topic"] = pd.NA

    category = frame["event_category"].astype("string")
    if "is_learning_event" in frame:
        learning = pd.to_numeric(frame["is_learning_event"], errors="coerce").eq(1)
    else:
        learning = category.isin(
            [
                "content_view",
                "lesson_interaction",
                "question_interaction",
                "quiz_attempt",
            ]
        )
    frame["_learning"] = learning
    frame["_practice"] = category.isin(["quiz_attempt", "question_interaction"])
    frame["_review"] = category.eq("quiz_review")

    rows = []
    for learner_id, group in frame.groupby("learner_id", sort=False):
        sessions = (
            group.groupby("_session_id", dropna=False)
            .agg(
                session_start=("timestamp", "min"),
                session_end=("timestamp", "max"),
            )
            .sort_values("session_start")
        )
        session_intervals = sessions["session_start"].diff().dt.total_seconds().dropna()
        topics = group.dropna(subset=["topic"])
        # A topic is revisited when the learner comes back to it in a later
        # session. Several views within one visit (overview, lesson, quiz)
        # are one visit, not a revisit.
        topic_sessions = (
            topics[topics["_learning"]].groupby("topic")["_session_id"].nunique()
        )
        topic_count = int(topics["topic"].nunique())
        revisit_rate = (
            float(topic_sessions.gt(1).sum() / topic_count) if topic_count else None
        )
        rows.append(
            {
                "learner_id": str(learner_id),
                "observation_start": group["timestamp"].min().isoformat(),
                "observation_end": group["timestamp"].max().isoformat(),
                "session_count": int(sessions.shape[0]),
                "average_session_interval_seconds": (
                    float(session_intervals.mean())
                    if not session_intervals.empty
                    else None
                ),
                "average_session_duration_seconds": float(
                    (sessions["session_end"] - sessions["session_start"])
                    .dt.total_seconds()
                    .mean()
                ),
                "learning_activity_count": int(group["_learning"].sum()),
                "practice_activity_count": int(group["_practice"].sum()),
                "review_activity_count": int(group["_review"].sum()),
                "distinct_topics_accessed": topic_count,
                "revisit_rate": revisit_rate,
                "behavior_summary_status": "available",
                "generated_at": generated_at,
            }
        )
    return pd.DataFrame(rows, columns=BEHAVIOR_COLUMNS)
