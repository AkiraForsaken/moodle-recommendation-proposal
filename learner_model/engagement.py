"""Moodle participation and completion summaries."""

import pandas as pd

ENGAGEMENT_COLUMNS = [
    "learner_id",
    "active_session_count",
    "total_learning_events",
    "total_activity_views",
    "estimated_learning_time_seconds",
    "completed_activity_count",
    "topic_completion_rate",
    "course_progress",
    "last_active_at",
    "days_since_last_activity",
    "observation_start",
    "observation_end",
    "engagement_status",
    "generated_at",
]


def build_engagement(
    events,
    quizzes,
    completions,
    activities,
    # topic_state,
    generated_at,
    as_of=None,
    completion_tracking_available=True,
):
    if not completions.empty and "learner_id" in completions.columns:
        completions = completions.copy()
        completions["learner_id"] = completions["learner_id"].astype(str)

    learner_ids = set()
    for frame in (events, quizzes, completions):  # topic_state
        if not frame.empty and "learner_id" in frame.columns:
            learner_ids.update(frame["learner_id"].dropna().astype(str))
    if not learner_ids:
        return pd.DataFrame(columns=ENGAGEMENT_COLUMNS)

    event_counts = {}
    if not events.empty:
        for learner_id, group in events.groupby("learner_id"):
            categories = group.get("event_category", pd.Series(dtype="string")).astype(
                "string"
            )
            if "is_learning_event" in group:
                total = int(
                    pd.to_numeric(group["is_learning_event"], errors="coerce")
                    .eq(1)
                    .sum()
                )
            else:
                total = int(
                    categories.isin(
                        [
                            "content_view",
                            "lesson_interaction",
                            "question_interaction",
                            "quiz_attempt",
                        ]
                    ).sum()
                )
            sessions = group["session_id"].nunique() if "session_id" in group else None
            timestamps = (
                pd.to_datetime(
                    pd.to_numeric(group["timestamp_unix"], errors="coerce"),
                    unit="s",
                    errors="coerce",
                    utc=True,
                )
                if "timestamp_unix" in group
                else pd.to_datetime(group["timestamp"], errors="coerce", utc=True)
            )
            valid = timestamps.dropna()
            event_counts[str(learner_id)] = {
                "active_session_count": None if sessions is None else int(sessions),
                "total_learning_events": total,
                "total_activity_views": int(categories.eq("content_view").sum()),
                "estimated_learning_time_seconds": _estimated_event_time(group),
                "last_active_at": valid.max().isoformat() if not valid.empty else None,
                "observation_start": (
                    valid.min().isoformat() if not valid.empty else None
                ),
                "observation_end": valid.max().isoformat() if not valid.empty else None,
            }
    if not quizzes.empty:
        if "duration_seconds" in quizzes:
            quiz_durations = pd.to_numeric(
                quizzes["duration_seconds"], errors="coerce"
            ).fillna(0)
            for learner_id, indices in quizzes.groupby("learner_id").groups.items():
                current = event_counts.setdefault(str(learner_id), {})
                current["estimated_learning_time_seconds"] = current.get(
                    "estimated_learning_time_seconds", 0
                ) + int(quiz_durations.loc[indices].sum())
        quiz_time_column = (
            "finish_time_unix"
            if "finish_time_unix" in quizzes
            else "finish_time" if "finish_time" in quizzes else None
        )
        if quiz_time_column:
            for learner_id, group in quizzes.groupby("learner_id"):
                times = (
                    pd.to_datetime(
                        pd.to_numeric(group[quiz_time_column], errors="coerce"),
                        unit="s",
                        errors="coerce",
                        utc=True,
                    )
                    if quiz_time_column.endswith("_unix")
                    else pd.to_datetime(
                        group[quiz_time_column], errors="coerce", utc=True
                    )
                ).dropna()
                if times.empty:
                    continue
                current = event_counts.setdefault(str(learner_id), {})
                existing_last = current.get("last_active_at")
                existing_first = current.get("observation_start")
                last = pd.Timestamp(existing_last) if existing_last else None
                first = pd.Timestamp(existing_first) if existing_first else None
                current["last_active_at"] = (
                    max(last, times.max()).isoformat()
                    if last
                    else times.max().isoformat()
                )
                current["observation_start"] = (
                    min(first, times.min()).isoformat()
                    if first
                    else times.min().isoformat()
                )
                current["observation_end"] = current["last_active_at"]

    # topic_aggregates = {}
    # if not topic_state.empty:
    #     for learner_id, group in topic_state.groupby("learner_id"):
    #         topic_aggregates[str(learner_id)] = {
    #             "total_activity_views": int(
    #                 pd.to_numeric(group["activity_views"], errors="coerce")
    #                 .fillna(0)
    #                 .sum()
    #             ),
    #             "estimated_learning_time_seconds": int(
    #                 pd.to_numeric(group["lesson_time_estimate"], errors="coerce")
    #                 .fillna(0)
    #                 .sum()
    #                 + pd.to_numeric(group["quiz_time"], errors="coerce").fillna(0).sum()
    #             ),
    #         }

    trackable = activities
    if "completion_tracking" in trackable:
        trackable = trackable[
            pd.to_numeric(trackable["completion_tracking"], errors="coerce").gt(0)
        ]

    course_trackable_count = (
        int(trackable["course_module_id"].nunique())
        if "course_module_id" in trackable and "completion_tracking" in trackable
        else 0
    )
    completion_summary = {}
    if (
        completion_tracking_available
        and not completions.empty
        and {"learner_id", "course_module_id", "completion_state"}.issubset(
            completions.columns
        )
        and "course_module_id" in trackable
    ):
        tracked = completions.merge(
            trackable[["course_module_id"]].drop_duplicates(),
            on="course_module_id",
            how="inner",
        )
        if not tracked.empty:
            tracked["_complete"] = pd.to_numeric(
                tracked["completion_state"], errors="coerce"
            ).isin([1, 2])
            # Ensure groupby keys are converted to string dictionaries explicitly
            completion_summary = {
                str(k): v
                for k, v in tracked.groupby("learner_id")
                .agg(
                    completed_activity_count=("_complete", "sum"),
                    observed_trackable_activity_count=("course_module_id", "nunique"),
                )
                .to_dict("index")
                .items()
            }

    cutoff_value = generated_at if as_of is None else as_of
    if isinstance(cutoff_value, dict):
        cutoff_by_learner = {
            str(key): pd.to_datetime(value, unit="s", utc=True)
            for key, value in cutoff_value.items()
        }
    elif isinstance(cutoff_value, (int, float)):
        cutoff_by_learner = pd.to_datetime(cutoff_value, unit="s", utc=True)
    elif isinstance(cutoff_value, str) and cutoff_value.strip().lstrip("-").isdigit():
        cutoff_by_learner = pd.to_datetime(int(cutoff_value), unit="s", utc=True)
    else:
        cutoff_by_learner = pd.to_datetime(cutoff_value, errors="coerce", utc=True)
    rows = []
    for learner_id in sorted(learner_ids):
        event = event_counts.get(learner_id, {})
        # topic = topic_aggregates.get(learner_id, {})
        learner_completion = completion_summary.get(learner_id)
        completed_count = (
            int(learner_completion["completed_activity_count"])
            if learner_completion
            else None
        )
        tracked_count = (
            int(learner_completion["observed_trackable_activity_count"])
            if learner_completion
            else 0
        )
        rate = float(completed_count / tracked_count) if tracked_count else None
        # Moodle's own notion of progress: completed / every activity in the
        # course that tracks completion (not only those the learner touched).
        progress = (
            float(completed_count / course_trackable_count)
            if completed_count is not None and course_trackable_count
            else None
        )
        last_active = event.get("last_active_at")
        days_since = None
        cutoff = (
            cutoff_by_learner.get(learner_id)
            if isinstance(cutoff_by_learner, dict)
            else cutoff_by_learner
        )
        if last_active and cutoff is not None and not pd.isna(cutoff):
            days_since = max(
                0.0,
                (cutoff - pd.Timestamp(last_active)).total_seconds() / 86400,
            )
        has_events = learner_id in event_counts
        rows.append(
            {
                "learner_id": learner_id,
                "active_session_count": event.get("active_session_count", 0),
                "total_learning_events": event.get("total_learning_events", 0),
                "total_activity_views": event.get("total_activity_views", 0),
                "estimated_learning_time_seconds": event.get(
                    "estimated_learning_time_seconds", 0
                ),
                "completed_activity_count": (
                    completed_count
                    if completion_tracking_available and learner_completion
                    else None
                ),
                "topic_completion_rate": rate,
                "course_progress": progress,
                "last_active_at": last_active,
                "days_since_last_activity": days_since,
                "observation_start": event.get("observation_start"),
                "observation_end": event.get("observation_end"),
                "engagement_status": (
                    "available"
                    if has_events and completion_tracking_available
                    else "partial" if has_events else "unavailable"
                ),
                "generated_at": generated_at,
            }
        )
    return pd.DataFrame(rows, columns=ENGAGEMENT_COLUMNS)


def _estimated_event_time(events, max_delta_seconds=1800):
    """Sum bounded learning-event gaps; quiz duration is added separately."""
    if "seconds_since_prev_event" not in events:
        return 0
    category = events.get(
        "event_category", pd.Series(index=events.index, dtype="string")
    )
    learning = (
        pd.to_numeric(events["is_learning_event"], errors="coerce").eq(1)
        if "is_learning_event" in events
        else category.isin(
            ["content_view", "lesson_interaction", "question_interaction"]
        )
    )
    deltas = pd.to_numeric(events["seconds_since_prev_event"], errors="coerce")
    valid = learning & deltas.gt(0) & deltas.le(max_delta_seconds)
    if "event_category" in events:
        valid &= category.isin(
            ["content_view", "lesson_interaction", "question_interaction"]
        )
    return int(deltas.loc[valid].sum())
