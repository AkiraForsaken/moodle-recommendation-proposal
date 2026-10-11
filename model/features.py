import pandas as pd


def extract_quiz_features(quizzes, activity_lookup):
    """
    Extract performance features and timing from quiz attempts.
    """
    quizzes = quizzes.copy()
    quizzes["quiz_id"] = pd.to_numeric(quizzes["quiz_id"], errors="coerce")
    # latest_score uses .last(), so attempts must be in attempt order
    quizzes = quizzes.sort_values(["learner_id", "quiz_id", "attempt"])

    # Aggregate basic quiz features per learner and quiz
    quiz_summary = quizzes.groupby(
        ["learner_id", "quiz_id", "quiz_name"], as_index=False
    ).agg(
        quiz_attempts=("attempt", "count"),
        best_score=("normalized_score", "max"),
        latest_score=("normalized_score", "last"),
        quiz_time=("duration_seconds", "sum"),
        latest_quiz_time=("duration_seconds", "last"),
    )

    # Filter catalog for quizzes to get topic and sequence positioning
    quiz_topic_lookup = activity_lookup[activity_lookup["activity_type"] == "quiz"][
        ["activity_id", "topic", "sequence"]
    ].rename(columns={"activity_id": "quiz_id", "sequence": "sequence_position"})

    quiz_topic_lookup["quiz_id"] = pd.to_numeric(
        quiz_topic_lookup["quiz_id"], errors="coerce"
    )

    # Merge topics back to the quiz features
    quiz_summary = quiz_summary.merge(quiz_topic_lookup, on="quiz_id", how="left")

    return quiz_summary


def extract_event_features(events, activity_lookup):
    """
    Extract engagement features, time estimates, and interactions using the
    explicit event_category, session_id and delta columns produced by the collector
    """
    events = events.copy()
    activity_lookup = activity_lookup.copy()

    # Cast join keys to numeric to prevent string vs float merge mismatches
    events["course_module_id"] = pd.to_numeric(
        events["course_module_id"], errors="coerce"
    )
    activity_lookup["course_module_id"] = pd.to_numeric(
        activity_lookup["course_module_id"], errors="coerce"
    )

    # Map topics and sequences to events using the catalog
    events_merged = events.merge(
        activity_lookup[["course_module_id", "topic", "sequence"]],
        on="course_module_id",
        how="left",
    )

    # Filter to events that have a recognized topic mapping
    topic_events = events_merged.dropna(subset=["topic"]).copy()

    topic_events["timestamp"] = pd.to_datetime(
        topic_events["timestamp"], errors="coerce"
    )
    topic_events = topic_events.sort_values(by=["learner_id", "topic", "timestamp"])

    # 1. Base Activity View Counts & Start/End Times (content_view category)
    content_views = topic_events[topic_events["event_category"] == "content_view"]
    view_summary = content_views.groupby(["learner_id", "topic"], as_index=False).agg(
        activity_views=("log_id", "count"),
    )
    event_summary = topic_events.groupby(["learner_id", "topic"], as_index=False).agg(
        first_activity=("timestamp", "min"),
        last_activity=("timestamp", "max"),
    )
    event_summary = event_summary.merge(
        view_summary, on=["learner_id", "topic"], how="left"
    )
    event_summary["activity_views"] = (
        event_summary["activity_views"].fillna(0).astype(int)
    )

    # 2. Count distinct lesson/content sessions using session_id
    # lesson_sessions = the number of the learner's engagement sessions
    # (30-minute-gap rule) that touched this topic through a content_view or lesson_interaction
    # event. It is NOT a literal count of lesson_started / lesson_resumed
    # events. Consequence: it is capped by the learner's total session count
    # for the whole course, so a learner who finishes the course in one sitting
    # reports lesson_sessions = 1 for every topic.
    lesson_events = topic_events[
        topic_events["event_category"].isin(["lesson_interaction", "content_view"])
    ]
    lesson_summary = (
        lesson_events.groupby(["learner_id", "topic"])["session_id"]
        .nunique()
        .reset_index(name="lesson_sessions")
    )

    # 3. Count explicit Question Interactions
    question_events = topic_events[
        topic_events["event_category"] == "question_interaction"
    ]
    question_summary = (
        question_events.groupby(["learner_id", "topic"])
        .size()
        .reset_index(name="question_interactions")
    )

    # Merge metrics
    event_summary = event_summary.merge(
        lesson_summary, on=["learner_id", "topic"], how="left"
    )
    event_summary = event_summary.merge(
        question_summary, on=["learner_id", "topic"], how="left"
    )

    event_summary["lesson_sessions"] = (
        event_summary["lesson_sessions"].fillna(0).astype(int)
    )
    event_summary["question_interactions"] = (
        event_summary["question_interactions"].fillna(0).astype(int)
    )

    # 4. Estimate Lesson Time from pre-calculated seconds_since_prev_event
    lesson_time_df = estimate_lesson_time(topic_events)
    event_summary = event_summary.merge(
        lesson_time_df, on=["learner_id", "topic"], how="left"
    )
    event_summary["lesson_time_estimate"] = (
        event_summary["lesson_time_estimate"].fillna(0).astype(int)
    )

    return event_summary


# Event categories whose time counts toward lesson_time_estimate.
#
# quiz_attempt and quiz_review are deliberately EXCLUDED even though the
# collector flags quiz_attempt as a learning event (is_learning_event == 1):
# quiz duration is already measured precisely from attempt start/finish in
# quiz_attempts.csv and reported as quiz_time. Counting it here too would
# report the same interval under two feature names.
#
# These strings must match the categories emitted by collector/events.py.
ENGAGEMENT_TIME_CATEGORIES = {
    "content_view",
    "lesson_interaction",
    "question_interaction",
}


def estimate_lesson_time(topic_events_df, max_delta_seconds=1800):
    """
    Calculates estimated engagement time (seconds) by summing pre-calculated
    event deltas for learning events within valid session bounds (<= 1800s).

    Known limitation: each delta is the gap BEFORE an event, so it is attributed
    to the event that ended the gap (and that event's topic), not to the one
    the learner was viewing during the gap.
    """
    df = topic_events_df.copy()

    # Convert deltas to numeric
    df["seconds_since_prev_event"] = pd.to_numeric(
        df["seconds_since_prev_event"], errors="coerce"
    ).fillna(0)

    # Sum deltas for actual learning events where gap is within boundary
    valid_events = df[
        (df["is_learning_event"] == 1)
        & (df["seconds_since_prev_event"] > 0)
        & (df["seconds_since_prev_event"] <= max_delta_seconds)
    ]

    time_estimate = (
        valid_events.groupby(["learner_id", "topic"])["seconds_since_prev_event"]
        .sum()
        .reset_index(name="lesson_time_estimate")
    )

    return time_estimate


def extract_completion_features(completions, activity_lookup):
    """
    Extracts activity completion counts and topic completion percentage
    from course_modules_completion.csv.
    """
    if completions.empty:
        return pd.DataFrame(
            columns=[
                "learner_id",
                "topic",
                "completed_activities",
                "topic_completion_pct",
            ]
        )

    completions = completions.copy()
    completions["course_module_id"] = pd.to_numeric(
        completions["course_module_id"], errors="coerce"
    )

    # Filter completed activities (state 1 or 2 = complete/pass)
    completed_df = completions[completions["completion_state"].isin([1, 2])]

    completed_summary = (
        completed_df.groupby(["learner_id", "topic"])
        .agg(completed_activities=("course_module_id", "nunique"))
        .reset_index()
    )

    # Calculate total trackable modules per topic from catalog
    catalog_trackable = activity_lookup[activity_lookup["completion_tracking"] > 0]
    topic_totals = (
        catalog_trackable.groupby("topic")["course_module_id"].nunique().to_dict()
    )

    completed_summary["total_topic_modules"] = (
        completed_summary["topic"].map(topic_totals).fillna(1)
    )
    completed_summary["topic_completion_pct"] = (
        completed_summary["completed_activities"]
        / completed_summary["total_topic_modules"]
    ).round(4)

    return completed_summary[
        ["learner_id", "topic", "completed_activities", "topic_completion_pct"]
    ]
