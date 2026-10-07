import pandas as pd
import numpy as np


def get_uncompleted_catalog(learner_id, learner_model_df, catalog_df):
    """Returns candidate modules that the learner has not yet completed."""
    # Assuming sequence_position tracking determines completion boundary
    # For a stricter check, this could cross-reference course_modules_completion.csv
    student_state = learner_model_df[learner_model_df["learner_id"] == learner_id]
    if student_state.empty:
        return catalog_df

    current_seq = student_state["sequence_position"].max()
    return catalog_df[catalog_df["sequence"] >= current_seq].copy()


def rule_based_recommendation(learner_id, learner_model_df, catalog_df):
    """
    Knowledge-Based Routing: Recommends remediation, practice, or advancement
    based on the heuristic mastery_level of the current topic[cite: 3].
    """
    student_state = learner_model_df[learner_model_df["learner_id"] == learner_id]
    if student_state.empty:
        return pd.DataFrame(), "No learner state found."

    current_row = student_state.sort_values("sequence_position", ascending=False).iloc[
        0
    ]
    mastery = current_row["mastery_level"]
    current_seq = current_row["sequence_position"]
    topic = current_row["topic"]

    candidates = catalog_df.copy()

    if mastery == "low":
        reason = f"Low mastery in '{topic}'. Recommending remediation review."
        # Low Mastery: Recommends remediation, such as review lessons or basic concepts, for the current topic[cite: 3].
        recs = candidates[
            (candidates["sequence"] == current_seq)
            & (candidates["activity_type"].isin(["lesson", "page", "resource"]))
        ]
    elif mastery == "medium":
        reason = f"Medium mastery in '{topic}'. Recommending targeted practice."
        # Medium Mastery: Recommends additional practice, such as targeted quizzes[cite: 3].
        recs = candidates[
            (candidates["sequence"] == current_seq)
            & (candidates["activity_type"] == "quiz")
        ]
    elif mastery == "high":
        reason = f"High mastery in '{topic}'. Advancing to next sequence."
        # High Mastery: Advances the student to the first module of the next sequence position[cite: 3].
        recs = candidates[candidates["sequence"] > current_seq].sort_values("sequence")
    else:
        reason = f"Starting '{topic}'."
        recs = candidates[candidates["sequence"] == current_seq]

    return recs, reason


def popularity_based_recommendation(catalog_df, events_df):
    """
    Popularity-Based Filtering: Ranks learning activities based on global
    aggregate metrics across all users[cite: 3].
    """
    if events_df.empty:
        return catalog_df, "No events to calculate popularity."

    # Ranks candidate learning activities based on the highest overall activity_views[cite: 3].
    view_counts = events_df[events_df["event_category"] == "content_view"]
    view_counts = (
        view_counts.groupby("course_module_id")
        .size()
        .reset_index(name="global_activity_views")
    )

    ranked_catalog = catalog_df.merge(view_counts, on="course_module_id", how="left")
    ranked_catalog["global_activity_views"] = ranked_catalog[
        "global_activity_views"
    ].fillna(0)
    ranked_catalog = ranked_catalog.sort_values(
        "global_activity_views", ascending=False
    )

    reason = "Historically favored content based on global activity views."
    return ranked_catalog, reason


def content_based_recommendation(learner_id, learner_model_df, catalog_df):
    """
    Content-Based Filtering: Identifies semantically relevant content by matching
    the learner's current topic with metadata of uncompleted modules[cite: 3].
    """
    candidates = get_uncompleted_catalog(learner_id, learner_model_df, catalog_df)
    student_state = learner_model_df[learner_model_df["learner_id"] == learner_id]

    if student_state.empty:
        return (
            candidates,
            "No context available; returning sequential uncompleted modules.",
        )

    current_topic = student_state.sort_values(
        "sequence_position", ascending=False
    ).iloc[0]["topic"]

    # Recommends activities by matching the semantic tags (topic) of the learner's current context[cite: 3].
    recs = candidates[candidates["topic"] == current_topic]
    if recs.empty:
        recs = candidates  # Fallback if topic is finished

    reason = f"Content matched to current semantic topic: '{current_topic}'."
    return recs, reason


def collaborative_filtering_recommendation(learner_id, learner_model_df, catalog_df):
    """
    Collaborative Filtering: Identifies similar students using score_trend and
    lesson_time_estimate profiles to suggest modules they completed[cite: 3].
    """
    if learner_model_df.empty or len(learner_model_df["learner_id"].unique()) <= 1:
        return pd.DataFrame(), "Insufficient peer data for collaborative filtering."

    # Aggregate profiles per user
    user_profiles = (
        learner_model_df.groupby("learner_id")
        .agg(
            avg_score_trend=("score_trend", "mean"),
            total_lesson_time=("lesson_time_estimate", "sum"),
        )
        .reset_index()
    )

    target_user = user_profiles[user_profiles["learner_id"] == learner_id]
    if target_user.empty:
        return pd.DataFrame(), "No profile found for target user."

    # Identify similar students using comparable score_trend and lesson_time_estimate profiles[cite: 3].
    peers = user_profiles[user_profiles["learner_id"] != learner_id].copy()

    # Simple Euclidean distance for baseline
    target_trend = target_user["avg_score_trend"].values[0]
    target_time = target_user["total_lesson_time"].values[0]

    # Normalize before distance calc to prevent time (large scale) dominating trend (small scale)
    trend_std = user_profiles["avg_score_trend"].std() or 1
    time_std = user_profiles["total_lesson_time"].std() or 1

    peers["distance"] = np.sqrt(
        ((peers["avg_score_trend"] - target_trend) / trend_std) ** 2
        + ((peers["total_lesson_time"] - target_time) / time_std) ** 2
    )

    nearest_peer = peers.sort_values("distance").iloc[0]["learner_id"]

    # Suggests modules that similar students have successfully completed[cite: 3].
    peer_history = learner_model_df[learner_model_df["learner_id"] == nearest_peer]
    peer_advanced_seq = peer_history["sequence_position"].max()

    uncompleted = get_uncompleted_catalog(learner_id, learner_model_df, catalog_df)
    recs = uncompleted[uncompleted["sequence"] <= peer_advanced_seq]

    if recs.empty:
        recs = uncompleted

    reason = f"Recommended based on similarity to peer Learner ID {nearest_peer}."
    return recs, reason
