import pandas as pd
from pathlib import Path

# Local imports
from .features import (
    extract_quiz_features,
    extract_event_features,
    extract_completion_features,
)
from .mastery import compute_score_trend, evaluate_topic_state


def build_learner_model(data_dir=None, output_file=None):
    """
    Primary execution pipeline to build the learner model from CSV logs.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    if output_file is None:
        output_file = data_dir / "learner_model.csv"

    # 1. Load Data
    activities = pd.read_csv(data_dir / "activity_catalog.csv")
    events = pd.read_csv(data_dir / "learning_events.csv")
    quizzes = pd.read_csv(data_dir / "quiz_attempts.csv")

    completion_file = data_dir / "course_modules_completion.csv"
    completions = (
        pd.read_csv(completion_file) if completion_file.exists() else pd.DataFrame()
    )

    # Catalog preparation
    activity_lookup = activities[
        [
            "course_module_id",
            "activity_type",
            "activity_id",
            "activity_name",
            "topic",
            "sequence",
            "completion_tracking",
        ]
    ].copy()

    # 2. Extract Event, Quiz & Completion Features
    quiz_features = extract_quiz_features(quizzes, activity_lookup)
    event_features = extract_event_features(events, activity_lookup)
    completion_features = extract_completion_features(completions, activity_lookup)

    # 3. Compute Score Trends & Merge
    score_trends = compute_score_trend(quizzes)
    quiz_features = quiz_features.merge(
        score_trends, on=["learner_id", "quiz_id"], how="left"
    )

    # Combine event features and quiz features
    model = pd.merge(
        event_features, quiz_features, on=["learner_id", "topic"], how="outer"
    )

    # Merge completion features if available
    if not completion_features.empty:
        model = pd.merge(
            model, completion_features, on=["learner_id", "topic"], how="left"
        )

    # Clean up NaNs
    model["activity_views"] = model["activity_views"].fillna(0).astype(int)
    model["quiz_attempts"] = model["quiz_attempts"].fillna(0).astype(int)
    model["quiz_time"] = model["quiz_time"].fillna(0).astype(int)

    if "completed_activities" in model.columns:
        model["completed_activities"] = (
            model["completed_activities"].fillna(0).astype(int)
        )
        model["topic_completion_pct"] = model["topic_completion_pct"].fillna(0.0)
    else:
        model["completed_activities"] = 0
        model["topic_completion_pct"] = 0.0

    # 4. Evaluate Topic Mastery
    model = evaluate_topic_state(model)

    # 5. Schema Ordering and Alignment
    topic_seq_map = (
        activity_lookup.dropna(subset=["topic"])
        .groupby("topic")["sequence"]
        .min()
        .to_dict()
    )
    model["sequence_position"] = model["sequence_position"].fillna(
        model["topic"].map(topic_seq_map)
    )

    columns_ordered = [
        "learner_id",
        "topic",
        "sequence_position",
        "activity_views",
        "lesson_sessions",
        "question_interactions",
        "quiz_attempts",
        "best_score",
        "latest_score",
        "score_trend",
        "quiz_time",
        "lesson_time_estimate",
        "completed_activities",
        "topic_completion_pct",
        "mastery_level",
        "last_activity",
    ]

    final_cols = [c for c in columns_ordered if c in model.columns]
    model = model[final_cols]

    # Sort by learner and sequence position
    model = model.sort_values(["learner_id", "sequence_position"])

    # 6. Save Output
    model.to_csv(output_file, index=False)
    print(f"Learner Model created successfully: {output_file}")

    return model


if __name__ == "__main__":
    build_learner_model()
