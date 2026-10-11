import pandas as pd
from pathlib import Path

# Local imports
from .features import (
    extract_quiz_features,
    extract_event_features,
    extract_completion_features,
)
from .mastery import compute_score_trend, evaluate_topic_state
from .temporal import truncate_frames

COLUMNS_ORDERED = [
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


def load_collector_frames(data_dir=None):
    """Read the collector CSVs once. Returns (activities, events, quizzes, completions)."""
    data_dir = (
        Path(__file__).resolve().parent.parent / "data"
        if data_dir is None
        else Path(data_dir)
    )
    activities = pd.read_csv(data_dir / "activity_catalog.csv")
    events = pd.read_csv(data_dir / "learning_events.csv")
    quizzes = pd.read_csv(data_dir / "quiz_attempts.csv")
    completion_file = data_dir / "course_modules_completion.csv"
    completions = (
        pd.read_csv(completion_file) if completion_file.exists() else pd.DataFrame()
    )
    return activities, events, quizzes, completions


def build_learner_model_from_frames(
    activities, events, quizzes, completions=None, as_of=None
):
    """
    Pure function: collector frames in, learner model out. No file I/O, so an
    evaluation loop can call it once per cutoff without touching the disk.

    as_of: see learner_model.temporal.resolve_cutoff. Only data strictly before
           the cutoff is used. None (default) uses everything, exactly as before.
    """
    if completions is None:
        completions = pd.DataFrame()

    # 0. Time-aware snapshot (no-op when as_of is None)
    events, quizzes, completions, cutoff = truncate_frames(
        events, quizzes, completions, as_of
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

    # Nothing known yet at this cutoff -> empty model with the standard schema
    if events.empty and quizzes.empty:
        empty = pd.DataFrame(columns=COLUMNS_ORDERED)
        empty.attrs["as_of_unix"] = cutoff
        return empty

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

    final_cols = [c for c in COLUMNS_ORDERED if c in model.columns]
    model = model[final_cols]

    # Sort by learner and sequence position
    model = model.sort_values(["learner_id", "sequence_position"])

    model = model.reset_index(drop=True)
    model.attrs["as_of_unix"] = cutoff
    return model


def build_learner_model(data_dir=None, output_file=None, as_of=None):
    """
    Primary execution pipeline: collector CSVs -> learner_model CSV.

    With as_of set, the output goes to learner_model_asof_<unix>.csv unless
    output_file is given, so a snapshot can never overwrite the full model.
    """
    data_dir = (
        Path(__file__).resolve().parent.parent / "data"
        if data_dir is None
        else Path(data_dir)
    )
    frames = load_collector_frames(data_dir)
    model = build_learner_model_from_frames(*frames, as_of=as_of)

    if output_file is None:
        cutoff = model.attrs.get("as_of_unix")
        if cutoff is None:
            output_file = data_dir / "learner_model.csv"
        elif isinstance(cutoff, dict):
            output_file = data_dir / "learner_model_asof_per_learner.csv"
        else:
            output_file = data_dir / f"learner_model_asof_{cutoff}.csv"

    model.to_csv(output_file, index=False)
    print(f"Learner Model created successfully: {output_file}")
    return model


if __name__ == "__main__":
    build_learner_model()
