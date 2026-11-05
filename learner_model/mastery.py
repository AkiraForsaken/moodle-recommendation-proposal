import pandas as pd

# Thresholds for static rule-based mastery determination
MASTERY_HIGH = 0.80
MASTERY_MEDIUM = 0.60


def compute_score_trend(quizzes):
    """
    Calculates the performance trajectory based on chronological attempts.
    """
    # Ensure sequential ordering by attempt ID or timestamps
    quiz_sorted = quizzes.sort_values(["learner_id", "quiz_id", "attempt"])

    first_scores = (
        quiz_sorted.groupby(["learner_id", "quiz_id"])["normalized_score"]
        .first()
        .rename("first_score")
    )
    last_scores = (
        quiz_sorted.groupby(["learner_id", "quiz_id"])["normalized_score"]
        .last()
        .rename("last_score")
    )

    score_trend = pd.concat([first_scores, last_scores], axis=1).reset_index()
    score_trend["score_trend"] = score_trend["last_score"] - score_trend["first_score"]

    return score_trend[["learner_id", "quiz_id", "score_trend"]]


def determine_mastery_level(score):
    """
    Categorizes the best cognitive score into a categorical mastery band.
    """
    if pd.isna(score):
        return "unknown"
    if score >= MASTERY_HIGH:
        return "high"
    if score >= MASTERY_MEDIUM:
        return "medium"
    return "low"


def evaluate_topic_state(model_df):
    """
    Evaluates the combined feature frame to determine mastery levels.
    """
    if "best_score" in model_df.columns:
        model_df["mastery_level"] = model_df["best_score"].apply(
            determine_mastery_level
        )
    else:
        model_df["mastery_level"] = "unknown"
    return model_df
