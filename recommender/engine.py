import pandas as pd
from pathlib import Path

# Local imports
from .methods import (
    rule_based_recommendation,
    popularity_based_recommendation,
    content_based_recommendation,
    collaborative_filtering_recommendation,
    get_uncompleted_catalog,
)


def run_baseline_recommendations(learner_id, data_dir=None):
    """
    Reads the structured state from learner_model.csv and ranks the Next Best Activity
    based on multiple baseline approaches[cite: 3].
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    # 1. Load Required Artifacts
    try:
        learner_model = pd.read_csv(data_dir / "learner_model.csv")
        catalog = pd.read_csv(data_dir / "activity_catalog.csv")
        events = pd.read_csv(data_dir / "learning_events.csv")
    except FileNotFoundError as e:
        return f"Error loading data: {e}"

    print(f"--- Generating Baseline Recommendations for Learner {learner_id} ---")

    # 2. Rule-Based
    rule_recs, rule_reason = rule_based_recommendation(
        learner_id, learner_model, catalog
    )
    print(f"\n[Knowledge-Based] {rule_reason}")
    if not rule_recs.empty:
        print(
            f"Top Pick: {rule_recs.iloc[0]['activity_name']} (Type: {rule_recs.iloc[0]['activity_type']})"
        )

    # 3. Content-Based
    content_recs, content_reason = content_based_recommendation(
        learner_id, learner_model, catalog
    )
    print(f"\n[Content-Based] {content_reason}")
    if not content_recs.empty:
        print(
            f"Top Pick: {content_recs.iloc[0]['activity_name']} (Topic: {content_recs.iloc[0]['topic']})"
        )

    # 4. Popularity-Based
    uncompleted_catalog = get_uncompleted_catalog(learner_id, learner_model, catalog)
    pop_recs, pop_reason = popularity_based_recommendation(uncompleted_catalog, events)
    print(f"\n[Popularity-Based] {pop_reason}")
    if not pop_recs.empty:
        print(
            f"Top Pick: {pop_recs.iloc[0]['activity_name']} (Views: {pop_recs.iloc[0].get('global_activity_views', 0)})"
        )

    # 5. Collaborative Filtering
    cf_recs, cf_reason = collaborative_filtering_recommendation(
        learner_id, learner_model, catalog
    )
    print(f"\n[Collaborative Filtering] {cf_reason}")
    if not cf_recs.empty:
        print(f"Top Pick: {cf_recs.iloc[0]['activity_name']}")

    # Return structured output for Phase 4 Evaluation
    return {
        "rule_based": rule_recs,
        "content_based": content_recs,
        "popularity_based": pop_recs,
        "collaborative": cf_recs,
    }


if __name__ == "__main__":
    # Test execution for Learner ID 3
    run_baseline_recommendations(learner_id=3)
