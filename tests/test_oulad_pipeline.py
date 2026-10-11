"""
Pipeline Integration Test
Runs the complete loop: OULAD ETL -> Temporal Learner Model -> Recommender Engine
"""

from pathlib import Path
import pandas as pd

from adapters.oulad_adapter import transform_oulad_to_collector_schema
from model.build_model import build_learner_model
from recommender.data import RecommenderData
from recommender.engine import RecommendationEngine, run_baseline_recommendations


def run_pipeline_test():
    raw_oulad_path = Path("data/oulad_raw")
    processed_path = Path("data/oulad_processed")

    # Step 1: Run OULAD ETL
    print("=== STEP 1: Transforming OULAD Data ===")
    if not (processed_path / "activity_catalog.csv").exists():
        transform_oulad_to_collector_schema(raw_oulad_path, processed_path)
    else:
        print("Processed OULAD CSVs found. Skipping transform.")

    # Step 2: Test Temporal Learner Model Creation
    print("\n=== STEP 2: Building Temporal Learner Model ===")
    # Pick a point-in-time cutoff (e.g., day 100 after epoch)
    cutoff_dt = pd.Timestamp("2024-04-10 00:00:00")
    cutoff_unix = int(cutoff_dt.timestamp())

    # Generate point-in-time snapshot
    lm_asof = build_learner_model(data_dir=processed_path, as_of=cutoff_unix)
    print(f"Snapshot Learner Model rows (as_of {cutoff_dt}): {len(lm_asof)}")

    # Generate full model for fallback/comparison
    lm_full = build_learner_model(data_dir=processed_path, as_of=None)
    print(f"Full Learner Model rows: {len(lm_full)}")

    # Step 3: Initialize Recommender Data Layer
    print("\n=== STEP 3: Initializing Recommender Data Layer ===")
    data = RecommenderData.from_moodle_csv(
        data_dir=processed_path, learner_model_file="learner_model.csv"
    )
    sample_learner_id = data.learner_ids[0]
    print(
        f"Loaded {len(data.learner_ids)} learners into RecommenderData. Testing with Learner ID: {sample_learner_id}"
    )

    # Step 4: Run Baseline Recommendation Engine
    print(
        f"\n=== STEP 4: Running Baseline Recommenders for Learner {sample_learner_id} ==="
    )
    engine = RecommendationEngine(data)

    for method in engine.methods:
        recs = engine.recommend(learner_id=sample_learner_id, k=3, method=method)
        print(f"\n--- Method: {method} ---")
        for r in recs:
            fallback_flag = " [FALLBACK]" if r.via_fallback else ""
            review_flag = " ↺" if r.is_review else ""
            print(
                f"  Rank {r.rank}: CMID {r.course_module_id} ({r.activity_name} | {r.topic}){review_flag}{fallback_flag}"
            )
            print(f"    Score: {r.score:.4f} | Reason: {r.reason}")


if __name__ == "__main__":
    run_pipeline_test()
