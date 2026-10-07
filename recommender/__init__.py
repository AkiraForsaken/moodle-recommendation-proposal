"""
Recommendation Engine Module
Generates Next Best Activity rankings evaluating candidate modules using multiple baseline approaches[cite: 3].
"""

import argparse

from .engine import run_baseline_recommendations

__all__ = ["run_baseline_recommendations", "main"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline activity recommendations for a learner."
    )
    parser.add_argument(
        "--learner-id",
        type=int,
        default=3,
        help="learner ID to generate recommendations for (default: 3)",
    )
    parser.add_argument("--data-dir", help="folder holding the pipeline CSVs")
    args = parser.parse_args()

    run_baseline_recommendations(learner_id=args.learner_id, data_dir=args.data_dir)
