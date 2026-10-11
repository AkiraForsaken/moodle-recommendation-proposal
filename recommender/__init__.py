"""
Recommendation Engine Module

Ranks the Next Best Activity for a learner using four baseline methods
(rule-based, popularity, content-based, collaborative filtering) behind one
common interface. Reads the standardized learner model, so the same code runs
on the Moodle pipeline output and on the OULAD ETL output.
"""

import argparse
import sys

from .data import RecommenderData
from .engine import RecommendationEngine, run_baseline_recommendations

__all__ = ["RecommenderData", "RecommendationEngine", "run_baseline_recommendations", "main"]


def main():
    parser = argparse.ArgumentParser(description="Generate baseline activity recommendations.")
    parser.add_argument("--learner-id", type=int, default=3, help="learner ID (default: 3)")
    parser.add_argument("--data-dir", help="folder holding the pipeline CSVs")
    parser.add_argument("--k", type=int, default=5, help="list length (default: 5)")
    parser.add_argument("--method", action="append",
                        choices=["rule_based", "popularity", "content_based", "collaborative"],
                        help="restrict to a method (repeatable); default: all")
    args = parser.parse_args()
    try:
        run_baseline_recommendations(args.learner_id, args.data_dir, k=args.k, methods=args.method)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
