"""
Learner Model Module
Transforms raw Moodle analytics into a structured state representation.
"""

import argparse

from .build_model import build_learner_model

__all__ = ["build_learner_model", "main"]


def main():
    parser = argparse.ArgumentParser(
        description="Build the learner model from Moodle collector CSVs."
    )
    parser.add_argument("--data-dir", help="folder holding the collector CSVs")
    parser.add_argument(
        "--output", help="output CSV (default: <data-dir>/learner_model.csv)"
    )
    args = parser.parse_args()

    build_learner_model(data_dir=args.data_dir, output_file=args.output)
