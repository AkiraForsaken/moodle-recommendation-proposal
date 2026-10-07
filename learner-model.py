#!/usr/bin/env python3
"""
Thin CLI entry point for the learner model.

    python learner-model.py
    python learner-model.py --data-dir data_snapshot --output snapshot_model.csv

Reads the collector's CSVs from the data directory (default: ./data) and writes
learner_model.csv next to them. All logic lives in the `model/` package.
"""

from model import main


if __name__ == "__main__":
    main()
