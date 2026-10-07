#!/usr/bin/env python3
"""Thin CLI entry point for the recommendation engine.

    python recommendation-engine.py --learner-id 3

All logic lives in the `recommender/` package.
"""

from recommender import main

if __name__ == "__main__":
    main()
