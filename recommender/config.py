"""
Recommender configuration: constants only, no I/O.

Everything that encodes a *modelling decision* (which activity types can be
recommended, what counts as mastery, which features define "similar learners")
lives here so it can be documented and changed in one place.
"""

import os
from pathlib import Path

DATA_DIR = Path(
    os.getenv("RECSYS_DATA_DIR", Path(__file__).resolve().parent.parent / "data")
)

# --------------------------------------------------------------------------
# Mastery
# --------------------------------------------------------------------------
# Keep in sync with learner_model/mastery.py. These bands are only used when a
# numeric mastery probability is available (BKT/DKT); the V1 learner model
# already ships a categorical mastery_level.
MASTERY_HIGH = float(os.getenv("MOODLE_MASTERY_THRESHOLD", "0.80"))
MASTERY_MEDIUM = 0.60

# When this column exists in the learner model (Phase 2A/2B output, P(L_t)) it
# overrides the categorical mastery_level. No recommender code changes needed.
MASTERY_PROB_COLUMN = "p_mastery"

# --------------------------------------------------------------------------
# Catalog interpretation
# --------------------------------------------------------------------------
# What a learner *does* with an activity. Only types listed here can ever be
# recommended (forums, question banks, labels... are excluded on purpose).
#   learn    -> study material (remediation / first exposure)
#   practice -> assessment-like (consolidation / checking mastery)
ROLE_BY_TYPE = {
    "lesson": "learn",
    "resource": "learn",
    "page": "learn",
    "book": "learn",
    "url": "learn",
    "quiz": "practice",
    "assign": "practice",
}

# Moodle section 0 is the course-wide "General" block (syllabus, announcements).
# It is not a learning topic, so nothing from it is recommended.
EXCLUDED_SECTION_NUMBERS = {0}

# --------------------------------------------------------------------------
# Collaborative filtering
# --------------------------------------------------------------------------
# Plan specifies score_trend + lesson_time_estimate. best_score is added
# because score_trend is 0 for every single-attempt quiz and therefore carries
# no information about *level* on its own. Remove it to follow the plan strictly.
CF_FEATURES = ("best_score", "score_trend", "lesson_time_estimate")
CF_LOG_FEATURES = ("lesson_time_estimate",)  # heavy-tailed -> log1p before scaling
CF_NEIGHBOURS = 10
CF_MIN_SIMILARITY = 0.0
