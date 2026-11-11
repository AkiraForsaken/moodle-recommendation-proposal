"""Configuration for the complete learner-model build."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_STATES = (
    PROJECT_ROOT
    / "knowledge_tracing"
    / "artifacts"
    / "knowledge_states"
    / "learner_skill_state.csv"
)
DEFAULT_MOODLE_KNOWLEDGE_STATES = (
    PROJECT_ROOT
    / "knowledge_tracing"
    / "artifacts"
    / "moodle_kt"
    / "knowledge_states"
    / "learner_skill_state.csv"
)
DEFAULT_CONCEPT_MAP = PROJECT_ROOT / "data" / "processed" / "concept_map.csv"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MOODLE_ROSTER = PROJECT_ROOT / "data" / "learner_roster.csv"
DEFAULT_MAPPINGS_DIR = PROJECT_ROOT / "learner_model" / "mappings"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "learner_model" / "artifacts" / "complete_learner_model"
)

PROFILE_SCHEMA_VERSION = "1.0.0"
MAPPING_VERSION = "0.1.0-draft"

# These are auditable screening thresholds, not calibrated mastery cutoffs.
LOW_PREDICTED_CORRECTNESS = 0.60
# Reserved for future advancement/ranking use. It is deliberately
# NOT used by weak-area classification: a high prediction is not verified mastery.
HIGH_PREDICTED_CORRECTNESS = 0.80
MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN = 3
LOW_OBSERVED_ACCURACY = 0.60
