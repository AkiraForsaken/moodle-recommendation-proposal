"""Configuration for the learner-model build."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
# The only knowledge input: the contract file written by `knowledge_engine`.
# No benchmark dataset or benchmark learner state is ever read here.
DEFAULT_KNOWLEDGE_STATES = (
    PROJECT_ROOT
    / "knowledge_engine"
    / "artifacts"
    / "knowledge_states"
    / "learner_kc_state.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "learner_model" / "artifacts" / "learner_model"

KC_CATALOG_FILE = "kc_catalog.csv"
KC_TOPIC_MAP_FILE = "kc_topic_map.csv"
ROSTER_FILE = "learner_roster.csv"
# Written by the simulator next to its data; its presence marks the whole
# directory as simulated learners.
SIMULATION_MANIFEST_NAME = "simulation_manifest.json"
MOODLE_POPULATION = "moodle_collector"
SIMULATED_POPULATION = "simulated"

PROFILE_SCHEMA_VERSION = "2.0.0"
UNVERSIONED_MAPPING = "unversioned"

# These are auditable screening thresholds, not calibrated mastery cutoffs.
LOW_PREDICTED_CORRECTNESS = 0.60
# Reserved for future advancement/ranking use. It is deliberately
# NOT used by weak-area classification: a high prediction is not verified mastery.
HIGH_PREDICTED_CORRECTNESS = 0.80
MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN = 3
LOW_OBSERVED_ACCURACY = 0.60
