"""Paths and tunable settings for the knowledge engine.

Every number below is a documented configuration choice. None is calibrated on
learner data, and the thesis must not present them as empirically optimised.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SIMULATED_DIR = PROJECT_ROOT / "data" / "simulated"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_STATES_DIR = ARTIFACTS_DIR / "knowledge_states"
ACTIVE_MODEL_FILE = ARTIFACTS_DIR / "active_model.json"

STATE_FILE_NAME = "learner_kc_state.csv"
MANIFEST_FILE_NAME = "knowledge_state_manifest.json"
SIMULATION_MANIFEST_NAME = "simulation_manifest.json"
# Attempt-plan manifest written by `moodle_setup simulate-learners`.
PLAN_MANIFEST_NAME = "simulation_plan_manifest.json"

MOODLE_POPULATION = "moodle_collector"
SIMULATED_POPULATION = "simulated"

# The deep KT toolkit keeps every non-benchmark dataset under this name.
DEEP_DATASET_NAME = "moodle_kt"

# --- Phase 1: item-level BKT priors (fixed, uncalibrated) -------------------
# One observation is one graded answer to one question of the KC.
BKT_P_INIT = 0.30  # prior probability the KC is already known
BKT_P_LEARN = 0.10  # probability of learning between two answers on the KC
BKT_P_SLIP = 0.10  # P(wrong | known)
BKT_P_GUESS = 0.25  # P(right | not known); about one in four options
BKT_P_FORGET = 0.0  # classic BKT: no forgetting

# --- Phase 2: when is there enough data to try a deep model? ----------------
MIN_LEARNERS_WITH_ANSWERS = 100
MIN_ANSWER_EVENTS = 5_000
MIN_ANSWERS_PER_KC = 20
MIN_KC_COVERAGE = 0.80  # share of catalog KCs reaching MIN_ANSWERS_PER_KC

# --- Phase 3: promotion gate -------------------------------------------------
# The deep model replaces BKT only if its held-out AUC is higher by this much.
GATE_MIN_AUC_MARGIN = 0.01

# Evidence tiers, shared with the deep-model export: none / low / moderate / high.
EVIDENCE_TIER_BINS = [-1, 0, 2, 5, float("inf")]
EVIDENCE_TIER_LABELS = ["none", "low", "moderate", "high"]
