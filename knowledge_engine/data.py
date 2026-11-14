"""Read the collector-format inputs the knowledge engine needs."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import MOODLE_POPULATION, SIMULATED_POPULATION, SIMULATION_MANIFEST_NAME

ANSWER_EVENTS_FILE = "question_answer_events.csv"
ROSTER_FILE = "learner_roster.csv"
KC_CATALOG_FILE = "kc_catalog.csv"
QUESTION_KC_MAP_FILE = "moodle_question_kc_map.csv"

EVENT_COLUMNS = [
    "learner_id",
    "event_index",
    "item_id",
    "kc_id",
    "kc_dimension",
    "event_type",
    "timestamp",
    "correct",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_as_of(as_of):
    """Return the cutoff as a UTC Timestamp, or None. Naive datetimes are rejected."""
    if as_of is None:
        return None
    if isinstance(as_of, (int, float)) and not isinstance(as_of, bool):
        return pd.Timestamp(int(as_of), unit="s", tz="UTC")
    if isinstance(as_of, str) and as_of.strip().lstrip("-").isdigit():
        return pd.Timestamp(int(as_of), unit="s", tz="UTC")
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError(
            "as_of must be Unix seconds or a timezone-aware ISO datetime "
            f"(e.g. 2026-12-01T00:00:00Z); got {as_of!r}"
        )
    return cutoff.tz_convert("UTC")


def detect_population(data_dir):
    """A directory written by the simulator is never reported as real Moodle data."""
    if (Path(data_dir) / SIMULATION_MANIFEST_NAME).is_file():
        return SIMULATED_POPULATION
    return MOODLE_POPULATION


def load_answer_events(data_dir, as_of=None):
    """Graded answers, ordered per learner. A missing file means no answers yet.

    The cutoff is inclusive, matching the deep-model export.
    """
    path = Path(data_dir) / ANSWER_EVENTS_FILE
    if not path.is_file():
        return pd.DataFrame(columns=EVENT_COLUMNS + ["mapping_version"])
    events = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={
            "learner_id": "string",
            "item_id": "string",
            "kc_id": "string",
            "kc_dimension": "string",
            "event_type": "string",
            "mapping_version": "string",
        },
    )
    missing = set(EVENT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    events = events[events["event_type"].str.strip().eq("answer")].copy()
    events["correct"] = pd.to_numeric(events["correct"], errors="coerce")
    events = events[events["correct"].isin([0, 1])].copy()
    events["correct"] = events["correct"].astype(int)
    events["event_index"] = pd.to_numeric(events["event_index"], errors="raise").astype(
        int
    )
    events["timestamp"] = pd.to_numeric(events["timestamp"], errors="raise").astype(
        "int64"
    )
    events["kc_dimension"] = events["kc_dimension"].str.strip().str.lower()
    cutoff = normalize_as_of(as_of)
    if cutoff is not None:
        events = events[events["timestamp"].le(int(cutoff.timestamp()))].copy()
    if events.duplicated(["learner_id", "event_index"]).any():
        raise ValueError("Duplicate (learner_id, event_index) answer events found")
    if "mapping_version" not in events:
        events["mapping_version"] = pd.NA
    return events.sort_values(
        ["learner_id", "event_index"], kind="mergesort"
    ).reset_index(drop=True)


def load_kc_catalog(data_dir):
    """The course-owned list of Knowledge Components."""
    path = Path(data_dir) / KC_CATALOG_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"KC catalog not found: {path}. Run the Moodle collector (or the "
            "simulator) so the course's KCs are exported first."
        )
    catalog = pd.read_csv(path, dtype="string")
    missing = {"kc_id", "kc_dimension"}.difference(catalog.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if catalog["kc_id"].isna().any() or catalog["kc_id"].duplicated().any():
        raise ValueError(f"{path} must have one non-empty row per kc_id")
    catalog["kc_dimension"] = catalog["kc_dimension"].str.strip().str.lower()
    if catalog["kc_dimension"].isna().any() or catalog["kc_dimension"].eq("").any():
        raise ValueError(f"{path} contains a KC without a dimension")
    return catalog


def load_roster_ids(data_dir, as_of=None):
    """Learners enrolled at the cutoff, including those with no activity yet."""
    path = Path(data_dir) / ROSTER_FILE
    if not path.is_file():
        return []
    roster = pd.read_csv(path, dtype={"learner_id": "string"})
    if "learner_id" not in roster or roster["learner_id"].isna().any():
        raise ValueError("Learner roster must have non-empty learner_id values")
    if {"enrol_start_unix", "enrol_end_unix"}.issubset(roster.columns):
        cutoff = normalize_as_of(as_of)
        cutoff_unix = int((cutoff or pd.Timestamp.now(tz="UTC")).timestamp())
        start = pd.to_numeric(roster["enrol_start_unix"], errors="coerce").fillna(0)
        end = pd.to_numeric(roster["enrol_end_unix"], errors="coerce").fillna(0)
        roster = roster.loc[start.le(cutoff_unix) & (end.eq(0) | end.gt(cutoff_unix))]
    return sorted(roster["learner_id"].astype(str).unique().tolist())


def current_mapping_version(data_dir, events=None):
    """Version of the question-to-KC mapping the data dir currently carries."""
    path = Path(data_dir) / QUESTION_KC_MAP_FILE
    if path.is_file():
        versions = (
            pd.read_csv(path, dtype="string")["mapping_version"].dropna().unique()
        )
        if len(versions) > 1:
            raise ValueError(f"{path} must use exactly one mapping_version")
        if len(versions) == 1:
            return str(versions[0])
    if events is not None and not events.empty:
        versions = events["mapping_version"].dropna().unique()
        if len(versions) == 1:
            return str(versions[0])
    return None


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
