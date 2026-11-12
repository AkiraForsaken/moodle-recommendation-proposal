"""Build learner profiles for one course from its own Moodle evidence.

Inputs (all from the course, none from a benchmark dataset):

    <data-dir>/*.csv               Moodle collector exports
    learner_kc_state.csv           knowledge-engine contract (optional)

Every enrolled learner gets a profile. Without a knowledge-state file the
knowledge, skills and weak-area components are ``unavailable`` and the rest of
the profile is still built; with it, a learner who has answered nothing has a
``cold_start_only`` knowledge component (priors, not evidence).
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .behavior import build_behavior
from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_KNOWLEDGE_STATES,
    DEFAULT_OUTPUT_DIR,
    HIGH_PREDICTED_CORRECTNESS,
    KC_CATALOG_FILE,
    KC_TOPIC_MAP_FILE,
    LOW_OBSERVED_ACCURACY,
    LOW_PREDICTED_CORRECTNESS,
    MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN,
    MOODLE_POPULATION,
    PROFILE_SCHEMA_VERSION,
    ROSTER_FILE,
    SIMULATED_POPULATION,
    SIMULATION_MANIFEST_NAME,
    UNVERSIONED_MAPPING,
)
from .engagement import build_engagement
from .history import build_history
from .knowledge import load_knowledge_states, summarize_dimensions
from .profile import USABLE_STATUSES, build_profiles
from .schemas import PROFILE_COLUMNS
from .skills import (
    build_kc_skill_map,
    ensure_kc_coverage,
    load_kc_catalog,
    load_kc_topic_map,
    summarize_skills,
)
from .temporal import truncate_frames
from .weak_areas import CONCERN_STATUSES, identify_weak_areas

COLLECTOR_FILES = {
    "activity_catalog": "activity_catalog.csv",
    "learning_events": "learning_events.csv",
    "quiz_attempts": "quiz_attempts.csv",
    "course_modules_completion": "course_modules_completion.csv",
}
OPTIONAL_INPUT_FILES = {
    "learner_roster": ROSTER_FILE,
    "kc_catalog": KC_CATALOG_FILE,
    "kc_topic_map": KC_TOPIC_MAP_FILE,
    "kc_activity_map": "kc_activity_map.csv",
    "question_answer_events": "question_answer_events.csv",
    "moodle_question_kc_mapping": "moodle_question_kc_map.csv",
    "question_answer_mapping_report": "question_answer_mapping_report.csv",
    "simulation_manifest": SIMULATION_MANIFEST_NAME,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_population(frame, population):
    result = frame.copy()
    if "source_population" in result.columns:
        result["source_population"] = population
    else:
        result.insert(0, "source_population", population)
    return result


def detect_population(data_dir):
    """A directory written by the simulator is never reported as real Moodle data."""
    if (Path(data_dir) / SIMULATION_MANIFEST_NAME).is_file():
        return SIMULATED_POPULATION
    return MOODLE_POPULATION


def _read_optional(path, **kwargs):
    """A collector export that does not exist yet is an empty table, not an error."""
    return pd.read_csv(path, **kwargs) if Path(path).is_file() else pd.DataFrame()


def _load_collector_frames(data_dir):
    """Load the Moodle collector exports used by behavioral components."""
    activities = _read_optional(data_dir / COLLECTOR_FILES["activity_catalog"])
    events = _read_optional(data_dir / COLLECTOR_FILES["learning_events"])
    quizzes = _read_optional(data_dir / COLLECTOR_FILES["quiz_attempts"])
    completions = _read_optional(
        data_dir / COLLECTOR_FILES["course_modules_completion"]
    )
    roster = _read_optional(data_dir / ROSTER_FILE, dtype={"learner_id": "string"})
    return activities, events, quizzes, completions, roster


def _active_roster_ids(roster, cutoff_unix):
    """Learners enrolled at the cutoff, including those with no activity yet."""
    if roster.empty:
        return set()
    if "learner_id" not in roster or roster["learner_id"].isna().any():
        raise ValueError("Learner roster must have non-empty learner_id values")
    if {"enrol_start_unix", "enrol_end_unix"}.issubset(roster.columns):
        cutoff = (
            cutoff_unix
            if cutoff_unix is not None
            else int(pd.Timestamp.now(tz="UTC").timestamp())
        )
        starts = pd.to_numeric(roster["enrol_start_unix"], errors="coerce").fillna(0)
        ends = pd.to_numeric(roster["enrol_end_unix"], errors="coerce").fillna(0)
        roster = roster.loc[starts.le(cutoff) & (ends.eq(0) | ends.gt(cutoff))]
    return set(roster["learner_id"].dropna().astype(str))


def _latest_timestamp_map(frames):
    latest = {}
    for frame, time_column, is_unix in frames:
        if frame.empty or not {"learner_id", time_column}.issubset(frame.columns):
            continue
        times = (
            pd.to_datetime(
                pd.to_numeric(frame[time_column], errors="coerce"),
                unit="s",
                errors="coerce",
                utc=True,
            )
            if is_unix
            else pd.to_datetime(frame[time_column], errors="coerce", utc=True)
        )
        current = pd.DataFrame(
            {"learner_id": frame["learner_id"].astype(str), "timestamp": times}
        ).dropna(subset=["timestamp"])
        for learner_id, timestamp in (
            current.groupby("learner_id")["timestamp"].max().items()
        ):
            previous = latest.get(learner_id)
            if previous is None or timestamp > previous:
                latest[learner_id] = timestamp
    return {
        learner_id: timestamp.isoformat() for learner_id, timestamp in latest.items()
    }


def _time_column(frame, unix_name, text_name):
    """(frame, column, is_unix) for the timestamp column the frame carries."""
    return (frame, unix_name, True) if unix_name in frame else (frame, text_name, False)


def normalize_as_of(as_of):
    """Return ``(iso_utc, unix_seconds)`` for a cutoff, or ``(None, None)``.

    A cutoff must be Unix seconds or a timezone-aware datetime string. Naive
    strings are rejected: the knowledge loader would read them as UTC while the
    collector helper reads them as collector-local time, so the components
    would silently use different instants.
    """
    if as_of is None:
        return None, None
    if isinstance(as_of, (int, float)) and not isinstance(as_of, bool):
        stamp = pd.Timestamp(int(as_of), unit="s", tz="UTC")
    elif isinstance(as_of, str) and as_of.strip().lstrip("-").isdigit():
        stamp = pd.Timestamp(int(as_of), unit="s", tz="UTC")
    else:
        stamp = pd.Timestamp(as_of)
        if pd.isna(stamp):
            raise ValueError(f"Invalid as_of cutoff: {as_of!r}")
        if stamp.tzinfo is None:
            raise ValueError(
                "as_of must be Unix seconds or a timezone-aware ISO datetime "
                f"(e.g. 2026-12-01T00:00:00Z); got {as_of!r}"
            )
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat(), int(stamp.timestamp())


def _duplicates(frame, keys):
    return not frame.empty and frame.duplicated(keys).any()


def validate_outputs(outputs, as_of):
    """Raise on structural or semantic violations before artifacts are written.

    ``as_of`` is the normalized ISO cutoff (or None).
    """
    failures = []
    checks = []

    def check(passed, ok_message, fail_message):
        if passed:
            checks.append(ok_message)
        else:
            failures.append(fail_message)

    profiles = outputs["learner_profile"]
    missing_profile_columns = set(PROFILE_COLUMNS).difference(profiles.columns)
    check(
        not missing_profile_columns,
        "profile schema contains every required column",
        f"profile schema is missing columns: {sorted(missing_profile_columns)}",
    )
    check(
        not _duplicates(profiles, ["source_population", "learner_id", "as_of"]),
        "profile keys are unique within the snapshot",
        "duplicate population/learner/snapshot profiles",
    )

    kc_states = outputs["learner_kc_state"]
    check(
        not _duplicates(kc_states, ["source_population", "learner_id", "kc_id"]),
        "learner-KC keys are unique",
        "duplicate source-population/learner/KC states",
    )
    for name, keys in {
        "learner_skill_summary": ["source_population", "learner_id", "skill_id"],
        "learner_dimension_summary": [
            "source_population",
            "learner_id",
            "kc_dimension",
        ],
        "learner_behavior": ["source_population", "learner_id"],
        "learner_engagement": ["source_population", "learner_id"],
        "learner_history": ["source_population", "learner_id"],
        "weak_areas": ["source_population", "learner_id", "entity_type", "entity_id"],
    }.items():
        check(
            not _duplicates(outputs[name], keys),
            f"{name} keys are unique",
            f"duplicate keys in {name} on {keys}",
        )

    range_ok = True
    for column in ("observed_accuracy", "predicted_correctness", "p_mastery"):
        if column in kc_states:
            values = pd.to_numeric(kc_states[column], errors="coerce")
            if (values.notna() & ~values.between(0, 1)).any():
                failures.append(f"{column} is outside [0, 1]")
                range_ok = False
    if range_ok:
        checks.append("knowledge probabilities are within [0, 1]")
    for name, column in (
        ("learner_behavior", "revisit_rate"),
        ("learner_engagement", "topic_completion_rate"),
        ("learner_engagement", "course_progress"),
    ):
        frame = outputs[name]
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            if (values.notna() & ~values.between(0, 1)).any():
                failures.append(f"{name}.{column} is outside [0, 1]")
    count_columns = {
        "learner_behavior": [
            "session_count",
            "learning_activity_count",
            "practice_activity_count",
            "review_activity_count",
        ],
        "learner_engagement": [
            "active_session_count",
            "total_learning_events",
            "total_activity_views",
            "estimated_learning_time_seconds",
        ],
        "learner_history": [
            "total_learning_events",
            "total_quiz_attempts",
            "historical_learning_days",
        ],
    }
    for name, columns in count_columns.items():
        for column in columns:
            if column in outputs[name]:
                values = pd.to_numeric(outputs[name][column], errors="coerce")
                if values.lt(0).any():
                    failures.append(f"{name}.{column} contains negative values")

    # Cold-start rows must never carry a fabricated observed accuracy of 0.
    if {"state_status", "observed_accuracy", "attempt_count"}.issubset(
        kc_states.columns
    ):
        no_attempts = pd.to_numeric(kc_states["attempt_count"], errors="coerce").eq(0)
        check(
            kc_states.loc[no_attempts, "observed_accuracy"].isna().all(),
            "KC rows without direct attempts have no observed accuracy",
            "KC rows with zero attempts carry an observed_accuracy value",
        )

    if as_of is not None and not kc_states.empty:
        attempts = pd.to_datetime(
            kc_states["last_attempt_at"], errors="coerce", utc=True
        )
        has_attempts = pd.to_numeric(kc_states["attempt_count"], errors="coerce").gt(0)
        bad = has_attempts & (attempts.isna() | attempts.gt(pd.Timestamp(as_of)))
        check(
            not bad.any(),
            "KC last-attempt times respect the snapshot cutoff",
            "KC states contain a missing or post-cutoff last_attempt_at",
        )

    # Cross-component: every component row belongs to a profile.
    keys = set(zip(profiles["source_population"], profiles["learner_id"].astype(str)))
    orphan = []
    for name in (
        "learner_kc_state",
        "learner_skill_summary",
        "learner_dimension_summary",
        "learner_behavior",
        "learner_engagement",
        "learner_history",
        "weak_areas",
    ):
        frame = outputs[name]
        if frame.empty:
            continue
        pairs = set(zip(frame["source_population"], frame["learner_id"].astype(str)))
        if pairs - keys:
            orphan.append(name)
    check(
        not orphan,
        "every component row belongs to a profile",
        f"component rows without a matching profile: {orphan}",
    )

    weak = outputs["weak_areas"]
    kc_keys = set(
        zip(
            kc_states["source_population"],
            kc_states["learner_id"].astype(str),
            kc_states["kc_id"].astype(str),
        )
    )
    weak_keys = set(
        zip(
            weak["source_population"],
            weak["learner_id"].astype(str),
            weak["entity_id"].astype(str),
        )
    )
    check(
        weak_keys == kc_keys,
        "every weak-area row traces to a learner-KC state and vice versa",
        "weak-area rows do not match learner-KC states one-to-one",
    )
    invalid_concerns = set(weak["concern_status"].dropna()) - set(CONCERN_STATUSES)
    check(
        not invalid_concerns and weak["reason"].notna().all(),
        "concern statuses are valid and every row has a reason",
        f"invalid concern statuses or missing reasons: {sorted(invalid_concerns)}",
    )
    if not kc_states.empty and "state_status" in kc_states:
        merged = weak.merge(
            kc_states[["source_population", "learner_id", "kc_id", "attempt_count"]],
            left_on=["source_population", "learner_id", "entity_id"],
            right_on=["source_population", "learner_id", "kc_id"],
        )
        cold_evidence = merged["attempt_count"].eq(0) & merged["concern_status"].eq(
            "evidence_based_concern"
        )
        check(
            not cold_evidence.any(),
            "no cold-start KC is labelled an evidence-based concern",
            "cold-start KC labelled evidence_based_concern",
        )

    skills = outputs["learner_skill_summary"]
    if not skills.empty:
        check(
            skills["kc_count"].gt(0).all()
            and skills["estimated_kc_count"]
            .add(skills["cold_start_kc_count"])
            .eq(skills["kc_count"])
            .all(),
            "skill summaries trace to their KC rows (estimated + cold-start = KC count)",
            "skill summary KC counts are inconsistent",
        )

    populations = set(profiles["source_population"].dropna())
    for frame in outputs.values():
        if "source_population" in frame and not frame.empty:
            populations |= set(frame["source_population"].dropna())
    check(
        len(populations) <= 1,
        "all outputs belong to one source population",
        f"outputs mix source populations: {sorted(populations)}",
    )

    valid_status = USABLE_STATUSES | {"unavailable", "no_observations"}
    for column in [c for c in profiles.columns if c.endswith("_status")]:
        if set(profiles[column].dropna()) - valid_status:
            failures.append(f"profile.{column} has unknown status values")

    if failures:
        raise ValueError("Learner-model validation failed: " + "; ".join(failures))
    return checks


def _empty_knowledge():
    states = pd.DataFrame(
        columns=[
            "learner_id",
            "kc_id",
            "kc_dimension",
            "attempt_count",
            "evidence_count",
            "observed_accuracy",
            "predicted_correctness",
            "p_mastery",
            "last_attempt_at",
            "state_status",
            "model_name",
            "model_version",
            "estimator_phase",
        ]
    )
    states.attrs.update(
        source_learner_ids=[],
        source_population=None,
        knowledge_snapshot_as_of=None,
        dropped_after_cutoff=0,
        dropped_without_timestamp=0,
    )
    return states


def _counts(series):
    return {str(k): int(v) for k, v in series.value_counts().items()}


def build_complete_learner_model(
    data_dir=DEFAULT_DATA_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    knowledge_state_file=DEFAULT_KNOWLEDGE_STATES,
    as_of=None,
    generated_at=None,
):
    """Build every learner-model component for the course in ``data_dir``.

    ``knowledge_state_file`` is the knowledge-engine contract. If it does not
    exist, profiles are still built and knowledge is ``unavailable``. Knowledge
    states of another population than ``data_dir`` are rejected, so simulated
    learners are never joined to real Moodle activity.

    ``as_of`` cuts off every time-dependent component. ``generated_at`` may be
    pinned so that two builds of the same inputs are byte-identical.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    knowledge_state_file = Path(knowledge_state_file)
    as_of_iso, as_of_unix = normalize_as_of(as_of)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Learner-model data directory does not exist: {data_dir}"
        )
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    population = detect_population(data_dir)

    # ---- knowledge, skills, weak areas (from the knowledge-engine contract)
    knowledge_available = knowledge_state_file.is_file()
    if knowledge_available:
        states = load_knowledge_states(knowledge_state_file, as_of=as_of_iso)
        state_population = states.attrs["source_population"]
        if state_population not in (None, population):
            raise ValueError(
                f"Knowledge states describe {state_population!r} learners but "
                f"{data_dir} holds {population!r} data; refusing to combine them."
            )
    else:
        states = _empty_knowledge()

    kc_catalog = load_kc_catalog(data_dir)
    kc_topic_map, topic_map_version = load_kc_topic_map(data_dir)
    state_versions = (
        states["mapping_version"].dropna().unique()
        if "mapping_version" in states
        else []
    )
    mapping_version = (
        str(state_versions[0])
        if len(state_versions) == 1
        else topic_map_version or UNVERSIONED_MAPPING
    )
    kc_skill_map = ensure_kc_coverage(
        states,
        build_kc_skill_map(kc_catalog, kc_topic_map, mapping_version),
        mapping_version,
    )
    kc_names = dict(
        zip(kc_catalog["kc_id"].astype(str), kc_catalog["kc_name"].astype(str))
    )
    if not states.empty:
        states.insert(
            states.columns.get_loc("kc_id") + 1,
            "kc_name",
            [kc_names.get(kc_id, kc_id) for kc_id in states["kc_id"].astype(str)],
        )
    dimensions = summarize_dimensions(states)
    skill_summary = summarize_skills(states, kc_skill_map, mapping_version)
    weak_areas = identify_weak_areas(states, generated_at, mapping_version, kc_names)

    # ---- behavior, engagement, history (from the collector exports)
    activities, events, quizzes, completions, roster = _load_collector_frames(data_dir)
    completion_tracking_available = not completions.empty
    # Apply one collector cutoff to every time-dependent component. The
    # collector pipeline's established strict-before-T policy is preserved.
    events_snapshot, quizzes_snapshot, completions_snapshot, cutoff = truncate_frames(
        events, quizzes, completions, as_of_unix
    )
    learner_ids = _active_roster_ids(roster, as_of_unix)
    for frame in (events_snapshot, quizzes_snapshot, completions_snapshot):
        if frame is not None and not frame.empty and "learner_id" in frame.columns:
            learner_ids.update(frame["learner_id"].dropna().astype(str))
    learner_ids.update(states.attrs["source_learner_ids"])
    latest = _latest_timestamp_map(
        [
            _time_column(events_snapshot, "timestamp_unix", "timestamp"),
            _time_column(quizzes_snapshot, "finish_time_unix", "finish_time"),
            _time_column(completions_snapshot, "time_modified_unix", "time_modified"),
            (states, "last_attempt_at", False),
        ]
    )
    behavior = build_behavior(events_snapshot, activities, generated_at)
    engagement = build_engagement(
        events_snapshot,
        quizzes_snapshot,
        completions_snapshot,
        activities,
        generated_at,
        as_of=cutoff if cutoff is not None else generated_at,
        completion_tracking_available=completion_tracking_available,
    )
    history = build_history(
        events_snapshot,
        quizzes_snapshot,
        completions_snapshot,
        activities,
        generated_at,
    )

    outputs = {
        "learner_kc_state": states,
        "learner_skill_summary": skill_summary,
        "learner_dimension_summary": dimensions,
        "learner_behavior": behavior,
        "learner_engagement": engagement,
        "learner_history": history,
        "weak_areas": weak_areas,
    }
    outputs = {
        name: _with_population(frame, population) for name, frame in outputs.items()
    }
    profiles = build_profiles(
        population,
        learner_ids,
        {
            "knowledge": outputs["learner_kc_state"],
            "skills": outputs["learner_skill_summary"],
            "weak_areas": outputs["weak_areas"],
            "behavior": outputs["learner_behavior"],
            "engagement": outputs["learner_engagement"],
            "history": outputs["learner_history"],
        },
        generated_at,
        as_of=as_of_iso,
        latest_timestamps=latest,
    )
    outputs = {"learner_profile": profiles, **outputs}
    checks = validate_outputs(outputs, as_of_iso)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    kc_skill_map.to_csv(output_dir / "kc_skill_map.csv", index=False)

    input_files = {
        name: data_dir / filename
        for name, filename in {**COLLECTOR_FILES, **OPTIONAL_INPUT_FILES}.items()
        if (data_dir / filename).is_file()
    }
    if knowledge_available:
        input_files["knowledge_state_file"] = knowledge_state_file
        engine_manifest = knowledge_state_file.with_name(
            "knowledge_state_manifest.json"
        )
        if engine_manifest.is_file():
            input_files["knowledge_state_manifest"] = engine_manifest
    components = ["behavior", "engagement", "history"]
    if knowledge_available:
        components = ["knowledge", "skills", "weak_areas", *components]
    manifest = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "as_of": as_of_iso,
        "source_population": population,
        "learner_count": len(profiles),
        "components": components,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(input_files.items())
        },
        "knowledge_provenance": (
            {
                "model_names": sorted(states["model_name"].dropna().unique().tolist()),
                "model_versions": sorted(
                    states["model_version"].dropna().unique().tolist()
                ),
                "estimator_phases": sorted(
                    int(v) for v in states["estimator_phase"].dropna().unique()
                ),
                "checkpoint_ids": sorted(
                    states["checkpoint_id"].dropna().unique().tolist()
                ),
                "knowledge_snapshot_as_of": states.attrs["knowledge_snapshot_as_of"],
                "state_status_counts": _counts(states["state_status"]),
                "dropped_after_cutoff": states.attrs["dropped_after_cutoff"],
                "dropped_without_last_attempt_at": states.attrs[
                    "dropped_without_timestamp"
                ],
            }
            if knowledge_available
            else None
        ),
        "weak_area_counts": _counts(weak_areas["concern_status"]),
        "mapping": {
            "version": mapping_version,
            "kc_count": int(kc_catalog["kc_id"].nunique()),
            "skill_count": int(kc_skill_map["skill_id"].nunique()),
            "status_counts": _counts(kc_skill_map["mapping_status"]),
        },
        "knowledge_semantics": {
            "predicted_correctness": "probability of answering the next question on the KC correctly",
            "p_mastery": "BKT latent probability that the KC is known; null for other estimators",
            "cold_start": "prior for a KC with no answers; not individual evidence or mastery",
        },
        "configuration": {
            "low_predicted_correctness_threshold": LOW_PREDICTED_CORRECTNESS,
            "high_predicted_correctness_threshold_reserved_unused": HIGH_PREDICTED_CORRECTNESS,
            "minimum_attempts_for_evidence_concern": MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN,
            "low_observed_accuracy_threshold": LOW_OBSERVED_ACCURACY,
        },
        "output_counts": {name: len(frame) for name, frame in outputs.items()},
        "validation_checks": checks,
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_validation_report(
        output_dir / "validation_report.md", outputs, manifest, kc_skill_map, checks
    )
    return outputs


def _write_validation_report(path, outputs, manifest, kc_skill_map, checks):
    profiles = outputs["learner_profile"]
    population = manifest["source_population"]
    lines = [
        "# Learner-model validation report",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Snapshot cutoff: `{manifest['as_of'] or 'latest'}`",
        f"- Source population: `{population}`"
        + (
            " (SIMULATED learners: pipeline demonstration, not evidence about real learners)"
            if population == SIMULATED_POPULATION
            else ""
        ),
        f"- Mapping version: `{manifest['mapping']['version']}`",
        "- `predicted_correctness` is an answer-correctness estimate, not a mastery probability.",
        "",
        "## Output record counts",
        "",
        "| Output | Rows | Null cells |",
        "|---|---:|---:|",
        *[
            f"| `{name}.csv` | {len(frame)} | {int(frame.isna().sum().sum())} |"
            for name, frame in outputs.items()
        ],
        "",
        "## Component availability",
        "",
        "| Component | Status | Learners |",
        "|---|---|---:|",
    ]
    for column in [c for c in PROFILE_COLUMNS if c.endswith("_status")]:
        for status, count in profiles[column].value_counts(dropna=False).items():
            lines.append(
                f"| {column.removesuffix('_status')} | {status} | {int(count)} |"
            )
    provenance = manifest["knowledge_provenance"]
    lines += ["", "## Knowledge"]
    if provenance is None:
        lines += [
            "",
            "No knowledge-state file was supplied: knowledge, skills and weak areas are "
            "`unavailable`. Run `python -m knowledge_engine estimate` first.",
        ]
    else:
        lines += [
            "",
            f"- Estimator: `{', '.join(provenance['model_names'])}` "
            f"(phase {', '.join(map(str, provenance['estimator_phases']))})",
            "",
            "| KC state status | Rows |",
            "|---|---:|",
            *[f"| `{k}` | {v} |" for k, v in provenance["state_status_counts"].items()],
            "",
            "| Concern status | Rows |",
            "|---|---:|",
            *[f"| `{k}` | {v} |" for k, v in manifest["weak_area_counts"].items()],
        ]
    lines += [
        "",
        "## Mapping coverage",
        "",
        "| Mapping status | KC-skill rows |",
        "|---|---:|",
        *[f"| `{k}` | {v} |" for k, v in manifest["mapping"]["status_counts"].items()],
        "",
        "## Checks",
        "",
        *[f"- PASS: {check}" for check in checks],
        "- Validation failures: none (the builder stops before writing outputs if a check fails).",
        "",
        "## Reading these outputs",
        "",
        "- A `cold_start` KC row is a prior shared by every learner without answers on that KC.",
        "- Phase 1 estimates use fixed, uncalibrated BKT parameters.",
        "- Weak-area thresholds are configuration choices, not empirically calibrated cutoffs.",
        "- Weak-area rows cover every learner-KC state; filter on `concern_status` to list concerns.",
        "- KC cutoffs are inclusive; collector cutoffs are strictly before the cutoff.",
        "- Behavior, engagement and history describe participation, not knowledge.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Build learner profiles from Moodle collector data and knowledge states."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--knowledge-states",
        type=Path,
        default=DEFAULT_KNOWLEDGE_STATES,
        help="knowledge-engine learner_kc_state.csv (optional: profiles build without it)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--as-of",
        help="snapshot cutoff: Unix seconds or timezone-aware ISO datetime",
    )
    parser.add_argument(
        "--generated-at",
        help="pin the generation timestamp (ISO) for byte-identical rebuilds",
    )
    args = parser.parse_args()
    outputs = build_complete_learner_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        knowledge_state_file=args.knowledge_states,
        as_of=args.as_of,
        generated_at=args.generated_at,
    )
    print(f"Learner model written to {args.output_dir}")
    for name, frame in outputs.items():
        print(f"  {name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
