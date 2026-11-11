"""Reproducible orchestration for the Milestone 4 learner model."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .behavior import build_behavior
from .config import (
    DEFAULT_CONCEPT_MAP,
    DEFAULT_DATA_DIR,
    DEFAULT_KNOWLEDGE_STATES,
    DEFAULT_MAPPINGS_DIR,
    DEFAULT_MOODLE_KNOWLEDGE_STATES,
    DEFAULT_MOODLE_ROSTER,
    DEFAULT_OUTPUT_DIR,
    HIGH_PREDICTED_CORRECTNESS,
    LOW_OBSERVED_ACCURACY,
    LOW_PREDICTED_CORRECTNESS,
    MAPPING_VERSION,
    MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN,
    PROFILE_SCHEMA_VERSION,
)
from .engagement import build_engagement
from .history import build_history
from .knowledge import load_knowledge_states, summarize_dimensions
from .profile import USABLE_STATUSES, build_profiles
from .schemas import PROFILE_COLUMNS
from .skills import (
    build_taxonomy,
    ensure_kc_coverage,
    load_kc_mapping,
    summarize_skills,
    write_mapping_artifacts,
)
from .temporal import truncate_frames
from .weak_areas import CONCERN_STATUSES, identify_weak_areas

ENGLISH_POPULATION = "english_learning_events"
MOODLE_POPULATION = "moodle_collector"


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


def _load_collector_frames(data_dir):
    """Load the Moodle collector exports used by behavioral components."""
    activities = pd.read_csv(data_dir / "activity_catalog.csv")
    events = pd.read_csv(data_dir / "learning_events.csv")
    quizzes = pd.read_csv(data_dir / "quiz_attempts.csv")
    completion_file = data_dir / "course_modules_completion.csv"
    completions = (
        pd.read_csv(completion_file) if completion_file.exists() else pd.DataFrame()
    )
    roster_file = data_dir / "learner_roster.csv"
    roster = (
        pd.read_csv(roster_file, dtype={"learner_id": "string"})
        if roster_file.exists()
        else pd.DataFrame()
    )
    return activities, events, quizzes, completions, roster


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


def normalize_as_of(as_of):
    """Return ``(iso_utc, unix_seconds)`` for a cutoff, or ``(None, None)``.

    A cutoff must be Unix seconds or a timezone-aware datetime string. Naive
    strings are rejected: the English knowledge loader would read them as UTC
    while the collector helper reads them as collector-local time, so the two
    populations would silently use different instants.
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
                f"(e.g. 2023-12-01T00:00:00Z); got {as_of!r}"
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
        "profile keys are unique within population and snapshot",
        "duplicate population/learner/snapshot profiles",
    )

    kc_states = outputs["learner_kc_state"]
    check(
        not _duplicates(kc_states, ["source_population", "learner_id", "kc_id"]),
        "learner-KC keys are unique within population",
        "duplicate source-population/learner/KC states",
    )
    for name, keys in {
        "learner_skill_summary": ["source_population", "learner_id", "skill_id"],
        "learner_dimension_summary": [
            "source_population",
            "learner_id",
            "kc_dimension",
        ],
        # "learner_topic_state": ["source_population", "learner_id", "topic"],
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
    for column in ("observed_accuracy", "predicted_correctness"):
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
        cutoffs = as_of if isinstance(as_of, dict) else None
        populations = (
            kc_states["source_population"].dropna().unique()
            if cutoffs is not None and "source_population" in kc_states
            else [None]
        )
        bad = pd.Series(False, index=kc_states.index)
        attempts = pd.to_datetime(
            kc_states["last_attempt_at"], errors="coerce", utc=True
        )
        has_attempts = pd.to_numeric(kc_states["attempt_count"], errors="coerce").gt(0)
        for population in populations:
            mask = (
                kc_states["source_population"].eq(population)
                if population is not None
                else pd.Series(True, index=kc_states.index)
            )
            cutoff_value = cutoffs.get(population) if cutoffs is not None else as_of
            if cutoff_value is None:
                continue
            cutoff = pd.Timestamp(cutoff_value)
            bad |= mask & (has_attempts & (attempts.isna() | attempts.gt(cutoff)))
        check(
            not bad.any(),
            "KC last-attempt cutoffs are enforced per source population",
            "KC states contain a missing or post-cutoff last_attempt_at",
        )

    # Cross-component: every component row belongs to a profile in its population.
    keys = set(zip(profiles["source_population"], profiles["learner_id"].astype(str)))
    orphan = []
    for name in (
        "learner_kc_state",
        "learner_skill_summary",
        "learner_dimension_summary",
        # "learner_topic_state",
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
        "every component row belongs to a profile in its own population",
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
            kc_states[
                ["source_population", "learner_id", "kc_id", "attempt_count"]
            ],
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
            #     )
            # topic_state = outputs["learner_topic_state"]
            # if not topic_state.empty:
            #     check(
            #         "p_mastery" in topic_state and "source_population" in topic_state,
            #         "topic p_mastery remains in the Moodle topic-state table only",
            #         "Moodle topic-state schema lacks p_mastery or source-population provenance",
            #     )
            # check(
            #     "p_mastery" not in kc_states.columns,
            #     "KC state table has no p_mastery column",
            #     "learner_kc_state must not contain p_mastery",
        )

    valid_status = USABLE_STATUSES | {"unavailable", "no_observations"}
    for column in [c for c in profiles.columns if c.endswith("_status")]:
        if set(profiles[column].dropna()) - valid_status:
            failures.append(f"profile.{column} has unknown status values")

    if failures:
        raise ValueError("Learner-model validation failed: " + "; ".join(failures))
    return checks


def build_complete_learner_model(
    data_dir=DEFAULT_DATA_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    knowledge_state_file=DEFAULT_KNOWLEDGE_STATES,
    concept_map_file=DEFAULT_CONCEPT_MAP,
    as_of=None,
    mappings_dir=DEFAULT_MAPPINGS_DIR,
    generated_at=None,
    knowledge_as_of=None,
    moodle_as_of=None,
    moodle_knowledge_state_file=DEFAULT_MOODLE_KNOWLEDGE_STATES,
):
    """Build both source populations without matching their learner identifiers.

    ``mappings_dir`` holds the versioned KC mapping. An existing
    ``kc_topic_map.csv`` there (e.g. supervisor-reviewed) is loaded and used;
    a draft is generated only when it is absent. ``generated_at`` may be pinned
    so that two builds of the same inputs are byte-identical.

    The two populations live on different time axes (the English benchmark is
    synthetic 2023 data; the Moodle collector is current). ``as_of`` is the
    default cutoff for both; ``knowledge_as_of`` / ``moodle_as_of`` override it
    per population.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    knowledge_state_file = Path(knowledge_state_file)
    moodle_knowledge_state_file = Path(moodle_knowledge_state_file)
    concept_map_file = Path(concept_map_file)
    mappings_dir = Path(mappings_dir)
    knowledge_as_of_iso, _ = normalize_as_of(
        knowledge_as_of if knowledge_as_of is not None else as_of
    )
    moodle_as_of_iso, moodle_as_of_unix = normalize_as_of(
        moodle_as_of if moodle_as_of is not None else as_of
    )
    required_inputs = [
        knowledge_state_file,
        concept_map_file,
        data_dir / "activity_catalog.csv",
        data_dir / "learning_events.csv",
        data_dir / "quiz_attempts.csv",
    ]
    for path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required learner-model input does not exist: {path}"
            )
    if (
        moodle_knowledge_state_file != DEFAULT_MOODLE_KNOWLEDGE_STATES
        and not moodle_knowledge_state_file.is_file()
    ):
        raise FileNotFoundError(
            f"Moodle knowledge-state file does not exist: {moodle_knowledge_state_file}"
        )

    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    states = load_knowledge_states(knowledge_state_file, as_of=knowledge_as_of_iso)
    source_english_learners = states.attrs["source_learner_ids"]
    english_latest = _latest_timestamp_map([(states, "last_attempt_at", False)])
    dimensions = summarize_dimensions(states)
    concept_map = pd.read_csv(concept_map_file, dtype={"kc_id": "string"})
    taxonomy, draft_mapping = build_taxonomy(concept_map)
    mapping_file = mappings_dir / "kc_topic_map.csv"
    taxonomy_file = mappings_dir / "skill_taxonomy.csv"
    taxonomy_status = "needs_review"
    if taxonomy_file.is_file():
        existing_taxonomy = pd.read_csv(
            taxonomy_file, dtype={"taxonomy_status": "string"}
        )
        if "taxonomy_status" not in existing_taxonomy:
            raise ValueError(
                f"Skill taxonomy {taxonomy_file} is missing taxonomy_status"
            )
        if (
            not existing_taxonomy.empty
            and existing_taxonomy["taxonomy_status"].eq("reviewed").all()
        ):
            taxonomy_status = "reviewed"
    if mapping_file.is_file():
        kc_topic_map, mapping_version = load_kc_mapping(mapping_file)
        mapping_origin = "existing_file"
    else:
        kc_topic_map, mapping_version = draft_mapping, MAPPING_VERSION
        mapping_origin = "generated_draft"
    kc_topic_map = ensure_kc_coverage(states, kc_topic_map, mapping_version)
    skill_summary = summarize_skills(states, kc_topic_map, mapping_version)
    weak_areas = identify_weak_areas(states, generated_at, mapping_version)

    moodle_states = pd.DataFrame()
    if moodle_knowledge_state_file.is_file():
        moodle_states = load_knowledge_states(
            moodle_knowledge_state_file, as_of=moodle_as_of_iso
        )
    moodle_dimensions = summarize_dimensions(moodle_states)
    moodle_skill_summary = summarize_skills(
        moodle_states,
        ensure_kc_coverage(moodle_states, kc_topic_map, mapping_version),
        mapping_version,
    )
    moodle_weak_areas = identify_weak_areas(
        moodle_states, generated_at, mapping_version
    )

    activities, events, quizzes, completions, roster = _load_collector_frames(data_dir)
    completion_file = data_dir / "course_modules_completion.csv"
    completion_tracking_available = completion_file.is_file() and not completions.empty
    source_moodle_learners = set()
    if not roster.empty:
        if "learner_id" not in roster or roster["learner_id"].isna().any():
            raise ValueError("Learner roster must have non-empty learner_id values")
        if {"enrol_start_unix", "enrol_end_unix"}.issubset(roster.columns):
            roster_cutoff = (
                moodle_as_of_unix
                if moodle_as_of_unix is not None
                else int(pd.Timestamp.now(tz="UTC").timestamp())
            )
            starts = pd.to_numeric(roster["enrol_start_unix"], errors="coerce").fillna(0)
            ends = pd.to_numeric(roster["enrol_end_unix"], errors="coerce").fillna(0)
            active = starts.le(roster_cutoff) & (
                ends.eq(0) | ends.gt(roster_cutoff)
            )
            roster = roster.loc[active]
        source_moodle_learners.update(roster["learner_id"].dropna().astype(str))

    # Apply one collector cutoff to every time-dependent component. The
    # collector pipeline's established strict-before-T policy is preserved.
    events_snapshot, quizzes_snapshot, completions_snapshot, cutoff = truncate_frames(
        events, quizzes, completions, moodle_as_of_unix
    )
    for frame in (events_snapshot, quizzes_snapshot, completions_snapshot):
        if frame is not None and not frame.empty and "learner_id" in frame.columns:
            source_moodle_learners.update(frame["learner_id"].dropna().astype(str))
    if "learner_id" in moodle_states:
        source_moodle_learners.update(
            moodle_states["learner_id"].dropna().astype(str)
        )
    moodle_latest = _latest_timestamp_map(
        [
            (
                events_snapshot,
                (
                    "timestamp_unix"
                    if "timestamp_unix" in events_snapshot
                    else "timestamp"
                ),
                "timestamp_unix" in events_snapshot,
            ),
            (
                quizzes_snapshot,
                (
                    "finish_time_unix"
                    if "finish_time_unix" in quizzes_snapshot
                    else "finish_time"
                ),
                "finish_time_unix" in quizzes_snapshot,
            ),
            (
                completions_snapshot,
                (
                    "time_modified_unix"
                    if "time_modified_unix" in completions_snapshot
                    else "time_modified"
                ),
                "time_modified_unix" in completions_snapshot,
            ),
        ]
    )
    # topic_state = build_learner_model_from_frames(
    #     activities,
    #     events_snapshot,
    #     quizzes_snapshot,
    #     completions_snapshot,
    #     generated_at=generated_at,
    # )
    # if topic_state.empty:
    #     topic_state = _empty_topic_state()
    # topic_state = _with_population(topic_state, MOODLE_POPULATION)

    behavior = _with_population(
        build_behavior(events_snapshot, activities, generated_at), MOODLE_POPULATION
    )
    engagement = _with_population(
        build_engagement(
            events_snapshot,
            quizzes_snapshot,
            completions_snapshot,
            activities,
            # topic_state.drop(columns="source_population", errors="ignore"),
            generated_at,
            as_of=cutoff if cutoff is not None else generated_at,
            completion_tracking_available=completion_tracking_available,
        ),
        MOODLE_POPULATION,
    )
    history = _with_population(
        build_history(
            events_snapshot,
            quizzes_snapshot,
            completions_snapshot,
            activities,
            generated_at,
        ),
        MOODLE_POPULATION,
    )

    knowledge_output = pd.concat(
        [
            _with_population(states, ENGLISH_POPULATION),
            _with_population(moodle_states, MOODLE_POPULATION),
        ],
        ignore_index=True,
    )
    skill_output = pd.concat(
        [
            _with_population(skill_summary, ENGLISH_POPULATION),
            _with_population(moodle_skill_summary, MOODLE_POPULATION),
        ],
        ignore_index=True,
    )
    weak_output = pd.concat(
        [
            _with_population(weak_areas, ENGLISH_POPULATION),
            _with_population(moodle_weak_areas, MOODLE_POPULATION),
        ],
        ignore_index=True,
    )
    dimension_output = pd.concat(
        [
            _with_population(dimensions, ENGLISH_POPULATION),
            _with_population(moodle_dimensions, MOODLE_POPULATION),
        ],
        ignore_index=True,
    )
    component_frames = {
        "knowledge": knowledge_output,
        "skills": skill_output,
        "weak_areas": weak_output,
        "behavior": behavior,
        "engagement": engagement,
        "history": history,
    }
    profiles = build_profiles(
        {
            ENGLISH_POPULATION: source_english_learners,
            MOODLE_POPULATION: source_moodle_learners,
        },
        component_frames,
        generated_at,
        as_of={
            ENGLISH_POPULATION: knowledge_as_of_iso,
            MOODLE_POPULATION: moodle_as_of_iso,
        },
        latest_timestamps={
            ENGLISH_POPULATION: english_latest,
            MOODLE_POPULATION: moodle_latest,
        },
    )
    outputs = {
        "learner_profile": profiles,
        "learner_kc_state": knowledge_output,
        "learner_skill_summary": skill_output,
        "learner_dimension_summary": dimension_output,
        # "learner_topic_state": topic_state,
        "learner_behavior": behavior,
        "learner_engagement": engagement,
        "learner_history": history,
        "weak_areas": weak_output,
    }
    checks = validate_outputs(
        outputs,
        {
            ENGLISH_POPULATION: knowledge_as_of_iso,
            MOODLE_POPULATION: moodle_as_of_iso,
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_mapping_artifacts(taxonomy, kc_topic_map, mappings_dir)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    input_files = {
        "knowledge_state_file": knowledge_state_file,
        "concept_map_file": concept_map_file,
        "kc_topic_map": mapping_file,
        "activity_catalog": data_dir / "activity_catalog.csv",
        "learning_events": data_dir / "learning_events.csv",
        "quiz_attempts": data_dir / "quiz_attempts.csv",
    }
    if moodle_knowledge_state_file.is_file():
        input_files["moodle_knowledge_state_file"] = moodle_knowledge_state_file
    roster_file = data_dir / "learner_roster.csv"
    if roster_file.is_file():
        input_files["learner_roster"] = roster_file
    answer_events_file = data_dir / "question_answer_events.csv"
    if answer_events_file.is_file():
        input_files["question_answer_events"] = answer_events_file
    question_mapping_file = data_dir / "moodle_question_kc_map.csv"
    if question_mapping_file.is_file():
        input_files["moodle_question_kc_mapping"] = question_mapping_file
    mapping_report_file = data_dir / "question_answer_mapping_report.csv"
    if mapping_report_file.is_file():
        input_files["question_answer_mapping_report"] = mapping_report_file
    if completion_file.is_file():
        input_files["course_modules_completion"] = completion_file
    inference_manifest = knowledge_state_file.with_name("inference_manifest.json")
    if inference_manifest.is_file():
        input_files["inference_manifest"] = inference_manifest
    manifest = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "as_of": {
            ENGLISH_POPULATION: knowledge_as_of_iso,
            MOODLE_POPULATION: moodle_as_of_iso,
        },
        "source_populations": {
            ENGLISH_POPULATION: {
                "learner_count": len(source_english_learners),
                "components": ["knowledge", "skills", "weak_areas"],
            },
            MOODLE_POPULATION: {
                "learner_count": len(source_moodle_learners),
                "components": (
                    ["knowledge", "skills", "weak_areas", "behavior", "engagement", "history"]
                    if not moodle_states.empty
                    else ["behavior", "engagement", "history"]
                ),
            },
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in input_files.items()
        },
        "knowledge_provenance": {
            "model_names": sorted(states["model_name"].dropna().unique().tolist()),
            "model_versions": sorted(
                states["model_version"].dropna().unique().tolist()
            ),
            "checkpoint_ids": sorted(
                states["checkpoint_id"].dropna().unique().tolist()
            ),
            "checkpoint_sha256": sorted(
                states["checkpoint_sha256"].dropna().unique().tolist()
            ),
            "aggregation_methods": sorted(
                states["aggregation_method"].dropna().unique().tolist()
            ),
            "inference_timestamps": sorted(
                states["inference_timestamp"].dropna().astype(str).unique().tolist()
            ),
            "knowledge_snapshot_as_of": states.attrs["knowledge_snapshot_as_of"],
            "state_status_counts": {
                str(k): int(v) for k, v in states["state_status"].value_counts().items()
            },
        },
        "moodle_knowledge_provenance": (
            {
                "model_names": sorted(
                    moodle_states["model_name"].dropna().unique().tolist()
                ),
                "model_versions": sorted(
                    moodle_states["model_version"].dropna().unique().tolist()
                ),
                "checkpoint_ids": sorted(
                    moodle_states["checkpoint_id"].dropna().unique().tolist()
                ),
                "mapping_versions": sorted(
                    moodle_states["mapping_version"].dropna().unique().tolist()
                )
                if "mapping_version" in moodle_states
                else [],
                "state_status_counts": {
                    str(k): int(v)
                    for k, v in moodle_states["state_status"].value_counts().items()
                },
                "cold_start_semantics": (
                    "Model prior for an empty response history; not individual evidence or mastery."
                ),
            }
            if not moodle_states.empty
            else None
        ),
        "weak_area_counts": {
            str(k): int(v)
            for k, v in weak_areas["concern_status"].value_counts().items()
        },
        "knowledge_cutoff": {
            "dropped_after_cutoff": states.attrs["dropped_after_cutoff"],
            "dropped_without_last_attempt_at": states.attrs[
                "dropped_without_timestamp"
            ],
        },
        "mapping": {
            "version": mapping_version,
            "origin": mapping_origin,
            "mappings_dir": str(mappings_dir),
            "status_counts": {
                str(k): int(v)
                for k, v in kc_topic_map["mapping_status"].value_counts().items()
            },
            "skill_taxonomy_status": taxonomy_status,
            "kc_topic_mapping_status": (
                "needs_review"
                if kc_topic_map["mapping_status"].ne("mapped").any()
                else "mapped"
            ),
            "topic_mapping_note": (
                "Mappings are metadata only; English and Moodle learner populations "
                "are not joined by this build."
            ),
        },
        "knowledge_semantics": {
            "predicted_correctness": "probability of answering representative questions correctly",
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

    validation = [
        "# Complete learner-model validation report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Snapshot cutoff (English KC states): `{knowledge_as_of_iso or 'latest'}`",
        f"- Snapshot cutoff (Moodle collector): `{moodle_as_of_iso or 'latest'}`",
        f"- Mapping version: `{mapping_version}` ({mapping_origin})",
        "- Source learner populations are kept separate; equal identifier strings are not joined.",
        "- `predicted_correctness` is an answer-correctness estimate, not a mastery probability.",
        "",
        "## Output record counts",
        "",
        "| Output | Rows |",
        "|---|---:|",
    ]
    validation.extend(
        f"| `{name}.csv` | {len(frame)} |" for name, frame in outputs.items()
    )
    validation.extend(
        [
            "",
            "## Missing-value summary",
            "",
            "| Output | Null cells | Columns with nulls |",
            "|---|---:|---:|",
            *[
                f"| `{name}.csv` | {int(frame.isna().sum().sum())} | "
                f"{int(frame.isna().any().sum())} |"
                for name, frame in outputs.items()
            ],
            "",
            "## Component availability",
            "",
            "| Population | Component | Status | Learners |",
            "|---|---|---|---:|",
        ]
    )
    status_columns = {
        "knowledge": "knowledge_status",
        "skills": "skills_status",
        "weak areas": "weak_areas_status",
        "behavior": "behavior_status",
        "engagement": "engagement_status",
        "history": "history_status",
    }
    for population, population_profiles in profiles.groupby(
        "source_population", sort=True
    ):
        for component, column in status_columns.items():
            counts = population_profiles[column].value_counts(dropna=False)
            for status, count in counts.items():
                validation.append(
                    f"| `{population}` | {component} | {status} | {int(count)} |"
                )
    validation.extend(
        [
            "",
            "## Mapping coverage",
            "",
            "| Mapping status | KC rows |",
            "|---|---:|",
            *[
                f"| `{status}` | {int(count)} |"
                for status, count in kc_topic_map["mapping_status"]
                .value_counts(dropna=False)
                .items()
            ],
            "",
            "## Knowledge states and weak-area screening",
            "",
            "| KC state status | Rows |",
            "|---|---:|",
            *[
                f"| `{k}` | {int(v)} |"
                for k, v in states["state_status"].value_counts().items()
            ],
            "",
            "| Concern status | Rows |",
            "|---|---:|",
            *[
                f"| `{k}` | {int(v)} |"
                for k, v in weak_areas["concern_status"].value_counts().items()
            ],
            "",
            "## Checks",
            "",
            *[f"- PASS: {check}" for check in checks],
            "- Validation failures: none (the builder stops before writing outputs if a check fails).",
            "",
            f"- KC rows excluded after cutoff: {states.attrs['dropped_after_cutoff']}",
            "- KC rows excluded because `last_attempt_at` was missing at a requested cutoff: "
            f"{states.attrs['dropped_without_timestamp']}",
            f"- KC-to-topic mapping rows pending review: {int(kc_topic_map['mapping_status'].eq('needs_review').sum())}",
            "- English course topic links are intentionally unset pending the Moodle refactor.",
            "- Dimension-group skill summaries are drafts from KC metadata, not supervisor-validated skill ground truth.",
            "- Moodle-specific KT states are joined only to Moodle learner IDs; English benchmark IDs remain a separate population.",
            "- Cold-start Moodle predictions are model priors for empty response histories, not individual evidence or mastery.",
            "- Behavior, engagement, and history describe Moodle collector data.",
            "- KC cutoffs are inclusive (Milestone 3 convention); collector cutoffs are strictly before the cutoff (existing temporal policy).",
            "- Weak-area rows cover every learner-KC state, including `no_current_concern` and `insufficient_evidence`; filter on `concern_status` to list concerns.",
            "- Weak-area thresholds are configuration choices, not empirically calibrated cutoffs.",
            "- Most estimated KC states have fewer direct attempts than the evidence threshold; see the concern counts above.",
            "",
        ]
    )
    (output_dir / "validation_report.md").write_text(
        "\n".join(validation), encoding="utf-8"
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Build the population-separated Milestone 4 learner model."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--knowledge-states", type=Path, default=DEFAULT_KNOWLEDGE_STATES
    )
    parser.add_argument(
        "--moodle-knowledge-states",
        type=Path,
        default=DEFAULT_MOODLE_KNOWLEDGE_STATES,
    )
    parser.add_argument("--concept-map", type=Path, default=DEFAULT_CONCEPT_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mappings-dir", type=Path, default=DEFAULT_MAPPINGS_DIR)
    parser.add_argument(
        "--as-of",
        help="snapshot cutoff: Unix seconds or timezone-aware ISO datetime",
    )
    parser.add_argument(
        "--knowledge-as-of", help="override --as-of for English KC states"
    )
    parser.add_argument(
        "--moodle-as-of", help="override --as-of for Moodle collector data"
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
        moodle_knowledge_state_file=args.moodle_knowledge_states,
        concept_map_file=args.concept_map,
        as_of=args.as_of,
        mappings_dir=args.mappings_dir,
        generated_at=args.generated_at,
        knowledge_as_of=args.knowledge_as_of,
        moodle_as_of=args.moodle_as_of,
    )
    print(f"Complete learner model written to {args.output_dir}")
    for name, frame in outputs.items():
        print(f"  {name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
