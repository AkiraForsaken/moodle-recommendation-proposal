"""Export scored Moodle question attempts in the knowledge-tracing event format."""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .config import OUTPUT_DIR, QUESTION_KC_MAPPING_FILE, t
from .db import unix_to_local, write_csv

ANSWER_EVENT_COLUMNS = [
    "learner_id",
    "event_index",
    "item_id",
    "question_bank_entry_id",
    "kc_id",
    "kc_dimension",
    "event_type",
    "timestamp",
    "response_time_s",
    "correct",
    "raw_fraction",
    "max_mark",
    "quiz_attempt_id",
    "question_attempt_id",
    "quiz_id",
    "attempt_number",
    "mapping_version",
    "mapping_source",
]
ROSTER_COLUMNS = ["learner_id", "course_id", "enrol_start_unix", "enrol_end_unix"]
# Accounts created by `python -m moodle_setup simulate-learners` carry this
# ID-number prefix. An export that contains any of them is labelled simulated.
SIMULATED_IDNUMBER_PREFIX = "SIM-"
SIMULATION_MANIFEST_NAME = "simulation_manifest.json"
# Written by `moodle_setup simulate-learners`: the attempt plan's manifest,
# including the generator's ceiling AUC that `knowledge_engine train` reports.
PLAN_MANIFEST_NAME = "simulation_plan_manifest.json"
MAPPING_REPORT_COLUMNS = [
    "status",
    "total_scored_question_attempts",
    "mapped_question_attempts",
    "unmapped_question_attempts",
    "mapping_version",
]
MAPPING_COLUMNS = {
    "question_bank_entry_id",
    "kc_id",
    "kc_dimension",
    "mapping_version",
    "mapping_source",
}


def load_question_kc_mapping(path=QUESTION_KC_MAPPING_FILE):
    """Load a curated, versioned Moodle question-bank-item to KC crosswalk."""
    path = Path(path)
    if not path.is_file():
        return None

    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Question-to-KC mapping is empty: {path}")
        missing = MAPPING_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Question-to-KC mapping {path} is missing columns: {sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Question-to-KC mapping has no rows: {path}")
    mapping = {}
    versions = set()
    for row in rows:
        question_id = row["question_bank_entry_id"].strip()
        kc_id = row["kc_id"].strip()
        dimension = row["kc_dimension"].strip().lower()
        version = row["mapping_version"].strip()
        source = row["mapping_source"].strip()
        if not all((question_id, kc_id, dimension, version, source)):
            raise ValueError(
                f"Question-to-KC mapping {path} contains an empty required value"
            )
        if question_id in mapping:
            raise ValueError(
                f"Question-to-KC mapping contains duplicate item {question_id}"
            )
        mapping[question_id] = {
            "kc_id": kc_id,
            "kc_dimension": dimension,
            "mapping_version": version,
            "mapping_source": source,
        }
        versions.add(version)
    if len(versions) != 1:
        raise ValueError(
            f"Question-to-KC mapping {path} must use exactly one mapping_version"
        )
    return mapping


def collect_learner_roster(cursor, course_id, user_filter):
    """Return active enrolled users, including learners without activity yet.

    Also labels the export: if any enrolled learner is a simulated account,
    ``simulation_manifest.json`` is written next to the CSVs (and a stale one
    is removed otherwise), so downstream outputs say ``simulated`` and are
    never mistaken for real learners. A mixed cohort is labelled simulated.
    """
    cursor.execute(
        f"""
        SELECT DISTINCT ue.userid, ue.timestart, ue.timeend, u.idnumber
        FROM {t('user_enrolments')} ue
        JOIN {t('enrol')} e ON e.id = ue.enrolid
        JOIN {t('user')} u ON u.id = ue.userid
        WHERE e.courseid = ?
          AND e.status = 0
          AND ue.status = 0
          AND u.deleted = 0
        ORDER BY ue.userid
        """,
        (course_id,),
    )
    rows, simulated = [], 0
    for user_id, start_time, end_time, idnumber in cursor.fetchall():
        if user_filter is not None and user_id not in user_filter:
            continue
        simulated += str(idnumber or "").startswith(SIMULATED_IDNUMBER_PREFIX)
        rows.append(
            {
                "learner_id": str(user_id),
                "course_id": course_id,
                "enrol_start_unix": int(start_time or 0),
                "enrol_end_unix": int(end_time or 0),
            }
        )
    learner_ids = [row["learner_id"] for row in rows]
    write_csv("learner_roster.csv", rows, ROSTER_COLUMNS)
    _label_simulated_export(simulated, len(rows))
    return learner_ids


def _label_simulated_export(simulated_learners, total_learners):
    marker = OUTPUT_DIR / SIMULATION_MANIFEST_NAME
    if not simulated_learners:
        marker.unlink(missing_ok=True)
        # The simulated learners were removed: their plan no longer describes the data.
        (OUTPUT_DIR / PLAN_MANIFEST_NAME).unlink(missing_ok=True)
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "simulated": True,
                "source": "Moodle accounts with a SIM- ID number (moodle_setup simulate-learners)",
                "simulated_learners": simulated_learners,
                "real_learners": total_learners - simulated_learners,
                "warning": (
                    "This export contains simulated learners. Results describe the "
                    "simulator, not real learners, and are not validity evidence."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"NOTE: {simulated_learners} of {total_learners} enrolled learners are simulated; "
        "this export is labelled 'simulated'."
    )


def _read_scored_question_attempts(cursor, course_id):
    """Read the final graded step and response bounds for completed attempts."""
    cursor.execute(
        f"""
        SELECT
            qa.userid,
            qa.id,
            qa.quiz,
            qa.attempt,
            qat.id,
            qbe.id,
            first_step.first_time,
            graded_step.timecreated,
            graded_step.fraction,
            qat.maxmark
        FROM {t('quiz_attempts')} qa
        JOIN {t('quiz')} q ON q.id = qa.quiz
        JOIN {t('question_attempts')} qat
          ON qat.questionusageid = qa.uniqueid
        JOIN {t('question_versions')} qv ON qv.questionid = qat.questionid
        JOIN {t('question_bank_entries')} qbe
          ON qbe.id = qv.questionbankentryid
        JOIN (
            SELECT questionattemptid, MIN(timecreated) AS first_time
            FROM {t('question_attempt_steps')}
            GROUP BY questionattemptid
        ) first_step ON first_step.questionattemptid = qat.id
        JOIN (
            SELECT questionattemptid, MAX(sequencenumber) AS graded_sequence
            FROM {t('question_attempt_steps')}
            WHERE fraction IS NOT NULL
            GROUP BY questionattemptid
        ) latest_graded
          ON latest_graded.questionattemptid = qat.id
        JOIN {t('question_attempt_steps')} graded_step
          ON graded_step.questionattemptid = qat.id
         AND graded_step.sequencenumber = latest_graded.graded_sequence
        WHERE q.course = ?
          AND qa.preview = 0
          AND qa.state = 'finished'
        ORDER BY qa.userid, graded_step.timecreated, qa.id, qat.slot, qat.id
        """,
        (course_id,),
    )
    return cursor.fetchall()


def _write_mapping_report(path, total, mapped, version, status):
    row = {
        "status": status,
        "total_scored_question_attempts": total,
        "mapped_question_attempts": mapped,
        "unmapped_question_attempts": total - mapped,
        "mapping_version": version or "",
    }
    with Path(path).open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=MAPPING_REPORT_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def collect_question_answer_events(
    cursor,
    course_id,
    user_filter,
    mapping_file=QUESTION_KC_MAPPING_FILE,
):
    """Write one canonical answer event for each completed, graded item attempt.

    Moodle can award partial credit. ``correct`` is 1 only for full credit;
    ``raw_fraction`` preserves the graded fraction for other analyses.
    Responses without a graded fraction and items absent from the curated
    crosswalk are excluded from the KT event CSV and counted in its report.
    """
    mapping = load_question_kc_mapping(mapping_file)
    if mapping is None:
        write_csv("question_answer_events.csv", [], ANSWER_EVENT_COLUMNS)
        _write_mapping_report(
            OUTPUT_DIR / "question_answer_mapping_report.csv",
            0,
            0,
            None,
            "skipped_missing_mapping",
        )
        print(
            "WARNING: question-answer KT export skipped; configure "
            f"MOODLE_QUESTION_KC_MAPPING with a reviewed crosswalk ({mapping_file})."
        )
        return []

    db_rows = _read_scored_question_attempts(cursor, course_id)
    eligible = [row for row in db_rows if user_filter is None or row[0] in user_filter]
    mapped_by_learner = defaultdict(list)
    unmapped_count = 0
    for row in eligible:
        (
            learner_id,
            quiz_attempt_id,
            quiz_id,
            attempt_number,
            question_attempt_id,
            bank_entry_id,
            first_time,
            graded_time,
            fraction,
            max_mark,
        ) = row
        link = mapping.get(str(bank_entry_id))
        if link is None:
            unmapped_count += 1
            continue
        score_fraction = float(fraction)
        if not math.isfinite(score_fraction):
            raise ValueError(
                f"Moodle returned a non-finite score fraction for question attempt "
                f"{question_attempt_id}"
            )
        event_time = int(graded_time)
        # Moodle stores -1 as the first-step time of a question that was never
        # shown on screen; that is not a time, so no response time is derived.
        response_time = (
            max(0, event_time - int(first_time))
            if first_time is not None and int(first_time) > 0
            else None
        )
        mapped_by_learner[str(learner_id)].append(
            {
                "learner_id": str(learner_id),
                "item_id": f"moodle-qbe:{bank_entry_id}",
                "question_bank_entry_id": str(bank_entry_id),
                "kc_id": link["kc_id"],
                "kc_dimension": link["kc_dimension"],
                "event_type": "answer",
                "timestamp": event_time,
                "response_time_s": response_time,
                "correct": int(score_fraction >= 1.0),
                "raw_fraction": score_fraction,
                "max_mark": float(max_mark) if max_mark is not None else None,
                "quiz_attempt_id": quiz_attempt_id,
                "question_attempt_id": question_attempt_id,
                "quiz_id": quiz_id,
                "attempt_number": attempt_number,
                "mapping_version": link["mapping_version"],
                "mapping_source": link["mapping_source"],
            }
        )

    rows = []
    for learner_id in sorted(mapped_by_learner):
        events = sorted(
            mapped_by_learner[learner_id],
            key=lambda event: (
                event["timestamp"],
                event["quiz_attempt_id"],
                event["question_attempt_id"],
            ),
        )
        for event_index, event in enumerate(events, start=1):
            event["event_index"] = event_index
            event["timestamp"] = int(event["timestamp"])
            rows.append(event)

    write_csv("question_answer_events.csv", rows, ANSWER_EVENT_COLUMNS)
    version = next(iter(mapping.values()))["mapping_version"]
    _write_mapping_report(
        OUTPUT_DIR / "question_answer_mapping_report.csv",
        len(eligible),
        len(rows),
        version,
        "complete" if not unmapped_count else "partial_mapping",
    )
    if unmapped_count:
        print(
            f"WARNING: excluded {unmapped_count:,} graded question attempts with no "
            "question-to-KC mapping; see question_answer_mapping_report.csv."
        )
    return rows
