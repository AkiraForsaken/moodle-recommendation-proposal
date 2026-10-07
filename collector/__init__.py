"""
Moodle data collector for the Learner Model v1 pipeline.

Outputs (written to ../data relative to this package):
    activity_catalog.csv        course modules -> topic + sequence
    learning_events.csv         filtered, context-resolved, session-tagged events
    learning_sessions.csv       one row per continuous learning session
    quiz_attempts.csv           finished non-preview attempts, normalized + duration
    course_modules_completion.csv   activity completion states (may be empty)

Design notes relevant to the thesis:
  * No quiz IDs are hard-coded. Topics come from Moodle course sections.
  * Events are selected from a documented whitelist and classified into
    behavioural categories (events.py), so "question interactions" are
    countable separately from ordinary page/quiz views.
  * Every event is mapped to a course_module_id through an explicit
    resolution chain (context_resolution.py), and the method used is recorded
    in `cmid_source` so the mapping can be audited rather than trusted blindly.
  * Sessions are derived from inter-event gaps (sessions.py, default 30 min)
    so engagement time is an explicit, documented estimate.

Module map:
    config.py              env-driven constants, table-prefix helper t()
    events.py              event whitelist + classify_event()
    sessions.py            assign_sessions(), summarize_sessions()
    context_resolution.py  event -> course_module_id resolution chain
    db.py                  cursor helpers, participant filter, CSV writer
    collect_*.py           one collection routine per output CSV

Entry points: run_collection(cursor) for a caller-supplied cursor (tests,
notebooks); main() for the full connect -> collect -> summarise run.
"""

from .collect_activities import collect_activity_catalog
from .collect_completion import collect_course_modules_completion
from .collect_events import collect_learning_events
from .collect_quizzes import collect_quiz_attempts
from .config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    INCLUDE_ROLES,
    LOCAL_TZ_NAME,
    MASTERY_THRESHOLD,
    SESSION_GAP_SECONDS,
)
from .db import choose_course, resolve_participants

__all__ = ["run_collection", "main"]


def run_collection(cursor):
    """
    Run every collection step, in dependency order, on an open cursor.

    Writes all CSVs to OUTPUT_DIR and returns (course_fullname, course_shortname)
    for the run summary. Contains no connection handling, so it can be driven by
    a fake cursor in tests.
    """
    course_id, fullname, shortname = choose_course(cursor)

    print("\nResolving participants...")
    user_filter, _ = resolve_participants(cursor, course_id)

    print("\nCollecting activity catalog...")
    catalog_rows = collect_activity_catalog(cursor, course_id)

    print("\nCollecting learning events...")
    collect_learning_events(cursor, course_id, catalog_rows, user_filter)

    print("\nCollecting quiz attempts...")
    collect_quiz_attempts(cursor, course_id, catalog_rows, user_filter)

    print("\nCollecting activity completion...")
    collect_course_modules_completion(cursor, course_id, catalog_rows, user_filter)

    return fullname, shortname


def main():
    if not DB_PASSWORD:
        raise RuntimeError("MOODLE_DOCKER_DBPASS is not set.")

    # Imported here (not at module top) so the rest of the package - and its
    # tests - can be imported on machines without the MariaDB driver installed.
    import mariadb

    print("Connecting to Moodle MariaDB...")
    conn = mariadb.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    cursor = conn.cursor()
    try:
        fullname, shortname = run_collection(cursor)

        print("\nCollection complete.")
        print(f"Course: {fullname} ({shortname})")
        print(f"Timezone: {LOCAL_TZ_NAME}")
        print(f"Mastery threshold: {MASTERY_THRESHOLD:.0%}")
        print(f"Session gap: {SESSION_GAP_SECONDS}s")
        print(f"Role filter: {INCLUDE_ROLES}")
    finally:
        cursor.close()
        conn.close()
