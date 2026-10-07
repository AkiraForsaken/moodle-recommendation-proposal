#!/usr/bin/env python3
"""
Moodle data collector for the Learner Model v1 pipeline.

Outputs (written to ../data relative to this file):
    activity_catalog.csv        course modules -> topic + sequence
    learning_events.csv         filtered, context-resolved, session-tagged events
    learning_sessions.csv       one row per continuous learning session
    quiz_attempts.csv           finished non-preview attempts, normalized + duration
    course_modules_completion.csv   activity completion states (may be empty)

Design notes relevant to the thesis:
  * No quiz IDs are hard-coded. Topics come from Moodle course sections.
  * Events are selected from a documented whitelist and classified into
    behavioural categories, so "question interactions" are countable
    separately from ordinary page/quiz views.
  * Every event is mapped to a course_module_id through an explicit
    resolution chain, and the method used is recorded in `cmid_source`
    so the mapping can be audited rather than trusted blindly.
  * Sessions are derived from inter-event gaps (default 30 min) so
    engagement time is an explicit, documented estimate.
"""

import csv
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import mariadb

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DB_HOST = os.getenv("MOODLE_DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("MOODLE_DB_PORT", "3306"))
DB_NAME = os.getenv("MOODLE_DB_NAME", "moodle")
DB_USER = os.getenv("MOODLE_DB_USER", "moodle")
DB_PASSWORD = os.getenv("MOODLE_DOCKER_DBPASS")
DB_PREFIX = os.getenv("MOODLE_DB_PREFIX", "m_")

COURSE_ID_ENV = os.getenv("MOODLE_COURSE_ID")
LOCAL_TZ_NAME = os.getenv("MOODLE_TZ", "Asia/Ho_Chi_Minh")
MASTERY_THRESHOLD = float(os.getenv("MOODLE_MASTERY_THRESHOLD", "0.80"))

# Session boundary: consecutive events by the same learner that are this far
# apart (or closer) belong to the same continuous learning session.
SESSION_GAP_SECONDS = int(os.getenv("MOODLE_SESSION_GAP_SECONDS", "1800"))

# Which enrolled roles count as "learners". Set to "all" to disable filtering
# (useful when debugging, but teacher/admin authoring traffic will pollute the
# learner model - in the current test course the teacher account produced more
# events than the student).
INCLUDE_ROLES = os.getenv("MOODLE_INCLUDE_ROLES", "student")

OUTPUT_DIR = Path(
    os.getenv("MOODLE_OUTPUT_DIR", Path(__file__).resolve().parent.parent / "data")
)

LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)

CONTEXT_MODULE = 70

# --------------------------------------------------------------------------
# Event selection and classification
#
# Keep this list explicit and documented: the build plan requires a "small,
# documented set of meaningful events", not everything the log store holds.
# Names that do not exist in a given Moodle version simply match nothing.
# --------------------------------------------------------------------------

QUESTION_EVENTS = [
    r"\core\event\question_viewed",
    r"\core\event\question_answered",
    r"\mod_quiz\event\question_manually_graded",
    r"\mod_lesson\event\question_viewed",
    r"\mod_lesson\event\question_answered",
]

CONTENT_VIEW_EVENTS = [
    r"\mod_resource\event\course_module_viewed",
    r"\mod_page\event\course_module_viewed",
    r"\mod_url\event\course_module_viewed",
    r"\mod_book\event\course_module_viewed",
    r"\mod_lesson\event\course_module_viewed",
    r"\mod_quiz\event\course_module_viewed",
    r"\mod_assign\event\course_module_viewed",
    r"\mod_forum\event\course_module_viewed",
]

LESSON_EVENTS = [
    r"\mod_lesson\event\lesson_started",
    r"\mod_lesson\event\lesson_resumed",
    r"\mod_lesson\event\lesson_ended",
    r"\mod_lesson\event\content_page_viewed",
    r"\mod_lesson\event\highscore_added",
]

QUIZ_ATTEMPT_EVENTS = [
    r"\mod_quiz\event\attempt_started",
    r"\mod_quiz\event\attempt_updated",
    r"\mod_quiz\event\attempt_submitted",
    r"\mod_quiz\event\attempt_graded",
    r"\mod_quiz\event\attempt_summary_viewed",
    r"\mod_quiz\event\attempt_viewed",
    r"\mod_quiz\event\attempt_reviewed",
    r"\mod_quiz\event\attempt_abandoned",
    r"\mod_quiz\event\attempt_becameoverdue",
]

COMPLETION_EVENTS = [
    r"\core\event\course_module_completion_updated",
]

EVENT_NAME_WHITELIST = (
    QUESTION_EVENTS
    + CONTENT_VIEW_EVENTS
    + LESSON_EVENTS
    + QUIZ_ATTEMPT_EVENTS
    + COMPLETION_EVENTS
)

# Safety net for version-specific question events we may not have listed.
# NOTE: these run through SQL LIKE, so they must not contain backslashes
# (MariaDB treats "\" as an escape character inside LIKE patterns).
EVENT_NAME_PATTERNS = [
    "%question_viewed",
    "%question_answered",
    "%question_manually_graded",
    "%question_flagged",
]

# Explicitly excluded: authoring/teaching traffic such as slot_created,
# edit_page_viewed, report_viewed, page_created/updated/deleted,
# section_shuffle_updated, attempt_deleted, attempt_preview_started.
# These are course-construction events, not learning behaviour.

QUESTION_EVENT_SET = set(QUESTION_EVENTS)
CONTENT_VIEW_SET = set(CONTENT_VIEW_EVENTS)
LESSON_EVENT_SET = set(LESSON_EVENTS)
QUIZ_ATTEMPT_SET = set(QUIZ_ATTEMPT_EVENTS)
COMPLETION_EVENT_SET = set(COMPLETION_EVENTS)

# Which categories count as evidence of actual learning behaviour.
LEARNING_CATEGORIES = {
    "content_view",
    "lesson_interaction",
    "question_interaction",
    "quiz_attempt",
}


def classify_event(event_name, target, action):
    """
    Map a raw Moodle event name to one behavioural category.

    Returns one of:
        question_interaction | lesson_interaction | content_view
        quiz_attempt | quiz_review | completion | other
    """
    name = event_name or ""
    lowered = name.lower()

    if name in QUESTION_EVENT_SET or "question_" in lowered.rsplit("\\", 1)[-1]:
        return "question_interaction"
    if name in COMPLETION_EVENT_SET:
        return "completion"
    if name in LESSON_EVENT_SET:
        return "lesson_interaction"
    if name in QUIZ_ATTEMPT_SET:
        # Reviewing a graded attempt is feedback-seeking, not assessment.
        if lowered.endswith(("attempt_reviewed", "attempt_viewed")):
            return "quiz_review"
        return "quiz_attempt"
    if name in CONTENT_VIEW_SET:
        return "content_view"
    if (target or "") == "course_module" and (action or "") == "viewed":
        return "content_view"
    return "other"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def t(name):
    """Prefix a Moodle table name (default prefix m_)."""
    return f"{DB_PREFIX}{name}"


def unix_to_local(value):
    if value in (None, 0):
        return ""
    dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
    return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def write_csv(filename, rows, fieldnames):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows -> {path}")


def table_exists(cursor, table_name):
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = ?
        """,
        (table_name,),
    )
    row = cursor.fetchone()
    return bool(row and row[0])


def existing_columns(cursor, table_name):
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = ?
        """,
        (table_name,),
    )
    return {r[0].lower() for r in cursor.fetchall()}


def choose_course(cursor):
    if COURSE_ID_ENV:
        cursor.execute(
            f"SELECT id, fullname, shortname FROM {t('course')} WHERE id = ?",
            (int(COURSE_ID_ENV),),
        )
    else:
        cursor.execute(f"""
            SELECT id, fullname, shortname
            FROM {t('course')}
            WHERE id > 1 AND visible = 1
            ORDER BY id
            LIMIT 1
        """)

    row = cursor.fetchone()
    if not row:
        raise RuntimeError("No course found. Set MOODLE_COURSE_ID.")
    print(f"Using course {row[0]}: {row[1]} ({row[2]})")
    return row


# --------------------------------------------------------------------------
# Participants (role filtering)
# --------------------------------------------------------------------------


def resolve_participants(cursor, course_id):
    """
    Return (userid_filter, role_map).

    userid_filter is None when no filtering should be applied, otherwise a set
    of user IDs holding one of INCLUDE_ROLES in this course's context.
    """
    cursor.execute(
        f"""
        SELECT ra.userid, r.shortname
        FROM {t('role_assignments')} ra
        JOIN {t('role')} r ON r.id = ra.roleid
        JOIN {t('context')} ctx ON ctx.id = ra.contextid
        WHERE ctx.contextlevel = 50 AND ctx.instanceid = ?
        """,
        (course_id,),
    )
    role_map = defaultdict(set)
    for user_id, shortname in cursor.fetchall():
        role_map[user_id].add(shortname)

    if role_map:
        summary = ", ".join(
            f"{uid}({'/'.join(sorted(roles))})"
            for uid, roles in sorted(role_map.items())
        )
        print(f"Course participants: {summary}")
    else:
        print("Course participants: no course-level role assignments found.")

    if INCLUDE_ROLES.strip().lower() in ("all", "", "none"):
        print("Role filter: disabled (MOODLE_INCLUDE_ROLES=all)")
        return None, role_map

    wanted = {r.strip().lower() for r in INCLUDE_ROLES.split(",") if r.strip()}
    keep = {
        uid for uid, roles in role_map.items() if {r.lower() for r in roles} & wanted
    }

    if not keep:
        print(
            f"Role filter: no users matched {sorted(wanted)} - falling back to "
            "no filtering. Check role assignments (they may sit at category level)."
        )
        return None, role_map

    print(f"Role filter: keeping {sorted(keep)} (roles: {sorted(wanted)})")
    return keep, role_map


# --------------------------------------------------------------------------
# Step 1 - Activity catalog
# --------------------------------------------------------------------------


def collect_activity_catalog(cursor, course_id):
    query = f"""
        SELECT
            cm.id AS course_module_id,
            cs.id AS section_id,
            cs.section AS section_number,
            COALESCE(NULLIF(cs.name, ''), CONCAT('Topic ', cs.section)) AS topic,
            m.name AS activity_type,
            cm.instance AS activity_id,
            CASE m.name
                WHEN 'quiz' THEN q.name
                WHEN 'lesson' THEN l.name
                WHEN 'resource' THEN r.name
                WHEN 'page' THEN p.name
                WHEN 'forum' THEN f.name
                WHEN 'assign' THEN a.name
                WHEN 'url' THEN u.name
                ELSE NULL
            END AS activity_name,
            cm.visible,
            cm.completion
        FROM {t('course_modules')} cm
        JOIN {t('modules')} m ON m.id = cm.module
        JOIN {t('course_sections')} cs ON cs.id = cm.section
        LEFT JOIN {t('quiz')} q ON m.name = 'quiz' AND q.id = cm.instance
        LEFT JOIN {t('lesson')} l ON m.name = 'lesson' AND l.id = cm.instance
        LEFT JOIN {t('resource')} r ON m.name = 'resource' AND r.id = cm.instance
        LEFT JOIN {t('page')} p ON m.name = 'page' AND p.id = cm.instance
        LEFT JOIN {t('forum')} f ON m.name = 'forum' AND f.id = cm.instance
        LEFT JOIN {t('assign')} a ON m.name = 'assign' AND a.id = cm.instance
        LEFT JOIN {t('url')} u ON m.name = 'url' AND u.id = cm.instance
        WHERE cm.course = ?
        ORDER BY cs.section, cm.id
    """
    cursor.execute(query, (course_id,))
    rows = []
    for i, row in enumerate(cursor.fetchall(), 1):
        (
            cmid,
            section_id,
            section_number,
            topic,
            activity_type,
            activity_id,
            name,
            visible,
            completion,
        ) = row
        rows.append(
            {
                "course_module_id": cmid,
                "section_id": section_id,
                "section_number": section_number,
                "activity_type": activity_type,
                "activity_id": activity_id,
                "activity_name": name or "",
                "topic": topic,
                "sequence": i,
                "visible": visible,
                "completion_tracking": completion,
            }
        )

    write_csv(
        "activity_catalog.csv",
        rows,
        [
            "course_module_id",
            "section_id",
            "section_number",
            "activity_type",
            "activity_id",
            "activity_name",
            "topic",
            "sequence",
            "visible",
            "completion_tracking",
        ],
    )
    return rows


# --------------------------------------------------------------------------
# Step 2 - Course-module resolution (context + object fallbacks)
# --------------------------------------------------------------------------


def load_context_index(cursor, course_id):
    """
    Load the course's context subtree once:
        {contextid: (contextlevel, instanceid, path)}

    The `path` field ('/1/3/25/28') lets us walk up from any deep context
    (question, block, user) to the enclosing CONTEXT_MODULE, which is what the
    learner model actually needs.
    """
    cursor.execute(
        f"""
        SELECT id, path
        FROM {t('context')}
        WHERE contextlevel = 50 AND instanceid = ?
        """,
        (course_id,),
    )
    course_ctx = cursor.fetchone()

    if course_ctx and course_ctx[1]:
        course_ctx_id, course_path = course_ctx
        cursor.execute(
            f"""
            SELECT id, contextlevel, instanceid, path
            FROM {t('context')}
            WHERE id = ? OR path LIKE ?
            """,
            (course_ctx_id, f"{course_path}/%"),
        )
    else:
        # No resolvable course context: fall back to the full tree.
        cursor.execute(f"SELECT id, contextlevel, instanceid, path FROM {t('context')}")

    index = {}
    for ctx_id, level, instance_id, path in cursor.fetchall():
        index[ctx_id] = (level, instance_id, path or "")
    print(f"Loaded {len(index):,} context records for module resolution.")
    return index


def build_object_resolvers(cursor, course_id, catalog_rows):
    """
    Build lookup tables that translate an event's objectid into a
    course_module_id when the context alone is not enough.
    """
    instance_to_cmid = {
        (row["activity_type"], row["activity_id"]): row["course_module_id"]
        for row in catalog_rows
    }

    resolvers = {}

    # quiz attempt id -> cmid  (attempt_* events put the attempt id in objectid)
    cursor.execute(
        f"""
        SELECT qa.id, q.id
        FROM {t('quiz_attempts')} qa
        JOIN {t('quiz')} q ON q.id = qa.quiz
        WHERE q.course = ?
        """,
        (course_id,),
    )
    resolvers["quiz_attempts"] = {
        att_id: instance_to_cmid.get(("quiz", quiz_id))
        for att_id, quiz_id in cursor.fetchall()
    }

    # quiz slot / section ids -> cmid
    for table, column in (("quiz_slots", "quizid"), ("quiz_sections", "quizid")):
        if not table_exists(cursor, t(table)):
            resolvers[table] = {}
            continue
        cursor.execute(
            f"""
            SELECT s.id, s.{column}
            FROM {t(table)} s
            JOIN {t('quiz')} q ON q.id = s.{column}
            WHERE q.course = ?
            """,
            (course_id,),
        )
        resolvers[table] = {
            row_id: instance_to_cmid.get(("quiz", quiz_id))
            for row_id, quiz_id in cursor.fetchall()
        }

    # lesson page id -> cmid (lesson question/content events reference pages)
    cursor.execute(
        f"""
        SELECT lp.id, l.id
        FROM {t('lesson_pages')} lp
        JOIN {t('lesson')} l ON l.id = lp.lessonid
        WHERE l.course = ?
        """,
        (course_id,),
    )
    resolvers["lesson_pages"] = {
        page_id: instance_to_cmid.get(("lesson", lesson_id))
        for page_id, lesson_id in cursor.fetchall()
    }

    # Direct module tables: objectid is the activity instance id.
    for module_name in (
        "quiz",
        "lesson",
        "resource",
        "page",
        "forum",
        "assign",
        "url",
        "book",
    ):
        resolvers[module_name] = {
            activity_id: cmid
            for (mod, activity_id), cmid in instance_to_cmid.items()
            if mod == module_name
        }

    # course_modules_completion / course_modules: objectid is already a cmid.
    valid_cmids = {row["course_module_id"] for row in catalog_rows}
    resolvers["course_modules"] = {cmid: cmid for cmid in valid_cmids}

    return resolvers


def resolve_course_module_id(
    context_id,
    context_level,
    context_instance_id,
    object_table,
    object_id,
    context_index,
    resolvers,
    valid_cmids,
):
    """
    Resolution chain, most reliable first. Returns (cmid, source).

    1. context_module   - the event is already at module context
    2. context_ancestor - walk the context path up to the enclosing module
    3. object_table     - translate objectid via the object's own table
    4. unresolved       - record it rather than silently dropping it
    """
    # 1. Direct module context.
    if context_level == CONTEXT_MODULE and context_instance_id in valid_cmids:
        return context_instance_id, "context_module"

    # 2. Walk the context path upwards.
    entry = context_index.get(context_id)
    if entry:
        _, _, path = entry
        ancestors = [int(p) for p in path.strip("/").split("/") if p]
        for ancestor_id in reversed(ancestors):
            if ancestor_id == context_id:
                continue
            ancestor = context_index.get(ancestor_id)
            if not ancestor:
                continue
            level, instance_id, _ = ancestor
            if level == CONTEXT_MODULE and instance_id in valid_cmids:
                return instance_id, "context_ancestor"

    # 3. Fall back to the object itself.
    if object_table and object_id:
        table_map = resolvers.get(object_table)
        if table_map:
            cmid = table_map.get(object_id)
            if cmid in valid_cmids:
                return cmid, f"object_table:{object_table}"

    return None, "unresolved"


# --------------------------------------------------------------------------
# Step 4 - Session boundaries
# --------------------------------------------------------------------------


def assign_sessions(rows, gap_seconds=SESSION_GAP_SECONDS):
    """
    Tag each event with a session id based on the gap to the learner's previous
    event. A gap <= gap_seconds continues the current session; anything larger
    starts a new one. Events are returned sorted by learner, then time.
    """
    rows.sort(key=lambda r: (r["learner_id"], r["timestamp_unix"], r["log_id"]))

    prev_learner = None
    prev_time = None
    session_counter = 0
    seq_in_session = 0

    for row in rows:
        learner = row["learner_id"]
        now = int(row["timestamp_unix"])

        if learner != prev_learner:
            session_counter = 1
            seq_in_session = 1
            delta = ""
            is_start = 1
        else:
            delta = now - prev_time
            if delta > gap_seconds:
                session_counter += 1
                seq_in_session = 1
                is_start = 1
            else:
                seq_in_session += 1
                is_start = 0

        row["seconds_since_prev_event"] = delta
        row["session_id"] = f"u{learner}-s{session_counter:03d}"
        row["session_event_seq"] = seq_in_session
        row["is_session_start"] = is_start

        prev_learner = learner
        prev_time = now

    return rows


def summarize_sessions(rows):
    """One row per session: bounds, duration, event mix and topics touched."""
    sessions = {}
    for row in rows:
        sid = row["session_id"]
        session = sessions.get(sid)
        if session is None:
            session = {
                "session_id": sid,
                "learner_id": row["learner_id"],
                "course_id": row["course_id"],
                "start_time_unix": row["timestamp_unix"],
                "end_time_unix": row["timestamp_unix"],
                "event_count": 0,
                "learning_event_count": 0,
                "question_interactions": 0,
                "topics": [],
                "course_module_ids": set(),
            }
            sessions[sid] = session

        session["end_time_unix"] = row["timestamp_unix"]
        session["event_count"] += 1
        if row["event_category"] in LEARNING_CATEGORIES:
            session["learning_event_count"] += 1
        if row["event_category"] == "question_interaction":
            session["question_interactions"] += 1

        topic = row.get("_topic")
        if topic and topic not in session["topics"]:
            session["topics"].append(topic)
        if row["course_module_id"] != "":
            session["course_module_ids"].add(row["course_module_id"])

    out = []
    for session in sessions.values():
        duration = int(session["end_time_unix"]) - int(session["start_time_unix"])
        out.append(
            {
                "session_id": session["session_id"],
                "learner_id": session["learner_id"],
                "course_id": session["course_id"],
                "start_time": unix_to_local(session["start_time_unix"]),
                "end_time": unix_to_local(session["end_time_unix"]),
                "start_time_unix": session["start_time_unix"],
                "end_time_unix": session["end_time_unix"],
                "duration_seconds": duration,
                "event_count": session["event_count"],
                "learning_event_count": session["learning_event_count"],
                "question_interactions": session["question_interactions"],
                "distinct_modules": len(session["course_module_ids"]),
                "topics": " | ".join(session["topics"]),
            }
        )

    out.sort(key=lambda r: (r["learner_id"], r["start_time_unix"]))
    return out


# --------------------------------------------------------------------------
# Step 2/4 - Learning events
# --------------------------------------------------------------------------


def collect_learning_events(cursor, course_id, catalog_rows, user_filter):
    context_index = load_context_index(cursor, course_id)
    resolvers = build_object_resolvers(cursor, course_id, catalog_rows)
    valid_cmids = {row["course_module_id"] for row in catalog_rows}
    topic_by_cmid = {row["course_module_id"]: row["topic"] for row in catalog_rows}
    type_by_cmid = {
        row["course_module_id"]: row["activity_type"] for row in catalog_rows
    }

    name_placeholders = ", ".join(["?"] * len(EVENT_NAME_WHITELIST))
    pattern_clause = " OR ".join(["l.eventname LIKE ?"] * len(EVENT_NAME_PATTERNS))

    query = f"""
        SELECT
            l.id, l.userid, l.courseid, l.eventname, l.action, l.target,
            l.objecttable, l.objectid, l.contextid, l.contextlevel,
            l.contextinstanceid, l.timecreated
        FROM {t('logstore_standard_log')} l
        WHERE l.courseid = ?
          AND l.userid > 0
          AND (
                l.eventname IN ({name_placeholders})
                OR {pattern_clause}
              )
        ORDER BY l.userid, l.timecreated, l.id
    """
    params = [course_id] + EVENT_NAME_WHITELIST + EVENT_NAME_PATTERNS
    cursor.execute(query, params)

    rows = []
    source_counts = defaultdict(int)
    category_counts = defaultdict(int)
    skipped_users = 0

    for row in cursor.fetchall():
        (
            log_id,
            learner_id,
            event_course_id,
            event_name,
            action,
            target,
            object_table,
            object_id,
            context_id,
            context_level,
            context_instance_id,
            timecreated,
        ) = row

        if user_filter is not None and learner_id not in user_filter:
            skipped_users += 1
            continue

        cmid, cmid_source = resolve_course_module_id(
            context_id,
            context_level,
            context_instance_id,
            object_table,
            object_id,
            context_index,
            resolvers,
            valid_cmids,
        )
        category = classify_event(event_name, target, action)

        source_counts[cmid_source] += 1
        category_counts[category] += 1

        rows.append(
            {
                "log_id": log_id,
                "learner_id": learner_id,
                "course_id": event_course_id,
                "course_module_id": cmid if cmid is not None else "",
                "cmid_source": cmid_source,
                "event_name": event_name or "",
                "event_category": category,
                "is_learning_event": int(category in LEARNING_CATEGORIES),
                "action": action or "",
                "target": target or "",
                "object_table": object_table or "",
                "object_id": object_id if object_id is not None else "",
                "context_id": context_id if context_id is not None else "",
                "context_level": context_level if context_level is not None else "",
                "context_instance_id": (
                    context_instance_id if context_instance_id is not None else ""
                ),
                "timestamp": unix_to_local(timecreated),
                "timestamp_unix": timecreated,
                # internal only, used for the session summary
                "_topic": topic_by_cmid.get(cmid, ""),
                "_activity_type": type_by_cmid.get(cmid, ""),
            }
        )

    assign_sessions(rows)
    session_rows = summarize_sessions(rows)

    for row in rows:
        row.pop("_topic", None)
        row.pop("_activity_type", None)

    write_csv(
        "learning_events.csv",
        rows,
        [
            "log_id",
            "learner_id",
            "course_id",
            "course_module_id",
            "cmid_source",
            "event_name",
            "event_category",
            "is_learning_event",
            "action",
            "target",
            "object_table",
            "object_id",
            "context_id",
            "context_level",
            "context_instance_id",
            "timestamp",
            "timestamp_unix",
            "seconds_since_prev_event",
            "session_id",
            "session_event_seq",
            "is_session_start",
        ],
    )

    write_csv(
        "learning_sessions.csv",
        session_rows,
        [
            "session_id",
            "learner_id",
            "course_id",
            "start_time",
            "end_time",
            "start_time_unix",
            "end_time_unix",
            "duration_seconds",
            "event_count",
            "learning_event_count",
            "question_interactions",
            "distinct_modules",
            "topics",
        ],
    )

    if skipped_users:
        print(f"  Skipped {skipped_users:,} events from non-learner accounts.")
    print(
        "  Event categories: "
        + ", ".join(f"{k}={v}" for k, v in sorted(category_counts.items()))
    )
    print(
        "  Module resolution: "
        + ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items()))
    )
    if source_counts.get("unresolved"):
        print(
            f"  WARNING: {source_counts['unresolved']} events could not be mapped "
            "to a course module; they are kept with an empty course_module_id."
        )

    return rows


# --------------------------------------------------------------------------
# Step 3 - Quiz attempts
# --------------------------------------------------------------------------


def collect_quiz_attempts(cursor, course_id, catalog_rows, user_filter):
    topic_by_quiz = {
        row["activity_id"]: row["topic"]
        for row in catalog_rows
        if row["activity_type"] == "quiz"
    }
    seq_by_quiz = {
        row["activity_id"]: row["sequence"]
        for row in catalog_rows
        if row["activity_type"] == "quiz"
    }

    query = f"""
        SELECT
            qa.userid, q.id, q.name, qa.attempt, qa.state,
            qa.sumgrades, q.sumgrades, qa.timestart, qa.timefinish
        FROM {t('quiz_attempts')} qa
        JOIN {t('quiz')} q ON q.id = qa.quiz
        WHERE q.course = ?
          AND qa.preview = 0
          AND qa.state = 'finished'
        ORDER BY qa.userid, qa.timestart, qa.id
    """
    cursor.execute(query, (course_id,))
    rows = []
    for row in cursor.fetchall():
        (
            learner_id,
            quiz_id,
            quiz_name,
            attempt,
            state,
            score,
            max_score,
            timestart,
            timefinish,
        ) = row

        if user_filter is not None and learner_id not in user_filter:
            continue

        max_value = float(max_score or 0)
        score_value = float(score or 0)
        normalized = "" if max_value == 0 else round(score_value / max_value, 4)
        mastery = 0 if normalized == "" else int(normalized >= MASTERY_THRESHOLD)
        duration = ""
        if timestart and timefinish:
            duration = int(timefinish - timestart)

        rows.append(
            {
                "learner_id": learner_id,
                "quiz_id": quiz_id,
                "quiz_name": quiz_name,
                "topic": topic_by_quiz.get(quiz_id, ""),
                "sequence_position": seq_by_quiz.get(quiz_id, ""),
                "attempt": attempt,
                "state": state,
                "score": score_value,
                "max_score": max_value,
                "normalized_score": normalized,
                "mastery_threshold_reached": mastery,
                "start_time": unix_to_local(timestart),
                "finish_time": unix_to_local(timefinish),
                "duration_seconds": duration,
                "start_time_unix": timestart,
                "finish_time_unix": timefinish,
            }
        )

    write_csv(
        "quiz_attempts.csv",
        rows,
        [
            "learner_id",
            "quiz_id",
            "quiz_name",
            "topic",
            "sequence_position",
            "attempt",
            "state",
            "score",
            "max_score",
            "normalized_score",
            "mastery_threshold_reached",
            "start_time",
            "finish_time",
            "duration_seconds",
            "start_time_unix",
            "finish_time_unix",
        ],
    )
    return rows


# --------------------------------------------------------------------------
# Activity completion
# --------------------------------------------------------------------------

COMPLETION_STATE_LABELS = {
    0: "incomplete",
    1: "complete",
    2: "complete_pass",
    3: "complete_fail",
}

COMPLETION_OPTIONAL_COLUMNS = [
    "completionstate",
    "overrideby",
    "timemodified",
    "timecompleted",
    "viewed",
]


def collect_course_modules_completion(cursor, course_id, catalog_rows, user_filter):
    """
    Export activity completion states.

    Completion tracking may be switched off, in which case this writes a
    header-only CSV. That is intentional: the pipeline stays wired up so the
    data flows the moment completion is enabled in Moodle.

    The column set is introspected because Moodle 4.3+ moved the `viewed` flag
    out of course_modules_completion into course_modules_viewed.
    """
    table = t("course_modules_completion")
    fieldnames = [
        "learner_id",
        "course_module_id",
        "activity_type",
        "activity_name",
        "topic",
        "sequence",
        "completion_state",
        "completion_state_label",
        "viewed",
        "overridden_by",
        "time_modified",
        "time_modified_unix",
    ]

    if not table_exists(cursor, table):
        print(f"  {table} not found - writing empty file.")
        write_csv("course_modules_completion.csv", [], fieldnames)
        return []

    available = existing_columns(cursor, table)
    selected = [c for c in COMPLETION_OPTIONAL_COLUMNS if c in available]
    select_sql = ", ".join(f"cmc.{c}" for c in selected)

    has_viewed_table = table_exists(cursor, t("course_modules_viewed"))
    viewed_join = ""
    viewed_select = ""
    if has_viewed_table and "viewed" not in available:
        viewed_select = ", cmv.timecreated AS viewed_time"
        viewed_join = f"""
            LEFT JOIN {t('course_modules_viewed')} cmv
                   ON cmv.coursemoduleid = cmc.coursemoduleid
                  AND cmv.userid = cmc.userid
        """

    query = f"""
        SELECT cmc.userid, cmc.coursemoduleid{',' if selected else ''} {select_sql}
               {viewed_select}
        FROM {table} cmc
        JOIN {t('course_modules')} cm ON cm.id = cmc.coursemoduleid
        {viewed_join}
        WHERE cm.course = ?
        ORDER BY cmc.userid, cmc.coursemoduleid
    """
    cursor.execute(query, (course_id,))
    fetched = cursor.fetchall()

    catalog_by_cmid = {row["course_module_id"]: row for row in catalog_rows}
    column_order = (
        ["userid", "coursemoduleid"]
        + selected
        + (["viewed_time"] if viewed_select else [])
    )

    rows = []
    for raw in fetched:
        record = dict(zip(column_order, raw))
        learner_id = record["userid"]
        if user_filter is not None and learner_id not in user_filter:
            continue

        cmid = record["coursemoduleid"]
        catalog = catalog_by_cmid.get(cmid, {})
        state = record.get("completionstate")
        timemodified = record.get("timemodified") or record.get("timecompleted")

        if "viewed" in record:
            viewed = record.get("viewed")
        elif "viewed_time" in record:
            viewed = 1 if record.get("viewed_time") else 0
        else:
            viewed = ""

        rows.append(
            {
                "learner_id": learner_id,
                "course_module_id": cmid,
                "activity_type": catalog.get("activity_type", ""),
                "activity_name": catalog.get("activity_name", ""),
                "topic": catalog.get("topic", ""),
                "sequence": catalog.get("sequence", ""),
                "completion_state": state if state is not None else "",
                "completion_state_label": COMPLETION_STATE_LABELS.get(state, "unknown"),
                "viewed": viewed,
                "overridden_by": record.get("overrideby") or "",
                "time_modified": unix_to_local(timemodified),
                "time_modified_unix": timemodified or "",
            }
        )

    write_csv("course_modules_completion.csv", rows, fieldnames)

    if not rows:
        tracked = sum(1 for r in catalog_rows if r.get("completion_tracking"))
        print(
            "  No completion records. "
            f"{tracked}/{len(catalog_rows)} modules have completion tracking enabled; "
            "check Site admin > Advanced features > Enable completion tracking."
        )
    return rows


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main():
    if not DB_PASSWORD:
        raise RuntimeError("MOODLE_DOCKER_DBPASS is not set.")

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

        print("\nCollection complete.")
        print(f"Course: {fullname} ({shortname})")
        print(f"Timezone: {LOCAL_TZ_NAME}")
        print(f"Mastery threshold: {MASTERY_THRESHOLD:.0%}")
        print(f"Session gap: {SESSION_GAP_SECONDS}s")
        print(f"Role filter: {INCLUDE_ROLES}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
