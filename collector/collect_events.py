"""Step 2/4 - learning events: filtered, context-resolved, session-tagged."""

from collections import defaultdict

from .config import t
from .context_resolution import (
    build_object_resolvers,
    load_context_index,
    resolve_course_module_id,
)
from .db import unix_to_local, write_csv
from .events import (
    EVENT_NAME_PATTERNS,
    EVENT_NAME_WHITELIST,
    LEARNING_CATEGORIES,
    classify_event,
)
from .sessions import assign_sessions, summarize_sessions


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
