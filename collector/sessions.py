"""
Session boundaries.

A session is a run of one learner's events in which no two consecutive events
are more than SESSION_GAP_SECONDS apart (default 30 minutes). This module holds
that rule and the per-session summary. Pure logic - no database access.
"""

from .config import SESSION_GAP_SECONDS
from .db import unix_to_local
from .events import LEARNING_CATEGORIES

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
