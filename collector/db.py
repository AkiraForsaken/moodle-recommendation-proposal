"""
Database-adjacent helpers: cursor queries, participant/role filtering, CSV output.

Functions take an open DB-API cursor as an argument; this module never opens a
connection itself, so it can be exercised with a fake cursor in tests.
"""

import csv
from collections import defaultdict
from datetime import datetime, timezone

from .config import COURSE_ID_ENV, INCLUDE_ROLES, LOCAL_TZ, OUTPUT_DIR, t

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


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
