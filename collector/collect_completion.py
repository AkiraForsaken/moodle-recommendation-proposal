"""Activity completion states (may legitimately be empty - see docstring below)."""

from .config import t
from .db import existing_columns, table_exists, unix_to_local, write_csv

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
