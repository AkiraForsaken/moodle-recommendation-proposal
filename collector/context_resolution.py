"""
Course-module resolution: map a log event to the course module it belongs to.

The chain (module context -> context ancestor -> object table -> unresolved)
is recorded per event in `cmid_source` so the mapping can be audited.
"""

from .config import CONTEXT_MODULE, t
from .db import table_exists

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
