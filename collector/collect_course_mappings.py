"""Derive the course's KC catalog and mappings from the Moodle question bank.

Authoring convention (the course designer's side of the contract):

  * A question category with a non-empty ``idnumber`` is one Knowledge
    Component: ``idnumber`` -> kc_id, ``name`` -> kc_name, ``info`` -> learning
    objective.
  * Its parent category's name is the KC dimension (grammar, vocabulary, ...).
    A KC category directly under the question bank's top category gets the
    dimension ``general``.
  * A question belongs to exactly one category, hence to at most one KC.

Questions in categories without an ``idnumber`` are left unmapped; the answer
exporter counts them in ``question_answer_mapping_report.csv``.

Outputs (written to the collector output directory):

    kc_catalog.csv              kc_id, kc_name, kc_dimension, learning_objective
    moodle_question_kc_map.csv  question_bank_entry_id -> kc_id (versioned)
    kc_topic_map.csv            kc_id -> course topic, from the quizzes that
                                use (or have used) the KC's questions
    kc_activity_map.csv         kc_id -> activities: role "learn" for a
                                resource whose course-module ID number is the
                                kc_id, role "practice" for a quiz that uses the
                                KC's questions (question_count of them)

A course module (page, URL, book, ...) whose ID number equals a catalog kc_id
is that KC's learning resource. This is how a recommender finds what to show
for a weak KC without any hand-maintained mapping.

Nothing here reads any benchmark dataset: KC identity belongs to the course.
"""

import hashlib
import re
from pathlib import Path

from .config import (
    QUESTION_KC_MAPPING_FILE,
    QUESTION_KC_MAPPING_IS_OVERRIDE,
    t,
)
from .db import write_csv

KC_CATALOG_COLUMNS = ["kc_id", "kc_name", "kc_dimension", "learning_objective"]
QUESTION_KC_MAP_COLUMNS = [
    "question_bank_entry_id",
    "kc_id",
    "kc_dimension",
    "mapping_version",
    "mapping_source",
]
KC_TOPIC_MAP_COLUMNS = ["kc_id", "topic", "mapping_version"]
KC_ACTIVITY_MAP_COLUMNS = [
    "kc_id",
    "course_module_id",
    "activity_type",
    "activity_name",
    "topic",
    "role",
    "question_count",
    "mapping_source",
    "mapping_version",
]
ROLE_LEARN = "learn"
ROLE_PRACTICE = "practice"
LEARN_MAPPING_SOURCE = "course_module_idnumber"
PRACTICE_MAPPING_SOURCE = "quiz_question_usage"
MAPPING_SOURCE = "moodle_question_category"
DEFAULT_DIMENSION = "general"
CONTEXT_COURSE = 50
CONTEXT_MODULE = 70

_ENTRY_COLUMNS = """
            qbe.id,
            qc.idnumber,
            qc.name,
            qc.info,
            parent.name,
            parent.parent
"""


def _entry_joins():
    return f"""
        FROM {t('question_bank_entries')} qbe
        JOIN {t('question_categories')} qc ON qc.id = qbe.questioncategoryid
        LEFT JOIN {t('question_categories')} parent ON parent.id = qc.parent
    """


def _plain_text(value):
    """Category ``info`` is stored as HTML; keep a single readable line."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(value))).strip()


def _read_course_bank_entries(cursor, course_id):
    """Question-bank entries whose category lives in this course.

    Moodle 5 keeps question banks in module contexts (level 70); earlier
    versions used the course context (level 50). Both are accepted.
    """
    cursor.execute(
        f"""
        SELECT {_ENTRY_COLUMNS}
        {_entry_joins()}
        JOIN {t('context')} ctx ON ctx.id = qc.contextid
        LEFT JOIN {t('course_modules')} cm
          ON ctx.contextlevel = {CONTEXT_MODULE} AND cm.id = ctx.instanceid
        WHERE (ctx.contextlevel = {CONTEXT_MODULE} AND cm.course = ?)
           OR (ctx.contextlevel = {CONTEXT_COURSE} AND ctx.instanceid = ?)
        ORDER BY qbe.id
        """,
        (course_id, course_id),
    )
    return cursor.fetchall()


def _read_entries_by_id(cursor, entry_ids):
    """Entries used by course quizzes but stored in a bank outside the course."""
    if not entry_ids:
        return []
    placeholders = ", ".join("?" for _ in entry_ids)
    cursor.execute(
        f"""
        SELECT {_ENTRY_COLUMNS}
        {_entry_joins()}
        WHERE qbe.id IN ({placeholders})
        ORDER BY qbe.id
        """,
        tuple(entry_ids),
    )
    return cursor.fetchall()


def _read_quiz_question_usage(cursor, course_id):
    """Distinct (quiz_id, question_bank_entry_id) pairs for the course.

    Fixed quiz slots give the designed usage; finished attempts add questions
    that were drawn through random slots.
    """
    cursor.execute(
        f"""
        SELECT qs.quizid, qr.questionbankentryid
        FROM {t('quiz_slots')} qs
        JOIN {t('quiz')} q ON q.id = qs.quizid
        JOIN {t('question_references')} qr
          ON qr.component = 'mod_quiz'
         AND qr.questionarea = 'slot'
         AND qr.itemid = qs.id
        WHERE q.course = ?
        UNION
        SELECT qa.quiz, qv.questionbankentryid
        FROM {t('quiz_attempts')} qa
        JOIN {t('quiz')} q ON q.id = qa.quiz
        JOIN {t('question_attempts')} qat ON qat.questionusageid = qa.uniqueid
        JOIN {t('question_versions')} qv ON qv.questionid = qat.questionid
        WHERE q.course = ? AND qa.preview = 0
        """,
        (course_id, course_id),
    )
    return [(int(quiz_id), int(entry_id)) for quiz_id, entry_id in cursor.fetchall()]


def _mapping_version(pairs):
    """Short content hash: changes exactly when the question->KC mapping does."""
    digest = hashlib.sha256()
    for entry_id, kc_id, dimension in sorted(pairs):
        digest.update(f"{entry_id},{kc_id},{dimension}\n".encode("utf-8"))
    return f"qcat-{digest.hexdigest()[:12]}"


def build_course_mappings(entry_rows):
    """Turn question-bank rows into ``(kc_catalog, question_kc_map)`` row lists.

    Raises ValueError when one kc_id is declared by categories that disagree on
    its name or dimension: KC identity must be unambiguous.
    """
    catalog = {}
    question_kcs = {}
    for entry_id, idnumber, name, info, parent_name, parent_parent in entry_rows:
        kc_id = (idnumber or "").strip()
        if not kc_id:
            continue
        # The bank's root category ("top") has parent 0 and is not a dimension.
        is_root_parent = parent_name is None or not parent_parent
        dimension = (
            DEFAULT_DIMENSION if is_root_parent else str(parent_name).strip().lower()
        ) or DEFAULT_DIMENSION
        entry = {
            "kc_id": kc_id,
            "kc_name": (name or "").strip() or kc_id,
            "kc_dimension": dimension,
            "learning_objective": _plain_text(info),
        }
        known = catalog.setdefault(kc_id, entry)
        if (known["kc_name"], known["kc_dimension"]) != (
            entry["kc_name"],
            entry["kc_dimension"],
        ):
            raise ValueError(
                f"KC {kc_id!r} is declared by question categories that disagree: "
                f"{known['kc_name']!r}/{known['kc_dimension']!r} vs "
                f"{entry['kc_name']!r}/{entry['kc_dimension']!r}"
            )
        if not known["learning_objective"]:
            known["learning_objective"] = entry["learning_objective"]
        question_kcs[str(entry_id)] = (kc_id, dimension)

    version = _mapping_version(
        (entry_id, kc_id, dimension)
        for entry_id, (kc_id, dimension) in question_kcs.items()
    )
    question_map = [
        {
            "question_bank_entry_id": entry_id,
            "kc_id": kc_id,
            "kc_dimension": dimension,
            "mapping_version": version,
            "mapping_source": MAPPING_SOURCE,
        }
        for entry_id, (kc_id, dimension) in sorted(
            question_kcs.items(), key=lambda item: (len(item[0]), item[0])
        )
    ]
    kc_catalog = [catalog[kc_id] for kc_id in sorted(catalog)]
    return kc_catalog, question_map


def build_kc_topic_map(question_map, usage, catalog_rows):
    """KC -> topic rows from the quizzes that use each KC's questions."""
    quiz_topic = {
        int(row["activity_id"]): row["topic"]
        for row in catalog_rows
        if row.get("activity_type") == "quiz" and row.get("topic")
    }
    entry_kc = {row["question_bank_entry_id"]: row["kc_id"] for row in question_map}
    version = question_map[0]["mapping_version"] if question_map else ""
    pairs = set()
    for quiz_id, entry_id in usage:
        kc_id = entry_kc.get(str(entry_id))
        topic = quiz_topic.get(quiz_id)
        if kc_id and topic:
            pairs.add((kc_id, topic))
    return [
        {"kc_id": kc_id, "topic": topic, "mapping_version": version}
        for kc_id, topic in sorted(pairs)
    ]


def build_kc_activity_map(kc_catalog, question_map, usage, catalog_rows):
    """KC -> activity rows: learning resources and the quizzes that practise it.

    Returns ``(rows, unknown_idnumbers)``. ``unknown_idnumbers`` lists
    non-quiz course-module ID numbers that match no catalog KC, so a typo in a
    resource's ID number is reported instead of silently ignored.
    """
    known = {row["kc_id"] for row in kc_catalog}
    version = question_map[0]["mapping_version"] if question_map else ""
    rows, unknown = [], []
    for activity in catalog_rows:
        idnumber = (activity.get("idnumber") or "").strip()
        if not idnumber or activity.get("activity_type") == "quiz":
            continue
        if idnumber not in known:
            unknown.append(idnumber)
            continue
        rows.append(
            {
                "kc_id": idnumber,
                "course_module_id": activity["course_module_id"],
                "activity_type": activity["activity_type"],
                "activity_name": activity.get("activity_name", ""),
                "topic": activity.get("topic") or "",
                "role": ROLE_LEARN,
                "question_count": "",
                "mapping_source": LEARN_MAPPING_SOURCE,
                "mapping_version": version,
            }
        )

    quiz_cm = {
        int(row["activity_id"]): row
        for row in catalog_rows
        if row.get("activity_type") == "quiz"
    }
    entry_kc = {row["question_bank_entry_id"]: row["kc_id"] for row in question_map}
    questions = {}
    for quiz_id, entry_id in sorted(set(usage)):
        kc_id = entry_kc.get(str(entry_id))
        if kc_id and quiz_id in quiz_cm:
            questions[(kc_id, quiz_id)] = questions.get((kc_id, quiz_id), 0) + 1
    for (kc_id, quiz_id), count in questions.items():
        activity = quiz_cm[quiz_id]
        rows.append(
            {
                "kc_id": kc_id,
                "course_module_id": activity.get("course_module_id", ""),
                "activity_type": "quiz",
                "activity_name": activity.get("activity_name", ""),
                "topic": activity.get("topic") or "",
                "role": ROLE_PRACTICE,
                "question_count": count,
                "mapping_source": PRACTICE_MAPPING_SOURCE,
                "mapping_version": version,
            }
        )
    rows.sort(key=lambda row: (row["kc_id"], row["role"], str(row["course_module_id"])))
    return rows, sorted(set(unknown))


def _catalog_from_override(mapping_file):
    """A hand-curated crosswalk names KCs only by id; reuse it as the catalog."""
    # Imported lazily: collect_question_answers imports this module's constants.
    from .collect_question_answers import load_question_kc_mapping

    mapping = load_question_kc_mapping(mapping_file)
    if mapping is None:
        return [], []
    question_map = [
        {"question_bank_entry_id": entry_id, **link}
        for entry_id, link in mapping.items()
    ]
    catalog = {}
    for link in mapping.values():
        catalog.setdefault(
            link["kc_id"],
            {
                "kc_id": link["kc_id"],
                "kc_name": link["kc_id"],
                "kc_dimension": link["kc_dimension"],
                "learning_objective": "",
            },
        )
    return [catalog[kc_id] for kc_id in sorted(catalog)], question_map


def collect_course_mappings(
    cursor,
    course_id,
    catalog_rows,
    mapping_file=QUESTION_KC_MAPPING_FILE,
    use_override=QUESTION_KC_MAPPING_IS_OVERRIDE,
):
    """Write kc_catalog / question->KC / KC->topic CSVs and return their rows.

    With ``use_override`` (``MOODLE_QUESTION_KC_MAPPING`` set), the curated
    crosswalk file is the source of truth and is never rewritten.
    """
    usage = _read_quiz_question_usage(cursor, course_id)
    if use_override:
        kc_catalog, question_map = _catalog_from_override(mapping_file)
        print(f"Question-to-KC mapping: curated override {mapping_file}")
    else:
        entry_rows = list(_read_course_bank_entries(cursor, course_id))
        known = {int(row[0]) for row in entry_rows}
        outside = sorted({entry_id for _, entry_id in usage} - known)
        entry_rows.extend(_read_entries_by_id(cursor, outside))
        kc_catalog, question_map = build_course_mappings(entry_rows)
        unmapped = len({int(row[0]) for row in entry_rows}) - len(question_map)
        write_csv(Path(mapping_file).name, question_map, QUESTION_KC_MAP_COLUMNS)
        if unmapped:
            print(
                f"WARNING: {unmapped:,} question-bank entries sit in categories "
                "without an idnumber and are not mapped to a KC."
            )
    kc_topic_map = build_kc_topic_map(question_map, usage, catalog_rows)
    kc_activity_map, unknown_idnumbers = build_kc_activity_map(
        kc_catalog, question_map, usage, catalog_rows
    )
    write_csv("kc_catalog.csv", kc_catalog, KC_CATALOG_COLUMNS)
    write_csv("kc_topic_map.csv", kc_topic_map, KC_TOPIC_MAP_COLUMNS)
    write_csv("kc_activity_map.csv", kc_activity_map, KC_ACTIVITY_MAP_COLUMNS)
    if unknown_idnumbers:
        print(
            "WARNING: activities have ID numbers that match no KC: "
            f"{unknown_idnumbers[:10]}"
        )
    print(
        f"Course mappings: {len(kc_catalog):,} KCs, {len(question_map):,} mapped "
        f"questions, {len(kc_topic_map):,} KC-topic links, "
        f"{sum(row['role'] == ROLE_LEARN for row in kc_activity_map):,} KC learning resources."
    )
    return kc_catalog, question_map, kc_topic_map
