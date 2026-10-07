"""Step 1 - activity catalog: course modules -> topic + sequence."""

from .config import t
from .db import write_csv


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
