"""Load and validate the English microlearning course content.

Source of truth (edit these, not Moodle):

    course/kc_catalog.csv        kc_id, unit, unit_name, dimension, kc_name,
                                 learning_objective
    course/questions/unitN.txt   one multiple-choice question per line:
                                 kc_id | stem | correct | wrong | wrong | wrong
    course/lessons/unitN.md      one short lesson per KC (format below)

The correct option is always written first; Moodle shuffles the options for
every attempt. Lines starting with ``#`` and blank lines are ignored.

Lesson format (one block per KC; text before the first ``## `` is ignored)::

    ## KC001
    video: https://...           optional; leave empty until a video exists
    ### Explanation
    One or more paragraphs.
    ### Examples
    - one example per bullet
    ### Common mistake
    - one mistake per bullet
    ### Try it
    - A self-check question => its answer

Inline ``**bold**`` and ``~~wrong form~~`` (struck through) are supported. Lessons must not reuse quiz questions, so
reading a lesson never shows a quiz answer.

``build_spec`` turns the content into the JSON the Moodle build script reads:
one question category per KC (ID number = kc_id) under a category per
dimension, one lesson page per KC (course-module ID number = kc_id), and one
practice quiz per unit and dimension.
"""

import csv
import html
import re
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent / "course"
COURSE_FULLNAME = "English Microlearning"
COURSE_SHORTNAME = "ENG-MICRO"
MIN_QUESTIONS_PER_KC = 4
OPTIONS_PER_QUESTION = 4
DIMENSION_ORDER = ["Grammar", "Vocabulary", "Reading", "Writing"]
CATALOG_COLUMNS = [
    "kc_id",
    "unit",
    "unit_name",
    "dimension",
    "kc_name",
    "learning_objective",
]
LESSON_SECTIONS = ["Explanation", "Examples", "Common mistake", "Try it"]
MIN_TRY_IT_ITEMS = 2
# A practice quiz counts as complete only with at least this share of the marks.
PASS_FRACTION = 0.7


def load_catalog(course_dir=COURSE_DIR):
    with (Path(course_dir) / "kc_catalog.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source:
        reader = csv.DictReader(source)
        missing = set(CATALOG_COLUMNS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"kc_catalog.csv is missing columns: {sorted(missing)}")
        return [{key: row[key].strip() for key in CATALOG_COLUMNS} for row in reader]


def load_questions(course_dir=COURSE_DIR):
    questions = []
    for path in sorted((Path(course_dir) / "questions").glob("unit*.txt")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 2 + OPTIONS_PER_QUESTION:
                raise ValueError(
                    f"{path.name}:{number}: expected kc_id | stem | correct | "
                    f"{OPTIONS_PER_QUESTION - 1} wrong options, got {len(parts)} fields"
                )
            questions.append(
                {
                    "kc_id": parts[0],
                    "stem": parts[1],
                    "options": parts[2:],  # options[0] is the correct one
                    "source": f"{path.name}:{number}",
                }
            )
    return questions


def load_lessons(course_dir=COURSE_DIR):
    """Parse ``lessons/unit*.md`` into ``[{kc_id, video, sections, source}]``.

    ``sections`` maps each ``### `` heading to its lines (blank lines kept as
    paragraph breaks). Structure is checked by ``validate``.
    """
    lessons, current, section = [], None, None
    for path in sorted((Path(course_dir) / "lessons").glob("unit*.md")):
        current = section = None
        for number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.rstrip()
            if line.startswith("## "):
                current = {
                    "kc_id": line[3:].strip(),
                    "video": "",
                    "sections": {},
                    "source": f"{path.name}:{number}",
                }
                lessons.append(current)
                section = None
            elif current is None:
                continue  # file preamble
            elif line.startswith("### "):
                section = line[4:].strip()
                if section in current["sections"]:
                    raise ValueError(
                        f"{path.name}:{number}: repeated section {section!r}"
                    )
                current["sections"][section] = []
            elif section is None and line.lower().startswith("video:"):
                current["video"] = line.split(":", 1)[1].strip()
            elif section is not None:
                current["sections"][section].append(line)
            elif line.strip():
                raise ValueError(f"{path.name}:{number}: text outside a ### section")
    return lessons


def _bullets(lines):
    return [line[2:].strip() for line in lines if line.startswith("- ")]


def _inline(text):
    """Escape HTML, then turn ``**bold**`` into <strong> and ``~~wrong~~`` into <del>."""
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"~~(.+?)~~", r"<del>\1</del>", text)


def _paragraphs(lines):
    blocks, block = [], []
    for line in [*lines, ""]:
        if line.strip():
            block.append(line.strip())
        elif block:
            blocks.append(" ".join(block))
            block = []
    return "".join(f"<p>{_inline(block)}</p>" for block in blocks)


def render_lesson(lesson, kc):
    """HTML for one lesson page. Self-check answers stay hidden until opened."""
    sections = lesson["sections"]
    parts = [f"<p><em>Learning goal:</em> {_inline(kc['learning_objective'])}</p>"]
    video = lesson["video"]
    if video:
        url = html.escape(video)
        if video.lower().split("?")[0].endswith((".mp4", ".webm")):
            parts.append(f'<video controls width="100%" src="{url}"></video>')
        parts.append(
            f'<p><a href="{url}">Watch the short video for this lesson</a></p>'
        )
    parts.append("<h3>Explanation</h3>" + _paragraphs(sections["Explanation"]))
    for heading in ("Examples", "Common mistake"):
        items = "".join(
            f"<li>{_inline(item)}</li>" for item in _bullets(sections[heading])
        )
        parts.append(f"<h3>{heading}</h3><ul>{items}</ul>")
    checks = []
    for item in _bullets(sections["Try it"]):
        question, answer = (part.strip() for part in item.split("=>", 1))
        checks.append(
            f"<li>{_inline(question)}<details><summary>Show answer</summary>"
            f"{_inline(answer)}</details></li>"
        )
    parts.append(f"<h3>Try it</h3><ol>{''.join(checks)}</ol>")
    parts.append(
        "<p>When you feel ready, practise this point in the unit's "
        f"{_inline(kc['dimension'])} practice quiz.</p>"
    )
    return "".join(parts)


def _lesson_problems(catalog, lessons):
    problems = []
    known = {kc["kc_id"] for kc in catalog}
    seen = {}
    for lesson in lessons:
        where, kc_id = lesson["source"], lesson["kc_id"]
        if kc_id not in known:
            problems.append(f"{where}: lesson for unknown kc_id {kc_id!r}")
            continue
        if kc_id in seen:
            problems.append(
                f"{where}: second lesson for {kc_id} (first at {seen[kc_id]})"
            )
            continue
        seen[kc_id] = where
        sections = lesson["sections"]
        missing = [name for name in LESSON_SECTIONS if name not in sections]
        extra = sorted(set(sections) - set(LESSON_SECTIONS))
        if missing:
            problems.append(f"{where}: {kc_id} lesson is missing sections {missing}")
            continue
        if extra:
            problems.append(f"{where}: {kc_id} lesson has unknown sections {extra}")
        if not any(line.strip() for line in sections["Explanation"]):
            problems.append(f"{where}: {kc_id} lesson has an empty explanation")
        for name in ("Examples", "Common mistake"):
            if not _bullets(sections[name]):
                problems.append(
                    f"{where}: {kc_id} lesson has no '- ' items under {name}"
                )
        checks = _bullets(sections["Try it"])
        if len(checks) < MIN_TRY_IT_ITEMS:
            problems.append(
                f"{where}: {kc_id} lesson needs {MIN_TRY_IT_ITEMS}+ Try it items"
            )
        if any("=>" not in item for item in checks):
            problems.append(
                f"{where}: {kc_id} Try it items must read 'question => answer'"
            )
        video = lesson["video"]
        if video and not video.startswith(("https://", "http://")):
            problems.append(f"{where}: {kc_id} video must be an http(s) URL")
    for kc_id in sorted(known - set(seen)):
        problems.append(f"{kc_id}: no lesson")
    return problems


def validate(catalog, questions, lessons=None):
    """Return a list of problems; an empty list means the content is usable.

    ``lessons`` is checked when given (``build_spec`` always gives it).
    """
    problems = []
    kc_ids = [kc["kc_id"] for kc in catalog]
    if len(set(kc_ids)) != len(kc_ids):
        problems.append("kc_catalog.csv contains duplicate kc_id values")
    for kc in catalog:
        if not all(kc.values()):
            problems.append(f"{kc['kc_id'] or '?'}: catalog row has an empty field")
        if kc["dimension"] not in DIMENSION_ORDER:
            problems.append(f"{kc['kc_id']}: unknown dimension {kc['dimension']!r}")
    names = {}
    for kc in catalog:
        names.setdefault((kc["unit"], kc["unit_name"]), None)
    units = {}
    for unit, unit_name in names:
        units.setdefault(unit, set()).add(unit_name)
    for unit, unit_names in units.items():
        if len(unit_names) > 1:
            problems.append(f"{unit} has several names: {sorted(unit_names)}")

    known = set(kc_ids)
    per_kc = dict.fromkeys(kc_ids, 0)
    stems = {}
    for question in questions:
        where = question["source"]
        if question["kc_id"] not in known:
            problems.append(f"{where}: unknown kc_id {question['kc_id']!r}")
            continue
        per_kc[question["kc_id"]] += 1
        if not question["stem"]:
            problems.append(f"{where}: empty stem")
        options = question["options"]
        if any(not option for option in options):
            problems.append(f"{where}: empty option")
        if len({option.lower() for option in options}) != len(options):
            problems.append(f"{where}: options are not distinct")
        key = (question["stem"].lower(), options[0].lower())
        if key in stems:
            problems.append(f"{where}: duplicates the question at {stems[key]}")
        stems[key] = where
    for kc_id, count in per_kc.items():
        if count < MIN_QUESTIONS_PER_KC:
            problems.append(
                f"{kc_id}: {count} question(s), need at least {MIN_QUESTIONS_PER_KC}"
            )
    if lessons is not None:
        problems.extend(_lesson_problems(catalog, lessons))
        stems = {question["stem"].lower() for question in questions}
        for lesson in lessons:
            for item in _bullets(lesson["sections"].get("Try it", [])):
                if item.split("=>", 1)[0].strip().lower() in stems:
                    problems.append(
                        f"{lesson['source']}: Try it item repeats a quiz question"
                    )
    return problems


def _unit_number(unit):
    return int("".join(ch for ch in unit if ch.isdigit()))


def build_spec(course_dir=COURSE_DIR):
    """Validated course specification for ``php/build_course.php``."""
    catalog, questions = load_catalog(course_dir), load_questions(course_dir)
    lessons = load_lessons(course_dir)
    problems = validate(catalog, questions, lessons)
    if problems:
        raise ValueError("Course content is invalid:\n  " + "\n  ".join(problems))
    units = sorted({(_unit_number(kc["unit"]), kc["unit_name"]) for kc in catalog})
    by_id = {kc["kc_id"]: kc for kc in catalog}
    counters = {}
    spec_questions = []
    for question in questions:
        counters[question["kc_id"]] = counters.get(question["kc_id"], 0) + 1
        spec_questions.append(
            {
                "kc_id": question["kc_id"],
                "name": f"{question['kc_id']} Q{counters[question['kc_id']]:02d}",
                "stem": question["stem"],
                "options": question["options"],
            }
        )
    return {
        "fullname": COURSE_FULLNAME,
        "shortname": COURSE_SHORTNAME,
        "dimensions": DIMENSION_ORDER,
        "units": [{"number": number, "name": name} for number, name in units],
        "kcs": [
            {
                "kc_id": kc["kc_id"],
                "unit": _unit_number(kc["unit"]),
                "dimension": kc["dimension"],
                "name": kc["kc_name"],
                "objective": kc["learning_objective"],
            }
            for kc in catalog
        ],
        "questions": spec_questions,
        "lessons": [
            {
                "kc_id": lesson["kc_id"],
                "video": lesson["video"],
                "html": render_lesson(lesson, by_id[lesson["kc_id"]]),
            }
            for lesson in lessons
        ],
        "pass_fraction": PASS_FRACTION,
    }
