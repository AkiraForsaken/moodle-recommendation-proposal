"""Course content validation and the Moodle attempt plan (no Moodle needed)."""

import csv
import json
import shutil

import pytest

from knowledge_engine import config as engine_config
from knowledge_engine.phase import _simulator_ceiling_auc
from knowledge_engine.simulator import simulate_plan
from moodle_setup import __main__ as cli
from moodle_setup import content


# --- course content ------------------------------------------------------------


def test_shipped_course_content_is_valid():
    spec = content.build_spec()

    assert len(spec["units"]) == 8
    assert len(spec["kcs"]) == 96
    per_kc = {}
    for question in spec["questions"]:
        per_kc[question["kc_id"]] = per_kc.get(question["kc_id"], 0) + 1
    assert set(per_kc) == {kc["kc_id"] for kc in spec["kcs"]}
    assert min(per_kc.values()) >= content.MIN_QUESTIONS_PER_KC
    # Question names are unique within the bank.
    names = [question["name"] for question in spec["questions"]]
    assert len(names) == len(set(names))
    # Every unit has KCs in all four dimensions, so it gets four quizzes.
    combos = {(kc["unit"], kc["dimension"]) for kc in spec["kcs"]}
    assert len(combos) == 8 * len(content.DIMENSION_ORDER)


@pytest.fixture
def course_copy(tmp_path):
    target = tmp_path / "course"
    shutil.copytree(content.COURSE_DIR, target)
    return target


def _append(course_dir, line, unit="unit1.txt"):
    with (course_dir / "questions" / unit).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _problems(course_dir):
    return content.validate(content.load_catalog(course_dir), content.load_questions(course_dir))


def test_validation_rejects_unknown_kc(course_copy):
    _append(course_copy, "KC999 | Stem? | a | b | c | d")
    assert any("unknown kc_id 'KC999'" in problem for problem in _problems(course_copy))


def test_validation_rejects_repeated_options(course_copy):
    _append(course_copy, "KC001 | A brand new stem? | goes | go | Goes | gone")
    assert any("options are not distinct" in problem for problem in _problems(course_copy))


def test_validation_rejects_duplicate_questions(course_copy):
    line = 'KC001 | Choose the correct form: "My sister ___ to work by bus every day." | goes | go | going | gone'
    _append(course_copy, line)
    assert any("duplicates the question" in problem for problem in _problems(course_copy))


def test_validation_requires_enough_questions_per_kc(course_copy):
    path = course_copy / "questions" / "unit1.txt"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("KC001 ")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert "KC001: 0 question(s), need at least 4" in _problems(course_copy)


def test_validation_rejects_unknown_dimension(course_copy):
    path = course_copy / "kc_catalog.csv"
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    rows[0]["dimension"] = "Listening"
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=content.CATALOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert any("unknown dimension 'Listening'" in problem for problem in _problems(course_copy))


def test_malformed_line_is_an_error(course_copy):
    _append(course_copy, "KC001 | Only two options? | yes | no")
    with pytest.raises(ValueError, match="expected kc_id"):
        content.build_spec(course_copy)


# --- attempt plan --------------------------------------------------------------


def _quizzes():
    """The shipped course as export_structure.php would report it (fake entry ids)."""
    spec = content.build_spec()
    unit_of = {kc["kc_id"]: (kc["unit"], content.DIMENSION_ORDER.index(kc["dimension"])) for kc in spec["kcs"]}
    grouped = {}
    for entry_id, question in enumerate(spec["questions"], start=1):
        grouped.setdefault(unit_of[question["kc_id"]], []).append(
            {"entry_id": entry_id, "kc_id": question["kc_id"]}
        )
    return [
        {"quiz_id": 100 + index, "entries": grouped[key]}
        for index, key in enumerate(sorted(grouped))
    ]


def test_plan_answers_every_question_of_each_attempt_in_time_order():
    quizzes = _quizzes()
    entries_of = {quiz["quiz_id"]: [e["entry_id"] for e in quiz["entries"]] for quiz in quizzes}
    plan, manifest = simulate_plan(quizzes, n_learners=30, seed=3, start_unix=1_700_000_000)

    assert len(plan) == 30
    assert manifest["answer_events"] == sum(len(a["answers"]) for l in plan for a in l["attempts"])
    assert 0.5 < manifest["ceiling_auc"] <= 1.0
    for learner in plan:
        assert learner["enrol_start"] >= 1_700_000_000
        previous = learner["enrol_start"]
        for attempt in learner["attempts"]:
            assert [a["entry_id"] for a in attempt["answers"]] == entries_of[attempt["quiz_id"]]
            times = [attempt["start"], *(a["time"] for a in attempt["answers"]), attempt["finish"]]
            assert times == sorted(times) and attempt["start"] > previous
            assert all(a["correct"] in (0, 1) and a["seconds"] >= 3 for a in attempt["answers"])
            previous = attempt["finish"]


def test_larger_cohort_extends_a_smaller_one():
    quizzes = _quizzes()
    small, _ = simulate_plan(quizzes, n_learners=20, seed=7, start_unix=1_700_000_000)
    large, _ = simulate_plan(quizzes, n_learners=60, seed=7, start_unix=1_700_000_000)
    assert large[:20] == small


def test_plan_needs_mapped_questions():
    with pytest.raises(ValueError, match="no quiz"):
        simulate_plan([{"quiz_id": 1, "entries": []}])


def test_plan_manifest_reaches_the_training_gate(tmp_path, capsys):
    _, manifest = simulate_plan(_quizzes(), n_learners=20, seed=7, start_unix=1_700_000_000)

    path = cli.save_plan_manifest(manifest, simulated_users=20, data_dir=tmp_path)
    assert path == tmp_path / engine_config.PLAN_MANIFEST_NAME
    # The collector's own marker carries no ceiling; the plan manifest supplies it.
    (tmp_path / engine_config.SIMULATION_MANIFEST_NAME).write_text(json.dumps({"simulated": True}))
    assert _simulator_ceiling_auc(tmp_path) == manifest["ceiling_auc"]

    # A smaller plan than the learners in Moodle must not replace it.
    _, smaller = simulate_plan(_quizzes(), n_learners=5, seed=1, start_unix=1_700_000_000)
    assert cli.save_plan_manifest(smaller, simulated_users=20, data_dir=tmp_path) is None
    assert _simulator_ceiling_auc(tmp_path) == manifest["ceiling_auc"]


def test_dry_run_does_not_touch_moodle(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run_php(script, *args):
        calls.append(script)
        return {"course_id": 9, "startdate": 1_700_000_000, "quizzes": _quizzes()}

    monkeypatch.setattr(cli.docker, "run_php", fake_run_php)
    plan_file = tmp_path / "plan.json"
    cli.main(["simulate-learners", "--learners", "5", "--dry-run", "--plan-out", str(plan_file),
              "--data-dir", str(tmp_path)])

    assert calls == ["export_structure.php"]
    saved = json.loads(plan_file.read_text())
    assert len(saved["learners"]) == 5 and saved["manifest"]["simulated"] is True
    assert not (tmp_path / engine_config.PLAN_MANIFEST_NAME).exists()


# --- lessons -------------------------------------------------------------------


def _lesson_block(kc_id, try_it="- Q one => A one\n- Q two => A two"):
    return (
        f"## {kc_id}\nvideo:\n### Explanation\nSome text.\n### Examples\n- An example\n"
        f"### Common mistake\n- A mistake\n### Try it\n{try_it}\n"
    )


def _lesson_problems(course_dir):
    return content.validate(
        content.load_catalog(course_dir),
        content.load_questions(course_dir),
        content.load_lessons(course_dir),
    )


def test_every_kc_has_one_rendered_lesson():
    spec = content.build_spec()
    assert sorted(lesson["kc_id"] for lesson in spec["lessons"]) == sorted(
        kc["kc_id"] for kc in spec["kcs"]
    )
    html = spec["lessons"][0]["html"]
    for heading in content.LESSON_SECTIONS:
        assert f"<h3>{heading}</h3>" in html
    assert "<details><summary>Show answer</summary>" in html
    assert spec["pass_fraction"] == content.PASS_FRACTION


def test_lesson_validation_catches_missing_duplicate_and_incomplete_lessons(course_copy):
    lessons = course_copy / "lessons"
    unit1 = lessons / "unit1.md"
    text = unit1.read_text(encoding="utf-8")
    # Drop KC002's lesson, repeat KC001's, and add one for an unknown KC.
    start, end = text.index("## KC002"), text.index("## KC003")
    unit1.write_text(text[:start] + text[end:], encoding="utf-8")
    (lessons / "unit9.md").write_text(
        _lesson_block("KC001") + _lesson_block("KC999") + "## KC003b\nvideo:\n### Explanation\nx\n",
        encoding="utf-8",
    )
    problems = _lesson_problems(course_copy)
    assert "KC002: no lesson" in problems
    assert any("second lesson for KC001" in p for p in problems)
    assert any("unknown kc_id 'KC999'" in p for p in problems)
    assert any("unknown kc_id 'KC003b'" in p for p in problems)


def test_lesson_needs_sections_answers_and_no_quiz_reuse(course_copy):
    lessons = course_copy / "lessons"
    for path in lessons.glob("unit*.md"):
        text = path.read_text(encoding="utf-8")
        if "## KC001" in text:
            start = text.index("## KC001")
            end = text.index("\n## ", start) + 1
            stem = 'Choose the correct form: "My sister ___ to work by bus every day."'
            text = text[:start] + _lesson_block("KC001", f"- {stem} => goes\n- No answer here") + text[end:]
            path.write_text(text, encoding="utf-8")
    problems = _lesson_problems(course_copy)
    assert any("KC001 Try it items must read 'question => answer'" in p for p in problems)
    assert any("Try it item repeats a quiz question" in p for p in problems)


def test_lesson_html_is_escaped_and_marks_wrong_forms():
    lesson = {
        "kc_id": "KC001",
        "video": "https://example.org/clip.mp4",
        "sections": {
            "Explanation": ["Use <b> **carefully**.", "", "Second paragraph."],
            "Examples": ["- ~~He go~~ → He **goes**"],
            "Common mistake": ["- A & B"],
            "Try it": ["- 1 < 2? => yes", "- Q => A"],
        },
    }
    html = content.render_lesson(lesson, {"learning_objective": "Goal", "dimension": "Grammar"})
    assert "<b>" not in html and "&lt;b&gt;" in html
    assert "<strong>carefully</strong>" in html and "<del>He go</del>" in html
    assert html.count("<p>") >= 3  # goal + two explanation paragraphs (+ footer)
    assert '<video controls width="100%" src="https://example.org/clip.mp4">' in html
    assert "A &amp; B" in html


# --- page behaviour in the plan ------------------------------------------------


def _course_structure():
    """The shipped course as export_structure.php reports it after a rebuild."""
    spec = content.build_spec()
    quizzes, lessons, overviews, cmid = [], {}, {}, 1000
    by_unit_dim = {}
    for entry_id, question in enumerate(spec["questions"], start=1):
        kc = next(k for k in spec["kcs"] if k["kc_id"] == question["kc_id"])
        by_unit_dim.setdefault((kc["unit"], content.DIMENSION_ORDER.index(kc["dimension"])), []).append(
            {"entry_id": entry_id, "kc_id": question["kc_id"]}
        )
    for unit in spec["units"]:
        cmid += 1
        overviews[str(unit["number"])] = cmid
        for dim in range(len(content.DIMENSION_ORDER)):
            for kc in spec["kcs"]:
                if kc["unit"] == unit["number"] and kc["dimension"] == content.DIMENSION_ORDER[dim]:
                    cmid += 1
                    lessons[kc["kc_id"]] = cmid
            cmid += 1
            quizzes.append(
                {
                    "quiz_id": 200 + len(quizzes),
                    "cmid": cmid,
                    "section": unit["number"],
                    "grade": 10.0,
                    "gradepass": 7.0,
                    "entries": by_unit_dim[(unit["number"], dim)],
                }
            )
    return quizzes, lessons, overviews


def _page_plan(n_learners=80, seed=5, **kwargs):
    quizzes, lessons, overviews = _course_structure()
    plan, manifest = simulate_plan(
        quizzes, n_learners=n_learners, seed=seed, start_unix=1_700_000_000,
        lessons=lessons, overviews=overviews, **kwargs,
    )
    return plan, manifest, quizzes, lessons, overviews


def test_page_views_happen_before_attempts_and_use_course_pages():
    plan, manifest, quizzes, lessons, overviews = _page_plan()
    lesson_cm = {cmid: kc for kc, cmid in lessons.items()}
    quiz_kcs = {q["quiz_id"]: {e["kc_id"] for e in q["entries"]} for q in quizzes}
    section_of = {q["quiz_id"]: str(q["section"]) for q in quizzes}
    pages = manifest["page_behaviour"]
    assert pages["lesson_views"] > 0 and pages["overview_views"] > 0
    assert pages["revisions"] > 0 and pages["reviews"] > 0
    previous_end = 0
    for learner in plan:
        previous_end = learner["enrol_start"]
        for attempt in learner["attempts"]:
            times = [view["time"] for view in attempt["views_before"]]
            assert times == sorted(times)
            assert all(previous_end < t < attempt["start"] for t in times)
            for view in attempt["views_before"]:
                if view["kind"] == "lesson":
                    assert lesson_cm[view["cmid"]] == view["kc_id"]
                    assert view["kc_id"] in quiz_kcs[attempt["quiz_id"]]
                else:
                    assert view["cmid"] == overviews[section_of[attempt["quiz_id"]]]
            end = attempt["review_at"] or attempt["finish"]
            assert end >= attempt["finish"]
            previous_end = end


def test_revision_retries_the_failed_quiz_after_reading_its_weak_kcs():
    plan, _, quizzes, _, _ = _page_plan()
    revisions = 0
    for learner in plan:
        for before, attempt in zip(learner["attempts"], learner["attempts"][1:]):
            if not attempt["revision"]:
                continue
            revisions += 1
            assert attempt["quiz_id"] == before["quiz_id"]
            assert sum(a["correct"] for a in before["answers"]) / len(before["answers"]) < 0.7
            per_kc = {}
            for answer in before["answers"]:
                per_kc.setdefault(answer["kc_id"], []).append(answer["correct"])
            weak = {kc for kc, values in per_kc.items() if sum(values) / len(values) < 0.7}
            assert {v["kc_id"] for v in attempt["views_before"] if v["kind"] == "lesson"} == weak
    assert revisions > 0


def test_plans_without_lessons_have_no_page_behaviour():
    quizzes, _, _ = _course_structure()
    plan, manifest = simulate_plan(quizzes, n_learners=30, seed=5, start_unix=1_700_000_000)
    assert manifest["page_behaviour"] is None
    assert all(not a["views_before"] and a["review_at"] is None for l in plan for a in l["attempts"])


def test_page_plans_still_extend_a_smaller_cohort():
    small, *_ = _page_plan(n_learners=15)
    large, *_ = _page_plan(n_learners=40)
    assert large[:15] == small


def test_lesson_gain_raises_accuracy_in_the_generator():
    # A generator property, not evidence about real learners.
    _, without, *_ = _page_plan(n_learners=150, lesson_gain=0.0)
    _, with_gain, *_ = _page_plan(n_learners=150, lesson_gain=2.0)
    assert with_gain["overall_accuracy"] > without["overall_accuracy"]
