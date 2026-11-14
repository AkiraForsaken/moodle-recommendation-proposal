import importlib

import pytest

# The package re-exports a function of the same name, so load the module itself.
mappings = importlib.import_module("collector.collect_course_mappings")


class RoutingCursor:
    """Fake cursor that answers each query by a distinctive table name."""

    def __init__(self, entries=(), usage=(), outside=()):
        self.results = {"entries": list(entries), "usage": list(usage), "outside": list(outside)}
        self.current = []

    def execute(self, query, params=()):
        if "quiz_slots" in query:
            self.current = self.results["usage"]
        elif "qbe.id IN" in query:
            self.current = self.results["outside"]
        else:
            self.current = self.results["entries"]

    def fetchall(self):
        return self.current


# (entry id, category idnumber, category name, info, parent name, parent.parent)
ENTRIES = [
    (10, "KC-G1", "Present simple -s", "<p>Apply <b>-s/-es</b> rules.</p>", "Grammar", 1),
    (11, "KC-G1", "Present simple -s", "", "Grammar", 1),
    (20, "KC-V1", "Daily routine words", None, "Vocabulary", 1),
    (30, "", "Scratch questions", None, "Grammar", 1),
    (40, "KC-X1", "Uncategorised skill", None, "top", 0),
]
CATALOG = [
    {"activity_type": "quiz", "activity_id": 5, "topic": "Unit 1"},
    {"activity_type": "quiz", "activity_id": 6, "topic": "Unit 2"},
    {"activity_type": "page", "activity_id": 5, "topic": "Unit 9"},
    {"activity_type": "quiz", "activity_id": 7, "topic": None},
]


def test_question_categories_become_kcs_dimensions_and_objectives():
    catalog, question_map = mappings.build_course_mappings(ENTRIES)

    assert [kc["kc_id"] for kc in catalog] == ["KC-G1", "KC-V1", "KC-X1"]
    grammar = catalog[0]
    assert grammar["kc_name"] == "Present simple -s"
    assert grammar["kc_dimension"] == "grammar"
    assert grammar["learning_objective"] == "Apply -s/-es rules."
    # A KC category directly under the bank root has no dimension category.
    assert catalog[2]["kc_dimension"] == "general"
    # The category without an idnumber is not a KC; its question stays unmapped.
    assert [row["question_bank_entry_id"] for row in question_map] == ["10", "11", "20", "40"]
    assert {row["mapping_source"] for row in question_map} == {"moodle_question_category"}
    assert len({row["mapping_version"] for row in question_map}) == 1


def test_mapping_version_changes_only_when_the_mapping_changes():
    _, first = mappings.build_course_mappings(ENTRIES)
    _, same = mappings.build_course_mappings(list(reversed(ENTRIES)))
    moved = [(11, "KC-V1", "Daily routine words", None, "Vocabulary", 1) if row[0] == 11 else row for row in ENTRIES]
    _, changed = mappings.build_course_mappings(moved)

    assert first[0]["mapping_version"] == same[0]["mapping_version"]
    assert first[0]["mapping_version"] != changed[0]["mapping_version"]


def test_conflicting_kc_declarations_are_rejected():
    conflict = ENTRIES + [(50, "KC-G1", "Present simple -s", None, "Reading", 1)]
    with pytest.raises(ValueError, match="KC-G1"):
        mappings.build_course_mappings(conflict)


def test_collect_writes_catalog_question_map_and_topic_links(monkeypatch):
    written = {}
    monkeypatch.setattr(
        mappings, "write_csv", lambda name, rows, columns: written.update({name: (list(rows), list(columns))})
    )
    # Entry 60 is used by a course quiz but lives in a bank outside the course.
    cursor = RoutingCursor(
        entries=ENTRIES,
        usage=[(5, 10), (6, 10), (5, 20), (7, 20), (5, 30), (6, 60)],
        outside=[(60, "KC-R1", "Read a timetable", None, "Reading", 1)],
    )

    catalog, question_map, topic_map = mappings.collect_course_mappings(
        cursor, 3, CATALOG, mapping_file="moodle_question_kc_map.csv", use_override=False
    )

    assert set(written) == {
        "moodle_question_kc_map.csv",
        "kc_catalog.csv",
        "kc_topic_map.csv",
        "kc_activity_map.csv",
    }
    assert written["kc_catalog.csv"][1] == mappings.KC_CATALOG_COLUMNS
    assert "KC-R1" in {kc["kc_id"] for kc in catalog}
    assert "60" in {row["question_bank_entry_id"] for row in question_map}
    # Quiz 7 has no topic and entry 30 has no KC, so neither creates a link.
    assert [(row["kc_id"], row["topic"]) for row in topic_map] == [
        ("KC-G1", "Unit 1"),
        ("KC-G1", "Unit 2"),
        ("KC-R1", "Unit 2"),
        ("KC-V1", "Unit 1"),
    ]


def test_curated_override_is_used_and_never_rewritten(tmp_path, monkeypatch):
    override = tmp_path / "curated.csv"
    override.write_text(
        "question_bank_entry_id,kc_id,kc_dimension,mapping_version,mapping_source\n"
        "10,KC001,Grammar,1.0,reviewed crosswalk\n",
        encoding="utf-8",
    )
    before = override.read_text(encoding="utf-8")
    written = {}
    monkeypatch.setattr(mappings, "write_csv", lambda name, rows, columns: written.update({name: list(rows)}))

    catalog, question_map, topic_map = mappings.collect_course_mappings(
        RoutingCursor(usage=[(5, 10)]), 3, CATALOG, mapping_file=override, use_override=True
    )

    assert override.read_text(encoding="utf-8") == before
    assert "curated.csv" not in written
    assert catalog == [
        {"kc_id": "KC001", "kc_name": "KC001", "kc_dimension": "grammar", "learning_objective": ""}
    ]
    assert question_map[0]["mapping_version"] == "1.0"
    assert topic_map == [{"kc_id": "KC001", "topic": "Unit 1", "mapping_version": "1.0"}]


def test_kc_activity_map_links_lessons_by_idnumber_and_quizzes_by_questions():
    kc_catalog, question_map = mappings.build_course_mappings(ENTRIES)
    catalog_rows = [
        {"course_module_id": 51, "activity_type": "quiz", "activity_id": 5, "activity_name": "Q1", "topic": "Unit 1", "idnumber": ""},
        {"course_module_id": 61, "activity_type": "quiz", "activity_id": 6, "activity_name": "Q2", "topic": "Unit 2", "idnumber": "KC-G1"},
        {"course_module_id": 70, "activity_type": "page", "activity_id": 9, "activity_name": "Lesson G1", "topic": "Unit 1", "idnumber": "KC-G1"},
        {"course_module_id": 71, "activity_type": "page", "activity_id": 10, "activity_name": "Overview", "topic": "Unit 1", "idnumber": ""},
        {"course_module_id": 72, "activity_type": "url", "activity_id": 11, "activity_name": "Typo", "topic": "Unit 1", "idnumber": "KC-G9"},
    ]
    usage = [(5, 10), (5, 11), (5, 20), (6, 10), (6, 10), (5, 30)]

    rows, unknown = mappings.build_kc_activity_map(kc_catalog, question_map, usage, catalog_rows)

    summary = [(r["kc_id"], r["role"], r["course_module_id"], r["question_count"]) for r in rows]
    assert summary == [
        ("KC-G1", "learn", 70, ""),
        ("KC-G1", "practice", 51, 2),
        ("KC-G1", "practice", 61, 1),
        ("KC-V1", "practice", 51, 1),
    ]
    # A quiz's own ID number never makes it a learning resource; a typo is reported.
    assert unknown == ["KC-G9"]
    assert {r["mapping_version"] for r in rows} == {question_map[0]["mapping_version"]}
