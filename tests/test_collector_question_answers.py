import csv

from collector import collect_question_answers as question_events


class FakeCursor:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.query = None
        self.params = None

    def execute(self, query, params=()):
        self.query = query
        self.params = params

    def fetchall(self):
        return self.rows


def _mapping(path):
    path.write_text(
        "question_bank_entry_id,kc_id,kc_dimension,mapping_version,mapping_source\n"
        "10,KC001,Grammar,1.0,reviewed crosswalk\n"
        "20,KC002,Reading,1.0,reviewed crosswalk\n",
        encoding="utf-8",
    )
    return path


def _graded_attempt(user, attempt_id, question_attempt_id, bank_entry_id, time, fraction):
    return (
        user,
        attempt_id,
        5,
        1,
        question_attempt_id,
        bank_entry_id,
        time - 30,
        time,
        fraction,
        1.0,
    )


def test_collect_question_answer_events_maps_ids_and_preserves_partial_credit(
    tmp_path, monkeypatch
):
    mapping_file = _mapping(tmp_path / "question_kc_map.csv")
    cursor = FakeCursor(
        [
            _graded_attempt(7, 100, 1000, 20, 200, 0.5),
            _graded_attempt(7, 100, 1001, 10, 100, 1.0),
            _graded_attempt(7, 101, 1002, 99, 300, 1.0),
        ]
    )
    written = {}
    monkeypatch.setattr(question_events, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        question_events,
        "write_csv",
        lambda name, rows, columns: written.update(
            {name: (list(rows), list(columns))}
        ),
    )

    rows = question_events.collect_question_answer_events(
        cursor, 5, {7}, mapping_file
    )

    assert [row["event_index"] for row in rows] == [1, 2]
    assert [row["item_id"] for row in rows] == ["moodle-qbe:10", "moodle-qbe:20"]
    assert [row["correct"] for row in rows] == [1, 0]
    assert rows[1]["raw_fraction"] == 0.5
    assert rows[0]["kc_dimension"] == "grammar"
    assert rows[0]["response_time_s"] == 30
    assert rows[0]["mapping_version"] == "1.0"
    report = list(csv.DictReader((tmp_path / "question_answer_mapping_report.csv").open()))
    assert report[0]["status"] == "partial_mapping"
    assert report[0]["unmapped_question_attempts"] == "1"


def test_missing_crosswalk_is_reported_without_breaking_other_exports(
    tmp_path, monkeypatch, capsys
):
    written = {}
    monkeypatch.setattr(question_events, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        question_events,
        "write_csv",
        lambda name, rows, columns: written.update({name: list(rows)}),
    )

    assert question_events.collect_question_answer_events(
        FakeCursor(), 5, None, tmp_path / "missing.csv"
    ) == []
    assert written["question_answer_events.csv"] == []
    assert "skipped_missing_mapping" in (
        tmp_path / "question_answer_mapping_report.csv"
    ).read_text()
    assert "export skipped" in capsys.readouterr().out


def test_roster_includes_active_enrolments_and_applies_role_filter(tmp_path, monkeypatch):
    cursor = FakeCursor([(7, 100, 0), (8, 100, 0), (9, 100, 0)])
    monkeypatch.setattr(question_events, "write_csv", lambda *_: None)

    learner_ids = question_events.collect_learner_roster(cursor, 5, {7, 9})

    assert learner_ids == ["7", "9"]
    assert cursor.params == (5,)
    assert "user_enrolments" in cursor.query
