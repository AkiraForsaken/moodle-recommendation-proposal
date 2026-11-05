import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from api.settings import Settings
from tests.conftest import TOKEN


def _run(client):
    r = client.post("/run-pipeline")
    assert r.status_code == 200, r.text
    return r.json()


# ---- auth ---------------------------------------------------------------
@pytest.mark.parametrize("method,path", [("post", "/run-pipeline"), ("get", "/analytics/heatmap"),
                                         ("get", "/recommend/2")])
def test_protected_endpoints_reject_missing_and_wrong_token(client, method, path):
    for headers in ({"X-API-Token": ""}, {"X-API-Token": "wrong" * 8}):
        r = getattr(client, method)(path, headers=headers)
        assert r.status_code == 401
    # the client's default header is the valid token, so this must NOT be 401
    assert getattr(client, method)(path).status_code != 401


def test_health_is_open_and_leaks_no_learner_data(client):
    r = client.get("/health", headers={"X-API-Token": ""})
    assert r.status_code == 200
    assert set(r.json()) <= {"status", "detail", "learners", "engine_ready", "loaded_at", "last_error"}


def test_settings_reject_weak_token(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "short")
    with pytest.raises(RuntimeError):
        Settings.from_env()


def test_settings_default_to_repository_collector(monkeypatch):
    from api.settings import PROJECT_ROOT

    monkeypatch.setenv("API_TOKEN", TOKEN)
    monkeypatch.delenv("API_COLLECTOR_COMMAND", raising=False)

    settings = Settings.from_env()

    assert settings.collector_command == [
        sys.executable,
        str(PROJECT_ROOT / "moodle-collector.py"),
    ]


def test_settings_allow_collector_command_override(monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    monkeypatch.setenv("API_COLLECTOR_COMMAND", "python custom_collector.py")

    assert Settings.from_env().collector_command == [
        "python",
        "custom_collector.py",
    ]


# ---- before the pipeline has run -------------------------------------------
def test_503_before_first_pipeline_run(client):
    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/analytics/heatmap").status_code == 503
    assert client.get("/recommend/2").status_code == 503


# ---- pipeline ---------------------------------------------------------------
def test_pipeline_builds_model_and_reports_counts(client):
    out = _run(client)
    assert out["status"] == "success" and out["learners"] == 11 and out["engine_ready"] is True
    assert out["collector"] == "ran"


def test_pipeline_runs_configured_collector_before_model_build(client, data_dir, monkeypatch):
    import api.main as m

    monkeypatch.setenv("MOODLE_OUTPUT_DIR", "/stale/collector/output")
    steps = []
    call_options = {}

    def run_collector(command, **kwargs):
        steps.append("collector")
        call_options["command"] = command
        call_options.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="collector output", stderr="")

    build_model = m.build_learner_model

    def build_after_collector(*args, **kwargs):
        steps.append("model")
        return build_model(*args, **kwargs)

    monkeypatch.setattr(m.subprocess, "run", run_collector)
    monkeypatch.setattr(m, "build_learner_model", build_after_collector)
    response = client.post("/run-pipeline")

    assert response.status_code == 200
    assert response.json()["collector"] == "ran"
    assert steps == ["collector", "model"]
    assert call_options["command"] == [sys.executable, "-c", "pass"]
    assert call_options["cwd"] == str(Path(m.__file__).resolve().parent.parent)
    assert call_options["env"]["MOODLE_OUTPUT_DIR"] == str(data_dir)


def test_pipeline_does_not_rebuild_model_when_collector_fails(client, monkeypatch):
    import api.main as m
    build_called = False

    def fail_collector(command, **kwargs):
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="database unavailable")

    def unexpected_model_build(*args, **kwargs):
        nonlocal build_called
        build_called = True
        raise AssertionError(
            f"model must not be rebuilt after collector failure ({len(args)} args, {len(kwargs)} kwargs)"
        )

    monkeypatch.setattr(m.subprocess, "run", fail_collector)
    monkeypatch.setattr(m, "build_learner_model", unexpected_model_build)
    response = client.post("/run-pipeline")

    assert response.status_code == 502
    assert response.json()["detail"] == "Collector step failed (see API log)"
    assert not build_called


def test_pipeline_reports_collector_timeout(client, monkeypatch):
    import api.main as m

    def timeout_collector(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(m.subprocess, "run", timeout_collector)
    response = client.post("/run-pipeline")

    assert response.status_code == 504
    assert response.json()["detail"] == "Collector step timed out"


def test_pipeline_refuses_to_rebuild_without_collector(data_dir):
    # The production settings factory always supplies the collector. An empty
    # command must not fall through to rebuilding from potentially stale CSVs.
    from fastapi.testclient import TestClient
    from api import create_app

    no_collector = TestClient(create_app(Settings(api_token=TOKEN, data_dir=data_dir)))
    no_collector.headers.update({"X-API-Token": TOKEN})

    response = no_collector.post("/run-pipeline")

    assert response.status_code == 500
    assert "refusing to use stale CSVs" in response.json()["detail"]


def test_pipeline_missing_collector_csvs_is_409(data_dir):
    from fastapi.testclient import TestClient
    from api import create_app
    (data_dir / "quiz_attempts.csv").unlink()
    c = TestClient(create_app(Settings(
        api_token=TOKEN,
        data_dir=data_dir,
        collector_command=[sys.executable, "-c", "pass"],
    )))
    r = c.post("/run-pipeline", headers={"X-API-Token": TOKEN})
    assert r.status_code == 409


def test_second_concurrent_pipeline_run_gets_409(client, monkeypatch):
    import api.main as m
    gate, started = threading.Event(), threading.Event()
    real = m.build_learner_model

    def slow(*a, **k):
        started.set()
        gate.wait(5)
        return real(*a, **k)

    monkeypatch.setattr(m, "build_learner_model", slow)
    results = {}
    t = threading.Thread(target=lambda: results.update(first=client.post("/run-pipeline")))
    t.start()
    assert started.wait(5)
    assert client.post("/run-pipeline").status_code == 409
    gate.set()
    t.join(10)
    assert results["first"].status_code == 200


# ---- heatmap -------------------------------------------------------------------
def test_heatmap_is_strict_json_and_uses_null_for_missing(client):
    _run(client)
    raw = client.get("/analytics/heatmap").text
    body = json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert body["topics"] == ["Variables and Data Types", "Conditions", "Loops", "Functions"]
    cells = [c for row in body["learners"] for c in row["topics"].values()]
    assert any(c["score_trend"] is None for c in cells)      # view-only learner
    assert sum(body["summary"].values()) == len(cells)
    assert set(body["summary"]) == {"high", "medium", "low", "unknown"}


def test_heatmap_values_match_learner_model(client):
    _run(client)
    body = client.get("/analytics/heatmap").json()
    row = next(r for r in body["learners"] if r["learner_id"] == 2)
    assert row["topics"]["Conditions"]["mastery_level"] == "low"
    assert row["topics"]["Variables and Data Types"]["mastery_level"] == "high"


def test_heatmap_drops_rows_without_topic(client, data_dir):
    """The OULAD export has ~86% blank-topic rows; they must not crash or show up."""
    _run(client)
    p = data_dir / "learner_model.csv"
    p.write_text(p.read_text() + "999,,,0,,,1,0.6,0.6,0.0,1800,,0,0.0,medium,\n")
    body = client.get("/analytics/heatmap").json()   # also exercises auto-reload
    assert all(r["learner_id"] != 999 for r in body["learners"])


def test_heatmap_hides_legacy_topic_zero_row(client, data_dir):
    _run(client)
    p = data_dir / "learner_model.csv"
    p.write_text(
        p.read_text()
        + "999,Topic 0,0,0,0,0,0,,,,0,0,0,0.0,unknown,\n"
    )
    body = client.get("/analytics/heatmap").json()
    assert "Topic 0" not in body["topics"]
    assert all("Topic 0" not in row["topics"] for row in body["learners"])


# ---- recommendations ---------------------------------------------------------------
def test_recommend_targets_weak_topic(client):
    _run(client)
    body = client.get("/recommend/2?k=3").json()
    assert body["focus"] == {"topic": "Conditions", "band": "low"}
    assert body["recommendations"][0]["topic"] == "Conditions"
    assert [r["rank"] for r in body["recommendations"]] == [1, 2, 3]
    assert "review" in body["recommendations"][0]["reason"].lower()


def test_recommend_flags_fallback_when_course_complete(client):
    _run(client)
    body = client.get("/recommend/1").json()
    assert body["focus"]["band"] == "complete"
    assert all(r["via_fallback"] for r in body["recommendations"])


def test_recommend_unknown_learner_gets_first_topic(client):
    _run(client)
    body = client.get("/recommend/9999").json()
    assert body["learner_known"] is False
    assert body["recommendations"][0]["topic"] == "Variables and Data Types"


@pytest.mark.parametrize("url,code", [("/recommend/0", 422), ("/recommend/-3", 422), ("/recommend/abc", 422),
                                      ("/recommend/2?k=0", 422), ("/recommend/2?k=999", 422),
                                      ("/recommend/2?method=bogus", 422)])
def test_recommend_validates_input(client, url, code):
    _run(client)
    assert client.get(url).status_code == code


def test_all_four_methods_serve(client):
    _run(client)
    for m in ("rule_based", "popularity", "content_based", "collaborative"):
        r = client.get(f"/recommend/4?method={m}")
        assert r.status_code == 200 and r.json()["method"] == m


# ---- store behaviour -----------------------------------------------------------------
def test_manual_model_rebuild_is_picked_up_without_restart(client, data_dir):
    _run(client)
    before = client.get("/analytics/heatmap").json()
    assert all(r["learner_id"] != 500 for r in before["learners"])
    p = data_dir / "learner_model.csv"
    time.sleep(0.01)
    p.write_text(p.read_text() + "500,Loops,12,1,1,0,1,0.9,0.9,0.0,10,10,0,0.0,high,2026-01-01 00:00:00\n")
    os.utime(p, None)
    after = client.get("/analytics/heatmap").json()
    assert any(r["learner_id"] == 500 for r in after["learners"])


def test_corrupt_reload_keeps_last_good_snapshot(client, data_dir):
    _run(client)
    good = client.get("/analytics/heatmap").json()
    (data_dir / "learner_model.csv").write_text("garbage\n")
    again = client.get("/analytics/heatmap")
    assert again.status_code == 200
    assert len(again.json()["learners"]) == len(good["learners"])
    assert client.get("/health").json()["last_error"]
