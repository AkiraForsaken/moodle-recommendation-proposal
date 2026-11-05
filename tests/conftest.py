import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import create_app  # noqa: E402
from api.settings import Settings  # noqa: E402
from tests.demo_data import write_collector_csvs  # noqa: E402

TOKEN = "t" * 32


@pytest.fixture()
def data_dir(tmp_path):
    return write_collector_csvs(tmp_path / "data")


@pytest.fixture()
def client(data_dir):
    app = create_app(Settings(
        api_token=TOKEN,
        data_dir=data_dir,
        collector_command=[sys.executable, "-c", "pass"],
    ))
    c = TestClient(app)
    c.headers.update({"X-API-Token": TOKEN})
    return c
