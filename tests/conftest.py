import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.demo_data import write_collector_csvs  # noqa: E402


@pytest.fixture()
def data_dir(tmp_path):
    return write_collector_csvs(tmp_path / "data")
