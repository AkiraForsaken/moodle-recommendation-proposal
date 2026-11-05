"""Environment-driven settings. No I/O besides reading os.environ."""

import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    api_token: str
    data_dir: Path = PROJECT_ROOT / "data"
    learner_model_file: str = "learner_model.csv"
    default_method: str = "rule_based"
    # Shell-free command that refreshes collector CSVs before the model is rebuilt.
    collector_command: List[str] = field(default_factory=list)
    collector_timeout: int = 300

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("API_TOKEN", "").strip()
        if len(token) < 16:
            raise RuntimeError(
                "API_TOKEN must be set to a secret of at least 16 characters, e.g.\n"
                "  export API_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
            )
        data_dir: Optional[str] = os.getenv("RECSYS_DATA_DIR")
        collector_command = os.getenv("API_COLLECTOR_COMMAND")
        command = (
            shlex.split(collector_command)
            if collector_command
            else [sys.executable, str(PROJECT_ROOT / "moodle-collector.py")]
        )
        return cls(
            api_token=token,
            data_dir=Path(data_dir) if data_dir else PROJECT_ROOT / "data",
            learner_model_file=os.getenv("API_LEARNER_MODEL_FILE", "learner_model.csv"),
            default_method=os.getenv("API_DEFAULT_METHOD", "rule_based"),
            collector_command=command,
            collector_timeout=int(os.getenv("API_COLLECTOR_TIMEOUT", "300")),
        )
