"""
Collector configuration: every environment-driven constant lives here.

Nothing in this module talks to the database. Values are read once at import
time, so set the MOODLE_* environment variables before importing the package.
"""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DB_HOST = os.getenv("MOODLE_DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("MOODLE_DB_PORT", "3306"))
DB_NAME = os.getenv("MOODLE_DB_NAME", "moodle")
DB_USER = os.getenv("MOODLE_DB_USER", "moodle")
DB_PASSWORD = os.getenv("MOODLE_DOCKER_DBPASS")
DB_PREFIX = os.getenv("MOODLE_DB_PREFIX", "m_")

COURSE_ID_ENV = os.getenv("MOODLE_COURSE_ID")
LOCAL_TZ_NAME = os.getenv("MOODLE_TZ", "Asia/Ho_Chi_Minh")
MASTERY_THRESHOLD = float(os.getenv("MOODLE_MASTERY_THRESHOLD", "0.80"))

# Session boundary: consecutive events by the same learner that are this far
# apart (or closer) belong to the same continuous learning session.
SESSION_GAP_SECONDS = int(os.getenv("MOODLE_SESSION_GAP_SECONDS", "1800"))

# Which enrolled roles count as "learners". Set to "all" to disable filtering
# (useful when debugging, but teacher/admin authoring traffic will pollute the
# learner model - in the current test course the teacher account produced more
# events than the student).
INCLUDE_ROLES = os.getenv("MOODLE_INCLUDE_ROLES", "student")

OUTPUT_DIR = Path(
    os.getenv("MOODLE_OUTPUT_DIR", Path(__file__).resolve().parent.parent / "data")
)

LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)

CONTEXT_MODULE = 70

# --------------------------------------------------------------------------
# Table-name helper
# --------------------------------------------------------------------------


def t(name):
    """Prefix a Moodle table name (default prefix m_)."""
    return f"{DB_PREFIX}{name}"
