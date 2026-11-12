"""Run the PHP scripts inside the Moodle web container.

The Moodle checkout is not modified: scripts and their JSON inputs are copied
to a scratch directory in the container and run with the container's PHP.
"""

import json
import os
import subprocess
from pathlib import Path

CONTAINER = os.getenv("MOODLE_WEBSERVER_CONTAINER", "moodle-docker-webserver-1")
REMOTE_DIR = "/tmp/moodle_setup"
PHP_DIR = Path(__file__).resolve().parent / "php"
RESULT_MARKER = "MOODLE_SETUP_RESULT:"


def _docker(*args, **kwargs):
    try:
        return subprocess.run(["docker", *args], check=True, **kwargs)
    except FileNotFoundError as error:
        raise SystemExit("docker is not installed or not on PATH") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"docker {' '.join(args[:2])} failed (exit {error.returncode}). Is the "
            f"container {CONTAINER!r} running? Set MOODLE_WEBSERVER_CONTAINER if it "
            "has another name."
        ) from error


def copy_in(local_path, remote_name=None):
    """Copy a file into the container scratch directory; return its path there."""
    local_path = Path(local_path)
    remote = f"{REMOTE_DIR}/{remote_name or local_path.name}"
    _docker("exec", CONTAINER, "mkdir", "-p", REMOTE_DIR, capture_output=True)
    _docker("cp", str(local_path), f"{CONTAINER}:{remote}", capture_output=True)
    return remote


def run_php(script, *args):
    """Run ``php/<script>`` in the container, streaming its output.

    A script reports its result as one line ``MOODLE_SETUP_RESULT:<json>``,
    which is parsed and returned (None if the script printed no result).
    """
    remote = copy_in(PHP_DIR / "lib.php")
    remote = copy_in(PHP_DIR / script)
    process = subprocess.Popen(
        ["docker", "exec", CONTAINER, "php", remote, *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result = None
    for line in process.stdout:
        if line.startswith(RESULT_MARKER):
            result = json.loads(line[len(RESULT_MARKER) :])
        else:
            print(line, end="", flush=True)
    if process.wait() != 0:
        raise SystemExit(
            f"{script} failed inside the Moodle container (exit {process.returncode})"
        )
    return result
