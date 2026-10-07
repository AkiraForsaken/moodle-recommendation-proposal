#!/usr/bin/env python3
"""
Thin CLI entry point for the Moodle collector.

    python moodle-collector.py

All logic lives in the `collector/` package (see collector/__init__.py for the
module map). Configuration is via MOODLE_* environment variables
(see collector/config.py).
"""

from collector import main

if __name__ == "__main__":
    main()
