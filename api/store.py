"""
Snapshot store: the learner model + fitted recommendation engine, kept in memory.

* Reloads automatically when a source CSV changes on disk (so `python -m model`
  run by hand is picked up without restarting the server).
* A failed reload keeps serving the last good snapshot instead of returning 500s.
* The model is read once per change, not once per request.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from recommender import RecommendationEngine, RecommenderData

from .settings import Settings

log = logging.getLogger("api.store")

_SOURCES = ("activity_catalog.csv", "learning_events.csv", "quiz_attempts.csv")


class NotReady(RuntimeError):
    """No usable snapshot (required CSVs missing or unreadable)."""


@dataclass
class Snapshot:
    model: pd.DataFrame
    engine: Optional[RecommendationEngine]
    engine_error: Optional[str]
    loaded_at: float


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self._snap: Optional[Snapshot] = None
        self._sig: Optional[Tuple] = None
        self.last_error: Optional[str] = None
        self.pipeline_lock = threading.Lock()  # one pipeline run at a time

    # -- file signature ------------------------------------------------------
    def _signature(self) -> Tuple:
        d = self.settings.data_dir
        names = (self.settings.learner_model_file,) + _SOURCES
        sig = []
        for n in names:
            p = d / n
            if p.exists():
                st = p.stat()
                sig.append((n, st.st_mtime_ns, st.st_size))
            else:
                sig.append((n, None, None))
        return tuple(sig)

    # -- loading ---------------------------------------------------------------
    def _load(self) -> Snapshot:
        d: Path = self.settings.data_dir
        lm_path = d / self.settings.learner_model_file
        if not lm_path.exists():
            raise NotReady(f"{lm_path.name} not found in {d}. Run the pipeline first.")
        model = pd.read_csv(lm_path)
        for col in ("learner_id", "topic", "mastery_level"):
            if col not in model.columns:
                raise NotReady(f"{lm_path.name} is missing required column '{col}'")

        engine, engine_error = None, None
        try:
            data = RecommenderData.from_moodle_csv(d, learner_model_file=self.settings.learner_model_file)
            engine = RecommendationEngine(data)
        except Exception as exc:  # heatmap can still be served without the engine
            engine_error = f"{type(exc).__name__}: {exc}"
            log.warning("Recommendation engine unavailable: %s", engine_error)
        return Snapshot(model, engine, engine_error, time.time())

    def snapshot(self, force: bool = False) -> Snapshot:
        with self._lock:
            sig = self._signature()
            if force or self._snap is None or sig != self._sig:
                try:
                    self._snap = self._load()
                    self._sig = sig
                    self.last_error = None
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    log.error("Reload failed: %s", self.last_error)
                    if self._snap is None:
                        raise NotReady(self.last_error) from exc
                    # keep serving the previous good snapshot
            return self._snap
