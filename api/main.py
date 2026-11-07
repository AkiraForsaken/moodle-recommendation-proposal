"""Routes. Every endpoint except /health requires the shared X-API-Token."""

import logging
import math
import os
import secrets
import subprocess
import time
from typing import Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query
from pydantic import BaseModel

from learner_model.build_model import build_learner_model
from recommender.config import MASTERY_PROB_COLUMN
from recommender.state import find_focus, read_mastery

from .settings import PROJECT_ROOT, Settings
from .store import NotReady, Store

log = logging.getLogger("api")
BANDS = ("high", "medium", "low", "unknown")


# --------------------------------------------------------------------------- schemas
class Cell(BaseModel):
    mastery_level: (
        str  # EFFECTIVE band: from p_mastery when present, else the CSV label
    )
    best_score: Optional[float] = None
    score_trend: Optional[float] = None
    p_mastery: Optional[float] = None  # BKT estimate; null = no evidence (cold start)


class LearnerRow(BaseModel):
    learner_id: int
    topics: Dict[str, Cell]


class Heatmap(BaseModel):
    generated_at: float
    topics: List[str]  # course order
    learners: List[LearnerRow]
    summary: Dict[str, int]


class Item(BaseModel):
    rank: int
    course_module_id: int
    activity_id: int
    activity_name: str
    activity_type: str
    topic: str
    score: float
    is_review: bool
    via_fallback: bool
    reason: str
    est_minutes: Optional[int] = None


class Focus(BaseModel):
    topic: Optional[str]
    band: str


class Recommendations(BaseModel):
    learner_id: int
    learner_known: bool
    method: str
    focus: Focus
    recommendations: List[Item]


# --------------------------------------------------------------------------- helpers
def _num(v) -> Optional[float]:
    """NaN / inf / missing -> None so the payload is valid JSON."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if math.isfinite(f) else None


def build_heatmap(model: pd.DataFrame) -> Heatmap:
    df = model.dropna(subset=["learner_id", "topic"]).copy()
    topic_labels = df["topic"].astype(str).str.strip().str.casefold()
    df = df.loc[topic_labels.ne("topic 0")]
    df["learner_id"] = pd.to_numeric(df["learner_id"], errors="coerce")
    df = df.dropna(subset=["learner_id"])
    df["learner_id"] = df["learner_id"].astype(int)
    df = df.drop_duplicates(["learner_id", "topic"], keep="last")

    if "sequence_position" in df.columns:
        order = (
            df.assign(_s=pd.to_numeric(df["sequence_position"], errors="coerce"))
            .groupby("topic")["_s"]
            .min()
            .sort_values(na_position="last")
            .index.tolist()
        )
    else:
        order = sorted(df["topic"].unique())

    rows: Dict[int, Dict[str, Cell]] = {}
    summary = {b: 0 for b in BANDS}
    for r in df.itertuples(index=False):
        # Same rule the recommender uses (recommender/state.py::read_mastery): a
        # finite p_mastery is binned at the 0.80 / 0.60 thresholds and wins over the
        # categorical label; otherwise the label is used. One rule for dashboard
        # and recommender, so they cannot disagree about a learner's band.
        band, prob = read_mastery(
            getattr(r, "mastery_level", None), getattr(r, MASTERY_PROB_COLUMN, None)
        )
        summary[band] += 1
        rows.setdefault(int(r.learner_id), {})[str(r.topic)] = Cell(
            mastery_level=band,
            best_score=_num(getattr(r, "best_score", None)),
            score_trend=_num(getattr(r, "score_trend", None)),
            p_mastery=_num(prob),
        )
    learners = [LearnerRow(learner_id=k, topics=v) for k, v in sorted(rows.items())]
    return Heatmap(
        generated_at=time.time(),
        topics=[str(t) for t in order],
        learners=learners,
        summary=summary,
    )


# --------------------------------------------------------------------------- app
def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings)
    app = FastAPI(title="Personalized Microlearning API", version="0.2.0")
    app.state.store = store

    # No CORS middleware on purpose: browsers never call this API. Moodle's PHP does.

    def require_token(x_api_token: Optional[str] = Header(default=None)) -> None:
        ok = x_api_token is not None and secrets.compare_digest(
            x_api_token.encode(), settings.api_token.encode()
        )
        if not ok:
            raise HTTPException(status_code=401, detail="Invalid or missing API token")

    def snapshot():
        try:
            return store.snapshot()
        except NotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.get("/health")
    def health():
        """Unauthenticated, no learner data. Safe for liveness checks."""
        try:
            snap = store.snapshot()
            return {
                "status": "ok",
                "learners": int(snap.model["learner_id"].nunique()),
                "engine_ready": snap.engine is not None,
                "loaded_at": snap.loaded_at,
                "last_error": store.last_error,
            }
        except NotReady as exc:
            return {"status": "degraded", "detail": str(exc)}

    @app.post("/run-pipeline", dependencies=[Depends(require_token)])
    def run_pipeline():
        """Refresh Moodle data, rebuild the learner model, and reload the snapshot."""
        if not store.pipeline_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Pipeline is already running")
        started = time.time()
        collector = "ran"
        try:
            if not settings.collector_command:
                raise HTTPException(
                    status_code=500,
                    detail="Collector command is not configured; refusing to use stale CSVs",
                )
            collector_env = os.environ.copy()
            collector_env["MOODLE_OUTPUT_DIR"] = str(settings.data_dir)
            proc = subprocess.run(
                settings.collector_command,
                capture_output=True,
                text=True,
                timeout=settings.collector_timeout,
                cwd=str(PROJECT_ROOT),
                env=collector_env,
                check=False,
            )
            if proc.returncode != 0:
                log.error(
                    "collector failed (exit %s): %s",
                    proc.returncode,
                    proc.stderr[-2000:],
                )
                raise HTTPException(
                    status_code=502,
                    detail="Collector step failed (see API log)",
                )
            log.info("collector completed: %s", proc.stdout[-2000:])
            try:
                built = build_learner_model(
                    data_dir=settings.data_dir,
                    output_file=settings.data_dir / settings.learner_model_file,
                )
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=409, detail=f"Collector CSVs missing: {exc.filename}"
                )
            snap = store.snapshot(force=True)
            return {
                "status": "success",
                "collector": collector,
                "rows_processed": int(len(built)),
                "learners": int(snap.model["learner_id"].nunique()),
                "engine_ready": snap.engine is not None,
                "duration_seconds": round(time.time() - started, 2),
            }
        except HTTPException:
            raise
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Collector step timed out")
        except Exception:
            log.exception("pipeline failed")
            raise HTTPException(status_code=500, detail="Pipeline failed (see API log)")
        finally:
            store.pipeline_lock.release()

    @app.get(
        "/analytics/heatmap",
        response_model=Heatmap,
        dependencies=[Depends(require_token)],
    )
    def heatmap():
        return build_heatmap(snapshot().model)

    @app.get(
        "/recommend/{learner_id}",
        response_model=Recommendations,
        dependencies=[Depends(require_token)],
    )
    def recommend(
        learner_id: int = Path(ge=1),
        k: int = Query(5, ge=1, le=20),
        method: Optional[str] = Query(None),
    ):
        snap = snapshot()
        if snap.engine is None:
            raise HTTPException(
                status_code=503, detail=f"Recommender unavailable: {snap.engine_error}"
            )
        engine = snap.engine
        method = method or settings.default_method
        if method not in engine.methods:
            raise HTTPException(
                status_code=422, detail=f"Unknown method; choose from {engine.methods}"
            )

        recs = engine.recommend(learner_id, k, method)
        catalog = engine.data.catalog
        minutes = (
            catalog.set_index("course_module_id")["estimated_minutes"]
            if "estimated_minutes" in catalog.columns
            else None
        )
        items = []
        for r in recs:
            est = None
            if minutes is not None and r.course_module_id in minutes.index:
                v = _num(minutes.loc[r.course_module_id])
                est = int(v) if v is not None else None
            items.append(
                Item(
                    rank=r.rank,
                    course_module_id=r.course_module_id,
                    activity_id=r.activity_id,
                    activity_name=r.activity_name,
                    activity_type=r.activity_type,
                    topic=r.topic,
                    score=_num(r.score) or 0.0,
                    is_review=r.is_review,
                    via_fallback=r.via_fallback,
                    reason=r.reason,
                    est_minutes=est,
                )
            )
        focus = find_focus(engine.data.state(learner_id), engine.data.topic_order)
        return Recommendations(
            learner_id=learner_id,
            learner_known=learner_id in engine.data.learner_ids,
            method=method,
            focus=Focus(topic=focus.topic, band=focus.band),
            recommendations=items,
        )

    return app
