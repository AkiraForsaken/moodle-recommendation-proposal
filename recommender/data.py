"""
Data layer: turns pipeline CSVs into one validated `RecommenderData` object.

The recommenders depend ONLY on this object. The OULAD ETL (Phase 2) must
produce frames that pass the same validation; then no recommender changes.

Three inputs:
  catalog        one row per course module (activity)
  learner_model  one row per (learner, topic)   <- learner STATE
  interactions   one row per (learner, activity) <- item-level HISTORY

Why interactions exist: learner_model.csv is aggregated per topic, so it cannot
say which activities a learner has seen or which activities are popular.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from . import config
from .state import LearnerState, TopicState, read_mastery

REQUIRED_COLUMNS = {
    "catalog": {
        "course_module_id",
        "activity_type",
        "activity_name",
        "topic",
        "sequence",
    },
    "learner_model": {"learner_id", "topic"},
}

INTERACTION_COLUMNS = [
    "learner_id",
    "course_module_id",
    "first_time",
    "last_time",
    "n_views",
    "n_events",
    "n_sessions",
    "best_score",
]


def _check_columns(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS[name] - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


# --------------------------------------------------------------------------
# Interactions
# --------------------------------------------------------------------------


def build_interactions(
    events: pd.DataFrame, quizzes: Optional[pd.DataFrame], catalog: pd.DataFrame
) -> pd.DataFrame:
    """Collapse learning_events (+ quiz_attempts) into one row per learner x activity."""
    ev = events.copy()
    if "is_learning_event" in ev.columns:
        ev = ev[ev["is_learning_event"] == 1]
    ev["course_module_id"] = pd.to_numeric(ev["course_module_id"], errors="coerce")
    ev = ev.dropna(subset=["course_module_id"])
    if ev.empty:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)

    ev["course_module_id"] = ev["course_module_id"].astype(int)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], errors="coerce")
    ev["_view"] = (ev["event_category"] == "content_view").astype(int)

    out = ev.groupby(["learner_id", "course_module_id"], as_index=False).agg(
        first_time=("timestamp", "min"),
        last_time=("timestamp", "max"),
        n_views=("_view", "sum"),
        n_events=("timestamp", "size"),
        n_sessions=("session_id", "nunique"),
    )

    out["best_score"] = np.nan
    if quizzes is not None and not quizzes.empty:
        quiz_cmid = (
            catalog[catalog["activity_type"] == "quiz"]
            .assign(
                activity_id=lambda d: pd.to_numeric(d["activity_id"], errors="coerce")
            )
            .set_index("activity_id")["course_module_id"]
        )
        q = quizzes.copy()
        q["course_module_id"] = pd.to_numeric(q["quiz_id"], errors="coerce").map(
            quiz_cmid
        )
        q["normalized_score"] = pd.to_numeric(q["normalized_score"], errors="coerce")
        q = q.dropna(subset=["course_module_id", "normalized_score"])
        q["course_module_id"] = q["course_module_id"].astype(int)
        best = q.groupby(["learner_id", "course_module_id"], as_index=False)[
            "normalized_score"
        ].max()
        out = out.drop(columns="best_score").merge(
            best.rename(columns={"normalized_score": "best_score"}),
            on=["learner_id", "course_module_id"],
            how="left",
        )
    return out[INTERACTION_COLUMNS]


# --------------------------------------------------------------------------
# RecommenderData
# --------------------------------------------------------------------------


class RecommenderData:
    def __init__(
        self,
        catalog: pd.DataFrame,
        learner_model: pd.DataFrame,
        interactions: Optional[pd.DataFrame] = None,
    ):
        _check_columns(catalog, "catalog")
        _check_columns(learner_model, "learner_model")
        if (
            "mastery_level" not in learner_model.columns
            and config.MASTERY_PROB_COLUMN not in learner_model.columns
        ):
            raise ValueError(
                f"learner_model needs 'mastery_level' or '{config.MASTERY_PROB_COLUMN}'"
            )

        self.catalog = self._prepare_catalog(catalog)
        self.topic_order: List[str] = (
            self.catalog.groupby("topic")["sequence"].min().sort_values().index.tolist()
        )

        lm = learner_model.copy()
        lm["learner_id"] = pd.to_numeric(lm["learner_id"], errors="coerce")
        lm = lm.dropna(subset=["learner_id"])
        lm["learner_id"] = lm["learner_id"].astype(int)
        self.learner_model = lm

        if interactions is None or interactions.empty:
            interactions = pd.DataFrame(columns=INTERACTION_COLUMNS)
        inter = interactions.copy()
        for col in INTERACTION_COLUMNS:
            if col not in inter.columns:
                inter[col] = np.nan
        inter["learner_id"] = pd.to_numeric(inter["learner_id"]).astype(int)
        inter["course_module_id"] = pd.to_numeric(inter["course_module_id"]).astype(int)
        for col in ("n_views", "n_events", "n_sessions"):
            inter[col] = pd.to_numeric(inter[col], errors="coerce").fillna(0)
        inter["best_score"] = pd.to_numeric(inter["best_score"], errors="coerce")
        # "Succeeded" = passed the mastery threshold on a scored activity, otherwise
        # simply engaged with it (completion tracking is off in the Moodle test course).
        inter["succeeded"] = np.where(
            inter["best_score"].notna(),
            inter["best_score"] >= config.MASTERY_HIGH,
            inter["n_events"] > 0,
        )
        self.interactions = inter[INTERACTION_COLUMNS + ["succeeded"]].reset_index(
            drop=True
        )

        self._seen: Dict[int, Set[int]] = (
            inter.groupby("learner_id")["course_module_id"].apply(set).to_dict()
        )
        self._states = self._build_states()

    # ---- construction -------------------------------------------------

    @classmethod
    def from_moodle_csv(cls, data_dir=None, learner_model_file="learner_model.csv"):
        d = Path(data_dir) if data_dir else config.DATA_DIR
        needed = ["activity_catalog.csv", learner_model_file, "learning_events.csv"]
        missing = [f for f in needed if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing in {d}: {', '.join(missing)}")

        catalog = pd.read_csv(d / "activity_catalog.csv")
        learner_model = pd.read_csv(d / learner_model_file)
        events = pd.read_csv(d / "learning_events.csv")
        quiz_file = d / "quiz_attempts.csv"
        quizzes = pd.read_csv(quiz_file) if quiz_file.exists() else None
        return cls(catalog, learner_model, build_interactions(events, quizzes, catalog))

    @staticmethod
    def _prepare_catalog(df: pd.DataFrame) -> pd.DataFrame:
        cat = df.copy()
        cat["course_module_id"] = pd.to_numeric(
            cat["course_module_id"], errors="coerce"
        )
        cat["sequence"] = pd.to_numeric(cat["sequence"], errors="coerce")
        cat = cat.dropna(subset=["course_module_id", "sequence", "topic"])
        cat["course_module_id"] = cat["course_module_id"].astype(int)
        if "activity_id" not in cat.columns:
            cat["activity_id"] = cat["course_module_id"]
        cat["activity_name"] = cat["activity_name"].fillna("").astype(str)

        keep = cat["activity_type"].isin(config.ROLE_BY_TYPE)
        if "visible" in cat.columns:
            keep &= pd.to_numeric(cat["visible"], errors="coerce").fillna(1) == 1
        if "section_number" in cat.columns:
            keep &= ~cat["section_number"].isin(config.EXCLUDED_SECTION_NUMBERS)
        cat = cat[keep].copy()
        cat["role"] = cat["activity_type"].map(config.ROLE_BY_TYPE)
        return cat.sort_values("sequence").reset_index(drop=True)

    def _build_states(self) -> Dict[int, LearnerState]:
        prob_col = config.MASTERY_PROB_COLUMN
        states: Dict[int, LearnerState] = {}
        for lid, grp in self.learner_model.groupby("learner_id"):
            topics = {}
            for row in grp.to_dict("records"):
                band, prob = read_mastery(row.get("mastery_level"), row.get(prob_col))
                last = pd.to_datetime(row.get("last_activity"), errors="coerce")
                topics[row["topic"]] = TopicState(row["topic"], band, prob, last)
            states[int(lid)] = LearnerState(int(lid), topics)
        return states

    # ---- access -------------------------------------------------------

    @property
    def learner_ids(self) -> List[int]:
        return sorted(set(self._states) | set(self._seen))

    def state(self, learner_id: int) -> LearnerState:
        lid = int(learner_id)
        return self._states.get(lid, LearnerState(lid, {}))

    def seen(self, learner_id: int) -> Set[int]:
        return self._seen.get(int(learner_id), set())

    def next_topic(self, topic: str) -> Optional[str]:
        order = self.topic_order
        i = order.index(topic) if topic in order else -1
        return order[i + 1] if 0 <= i < len(order) - 1 else None
