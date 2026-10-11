"""
Learner state: the recommender's view of the learner model.

This module is the single place that knows how mastery is stored. V1 stores a
categorical `mastery_level`; Phase 2 will store a probability `p_mastery`. The
recommenders only ever see a `LearnerState`, so swapping V1 -> BKT -> DKT is a
data change, not a code change.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from . import config

VALID_BANDS = ("low", "medium", "high")


def band_from_probability(p: float) -> str:
    if p >= config.MASTERY_HIGH:
        return "high"
    if p >= config.MASTERY_MEDIUM:
        return "medium"
    return "low"


def read_mastery(level, prob):
    """Return (band, probability-or-None). A probability wins over a label."""
    if prob is not None and not pd.isna(prob):
        return band_from_probability(float(prob)), float(prob)
    label = str(level).strip().lower() if level is not None and not pd.isna(level) else ""
    return (label if label in VALID_BANDS else "unknown"), None


@dataclass(frozen=True)
class TopicState:
    topic: str
    band: str  # low | medium | high | unknown
    mastery: Optional[float]
    last_activity: Optional[pd.Timestamp]


@dataclass(frozen=True)
class LearnerState:
    learner_id: int
    topics: Dict[str, TopicState]

    @property
    def is_new(self) -> bool:
        return not self.topics


@dataclass(frozen=True)
class Focus:
    """Where the learner should work next.

    band: low | medium | unknown  -> work on `topic` (it is engaged, not mastered)
          new                     -> `topic` has not been touched yet
          complete                -> every topic is mastered (`topic` is None)
    """

    topic: Optional[str]
    band: str
    mastery: Optional[float] = None


def find_focus(state: LearnerState, topic_order: List[str], mode: str = "frontier") -> Focus:
    """
    Pick the learner's focus topic.

    frontier (default): the earliest topic, in course order, that is not yet
        mastered. Unmastered gaps block later material, so they come first.
    latest: start from the most recently active topic and move forward past
        mastered ones. Follows the learner's own path instead of the syllabus.
    """
    if not topic_order:
        return Focus(None, "complete")

    start = 0
    if mode == "latest" and state.topics:
        engaged = [
            (ts.last_activity if pd.notna(ts.last_activity) else pd.Timestamp.min, t)
            for t, ts in state.topics.items()
            if t in topic_order
        ]
        if engaged:
            start = topic_order.index(max(engaged)[1])

    for topic in topic_order[start:]:
        ts = state.topics.get(topic)
        if ts is None:
            return Focus(topic, "new")
        if ts.band != "high":
            return Focus(topic, ts.band, ts.mastery)
    return Focus(None, "complete")
