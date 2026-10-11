"""Common output contract + template for every recommender."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List

import pandas as pd

OUTPUT_COLUMNS = [
    "learner_id", "rank", "method", "course_module_id", "activity_id",
    "activity_name", "activity_type", "topic", "score", "is_review",
    "via_fallback", "reason",
]


@dataclass
class Recommendation:
    learner_id: int
    rank: int
    method: str
    course_module_id: int
    activity_id: int
    activity_name: str
    activity_type: str
    topic: str
    score: float
    is_review: bool  # learner has already interacted with this activity
    reason: str
    via_fallback: bool = False

    def as_dict(self):
        return asdict(self)


def to_frame(recs: List[Recommendation]) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in recs], columns=OUTPUT_COLUMNS)


class BaseRecommender(ABC):
    """
    Subclasses implement `_score(learner_id)` and return a frame with columns
    [course_module_id, score, reason] (higher score = better). Everything else
    (metadata join, seen-policy, tie-breaking, top-K, output schema) is shared,
    so all methods are comparable by construction -- which Phase 4 needs.

    seen_policy:
      demote  unseen activities first, already-seen ones fill the rest (default)
      exclude never recommend something the learner already touched
      allow   no special treatment (the method handles revisits itself)
    """

    name = "base"

    def __init__(self, seen_policy: str = "demote"):
        if seen_policy not in ("demote", "exclude", "allow"):
            raise ValueError(f"unknown seen_policy {seen_policy!r}")
        self.seen_policy = seen_policy
        self.data = None

    def fit(self, data):
        self.data = data
        self._fit()
        return self

    def _fit(self) -> None:
        pass

    @abstractmethod
    def _score(self, learner_id: int) -> pd.DataFrame: ...

    def recommend(self, learner_id: int, k: int = 5) -> List[Recommendation]:
        if self.data is None:
            raise RuntimeError(f"{self.name}: call fit(data) first")
        learner_id = int(learner_id)
        scored = self._score(learner_id)
        if scored is None or scored.empty:
            return []

        meta = self.data.catalog[
            ["course_module_id", "activity_id", "activity_name", "activity_type", "topic", "sequence"]
        ]
        df = scored.merge(meta, on="course_module_id", how="inner")
        df["is_review"] = df["course_module_id"].isin(self.data.seen(learner_id))

        if self.seen_policy == "exclude":
            df = df[~df["is_review"]]
        if self.seen_policy == "demote":
            df = df.sort_values(["is_review", "score", "sequence"], ascending=[True, False, True])
        else:
            df = df.sort_values(["score", "sequence"], ascending=[False, True])
        df = df.drop_duplicates("course_module_id").head(k)

        return [
            Recommendation(
                learner_id=learner_id, rank=i, method=self.name,
                course_module_id=int(r.course_module_id), activity_id=int(r.activity_id),
                activity_name=r.activity_name, activity_type=r.activity_type, topic=r.topic,
                score=float(r.score), is_review=bool(r.is_review), reason=r.reason,
            )
            for i, r in enumerate(df.itertuples(index=False), start=1)
        ]
