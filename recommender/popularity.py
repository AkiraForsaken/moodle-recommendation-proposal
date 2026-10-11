"""Popularity baseline: what other learners used most."""

import pandas as pd

from .base import BaseRecommender

_METRICS = {"learners": "distinct learners", "views": "views", "sessions": "sessions"}


class PopularityRecommender(BaseRecommender):
    """
    metric="learners" (default) counts DISTINCT learners per activity, so a single
    heavy user cannot dominate (in the Moodle test data one learner produces
    ~70% of all events). "views" / "sessions" follow the plan's wording literally.
    """

    name = "popularity"

    def __init__(self, metric: str = "learners", seen_policy: str = "demote"):
        super().__init__(seen_policy)
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)}")
        self.metric = metric

    def _fit(self) -> None:
        inter = self.data.interactions
        self.n_learners = max(int(inter["learner_id"].nunique()), 1)
        self.stats = inter.groupby("course_module_id").agg(
            learners=("learner_id", "nunique"),
            views=("n_views", "sum"),
            sessions=("n_sessions", "sum"),
        )

    def _score(self, learner_id: int) -> pd.DataFrame:
        pool = self.data.catalog[["course_module_id"]].merge(
            self.stats, left_on="course_module_id", right_index=True, how="left"
        ).fillna(0)
        pool["score"] = pool[self.metric].astype(float)
        pool["reason"] = [
            f"Used by {int(l)} of {self.n_learners} learners ({int(v)} {_METRICS['views']})."
            for l, v in zip(pool["learners"], pool["views"])
        ]
        return pool[["course_module_id", "score", "reason"]]
