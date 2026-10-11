"""User-based collaborative filtering on learner-model profiles."""

import numpy as np
import pandas as pd

from . import config
from .base import BaseRecommender


class CollaborativeRecommender(BaseRecommender):
    """
    1. Profile = per-topic [best_score, score_trend, log lesson_time_estimate],
       min-max scaled to [0, 1]. Topics a learner never touched stay missing.
    2. Similarity to another learner = (1 - RMSE over co-observed dimensions)
       x (co-observed / union of observed dimensions), so learners who worked on
       the same topics at a similar level rank highest.
    3. Candidate score = similarity-weighted share of the k nearest neighbours
       who *succeeded* on it (passed the mastery threshold, or engaged for
       unscored content).

    Similarities are computed on demand for the target learner only, so memory
    stays O(learners x features) rather than O(learners^2) (OULAD has ~32k).
    """

    name = "collaborative"

    def __init__(self, features=config.CF_FEATURES, n_neighbors: int = config.CF_NEIGHBOURS,
                 min_similarity: float = config.CF_MIN_SIMILARITY, seen_policy: str = "demote"):
        super().__init__(seen_policy)
        self.features = tuple(features)
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity

    def _fit(self) -> None:
        lm = self.data.learner_model
        feats = [f for f in self.features if f in lm.columns]
        self._X, self._ids = None, np.array([], dtype=int)
        inter = self.data.interactions
        self._success = inter.loc[inter["succeeded"], ["learner_id", "course_module_id"]]
        if not feats or lm.empty:
            return

        df = lm[["learner_id", "topic"] + feats].copy()
        for f in feats:
            df[f] = pd.to_numeric(df[f], errors="coerce")
            if f in config.CF_LOG_FEATURES:
                df[f] = np.log1p(df[f].clip(lower=0))
        wide = df.pivot_table(index="learner_id", columns="topic", values=feats, aggfunc="first")
        arr = wide.to_numpy(dtype=float)
        if arr.size == 0:
            return
        lo, hi = np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        self._X = (arr - lo) / span
        self._ids = wide.index.to_numpy()

    def _similarities(self, learner_id: int):
        if self._X is None:
            return None
        pos = np.flatnonzero(self._ids == learner_id)
        if pos.size == 0:
            return None
        u = self._X[pos[0]]
        obs_u, obs_x = ~np.isnan(u), ~np.isnan(self._X)
        both, union = obs_x & obs_u, obs_x | obs_u
        n_both, n_union = both.sum(axis=1), union.sum(axis=1)
        sq = np.where(both, (self._X - np.nan_to_num(u)) ** 2, 0.0)
        rmse = np.sqrt(sq.sum(axis=1) / np.maximum(n_both, 1))
        sim = (1.0 - rmse) * (n_both / np.maximum(n_union, 1))
        sim[pos[0]] = 0.0  # never your own neighbour
        return pd.Series(sim, index=self._ids)

    def _score(self, learner_id: int) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["course_module_id", "score", "reason"])
        sims = self._similarities(learner_id)
        if sims is None:
            return empty
        neigh = sims[sims > self.min_similarity].nlargest(self.n_neighbors)
        if neigh.empty:
            return empty

        wins = self._success[self._success["learner_id"].isin(neigh.index)].copy()
        wins["w"] = wins["learner_id"].map(neigh)
        agg = wins.groupby("course_module_id").agg(
            support=("w", "sum"), n=("learner_id", "nunique")
        ).reset_index()
        agg["score"] = agg["support"] / neigh.sum()
        agg["reason"] = [
            f"{int(n)} of {len(neigh)} similar learners succeeded on this."
            for n in agg["n"]
        ]
        return agg[["course_module_id", "score", "reason"]]
