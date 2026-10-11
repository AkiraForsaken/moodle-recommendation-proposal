"""Orchestration: fit every recommender once, serve ranked Next Best Activity lists."""

from typing import Dict, Iterable, List, Optional

import pandas as pd

from .base import BaseRecommender, Recommendation, to_frame
from .collaborative import CollaborativeRecommender
from .content_based import ContentBasedRecommender
from .data import RecommenderData
from .popularity import PopularityRecommender
from .rule_based import RuleBasedRecommender


class RecommendationEngine:
    """
    The engine depends only on `RecommenderData`, so Moodle and OULAD frames go
    through exactly the same code. If a method returns fewer than k items (cold
    start, course complete, no neighbours) the gap is filled from `fallback`
    and those rows are flagged `via_fallback=True` -- never silently.
    """

    def __init__(self, data: RecommenderData,
                 recommenders: Optional[Iterable[BaseRecommender]] = None,
                 fallback: Optional[str] = "popularity"):
        recommenders = recommenders or [
            RuleBasedRecommender(), PopularityRecommender(),
            ContentBasedRecommender(), CollaborativeRecommender(),
        ]
        self.data = data
        self.fallback = fallback
        self.recommenders: Dict[str, BaseRecommender] = {r.name: r.fit(data) for r in recommenders}
        if fallback and fallback not in self.recommenders:
            raise ValueError(f"fallback {fallback!r} is not a registered method")

    @property
    def methods(self) -> List[str]:
        return list(self.recommenders)

    def recommend(self, learner_id: int, k: int = 5, method: str = "rule_based") -> List[Recommendation]:
        if method not in self.recommenders:
            raise KeyError(f"unknown method {method!r}; choose from {self.methods}")
        recs = self.recommenders[method].recommend(learner_id, k)

        if len(recs) < k and self.fallback and method != self.fallback:
            have = {r.course_module_id for r in recs}
            extra = self.recommenders[self.fallback].recommend(learner_id, k + len(have))
            for r in extra:
                if r.course_module_id in have or len(recs) >= k:
                    continue
                r.via_fallback = True
                r.method = method  # attribute to the REQUESTED method; source is in the reason
                r.reason = f"[{method} had no pick; {self.fallback} fallback] {r.reason}"
                recs.append(r)
        for i, r in enumerate(recs, start=1):
            r.rank = i
        return recs

    def recommend_all(self, learner_id: int, k: int = 5) -> pd.DataFrame:
        return to_frame([r for m in self.methods for r in self.recommend(learner_id, k, m)])


def run_baseline_recommendations(learner_id, data_dir=None, k: int = 5,
                                 methods: Optional[List[str]] = None, verbose: bool = True):
    """Back-compatible entry point. Returns {method: DataFrame} with one shared schema."""
    data = RecommenderData.from_moodle_csv(data_dir)  # raises FileNotFoundError if inputs are missing
    engine = RecommendationEngine(data)
    results = {}
    if verbose:
        print(f"--- Baseline recommendations for learner {learner_id} (top {k}) ---")
    for m in methods or engine.methods:
        frame = to_frame(engine.recommend(learner_id, k, m))
        results[m] = frame
        if verbose:
            print(f"\n[{m}]")
            if frame.empty:
                print("  (no recommendation)")
            for r in frame.itertuples():
                flag = " ↻" if r.is_review else ""
                print(f"  {r.rank}. {r.activity_name} ({r.activity_type}, {r.topic}){flag}\n     {r.reason}")
    return results
