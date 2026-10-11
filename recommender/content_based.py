"""Content-based baseline: match the learner's current topic to activity metadata."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender
from .state import find_focus


class ContentBasedRecommender(BaseRecommender):
    """
    Each activity is a TF-IDF vector over [name + topic (+ `tags` column if the
    catalog has one)]. The learner profile is the centroid of the focus topic's
    activities plus a smaller centroid of what the learner has already used.
    Candidates are ranked by cosine similarity to that profile.

    Character n-grams (not words) make "variable" match "variables". Real Moodle
    tags (m_tag_instance) would sharpen this: add a `tags` column in the collector.
    """

    name = "content_based"

    def __init__(self, history_weight: float = 0.3, focus_mode: str = "frontier",
                 seen_policy: str = "demote"):
        super().__init__(seen_policy)
        self.history_weight = history_weight
        self.focus_mode = focus_mode

    def _fit(self) -> None:
        cat = self.data.catalog
        text = cat["activity_name"] + " " + cat["topic"]
        if "tags" in cat.columns:
            text = text + " " + cat["tags"].fillna("").astype(str)
        self.matrix = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        ).fit_transform(text.str.lower())
        self._row = {int(c): i for i, c in enumerate(cat["course_module_id"])}

    def _score(self, learner_id: int) -> pd.DataFrame:
        data, cat = self.data, self.data.catalog
        focus = find_focus(data.state(learner_id), data.topic_order, self.focus_mode)

        profile = np.zeros(self.matrix.shape[1])
        if focus.topic is not None:
            rows = np.flatnonzero((cat["topic"] == focus.topic).to_numpy())
            profile += np.asarray(self.matrix[rows].mean(axis=0)).ravel()
        seen_rows = [self._row[c] for c in data.seen(learner_id) if c in self._row]
        if seen_rows:
            weight = self.history_weight if focus.topic is not None else 1.0
            profile += weight * np.asarray(self.matrix[seen_rows].mean(axis=0)).ravel()
        if not profile.any():
            return pd.DataFrame(columns=["course_module_id", "score", "reason"])

        sims = cosine_similarity(profile.reshape(1, -1), self.matrix).ravel()
        anchor = f"'{focus.topic}'" if focus.topic else "your learning history"
        return pd.DataFrame({
            "course_module_id": cat["course_module_id"].to_numpy(),
            "score": sims,
            "reason": [f"Content similar to {anchor} (similarity {s:.2f})." for s in sims],
        })
