"""Knowledge-based routing on the learner's mastery state."""

import pandas as pd

from .base import BaseRecommender
from .state import find_focus


def _fmt(focus) -> str:
    """'' for categorical mastery, ' (73%)' once BKT/DKT supplies a probability."""
    return f" ({focus.mastery:.0%})" if focus.mastery is not None else ""


class RuleBasedRecommender(BaseRecommender):
    """
    Find the focus topic, then order its activities by what the mastery band calls for:

      low      review material, then retry practice
      medium   practice first, then material
      unknown  material first, then practice (to measure mastery)
      new      material, then practice (first exposure to the topic)
      complete nothing (the engine falls back to another method)

    A one-item preview of the next topic is appended after low/medium/unknown.
    """

    name = "rule_based"

    def __init__(self, focus_mode: str = "frontier"):
        super().__init__(seen_policy="allow")  # revisits are intended (retry a quiz)
        self.focus_mode = focus_mode

    def _score(self, learner_id: int) -> pd.DataFrame:
        data = self.data
        focus = find_focus(data.state(learner_id), data.topic_order, self.focus_mode)
        if focus.topic is None:
            return pd.DataFrame(columns=["course_module_id", "score", "reason"])

        t, m = focus.topic, _fmt(focus)
        plans = {
            "low": [("learn", f"Mastery in '{t}' is low{m}: review the material first."),
                    ("practice", f"Retry practice once '{t}' has been reviewed.")],
            "medium": [("practice", f"Mastery in '{t}' is medium{m}: consolidate with practice."),
                       ("learn", f"Refresh the '{t}' material if practice feels shaky.")],
            "unknown": [("learn", f"No assessment evidence for '{t}' yet: start with the material."),
                        ("practice", f"Then check your understanding of '{t}'.")],
            "new": [("learn", f"Next topic: '{t}'. Start with the material."),
                    ("practice", f"Then check your understanding of '{t}'.")],
        }

        seen = data.seen(learner_id)
        cat = data.catalog
        in_topic = cat[cat["topic"] == t]
        rows, score = [], 100.0
        for role, reason in plans[focus.band]:
            bucket = in_topic[in_topic["role"] == role]
            bucket = pd.concat(  # unseen first, then revisits; each in course order
                [bucket[~bucket["course_module_id"].isin(seen)],
                 bucket[bucket["course_module_id"].isin(seen)]]
            )
            for r in bucket.itertuples():
                tag = " (revisit)" if r.course_module_id in seen else ""
                rows.append((r.course_module_id, score, reason + tag))
                score -= 1.0

        nxt = data.next_topic(t)
        if focus.band != "new" and nxt:
            ahead = cat[(cat["topic"] == nxt) & (cat["role"] == "learn")].head(1)
            for r in ahead.itertuples():
                rows.append((r.course_module_id, score, f"Preview the next topic: '{nxt}'."))
        return pd.DataFrame(rows, columns=["course_module_id", "score", "reason"])
