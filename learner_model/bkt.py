"""
Attempt-level Bayesian Knowledge Tracing (BKT) for the Moodle learner model.

WHAT THIS PRODUCES
    One `p_mastery` per (learner, topic): the BKT estimate P(L) that the learner
    has learned the topic, computed from that learner's finished quiz attempts
    in the topic, in time order. skill = topic (project decision; see
    Technical Handoff Section 10, step 1).

WHAT THE OBSERVATION IS (and is not)
    Each finished attempt is ONE binary observation:
        1 (pass)  if normalized_score >= MASTERY_HIGH (0.80)
        0 (fail)  otherwise
    This is a quiz-level proxy, NOT per-question correctness. It discards score
    magnitude (0.7 and 0.2 are both "fail"). Item-level BKT needs a collector
    extension (Handoff gap 5) and is a separate, later decision.

WHAT THE PARAMETERS ARE (and are not)
    The defaults in BKTParams are FIXED, UNCALIBRATED placeholders chosen by the
    developer. They were NOT fitted to data: the local Moodle export contains 3
    learners and 11 quiz attempts, which is far too little to estimate
    p_init / p_learn / p_slip / p_guess. Until real learner logs exist,
    p_mastery is a transparent, rule-like filter over pass/fail history, not an
    empirically validated measure of knowledge. Say so in the thesis.
    Note p_slip / p_guess here describe a WHOLE-QUIZ pass/fail, not one item:
    passing a 10-question quiz by guessing is much rarer than guessing one item,
    which is why p_guess is low.

WHAT p_mastery MEANS
    p_mastery = P(learner knows the topic at the NEXT opportunity), i.e. the
    posterior after the latest observation followed by the learning transition.
    It is a latent-knowledge estimate. It is not the probability of passing the
    next quiz (that is `predicted_correct` in `bkt_trace`, a different quantity,
    and is not written to the learner model).

LEAKAGE
    This module only sees the `quizzes` frame it is given. build_model passes the
    frame AFTER temporal.truncate_frames(as_of), so attempts finished at or after
    the cutoff cannot influence p_mastery. At every step the prediction is made
    from state BEFORE the current outcome is applied (see bkt_trace).

COLD START
    A (learner, topic) with no scored attempt gets NO row here; build_model gives
    it p_mastery = NaN, evidence_count = 0, state_status = "cold_start".
    "No evidence" is never turned into 0.0 or p_init.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .mastery import MASTERY_HIGH

MODEL_NAME = "bkt_attempt_level"
MODEL_VERSION = "0.1.0-fixed-params"
SOURCE_DATASET = "moodle_quiz_attempts"

STATE_ESTIMATED = "estimated"
STATE_COLD_START = "cold_start"
# "unavailable" (Handoff Section 7) is reserved for a future case where the model
# cannot run at all; this module never emits it.

BKT_COLUMNS = ["learner_id", "topic", "p_mastery", "evidence_count"]


@dataclass(frozen=True)
class BKTParams:
    """
    Standard four-parameter BKT plus an optional forgetting term.

    p_init    P(L0)  prior probability the topic is already known
    p_learn   P(T)   probability of learning between consecutive attempts
    p_slip    P(S)   P(fail the quiz | topic known)
    p_guess   P(G)   P(pass the quiz | topic not known)
    p_forget  P(F)   P(forget between attempts | known); 0 = classic BKT
    """

    p_init: float = 0.30
    p_learn: float = 0.30
    p_slip: float = 0.10
    p_guess: float = 0.10
    p_forget: float = 0.0

    def __post_init__(self):
        # slip/guess must be strictly inside (0, 1): that keeps every Bayes
        # denominator in bkt_step strictly positive.
        for name in ("p_slip", "p_guess"):
            v = getattr(self, name)
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {v}")
        for name in ("p_init", "p_learn"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if not 0.0 <= self.p_forget < 1.0:
            raise ValueError(f"p_forget must be in [0, 1), got {self.p_forget}")
        # slip + guess >= 1 makes a pass no more likely for a learner who knows
        # the topic than for one who does not (the model is not identifiable).
        if self.p_slip + self.p_guess >= 1.0:
            raise ValueError("p_slip + p_guess must be < 1")


DEFAULT_PARAMS = BKTParams()


def bkt_step(p_known: float, passed: bool, params: BKTParams):
    """
    One BKT update. Returns (predicted_pass, posterior, next_p_known):

      predicted_pass  P(pass) BEFORE seeing the outcome. Depends only on p_known.
      posterior       P(known | outcome), Bayes rule.
      next_p_known    posterior after the learning/forgetting transition; this is
                      the prior for the next attempt and the reported p_mastery.
    """
    s, g = params.p_slip, params.p_guess
    predicted_pass = p_known * (1.0 - s) + (1.0 - p_known) * g
    if passed:
        posterior = p_known * (1.0 - s) / predicted_pass
    else:
        posterior = p_known * s / (1.0 - predicted_pass)
    next_p = posterior * (1.0 - params.p_forget) + (1.0 - posterior) * params.p_learn
    return predicted_pass, posterior, next_p


def bkt_trace(
    outcomes: Sequence[int], params: BKTParams = DEFAULT_PARAMS
) -> List[dict]:
    """
    Step-by-step trace for one learner-topic sequence (for tests, notebooks and
    the thesis). `predicted_pass` at step t uses ONLY outcomes before t.
    """
    p, rows = params.p_init, []
    for t, o in enumerate(outcomes, start=1):
        predicted, posterior, nxt = bkt_step(p, bool(o), params)
        rows.append(
            dict(
                step=t,
                p_known_before=p,
                predicted_pass=predicted,
                outcome=int(bool(o)),
                posterior=posterior,
                p_mastery=nxt,
            )
        )
        p = nxt
    return rows


def attempt_observations(
    quizzes: pd.DataFrame,
    activity_lookup: pd.DataFrame,
    pass_threshold: float = MASTERY_HIGH,
) -> pd.DataFrame:
    """
    Finished quiz attempts -> one row per attempt: learner_id, topic, passed,
    in chronological order. Quizzes are mapped to topics exactly as
    features.extract_quiz_features does (quiz_id == activity_id of a quiz
    activity); attempts of quizzes without a topic (e.g. section 0) and attempts
    without a numeric score are dropped.
    """
    cols = ["learner_id", "topic", "passed"]
    if quizzes is None or quizzes.empty:
        return pd.DataFrame(columns=cols)

    # The real collector export already carries its own `topic` column. The
    # activity catalog is the single source of truth for topics (as in
    # features.extract_quiz_features), so drop the export's copy before joining.
    q = quizzes.drop(columns=["topic", "activity_id"], errors="ignore").copy()
    q["quiz_id"] = pd.to_numeric(q["quiz_id"], errors="coerce")
    q["normalized_score"] = pd.to_numeric(q["normalized_score"], errors="coerce")

    lookup = activity_lookup[activity_lookup["activity_type"] == "quiz"][
        ["activity_id", "topic"]
    ].copy()
    lookup["activity_id"] = pd.to_numeric(lookup["activity_id"], errors="coerce")
    lookup = lookup.dropna().drop_duplicates("activity_id")

    q = q.merge(lookup, left_on="quiz_id", right_on="activity_id", how="inner")
    q = q.dropna(subset=["topic", "normalized_score"])
    if q.empty:
        return pd.DataFrame(columns=cols)

    # Chronological order across ALL quizzes of a topic. finish time is the true
    # order; quiz_id / attempt only break ties (and stand in if the column is absent).
    order = [
        c
        for c in ("learner_id", "finish_time_unix", "quiz_id", "attempt")
        if c in q.columns
    ]
    q = q.sort_values(order, kind="stable")
    q["passed"] = (q["normalized_score"] >= pass_threshold).astype(int)
    return q[cols].reset_index(drop=True)


def compute_bkt_state(
    quizzes: pd.DataFrame,
    activity_lookup: pd.DataFrame,
    params: Optional[BKTParams] = None,
    pass_threshold: float = MASTERY_HIGH,
) -> pd.DataFrame:
    """
    Frame with BKT_COLUMNS: one row per (learner_id, topic) that has at least one
    scored attempt. Rows are absent (not zero) for topics without evidence.
    """
    params = params or DEFAULT_PARAMS
    obs = attempt_observations(quizzes, activity_lookup, pass_threshold)
    if obs.empty:
        return pd.DataFrame(columns=BKT_COLUMNS)

    out = []
    for (learner_id, topic), grp in obs.groupby(["learner_id", "topic"], sort=False):
        p = params.p_init
        for passed in grp["passed"].to_numpy():
            p = bkt_step(p, bool(passed), params)[2]
        out.append((learner_id, topic, p, len(grp)))

    state = pd.DataFrame(out, columns=BKT_COLUMNS)
    state["p_mastery"] = np.clip(state["p_mastery"].astype(float), 0.0, 1.0).round(6)
    return state
