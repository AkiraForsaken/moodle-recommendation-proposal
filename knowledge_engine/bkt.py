"""Phase 1 estimator: item-level Bayesian Knowledge Tracing per KC.

WHAT THE OBSERVATION IS
    One graded answer to one question of a KC, in the learner's own answer
    order. ``correct`` is 1 only for full credit (the collector's rule).

WHAT THE PARAMETERS ARE (and are not)
    The defaults in ``config.py`` are FIXED, UNCALIBRATED values of the size
    commonly reported for BKT. They are not fitted to this course. Until a
    model has been trained and validated on the course's own answers, these
    estimates are a transparent, rule-like filter over each learner's
    right/wrong history, not a validated measure of knowledge.

WHAT IS REPORTED
    p_mastery              P(KC known) after the latest answer and the learning
                           transition, i.e. going into the next opportunity.
    predicted_correctness  P(next answer on this KC is right), from p_mastery,
                           slip and guess. Same meaning as the deep model's
                           column, so the two phases can be compared.

COLD START
    A learner with no answer on a KC still gets a row: the prior, with
    attempt_count 0 and state_status "cold_start". It is a starting assumption
    shared by every such learner, never individual evidence.

LEAKAGE
    Each step's prediction is made from the state BEFORE that answer is
    applied (see ``next_answer_predictions``), so scoring BKT on held-out
    learners never uses the answer being predicted.
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from . import config
from .schemas import (
    KNOWLEDGE_STATE_COLUMNS,
    STATE_COLD_START,
    STATE_ESTIMATED,
    validate_states,
)

MODEL_NAME = "bkt_item_level"
MODEL_VERSION = "1.0.0-fixed-params"
AGGREGATION_METHOD = "bkt_next_answer_probability"
ESTIMATOR_PHASE = 1


@dataclass(frozen=True)
class BKTParams:
    """
    Standard four-parameter BKT plus an optional forgetting term.

    p_init    P(L0)  prior probability the KC is already known
    p_learn   P(T)   probability of learning between consecutive answers
    p_slip    P(S)   P(wrong answer | KC known)
    p_guess   P(G)   P(right answer | KC not known)
    p_forget  P(F)   P(forget between answers | known); 0 = classic BKT
    """

    p_init: float = config.BKT_P_INIT
    p_learn: float = config.BKT_P_LEARN
    p_slip: float = config.BKT_P_SLIP
    p_guess: float = config.BKT_P_GUESS
    p_forget: float = config.BKT_P_FORGET

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
        # slip + guess >= 1 makes a right answer no more likely for a learner
        # who knows the KC than for one who does not (not identifiable).
        if self.p_slip + self.p_guess >= 1.0:
            raise ValueError("p_slip + p_guess must be < 1")


DEFAULT_PARAMS = BKTParams()


def predicted_correct(p_known: float, params: BKTParams = DEFAULT_PARAMS) -> float:
    """P(right answer) for a learner who knows the KC with probability p_known."""
    return p_known * (1.0 - params.p_slip) + (1.0 - p_known) * params.p_guess


def bkt_step(p_known: float, correct: bool, params: BKTParams = DEFAULT_PARAMS):
    """
    One BKT update. Returns (predicted, posterior, next_p_known):

      predicted     P(right) BEFORE seeing the answer. Depends only on p_known.
      posterior     P(known | answer), Bayes rule.
      next_p_known  posterior after the learning/forgetting transition; this is
                    the prior for the next answer and the reported p_mastery.
    """
    predicted = predicted_correct(p_known, params)
    if correct:
        posterior = p_known * (1.0 - params.p_slip) / predicted
    else:
        posterior = p_known * params.p_slip / (1.0 - predicted)
    next_p = posterior * (1.0 - params.p_forget) + (1.0 - posterior) * params.p_learn
    return predicted, posterior, next_p


def bkt_trace(
    outcomes: Sequence[int], params: BKTParams = DEFAULT_PARAMS
) -> List[dict]:
    """Step-by-step trace of one learner-KC sequence (tests, notebooks, thesis)."""
    p, rows = params.p_init, []
    for step, outcome in enumerate(outcomes, start=1):
        predicted, posterior, nxt = bkt_step(p, bool(outcome), params)
        rows.append(
            dict(
                step=step,
                p_known_before=p,
                predicted_correct=predicted,
                outcome=int(bool(outcome)),
                posterior=posterior,
                p_mastery=nxt,
            )
        )
        p = nxt
    return rows


def final_mastery(outcomes: Sequence[int], params: BKTParams = DEFAULT_PARAMS) -> float:
    p = params.p_init
    for outcome in outcomes:
        p = bkt_step(p, bool(outcome), params)[2]
    return float(min(max(p, 0.0), 1.0))


def next_answer_predictions(concepts, responses, params: BKTParams = DEFAULT_PARAMS):
    """Predict each answer of one chronological sequence from the answers before it.

    ``concepts`` and ``responses`` are parallel sequences (any hashable KC key,
    0/1 outcome). Returns one probability per answer; the KC state is updated
    only after its prediction is recorded.
    """
    known, predictions = {}, []
    for concept, response in zip(concepts, responses):
        p = known.get(concept, params.p_init)
        predicted, _, nxt = bkt_step(p, bool(response), params)
        predictions.append(predicted)
        known[concept] = nxt
    return predictions


def estimate_states(
    events,
    kc_catalog,
    learner_ids,
    inference_timestamp,
    params: BKTParams = DEFAULT_PARAMS,
    as_of=None,
    mapping_version=None,
    source_population=config.MOODLE_POPULATION,
):
    """One contract row for every learner x catalog KC.

    ``events`` must already be cut off and ordered per learner (see
    ``data.load_answer_events``). Learners that appear only in ``events`` are
    included; a KC answered but absent from the catalog is an error, because
    KC identity belongs to the course catalog.
    """
    catalog = kc_catalog[["kc_id", "kc_dimension"]].astype(str)
    unknown = sorted(set(events["kc_id"].astype(str)) - set(catalog["kc_id"]))
    if unknown:
        raise ValueError(
            f"Answer events reference KCs absent from the KC catalog: {unknown[:10]}"
        )
    learners = sorted(
        set(map(str, learner_ids)) | set(events["learner_id"].astype(str))
    )

    evidence = {}
    ordered = events.sort_values(["learner_id", "event_index"], kind="mergesort")
    for (learner_id, kc_id), group in ordered.groupby(
        ["learner_id", "kc_id"], sort=False
    ):
        outcomes = group["correct"].to_numpy()
        evidence[(str(learner_id), str(kc_id))] = (
            len(outcomes),
            float(outcomes.mean()),
            final_mastery(outcomes, params),
            pd.Timestamp(
                int(group["timestamp"].iloc[-1]), unit="s", tz="UTC"
            ).isoformat(),
        )

    prior_prediction = predicted_correct(params.p_init, params)
    rows = []
    for learner_id in learners:
        for kc_id, dimension in catalog.itertuples(index=False):
            seen = evidence.get((learner_id, kc_id))
            if seen is None:
                attempts, accuracy, mastery, last_at = 0, None, params.p_init, None
            else:
                attempts, accuracy, mastery, last_at = seen
            rows.append(
                {
                    "learner_id": learner_id,
                    "kc_id": kc_id,
                    "kc_dimension": dimension,
                    "attempt_count": attempts,
                    "evidence_count": attempts,
                    "observed_accuracy": accuracy,
                    "predicted_correctness": round(
                        (
                            prior_prediction
                            if seen is None
                            else predicted_correct(mastery, params)
                        ),
                        6,
                    ),
                    "p_mastery": round(mastery, 6),
                    "last_attempt_at": last_at,
                    "state_status": (
                        STATE_COLD_START if seen is None else STATE_ESTIMATED
                    ),
                }
            )
    states = pd.DataFrame(
        rows,
        columns=[
            "learner_id",
            "kc_id",
            "kc_dimension",
            "attempt_count",
            "evidence_count",
            "observed_accuracy",
            "predicted_correctness",
            "p_mastery",
            "last_attempt_at",
            "state_status",
        ],
    )
    states["attempt_count"] = states["attempt_count"].astype(int)
    states["evidence_count"] = states["evidence_count"].astype(int)
    states["observed_accuracy"] = pd.to_numeric(states["observed_accuracy"])
    states["evidence_tier"] = pd.cut(
        states["attempt_count"],
        bins=config.EVIDENCE_TIER_BINS,
        labels=config.EVIDENCE_TIER_LABELS,
    ).astype(str)
    states["model_name"] = MODEL_NAME
    states["model_version"] = MODEL_VERSION
    states["estimator_phase"] = ESTIMATOR_PHASE
    states["checkpoint_id"] = None
    states["checkpoint_sha256"] = None
    states["aggregation_method"] = AGGREGATION_METHOD
    states["prediction_question_count"] = np.nan
    states["inference_timestamp"] = inference_timestamp
    states["as_of"] = as_of
    states["mapping_version"] = mapping_version
    states["source_population"] = source_population
    return validate_states(states[KNOWLEDGE_STATE_COLUMNS])
