"""
Time-aware snapshots of the raw collector frames.

The learner model is an aggregate over a learner's WHOLE history. For offline
evaluation of a next-activity recommender that is leakage: to predict what a
learner does at time T, the model may only know what happened BEFORE T.

`as_of` (a cutoff) gives every consumer one definition of "before T":

    events        timestamp_unix       < cutoff
    quiz attempts finish_time_unix     < cutoff   (a score exists only once finished)
    completions   time_modified_unix   < cutoff   (rows without a time are dropped)

The cut is STRICTLY before T, so setting T to the timestamp of the activity you
want to hide really does hide it.

Why truncating is safe: `session_id` and `seconds_since_prev_event` are computed
looking backwards only (gap to the previous event), so dropping later events
never changes the values of earlier ones. No recomputation is needed.

Dataset-agnostic: filtering uses unix columns. The OULAD ETL (which only has day
offsets) should emit pseudo-unix times (day * 86400) in the same columns.
"""

from collections.abc import Mapping
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

EVENT_TIME_COL = "timestamp_unix"
QUIZ_TIME_COL = "finish_time_unix"
COMPLETION_TIME_COL = "time_modified_unix"

Cutoff = Union[None, int, Dict[int, int]]
_EPOCH = pd.Timestamp("1970-01-01")


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(
        v, bool
    )


def _local_to_unix(value, events: Optional[pd.DataFrame]) -> int:
    """
    Convert a naive local datetime/string to unix seconds using the UTC offset
    that the collector actually used. The offset is read from the event row
    whose local time is nearest to `value`, so it is right across DST changes
    and needs no timezone configuration. Timezone-aware inputs convert directly.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return int(ts.timestamp())

    naive_epoch = (ts - _EPOCH).total_seconds()
    offset = 0.0  # no events to calibrate against -> treat as UTC
    if (
        events is not None
        and len(events)
        and {"timestamp", EVENT_TIME_COL} <= set(events.columns)
    ):
        local = pd.to_datetime(events["timestamp"], errors="coerce")
        unix = pd.to_numeric(events[EVENT_TIME_COL], errors="coerce")
        ok = (local.notna() & unix.notna()).to_numpy()
        if ok.any():
            local_epoch = ((local[ok] - _EPOCH).dt.total_seconds()).to_numpy()
            unix_ok = unix[ok].to_numpy(dtype=float)
            nearest = int(np.argmin(np.abs(local_epoch - naive_epoch)))
            offset = unix_ok[nearest] - local_epoch[nearest]
    return int(round(naive_epoch + offset))


def _resolve_scalar(value, events) -> int:
    if _is_number(value):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return _local_to_unix(value, events)


def resolve_cutoff(as_of, events: Optional[pd.DataFrame] = None) -> Cutoff:
    """
    Normalise `as_of` to unix seconds.

      None                          no cutoff (use all data)
      int / float                   unix seconds
      str / datetime / Timestamp    local time, converted with the collector's offset
      {learner_id: any of the above}  PER-LEARNER cutoffs. Learners not in the
                                    mapping are EXCLUDED from the snapshot.
    """
    if as_of is None:
        return None
    if isinstance(as_of, Mapping):
        return {int(k): _resolve_scalar(v, events) for k, v in as_of.items()}
    return _resolve_scalar(as_of, events)


def _before(
    df: Optional[pd.DataFrame], time_col: str, cutoff: Cutoff
) -> Optional[pd.DataFrame]:
    if cutoff is None or df is None or df.empty:
        return df
    if time_col not in df.columns:
        raise ValueError(
            f"Cannot apply as_of: column '{time_col}' is missing, so rows cannot be placed in time."
        )
    t = pd.to_numeric(df[time_col], errors="coerce")
    limit = df["learner_id"].map(cutoff) if isinstance(cutoff, dict) else cutoff
    # NaN time or NaN limit (learner not in mapping) compares False -> row is dropped
    return df[(t < limit).to_numpy()]


def truncate_frames(
    events: pd.DataFrame,
    quizzes: pd.DataFrame,
    completions: Optional[pd.DataFrame] = None,
    as_of=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Cutoff]:
    """Return (events, quizzes, completions, resolved_cutoff) as known just before `as_of`."""
    cutoff = resolve_cutoff(as_of, events)
    return (
        _before(events, EVENT_TIME_COL, cutoff),
        _before(quizzes, QUIZ_TIME_COL, cutoff),
        _before(completions, COMPLETION_TIME_COL, cutoff),
        cutoff,
    )
