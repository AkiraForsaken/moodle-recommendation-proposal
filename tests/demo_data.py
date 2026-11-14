"""
Synthetic collector output (activity_catalog / learning_events / quiz_attempts).

Used to provide deterministic collector CSVs for learner-model tests and by
`python -m tests.demo_data <dir>` for local experiments.

Column names follow the Moodle collector export contract.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOPICS = ["Variables and Data Types", "Conditions", "Loops", "Functions"]
BASE_UNIX = 1_790_000_000  # arbitrary fixed point in time


def make_catalog() -> pd.DataFrame:
    rows, cmid, seq = [], 100, 1
    # Section 0 is the course-wide block, not a learning topic.
    rows.append(dict(course_module_id=cmid, activity_type="forum", activity_id=1,
                     activity_name="Announcements", topic="General", sequence=0,
                     completion_tracking=0, visible=1, section_number=0))
    for s, topic in enumerate(TOPICS, start=1):
        for kind, label in (("page", "Reading"), ("lesson", "Lesson"), ("quiz", "Quiz")):
            cmid += 1
            rows.append(dict(course_module_id=cmid, activity_type=kind, activity_id=cmid - 100 + s * 10,
                             activity_name=f"{topic}: {label}", topic=topic, sequence=seq,
                             completion_tracking=0, visible=1, section_number=s))
            seq += 1
    return pd.DataFrame(rows)


def make_cohort(catalog: pd.DataFrame, seed: int = 7):
    """Create 11 learners with varied activity and quiz-score histories."""
    rng = np.random.default_rng(seed)
    learning = catalog[catalog["section_number"] > 0]
    events, quizzes, log_id = [], 0, 0
    profiles = {  # learner -> {topic: best quiz score or None (viewed only)}
        1: dict(zip(TOPICS, [0.95, 0.9, 0.85, 0.9])),    # mastered everything
        2: dict(zip(TOPICS, [0.95, 0.4, None, None])),   # low on topic 2
        3: dict(zip(TOPICS, [0.7, None, None, None])),   # medium on topic 1
        4: dict(zip(TOPICS, [0.9, 0.7, 0.3, None])),
        5: dict(zip(TOPICS, [0.85, 0.65, 0.9, 0.5])),
        6: dict(zip(TOPICS, [0.5, 0.5, None, None])),
        7: dict(zip(TOPICS, [0.9, 0.95, 0.6, None])),
        8: dict(zip(TOPICS, [None, None, None, None])),  # viewed only: 'unknown'
        9: dict(zip(TOPICS, [0.75, 0.55, None, None])),
        10: dict(zip(TOPICS, [0.92, 0.88, 0.81, 0.83])),
        11: dict(zip(TOPICS, [0.35, None, None, None])),
    }
    attempts = []
    for lid, prof in profiles.items():
        t = BASE_UNIX + lid * 100_000
        session = lid * 1000
        for topic, score in prof.items():
            sub = learning[learning["topic"] == topic]
            if score is None and lid != 8:
                continue
            session += 1
            prev = t
            for r in sub.itertuples():
                if r.activity_type == "quiz":
                    continue
                for _ in range(int(rng.integers(2, 5))):
                    t += int(rng.integers(30, 400))
                    log_id += 1
                    events.append(dict(
                        log_id=log_id, learner_id=lid, course_module_id=r.course_module_id,
                        timestamp=pd.Timestamp(t, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                        timestamp_unix=t, event_category="content_view",
                        session_id=session, seconds_since_prev_event=t - prev, is_learning_event=1))
                    prev = t
            if score is not None:
                q = sub[sub["activity_type"] == "quiz"].iloc[0]
                n_att = 2 if score < 0.8 else 1
                for a in range(1, n_att + 1):
                    t += 300
                    s = score if a == n_att else max(score - 0.2, 0.0)
                    attempts.append(dict(
                        learner_id=lid, quiz_id=q.activity_id, quiz_name=q.activity_name,
                        attempt=a, normalized_score=round(s, 2), duration_seconds=240,
                        finish_time_unix=t))
                    log_id += 1
                    events.append(dict(
                        log_id=log_id, learner_id=lid, course_module_id=q.course_module_id,
                        timestamp=pd.Timestamp(t, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                        timestamp_unix=t, event_category="quiz_attempt",
                        session_id=session, seconds_since_prev_event=300, is_learning_event=1))
    return pd.DataFrame(events), pd.DataFrame(attempts)


def write_collector_csvs(out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = make_catalog()
    events, quizzes = make_cohort(catalog)
    catalog.to_csv(out / "activity_catalog.csv", index=False)
    events.to_csv(out / "learning_events.csv", index=False)
    quizzes.to_csv(out / "quiz_attempts.csv", index=False)
    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data_demo"
    print("wrote", write_collector_csvs(target))
