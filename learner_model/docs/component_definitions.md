# Component definitions

## Knowledge and skills

Knowledge keeps every learner-KC row from the knowledge engine with its
provenance. Dimension summaries average non-null predictions and observed
accuracies separately; cold-start counts stay explicit. Skill summaries
aggregate through the course's KC catalog and topic links
(see `mapping_documentation.md`).

## Weak areas

Weak-area screening is KC-level and keeps prediction and evidence separate.
Rules, first match wins:

1. At least `MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN` (3) direct attempts and observed
   accuracy below `LOW_OBSERVED_ACCURACY` -> `evidence_based_concern`.
2. Same attempts, accuracy not low -> `no_current_concern` (direct evidence
   overrides a low prediction).
3. No attempts and a Phase 1 estimate -> `insufficient_evidence`. The value is
   the fixed prior shared by every new learner, so it is never a concern.
4. Fewer than 3 attempts and predicted correctness below
   `LOW_PREDICTED_CORRECTNESS` -> `prediction_based_concern` (a model signal,
   not verified difficulty and not low mastery).
5. Otherwise -> `insufficient_evidence`.

A high prediction is never promoted to a mastery label, and a cold-start KC can
never be an evidence-based concern. `HIGH_PREDICTED_CORRECTNESS` is reserved for
future advancement/ranking use and is not used here. `weak_areas.csv` holds one
row per learner-KC (including non-concerns, for accounting): filter on
`concern_status`. Thresholds are auditable configuration choices, not
empirically optimized cutoffs.

## Behavior

- Session count uses collector `session_id` values; if unavailable, sessions
  are reconstructed using a 30-minute inactivity gap.
- Average session duration is the mean session end minus start timestamp.
- Average session interval is the mean gap between consecutive session starts.
- Learning activity counts collector rows flagged `is_learning_event=1`, or the
  documented learning event categories when that flag is absent.
- Practice counts question interactions and quiz attempts; review counts
  `quiz_review` events.
- Revisit rate is the fraction of the learner's topics that they came back to
  in a later session (learning events for the topic in two or more sessions).
  Several views within one visit (overview, lesson, quiz) count once.

These are interaction patterns, not measures of learning effectiveness.

## Engagement

Activity views and bounded event-gap time are summarized directly from collector
events; quiz duration is added from quiz attempts. Active sessions and
learning-event counts come from collector events. Session counts are null (unknown), not 0, when the collector recorded no
session identifiers. Completion counts and completion rate remain
null when completion tracking data is absent; the absence of tracking is not
interpreted as non-completion. The rate denominator is the number of unique
trackable activities with completion records for that learner; both measures
remain null when that learner has no such records. `course_progress` is
Moodle's notion of progress: completed activities divided by every activity in
the course that tracks completion, whether or not the learner opened it. A
quiz counts as complete only with a passing grade (complete_pass); a failed
quiz (complete_fail) is not complete. Time spent is an estimate
based on existing collector deltas and is not equivalent to measured study time.
Quiz durations are not double-counted as event-gap time.

## History

History stores a summary, not a duplicate event log. It includes first/last
activity timestamps, event and quiz counts, completed activity count, distinct
mapped topics, active calendar days, session count, and latest activity type.
Collector and quiz rows are cut off using the existing temporal helper before
the history summary is computed.

## Topic-level data

Topics come from Moodle course sections. KCs are linked to topics through the
quizzes that use their questions (`kc_topic_map.csv`), and skills are reported
per dimension within a topic. No separate topic mastery state is emitted.
