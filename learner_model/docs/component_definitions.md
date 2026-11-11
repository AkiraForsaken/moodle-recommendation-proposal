# Milestone 4 component definitions

## Knowledge and skills

Knowledge preserves every eligible learner-KC source record and its model
provenance. English benchmark and Moodle-trained states retain separate source
populations; they are not merged or assumed comparable. Dimension summaries
average non-null predictions and observed accuracies separately; cold-start
counts remain explicit. Moodle learners with no answer history may have model
priors per KC, marked `cold_start` and not interpreted as individual evidence
or mastery. Skill summaries aggregate through the versioned KC-to-skill mapping
and report its review status.

## Weak areas

Weak-area screening is KC-level and keeps prediction and evidence separate.
Rules, first match wins:

1. At least `MIN_ATTEMPTS_FOR_EVIDENCE_CONCERN` (3) direct attempts and observed
   accuracy below `LOW_OBSERVED_ACCURACY` -> `evidence_based_concern`.
2. Same attempts, accuracy not low -> `no_current_concern` (direct evidence
   overrides a low prediction).
3. Fewer attempts and predicted correctness below `LOW_PREDICTED_CORRECTNESS`
   -> `prediction_based_concern` (a model signal, not verified difficulty and
   not low mastery).
4. Otherwise -> `insufficient_evidence`.

A high prediction is never promoted to a mastery label, and a cold-start KC can
never be an evidence-based concern. `HIGH_PREDICTED_CORRECTNESS` is reserved for
future advancement/ranking use and is not used here. `weak_areas.csv` holds one
row per learner-KC (including non-concerns, for accounting): filter on
`concern_status`. Defaults are auditable configuration choices, not empirically
optimized cutoffs; with the current benchmark most KC states have fewer than
three attempts, so `insufficient_evidence` dominates.

## Behavior

- Session count uses collector `session_id` values; if unavailable, sessions
  are reconstructed using a 30-minute inactivity gap.
- Average session duration is the mean session end minus start timestamp.
- Average session interval is the mean gap between consecutive session starts.
- Learning activity counts collector rows flagged `is_learning_event=1`, or the
  documented learning event categories when that flag is absent.
- Practice counts question interactions and quiz attempts; review counts
  `quiz_review` events.
- Revisit rate is the fraction of topics with more than one `content_view`
  among topics with mapped collector events.

These are interaction patterns, not measures of learning effectiveness.

## Engagement

Activity views and bounded event-gap time are summarized directly from collector
events; quiz duration is added from quiz attempts. Active sessions and
learning-event counts come from collector events. Session counts are null (unknown), not 0, when the collector recorded no
session identifiers. Completion counts and completion rate remain
null when completion tracking data is absent; the absence of tracking is not
interpreted as non-completion. The rate denominator is the number of unique
trackable activities with completion records for that learner; both measures
remain null when that learner has no such records. Time spent is an estimate
based on existing collector deltas and is not equivalent to measured study time.
Quiz durations are not double-counted as event-gap time.

## History

History stores a summary, not a duplicate event log. It includes first/last
activity timestamps, event and quiz counts, completed activity count, distinct
mapped topics, active calendar days, session count, and latest activity type.
Collector and quiz rows are cut off using the existing temporal helper before
the history summary is computed.

## Topic-level data

This pipeline does not emit a Moodle topic mastery state or combine English KC
predictions with Moodle learner records. Topic-level recommendation logic is
deferred until its population and mapping assumptions are implemented and
validated.
