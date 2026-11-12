# Learner-model data contract

## Inputs

| Input | Source | Required |
|---|---|---|
| `learner_roster.csv` | Moodle collector | no (recommended: lets new learners get profiles) |
| `activity_catalog.csv`, `learning_events.csv`, `quiz_attempts.csv`, `course_modules_completion.csv` | Moodle collector | no (a missing file is an empty table) |
| `kc_catalog.csv`, `kc_topic_map.csv` | Moodle collector (question categories and quiz usage) | no (needed for KC names and skills) |
| `learner_kc_state.csv` | `knowledge_engine` | no (knowledge is `unavailable` without it) |

The learner model reads the knowledge-state file as a contract. It does not
import `knowledge_engine` or `knowledge_tracing`, and does not know which
estimator produced the rows. No benchmark file is read.

## One population per build

Every output row carries `source_population`:

- `moodle_collector`: real learners from the Moodle collector.
- `simulated`: the data directory contains `simulation_manifest.json`.

A build fails if the knowledge states and the data directory describe
different populations, so simulated learners are never joined to real Moodle
activity. Simulated results demonstrate the pipeline only.

## Learners

A profile is built for every learner who is enrolled at the cutoff (roster),
appears in any collector export before the cutoff, or has a knowledge state.

## Knowledge semantics

`learner_kc_state.csv` keeps one row per learner and KC with its provenance.

- `observed_accuracy`, `attempt_count`: direct evidence from the learner's answers.
- `predicted_correctness`: estimated probability of answering the next question
  on the KC correctly. It is not a mastery probability.
- `p_mastery`: BKT's probability that the KC is known (Phase 1 only).
- `state_status`: `estimated`, or `cold_start` when the learner has no answer on
  the KC. A cold-start value is a prior, not evidence and not zero knowledge.
- `estimator_phase`, `model_name`: 1 / BKT with fixed priors, or 3 / the deep
  model that passed the promotion gate.

## Cutoff

`--as-of` must be Unix seconds or a timezone-aware ISO datetime; naive strings
are rejected. Knowledge states must have been computed with a cutoff at or
before it (their `as_of` column, or the latest `last_attempt_at` for a
full-history run); otherwise the build fails because the estimates would embed
future answers. Re-run `python -m knowledge_engine estimate --as-of ...`.
Cold-start rows have no timestamp and are valid at any cutoff. KC cutoffs are
inclusive; collector frames use strictly-before-cutoff.

## Outputs

- `learner_profile.csv`: one row per learner; component statuses,
  `knowledge_phase`, `knowledge_model`, `profile_completeness`.
- `learner_kc_state.csv`: learner-by-KC states with KC names.
- `learner_skill_summary.csv`, `learner_dimension_summary.csv`: aggregates that
  trace back to KC rows.
- `weak_areas.csv`: one row per learner and KC with a concern status and reason.
- `learner_behavior.csv`, `learner_engagement.csv`, `learner_history.csv`.
- `kc_skill_map.csv`: the KC-to-skill mapping used by this build.
- `build_manifest.json`, `validation_report.md`: provenance and checks. The
  builder stops before writing anything if a check fails.

## Component statuses in `learner_profile.csv`

| Status | Meaning |
|---|---|
| `available` | The component has individual evidence for this learner. |
| `partial` | Available with a known gap (for example engagement without completion tracking). |
| `cold_start_only` | Knowledge, skills or weak areas built from priors only. |
| `no_observations` | The collector has no activity for this learner yet. |
| `unavailable` | The component could not be produced (no knowledge-state file). |

`profile_completeness` lists the components with a usable status
(`available`, `partial`, `cold_start_only`).
