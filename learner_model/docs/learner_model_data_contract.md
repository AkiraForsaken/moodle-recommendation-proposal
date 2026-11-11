# Complete learner-model data contract

## Population boundaries

The initial build produces separate profiles for:

- `english_learning_events`: the English benchmark knowledge-state population.
- `moodle_collector`: learners observed in the local Moodle collector files.

The same identifier text in these populations does not imply the same person.
The profile key therefore includes `source_population`; this build never joins
the populations. Moodle behavior, engagement, and history are unavailable in
English profiles. English KC knowledge, skills, and weak areas are unavailable
in Moodle profiles.

When Moodle-specific KT states are available, they are joined to Moodle
behavior, engagement, and history using Moodle learner IDs only. The
`learner_roster.csv` export includes active enrolled users with no activity so
new learners can still receive profiles. Empty-history Moodle predictions are
marked `cold_start` and are model priors, not personal evidence, observed
performance, or mastery. Without a Moodle-trained KT model, Moodle knowledge
remains `unavailable`.

## Knowledge semantics and cutoff

`learner_kc_state.csv` retains the Milestone 3 source rows and provenance.
`observed_accuracy` is direct historical performance;
`predicted_correctness` is an estimated probability of answering representative
questions correctly. It is not a mastery probability.

### Cutoffs

The two populations are on different time axes (the English benchmark is
synthetic 2023 data; the Moodle collector is current), so a single instant
rarely fits both. `--as-of` is the default for both; `--knowledge-as-of` and
`--moodle-as-of` override it per population. A cutoff must be Unix seconds or a
timezone-aware ISO datetime; naive strings are rejected because the two
loaders would otherwise interpret them differently (UTC vs collector-local).

English KC states: the Milestone 3 predictions were computed from the events
M3 was allowed to see. That bound is the `as_of` column of the state file
(M3 `--as-of` run) or, for a full-history run, the latest `last_attempt_at`.
If it lies after the requested cutoff the build fails, because the predictions
would embed future events; re-run Milestone 3 with `--as-of`. The wall-clock
`inference_timestamp` is provenance only and is not compared to the cutoff.
Cold-start rows (no attempts, therefore no timestamp) are retained at any
cutoff. Only rows that report attempts but lack a timestamp are excluded, and
their count is recorded in the manifest. KC cutoffs are inclusive (M3
convention). Moodle collector frames use the existing strict-before-cutoff
behavior.

## Skills

`learner_model/mappings/kc_topic_map.csv` is the versioned source of truth for
KC-to-skill and KC-to-topic relationships. The builder loads and validates an
existing mapping and never overwrites it; a draft is generated only when it is
absent. Bump `mapping_version` when the mapping is reviewed or changed.

## Outputs

- `learner_profile.csv`: one profile per source population, learner, and build
  snapshot; component availability is explicit.
- `learner_kc_state.csv`: detailed learner-by-KC states, with
  `source_population` separating English benchmark and Moodle-trained states.
- `learner_skill_summary.csv`: aggregates by mapped skill and KC dimension.
- `learner_dimension_summary.csv`: traceable dimension-level knowledge metrics.
- `learner_behavior.csv`, `learner_engagement.csv`, and `learner_history.csv`:
  Moodle-only summaries.
- `weak_areas.csv`: KC-level evidence/prediction screening for each population.
- `mappings/skill_taxonomy.csv` and `mappings/kc_topic_map.csv`: draft mapping
  artifacts.
- `build_manifest.json` and `validation_report.md`: provenance and validation.

The original Milestone 3 state file is read-only input. The builder writes its
component outputs to the configured output directory.

## Rebuild

From the repository root:

```bash
python -m learner_model.builder
```

`python learner-model.py` is an equivalent CLI entry point.

Use `--data-dir`, `--knowledge-states`, `--concept-map`, `--mappings-dir`,
`--output-dir`, `--as-of` / `--knowledge-as-of` / `--moodle-as-of` to select
inputs, outputs, and snapshot cutoffs. `--generated-at` pins the generation
timestamp so two builds of identical inputs are byte-identical (without it,
`generated_at` and no-cutoff recency values use the current time).
The builder loads Moodle knowledge states from
`knowledge_tracing/artifacts/moodle_kt/knowledge_states/learner_skill_state.csv`
when that file exists; override it with `--moodle-knowledge-states`.

## Component statuses in `learner_profile.csv`

Each `*_status` reports the component's own status: `available`, `partial`
(e.g. engagement without completion tracking), `cold_start_only` (knowledge
with no estimated KC), `no_observations`, or `unavailable`.
`profile_completeness` lists only components with a usable status.
