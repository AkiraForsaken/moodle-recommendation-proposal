# Learner model

Builds one learner profile per enrolled learner of a Moodle course, from that
course's own data: knowledge states from `knowledge_engine`, and behavior,
engagement and history from the Moodle collector. No benchmark dataset is an
input.

## Build

From the repository root:

```bash
python moodle-collector.py              # Moodle -> data/*.csv
python -m knowledge_engine estimate     # data/ -> learner-by-KC knowledge states
python -m learner_model.builder         # -> learner_model/artifacts/learner_model/
```

Options: `--data-dir`, `--knowledge-states`, `--output-dir`, `--as-of` (Unix
seconds or a timezone-aware ISO datetime) and `--generated-at` (pin the
generation time for byte-identical rebuilds).

The knowledge-state file is optional. Without it, profiles are still built and
the knowledge, skills and weak-area components are `unavailable`.

## What a new learner gets

A learner who is enrolled but has done nothing still has a profile:

- knowledge: one row per course KC, `cold_start` (the shared prior),
  `knowledge_status = cold_start_only`
- behavior, engagement, history: `no_observations`
- weak areas: `insufficient_evidence` for every KC, never a fabricated concern

See [the data contract](docs/learner_model_data_contract.md),
[component definitions](docs/component_definitions.md) and
[KC and skill mapping](docs/mapping_documentation.md).

## Not in this package

Recommendation and the Moodle UI are not built yet. They will read the
profile outputs above; this package does not depend on them.
