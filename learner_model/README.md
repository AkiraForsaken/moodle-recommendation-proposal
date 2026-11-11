# Learner-model research pipeline

The current implementation builds a population-separated learner model with
knowledge states, skill summaries, explainable weak-area screening, Moodle
behavior and engagement summaries, and learner profiles.

## Build

From the repository root, use an environment with the packages in
`requirements.txt` installed:

```bash
python -m learner_model.builder
```

Equivalent entry point:

```bash
python learner-model.py
```

The default build reads English benchmark AKT states, the concept map, versioned
KC mappings, and Moodle collector exports. If Moodle-specific KT states exist,
they are joined only to Moodle learner IDs; the English benchmark model remains
a separate population. The builder writes CSV components, a build manifest, and a validation report under
`learner_model/artifacts/complete_learner_model/`.

Use `--data-dir`, `--knowledge-states`, `--concept-map`, `--mappings-dir`, and
`--output-dir` to select inputs and outputs. `--as-of` sets both population
cutoffs; `--knowledge-as-of` and `--moodle-as-of` override them independently.
Cutoffs accept Unix seconds or timezone-aware ISO datetimes. `--generated-at`
pins the generation timestamp for reproducible output.

See [the learner-model data contract](learner_model/docs/learner_model_data_contract.md)
and [component definitions](learner_model/docs/component_definitions.md) for
semantics, boundaries, and output schemas.

## Current scope

- English benchmark profiles remain separate from Moodle profiles; matching
  identifier strings are not treated as proof of identity. Moodle knowledge is
  included only when produced by the Moodle-specific training/inference path.
- `predicted_correctness` is a question-answering prediction, not a mastery
  probability.
- Moodle empty-history predictions are explicitly cold-start model priors, not
  individual evidence or mastery.
- Topic-level recommendation/API integrations that depended on the retired
  single-CSV model are deferred until they are designed for these component
  outputs and their population boundaries.

The old single-CSV learner model, quiz-level BKT/score-band mastery path, API
bridge, and recommender implementation have been retired. The current component
outputs are not wired to the retired Moodle-facing integration.
