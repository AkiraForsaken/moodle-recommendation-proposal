# Knowledge-state extraction

The extractor loads the selected AKT checkpoint and produces one record per
learner and Knowledge Component. It preserves observed accuracy separately
from the model-estimated probability of answering representative questions
correctly; `predicted_correctness` is not a mastery probability. Assigned
learners with no valid answer history remain cold-start rows; the prediction
does not fabricate historical evidence.

## Run

From the repository root, use an environment with pandas, NumPy, PyTorch, and
`pykt-toolkit==0.0.38` installed:

```bash
python knowledge_tracing/export_states.py
```

By default, the command reads `data/raw/english_learning_events.csv`, scores
every question mapped to each KC, prints progress by stage and learner, and
writes the following files under `knowledge_tracing/artifacts/knowledge_states/`:

- `learner_skill_state.csv`
- `inference_manifest.json`
- `knowledge_state_validation_report.md`

Use `--input`, `--output-dir`, `--questions-per-kc`, `--batch-size`, `--device`,
or `--as-of` to override defaults. `--questions-per-kc` controls the
deterministic sensitivity estimate (default 10); it does not limit primary
scoring, which always uses all questions. `--as-of` is an inclusive,
timezone-aware ISO-8601 cutoff, such as `2023-11-01T00:00:00Z`. Without it, the
script uses all answer events in the input.

Inference verifies that the current question and concept maps match the hashes
recorded with the selected checkpoint, checks that their contiguous indices
match the trained vocabulary sizes, and fails if output validation detects
inconsistent evidence or incomplete learner-KC coverage. Each state includes
the learner's split, as-of cutoff, and an evidence tier: none (0 attempts), low
(1-2), moderate (3-5), or high (6 or more).

`predicted_correctness` averages probabilities across every question mapped to
the KC. `predicted_correctness_10q` preserves the old deterministic 10-question
estimate as a sensitivity check from those same scores. Neither field is a
mastery probability, and comparisons with historical accuracy in the report
are descriptive rather than calibration or error estimates.

## Selected checkpoint

AKT seed 43 is selected by validation AUC across the saved architecture/seed
runs. It reached validation AUC `0.86697` at epoch 25. Its held-out test metrics
are recorded for generalization reporting only and were not used for selection.
