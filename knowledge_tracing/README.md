# Knowledge tracing toolkit

This package is the offline deep-KT toolkit (pyKT). It serves two separate
purposes that must not be confused:

| Dataset name | Purpose                                                                                                                                                                     | Feeds the Moodle learner model?                                                            |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `english_kt` | **Benchmark experiment.** Compares DKT / AKT / SAKT against baselines on the public English Learning Events dataset. Its results justify the choice of model in the thesis. | **No.** Its checkpoints, KC IDs and `learner_skill_state.csv` are experiment outputs only. |
| `moodle_kt`  | **Course model.** The same pipeline trained on a Moodle course's own answers (or on simulated answers for that course).                                                     | Yes, but only through `knowledge_engine`, after its promotion gate.                        |

A KT model depends on the items and KCs it was trained on, so a benchmark
checkpoint cannot score a Moodle course, and benchmark learners are not Moodle
learners. Nothing under `learner_model/` or `knowledge_engine/` reads the
benchmark files.

# Benchmark knowledge-state extraction

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
the KC for the English benchmark. For Moodle, it averages only questions
observed in the training fold; untrained question embeddings are not used to
produce KC estimates. The export fails if any mapped KC has no training-exposed
question. `predicted_correctness_10q` preserves a deterministic 10-question
sensitivity estimate from the same supported questions. The Moodle preparation
manifest records mapping version, file hash, and question coverage, including
mapped questions without training exposure and validation/test questions with
no training exposure. Review these coverage counts before interpreting
validation or test metrics: pyKT scores held-out events even when that
question's embedding had no training evidence. Neither field is a mastery
probability, and comparisons with historical accuracy in the report are
descriptive rather than calibration or error estimates.

## Selected checkpoint

AKT seed 43 is selected by validation AUC across the saved architecture/seed
runs. It reached validation AUC `0.86697` at epoch 25. Its held-out test metrics
are recorded for generalization reporting only and were not used for selection.

## Course (Moodle) data

Do not run these scripts by hand for a course. `knowledge_engine` drives them,
checks that enough answers exist, and decides whether the trained model may
replace the Phase 1 estimator:

```bash
python -m knowledge_engine status     # readiness and the active estimator
python -m knowledge_engine train      # prepare -> train -> evaluate -> gate
python -m knowledge_engine estimate   # learner-by-KC states for the learner model
```

Inputs come from the Moodle collector: `question_answer_events.csv`,
`learner_roster.csv`, and the question-to-KC mapping
`moodle_question_kc_map.csv`, which the collector derives from the course's
question categories. `correct` is 1 only for full credit; the raw Moodle
fraction is preserved. Unmapped graded responses are counted in
`question_answer_mapping_report.csv` and excluded.

The course maps, processed sequences, checkpoints and selection metadata use
separate paths (`data/processed/moodle_kt`, `artifacts/moodle_kt`,
`configs/moodle_kt_selected_model.json`); benchmark artifacts are not
overwritten. Training needs at least three learners with answers for distinct
train, validation and test folds, and far more for trustworthy metrics; the
readiness thresholds live in `knowledge_engine/config.py`.
