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

## Moodle English-course data

The Moodle collector can export graded item responses to
`data/question_answer_events.csv`. Before using the export, create
`data/moodle_question_kc_map.csv` with one reviewed row per Moodle question-bank
entry and these columns:

```text
question_bank_entry_id,kc_id,kc_dimension,mapping_version,mapping_source
```

The stable question-bank entry ID is distinct from a quiz slot. Unmapped graded
responses are counted in `data/question_answer_mapping_report.csv` and excluded
from the KT CSV. `correct` is 1 only for full credit; the raw Moodle fraction is
preserved. The collector also exports active enrolled users in
`data/learner_roster.csv`, including users with no question responses.

Train a separate model from the Moodle cohort, using the existing pipeline and
the curated taxonomy. The Moodle maps, processed sequences, checkpoints,
selection metadata, and knowledge-state outputs use separate paths; benchmark
artifacts are not overwritten.

```bash
python knowledge_tracing/prepare_data.py \
  --dataset-name moodle_kt \
  --input data/question_answer_events.csv \
  --question-mapping data/moodle_question_kc_map.csv
python knowledge_tracing/init_pykt_dataset.py \
  --dataset-name moodle_kt \
  --input data/processed/moodle_kt/pykt_formatted_sequences.csv \
  --output-dir data/processed/pykt_moodle
python knowledge_tracing/train.py --dataset-name moodle_kt --models akt --seeds 42 43 44
python knowledge_tracing/evaluate.py --dataset-name moodle_kt --models akt
python knowledge_tracing/export_states.py --dataset-name moodle_kt
python -m learner_model.builder
```

Moodle KT requires at least three learners with answer events for distinct
train, validation, and test folds. The ability to estimate trustworthy
generalization metrics depends on a substantially larger cohort and sufficient
outcome variation in each split. A roster-only learner receives the model's
empty-history prediction per KC, marked `cold_start`; it is a population prior,
not personal evidence or mastery. Profiles keep benchmark learners and Moodle
learners in separate source populations.
