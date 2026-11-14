# Knowledge engine

Produces the learner-by-KC knowledge states of a Moodle course from that
course's own graded answers. It is the only source of knowledge states for
`learner_model`.

## Phases

| Phase          | Estimator in charge                               | Condition                                                           |
| -------------- | ------------------------------------------------- | ------------------------------------------------------------------- |
| 1 Cold start   | Item-level BKT with fixed priors                  | Default. Works with zero answers.                                   |
| 2 Training     | BKT (a deep model is trained but not trusted yet) | Readiness thresholds met; the trained model did not beat BKT.       |
| 3 Personalised | Deep KT model (AKT)                               | The trained model beat BKT on held-out learners of the same course. |

The decision is stored in `artifacts/active_model.json`. Retraining repeats the
whole of Phase 2, including the gate.

## Commands

Run from the repository root.

```bash
python -m knowledge_engine status   [--data-dir data]
python -m knowledge_engine estimate [--data-dir data] [--as-of 2026-12-01T00:00:00Z]
python -m knowledge_engine train    [--data-dir data] [--epochs N] [--force]
python -m knowledge_engine simulate [--output-dir data/simulated] [--course-dir data] [--learners 300]
```

`estimate` and `status` need only pandas. `train`, and `estimate` once a deep
model is promoted, also need `requirements-kt.txt`.

## Inputs (collector format, in `--data-dir`)

| File                         | Required    | Use                                                    |
| ---------------------------- | ----------- | ------------------------------------------------------ |
| `kc_catalog.csv`             | yes         | The course's Knowledge Components                      |
| `learner_roster.csv`         | no          | Enrolled learners, so new learners get cold-start rows |
| `question_answer_events.csv` | no          | Graded answers; absent or empty means no evidence yet  |
| `moodle_question_kc_map.csv` | for `train` | Question to KC mapping and its version                 |

## Output

`artifacts/knowledge_states/learner_kc_state.csv` (one row per learner and KC;
columns and meaning in `schemas.py`) and `knowledge_state_manifest.json`.

- `predicted_correctness`: probability of answering the next question on the
  KC correctly. Produced by every estimator; the gate compares it.
- `p_mastery`: BKT's probability that the KC is known. Null for the deep model.
- `state_status = cold_start`: no answer on that KC. The row is a prior, not
  evidence about the learner.

## Safeguards

- A promoted model is used only for the population and question mapping it was
  trained on. Otherwise `estimate` falls back to BKT and records why. A model
  trained on simulated learners is therefore never applied to real ones, and a
  changed question bank requires a retrain.
- The gate scores BKT on exactly the prediction targets of the deep model's
  held-out test split and requires a margin (`GATE_MIN_AUC_MARGIN`).
- Simulator output carries `simulation_manifest.json`; everything derived from
  it is labelled `source_population = simulated`. The simulator refuses to
  write into a directory holding collector data.
- Simulated learners can also be injected into Moodle itself
  (`python -m moodle_setup simulate-learners`, see
  [moodle_setup/README.md](../moodle_setup/README.md)). They reach the engine
  through the collector, which writes the same manifest when an export holds
  `SIM-` accounts. `train` reads the generator's ceiling AUC from
  `simulation_plan_manifest.json` in that case.

## Limits to state when reporting results

- BKT parameters are fixed configuration values, not fitted to the course.
- Readiness thresholds and the gate margin are configuration choices.
- The simulator's answer model is not BKT, so the gate is not rigged, but
  results on simulated learners show only that the pipeline works. They are
  not evidence that the estimates are valid for real learners.
