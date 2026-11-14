# Personalized Microlearning Recommendation Using Moodle Learning Analytics

Builds a learner model (knowledge, skills, weak areas, behavior, engagement,
history) for the learners of a Moodle course, from that course's own data.

## Two distinct pipeline

|          | Benchmark experiment                      | Course pipeline                                         |
| -------- | ----------------------------------------- | ------------------------------------------------------- |
| Question | Which knowledge-tracing model works best? | What does each of our learners know?                    |
| Data     | Public English Learning Events dataset    | The Moodle course (or simulated learners for it)        |
| Code     | `knowledge_tracing/` with `english_kt`    | `collector/` -> `knowledge_engine/` -> `learner_model/` |
| Result   | Model comparison for the thesis           | Learner profiles                                        |

The benchmark justifies the choice of model. Its weights, KC IDs and learner
states are never used for Moodle learners.

## Course pipeline

```
Moodle DB --collector--> data/*.csv --knowledge_engine--> learner_kc_state.csv --learner_model--> profiles
```

```bash
pip install -r requirements.txt
python moodle-collector.py             # needs MOODLE_* settings, see collector/config.py
python -m knowledge_engine estimate
python -m learner_model.builder
pytest tests -q
```

Knowledge estimation moves through three phases:

1. **Cold start.** Item-level BKT with fixed priors. Every enrolled learner has
   a profile from day one; unanswered KCs hold a prior, not a score.
2. **Training.** Once enough answers exist (`python -m knowledge_engine status`),
   `python -m knowledge_engine train` trains a deep KT model on the course's
   answers. Needs `pip install -r requirements-kt.txt`.
3. **Personalised.** The deep model takes over only if it beats BKT on
   held-out learners of the course. Otherwise BKT stays in charge.

To exercise phases 2 and 3 before real data exists:

```bash
python -m knowledge_engine simulate --output-dir data/simulated
python -m knowledge_engine train    --data-dir data/simulated
python -m knowledge_engine estimate --data-dir data/simulated
python -m learner_model.builder     --data-dir data/simulated
```

Simulated results are labelled `simulated` throughout and show only that the
pipeline works.

## Preparing the Moodle course

The English test course and simulated learners for the local Moodle instance
are fully scripted. Run `python -m moodle_setup build-course`, then follow the
runbook in [moodle_setup/README.md](moodle_setup/README.md). It covers all
three phases on data that passes through Moodle itself. The course has one
short lesson per KC and activity completion on every activity (pages on view,
quizzes on a 70 % pass grade).

To prepare another course by hand, follow two conventions so the collector can
export the course's Knowledge Components and their resources:

- one question category per KC with an **ID number** (the `kc_id`), placed
  under a parent category named after its dimension (grammar, vocabulary, ...);
- optionally, a learning resource per KC whose **activity ID number** is the
  `kc_id`, so recommendations can point to it (`kc_activity_map.csv`).

Details:
[learner_model/docs/mapping_documentation.md](learner_model/docs/mapping_documentation.md).

## Packages

| Package              | Role                                                                  | Docs                                        |
| -------------------- | --------------------------------------------------------------------- | ------------------------------------------- |
| `collector/`         | Moodle database -> CSV exports, including the KC catalog and mappings | module docstring in `collector/__init__.py` |
| `knowledge_engine/`  | Learner-by-KC knowledge states, phases and promotion gate, simulator  | [README](knowledge_engine/README.md)        |
| `learner_model/`     | Learner profiles and their components                                 | [README](learner_model/README.md)           |
| `knowledge_tracing/` | Deep-KT toolkit: benchmark experiment and course-model training       | [README](knowledge_tracing/README.md)       |
| `moodle_setup/`      | Scripted English course and simulated learners in local Moodle        | [README](moodle_setup/README.md)            |

## Not built yet

The current state and the next steps are summarised in
[docs/HANDOFF.md](docs/HANDOFF.md).

Personalised recommendation, the student-facing Moodle block and the
profile view for teachers and administrators will be rebuilt on the learner
profiles above. The earlier topic-level recommender, API and Moodle plugins
were removed; they remain in git history (main repository before this change,
Moodle fork commit `2dfbc4dd519`).
