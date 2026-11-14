# moodle_setup: English course and simulated learners in local Moodle

Builds the **English Microlearning** course (`ENG-MICRO`) in the local
`moodle-docker` instance, and fills it with simulated learners whose quiz
attempts, page views and activity completion are stored in Moodle itself. The
course pipeline (`collector` -> `knowledge_engine` -> `learner_model`) can then
be tested on data that went through the whole chain, Moodle included.

Everything is scripted. Python prepares the content and the learner plan.
PHP scripts run inside the `webserver` container and use Moodle's own APIs. The
Moodle checkout is not modified: scripts and JSON inputs are copied to
`/tmp/moodle_setup` in the container with `docker cp`.

## What gets built

| Moodle object | Contents                                                                                                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Course        | `English Microlearning` (`ENG-MICRO`), topics format, completion tracking on, start date 150 days ago                                                                                            |
| Sections      | `Unit 1: Everyday Routines` ... `Unit 8`, one per unit                                                                                                                                           |
| Question bank | One course `qbank` with categories `top > Grammar / Vocabulary / Reading / Writing > one per KC`. Each KC category has **ID number = `kc_id`**, name = KC name, description = learning objective |
| Questions     | 390 single-answer multiple-choice questions (4 or 5 per KC), options shuffled                                                                                                                    |
| KC lessons    | One page per KC, `Lesson: <KC name>`, with **course-module ID number = `kc_id`**: explanation, examples, common mistake, self-check items with hidden answers, optional video                    |
| Per unit      | Overview page, then per dimension: that dimension's KC lessons followed by its practice quiz (`Unit N Grammar practice`, ...), immediate feedback, unlimited attempts                            |

That is 96 KCs, 96 lessons, 32 quizzes and 8 overview pages per course.

**Activity completion** (all set by the script):

| Activity                | Complete when                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| Overview and KC lessons | the learner views the page                                                                      |
| Practice quiz           | the learner reaches the **pass mark, 7/10 (70 %)**. Below it Moodle records `complete_fail` (3) |

The collector derives everything else from this layout at run time, with no
hand-maintained mapping file (see `collector/collect_course_mappings.py`):

- `kc_catalog.csv` and `moodle_question_kc_map.csv` from the question
  categories;
- `kc_topic_map.csv` from the quizzes that use each KC's questions;
- `kc_activity_map.csv` from both conventions: `role=learn` for the activity
  whose ID number is the `kc_id` (the lesson), `role=practice` for the quizzes
  holding the KC's questions.

A recommender can therefore go from a weak KC or a failed quiz straight to the
lesson to revise.

## Course content

Edit the content in these files, not in Moodle:

```
course/kc_catalog.csv        kc_id, unit, unit_name, dimension, kc_name, learning_objective
course/questions/unitN.txt   kc_id | stem | correct option | wrong | wrong | wrong
course/lessons/unitN.md      one lesson per KC (format below)
```

The correct option is always written first, and Moodle shuffles the options in
each attempt. Reading questions carry their short passage in the stem. Writing
questions are of the "choose the correctly written sentence / best revision"
kind, because free writing cannot be auto-graded for knowledge tracing.

Lesson format (`**bold**` and `~~wrong form~~` are supported):

```
## KC001
video: https://...            optional; empty until a video exists
### Explanation
One or more short paragraphs.
### Examples
- one example per bullet
### Common mistake
- one mistake per bullet
### Try it
- A self-check question => its answer (hidden until the learner opens it)
```

To turn a lesson into a video lesson later, put the video's URL in `video:`.
A direct `.mp4` / `.webm` link is embedded in the page; any other URL is shown
as a link. Then rebuild the course.

> **Review needed.** The questions and the lessons were drafted by an AI
> assistant. Check them (correct key, one defensible answer, level, accuracy of
> the explanations) before using any result in the thesis.

`python -m moodle_setup check-content` validates the content without touching
Moodle. It checks that every KC has at least 4 questions with 4 distinct
options, that no question is duplicated, that every KC has exactly one
complete lesson with at least 2 self-check items, and that no self-check item
repeats a quiz question (a lesson must not show quiz answers). `build-course`
runs the same checks first.

## Scripted vs. Moodle UI only

Everything this course needs is created by the scripts, including activity
completion, pass marks and the lesson ID numbers. What the scripts deliberately
do not do:

| Setting                                                        | Why not scripted                                                                                                                                   | Where to change it                                                        |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Site-wide **Enable completion tracking**                       | A site setting affecting every course. `build-course` checks it and stops with a message if it is off; it never changes it.                        | _Site administration > Advanced features_                                 |
| **Course completion** criteria (when the whole course is done) | Scriptable, but the learner model does not use it, so it is left unset.                                                                            | _Course > Course completion_                                              |
| Uploaded **video files**                                       | Videos do not exist yet. A URL in a lesson's `video:` field is enough; uploading files into Moodle would need the file API and is not implemented. | Lesson `video:` field (URL), or edit the page in Moodle (lost on rebuild) |

Anything changed by hand in the Moodle UI is lost on `build-course --rebuild`.
Put lasting changes into `course/` and the build script instead.

## Prerequisites

- The `moodle-docker` containers are running. The scripts expect the web
  container to be `moodle-docker-webserver-1`; set `MOODLE_WEBSERVER_CONTAINER`
  if yours is named differently (`docker ps`).
- The Python dependencies are installed: `pip install -r requirements.txt`
  (the collector needs its `mariadb` package), plus
  `pip install -r requirements-kt.txt` for `knowledge_engine train`.
- Your usual `MOODLE_*` collector settings are exported (see `collector/config.py`).
- **Back up the Moodle database first.** Injection writes to it. The database
  container runs MariaDB 11.4:

  ```bash
  # Uses the credentials Moodle itself connects with (the db container's own
  # MYSQL_PASSWORD may not match them).
  P=$(docker exec moodle-docker-webserver-1 printenv MOODLE_DOCKER_DBPASS)
  docker exec -e P="$P" moodle-docker-db-1 sh -c 'mariadb-dump -u moodle -p"$P" moodle' > moodle_backup.sql
  # restore: docker exec -i -e P="$P" moodle-docker-db-1 sh -c 'mariadb -u moodle -p"$P" moodle' < moodle_backup.sql
  ```

## Runbook

```bash
# 0. Build the course once. It prints the course id.
python -m moodle_setup build-course
export MOODLE_COURSE_ID=<id>

# 1. Phase 1 on an empty course: every enrolled learner gets a cold-start profile.
python moodle-collector.py
python -m knowledge_engine estimate
python -m learner_model.builder

# 2. Phase 1 with a few learners: not enough data for a deep model -> BKT.
python -m moodle_setup simulate-learners --learners 20 --seed 7 --plan-out data/plan_20.json
python moodle-collector.py
python -m knowledge_engine status                 # expect: not ready
python -m knowledge_engine estimate
python -m learner_model.builder

# 3. Phases 2-3: the full cohort. Same seed: learners 1-20 are kept, 21-300 added.
python -m moodle_setup simulate-learners --learners 300 --seed 7 --plan-out data/plan_300.json
python moodle-collector.py
python -m knowledge_engine status                 # expect: ready
python -m knowledge_engine train                  # tuned epochs, seeds 42 43 44, then the gate
python -m knowledge_engine estimate
python -m learner_model.builder

# 4. Clean up when done.
python -m moodle_setup remove-simulated
python moodle-collector.py                        # export without the simulated learners
```

Injecting 20 learners takes under a minute; 300 learners (about 2,300 attempts,
28,000 answers, 2,000 page views) take several minutes. Progress is printed
every 10 learners. If the run is interrupted, run the same command again:
learners whose attempts already exist are skipped, and an unfinished attempt is
deleted and redone.

**Keep the same `--seed`** when you extend a cohort. A plan for 300 learners
starts with exactly the 20 learners of the plan for 20 learners with the same
seed, which is what makes resuming and extending safe. With a different seed,
the existing learners' remaining attempts would come from a different learner.

`simulate-learners --dry-run --plan-out plan.json` creates the plan and
reports its size, page behaviour and ceiling AUC without writing anything to
Moodle.

### After changing the course

Changing the content (questions, lessons, catalog) means rebuilding the
course, which deletes every attempt in it:

```bash
python -m moodle_setup remove-simulated
python -m moodle_setup build-course --rebuild     # new course id: export MOODLE_COURSE_ID again
# then steps 1-3 of the runbook, including a retrain
```

The rebuild creates new question-bank entries, so the question-to-KC
`mapping_version` changes. A previously promoted deep model is then no longer
used: `estimate` falls back to BKT with the reason "mapping changed since
training; retrain required" until `train` is run on the new data. Moodle's
recycle bin keeps a backup of the deleted course (_Course category > Recycle
bin_) until it expires. Real test students must be enrolled again in the new
course.

## How the simulated learners are made

**Python decides, PHP replays.** `export_structure.php` reports the course's
quizzes (slots, question-bank entries, KCs, pass mark), the KC lessons (pages
whose ID number is a `kc_id`) and the unit overview pages.
`knowledge_engine.simulator.simulate_plan` decides every learner's sessions.
It uses the same answer generator as offline simulation: an item-response model
with practice gains and forgetting, deliberately not BKT. About 5 % of
learners enrol but never answer, so they are cold-start learners.

One session is one attempt at one real quiz, answering all its questions. Around
the attempts, each learner:

- views the unit's **overview** before their first attempt in a unit;
- with probability `study_propensity` (a learner trait) reads the **lessons**
  of some of a quiz's KCs before their first attempt at it;
- after **failing** a quiz, with probability `revise_propensity` makes the next
  session a **revision**: reads the lessons of the KCs they answered below the
  pass mark in that attempt, then retries the same quiz;
- sometimes **reviews** a finished attempt a few minutes later;
- moves on to the next quiz after passing, and sometimes goes back to an
  earlier one.

**Assumption:** reading a lesson adds `lesson_gain x learn_rate` to that KC's
practice gain (`--lesson-gain`, default 0.5, i.e. half the gain of the first
practice answer; `0` = lessons have no effect). This is a property of the
generator, chosen so recommendation logic has something to act on. It is not
evidence that the lessons help, and results that depend on it must say so.
The value is recorded in the plan manifest under `page_behaviour`.

`inject_attempts.php` replays the plan through Moodle's APIs:

- It creates users `sim0001`, `sim0002`, ... with ID number `SIM-0001`, ...
  (auth `nologin`, so they cannot sign in) and enrols them as students at the
  planned date.
- Page views go through `page_view()`, which logs the view and sets the page's
  "viewed" completion.
- Each attempt goes through `quiz_view()` and the normal quiz flow: create,
  start, answer each question at its simulated time (the planned right or
  wrong option, resolved through that attempt's shuffled order), submit,
  grade. Grading updates the quiz's pass-grade completion.
- Reviews fire the attempt's `attempt_reviewed` event.
- The script **stops** if Moodle's grade differs from the plan, or if a quiz's
  completion state is not `complete_pass` / `complete_fail` as the learner's
  best grade says.
- Moodle stamps everything with the current time. After each step, the rows the
  step created are moved to the simulated time: log rows, activity completion
  (`course_modules_completion`), module views (`course_modules_viewed`) and the
  first step of each question attempt. These are direct database updates,
  acceptable on a test instance but not something Moodle supports officially.

## Labelling

When any exported learner has a `SIM-` ID number, the collector writes
`simulation_manifest.json` into its output directory. The knowledge engine and
the learner model then label the whole export `simulated`, as they do for
offline simulation. A mixed export of real and simulated learners is
deliberately labelled `simulated` too.

After a successful injection, `simulate-learners` also writes
`simulation_plan_manifest.json` (seed, cohort size, accuracy, **ceiling AUC**,
page behaviour) into the collector output directory (`--data-dir`, default
`data/`). `knowledge_engine train` reports that ceiling next to the gate
decision. It is the AUC of the generator's own answer probabilities, so no
model can be expected to beat it on this data. When the collector finds no
simulated learners, it removes both manifests.

## Checks after each step

1. **After `build-course`**, in the browser: 8 unit sections; in each, the
   overview, then lessons and a quiz per dimension. Every activity shows its
   completion condition ("View", "Receive a passing grade"). The question bank
   has 4 dimension categories and 96 KC categories with ID numbers. Preview one
   quiz and one lesson.
2. **Collector on the empty course**: `kc_catalog.csv` has 96 rows,
   `moodle_question_kc_map.csv` 390, `kc_topic_map.csv` 96, and
   `kc_activity_map.csv` 192 (96 `learn` + 96 `practice`). Enrolled users'
   profiles are `cold_start_only`.
3. **After 20 learners** (seed 7: 164 attempts, 2,019 answers, 120 lesson
   views, 35 overview views, 31 reviews):
   - `question_answer_events.csv` matches the plan answer for answer;
   - `course_modules_completion.csv` has `complete` rows for viewed pages and
     `complete_pass` / `complete_fail` for quizzes, all with backdated times;
   - `learning_events.csv` holds `content_view` (lessons, overviews, quiz
     views), `quiz_attempt` and `quiz_review` events;
   - `status` says not ready, engagement is `available`, and outputs are
     labelled `simulated`.
4. **After 300 learners**: `status` says ready. `train` completes with three
   seeds and records the gate decision in
   `knowledge_engine/artifacts/active_model.json`. `estimate` and the builder
   produce Phase 3 or Phase 2 profiles, depending on the gate.
5. **After `remove-simulated`**: the course and the real users are intact.

## Answering quizzes by hand

To add real (non-simulated) answers, enrol an ordinary test student in
`ENG-MICRO` (_Participants -> Enrol users_), log in as that student, read some
lessons and take a few practice quizzes. Real accounts have no `SIM-` ID
number. Once the simulated learners are removed, the export is labelled
`moodle_collector` again.

## Commands

| Command                                                                                                   | Effect                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build-course [--rebuild]`                                                                                | Creates `ENG-MICRO`. Refuses if it exists. `--rebuild` deletes it first, **including all attempts in it**. Run `simulate-learners` again afterwards. |
| `simulate-learners [--learners N] [--seed S] [--lesson-gain G] [--plan-out F] [--dry-run] [--data-dir D]` | Plans and injects N simulated learners in total. Existing ones are kept.                                                                             |
| `remove-simulated`                                                                                        | Deletes every `SIM-` user, with their attempts, log rows and completion records. The course stays.                                                   |
| `check-content`                                                                                           | Validates the course content only.                                                                                                                   |

## Limits

- Results on simulated learners show that the pipeline and the gate work. They
  describe the generator, not real learners, and are not validity evidence.
  This includes any effect of the lessons, which is built into the generator.
- The PHP scripts were written against the Moodle 5.x source in this
  repository. A different Moodle version may need changes to the quiz and
  completion API calls in `php/inject_attempts.php`.
