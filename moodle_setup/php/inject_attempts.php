<?php
// Replay a simulated learner plan into Moodle through Moodle's own APIs.
//
// Usage: php inject_attempts.php <course shortname> <attempt_plan.json>
//
// For each planned learner: create the account (ID number SIM-<n>, cannot log
// in), enrol it as a student, then replay its sessions in time order:
//   * page views before an attempt (unit overview, KC lessons): page_view(),
//     which logs the view and sets "viewed" activity completion;
//   * the quiz attempt: quiz_view(), then start, answer every question at its
//     planned time, submit and grade; grading updates the quiz's
//     pass-grade completion (complete_pass / complete_fail);
//   * an optional review of the finished attempt (attempt_reviewed event).
//
// The script is resumable: attempts a learner already has are skipped, so
// running it again with a larger plan only adds the rest.
//
// Moodle stamps everything with the current time. After each step the rows
// it created are moved to the simulated time: log rows, activity completion
// and module-viewed rows, and the first step of each question attempt.

require(__DIR__ . '/lib.php');

$course = setup_get_course($argv[1] ?? '');
$plan = setup_read_json($argv[2] ?? '');
$learners = $plan['learners'] ?? [];

$coursecontext = \context_course::instance($course->id);
$studentrole = $DB->get_record('role', ['shortname' => 'student'], '*', MUST_EXIST);
$manual = enrol_get_plugin('manual');
$enrolinstance = $DB->get_record('enrol', ['courseid' => $course->id, 'enrol' => 'manual']);
if (!$enrolinstance) {
    $enrolinstance = $DB->get_record('enrol', ['id' => $manual->add_instance($course)], '*', MUST_EXIST);
}
$completion = new completion_info($course);
if (!$completion->is_enabled()) {
    cli_error('Completion tracking is off for this course. Rebuild it: python -m moodle_setup build-course --rebuild');
}

$modinfo = get_fast_modinfo($course);
$quizcms = $pagecms = [];
foreach ($modinfo->get_cms() as $cm) {
    if ($cm->modname === 'quiz') {
        $quizcms[$cm->instance] = $cm;
    } else if ($cm->modname === 'page') {
        $pagecms[$cm->id] = $cm;
    }
}
$coursequizids = array_keys($quizcms);
$pages = $DB->get_records_list('page', 'id', array_map(fn($cm) => $cm->instance, $pagecms) ?: [0]);

/**
 * A replay step: remember where the logs end and the wall-clock time, so the
 * rows the step creates can be found and moved to the simulated time.
 */
function step_begin(): array {
    global $DB;
    get_log_manager(true);   // Disposes the stores, which flushes their buffers.
    return [
        'logid' => (int) $DB->get_field_sql('SELECT COALESCE(MAX(id), 0) FROM {logstore_standard_log}'),
        'wallclock' => time(),
    ];
}

/**
 * Move what one step created to simulated time. Log rows for "started" and
 * "viewed" actions get $start, everything else $end; completion and viewed
 * rows the step touched get $end.
 */
function step_end(array $mark, int $userid, int $start, int $end): void {
    global $DB;
    get_log_manager(true);
    $DB->execute(
        "UPDATE {logstore_standard_log}
            SET timecreated = CASE WHEN action IN ('started', 'viewed') THEN :start ELSE :finish END
          WHERE id > :afterid AND (userid = :userid OR relateduserid = :relateduserid)",
        ['start' => $start, 'finish' => $end, 'afterid' => $mark['logid'],
            'userid' => $userid, 'relateduserid' => $userid]
    );
    $DB->execute(
        "UPDATE {course_modules_completion} SET timemodified = :t
          WHERE userid = :userid AND timemodified >= :wallclock",
        ['t' => $end, 'userid' => $userid, 'wallclock' => $mark['wallclock']]
    );
    $DB->execute(
        "UPDATE {course_modules_viewed} SET timecreated = :t
          WHERE userid = :userid AND timecreated >= :wallclock",
        ['t' => $start, 'userid' => $userid, 'wallclock' => $mark['wallclock']]
    );
}

$admin = get_admin();
$attemptsadded = $answersadded = $viewsadded = $reviewsadded = 0;
$total = count($learners);
foreach ($learners as $position => $learner) {
    $idnumber = SIM_IDNUMBER_PREFIX . sprintf('%04d', $learner['number']);
    $user = $DB->get_record('user', ['idnumber' => $idnumber, 'deleted' => 0]);
    if (!$user) {
        $username = 'sim' . sprintf('%04d', $learner['number']);
        $userid = user_create_user((object) [
            'username' => $username,
            'auth' => 'nologin',
            'confirmed' => 1,
            'mnethostid' => $CFG->mnet_localhost_id,
            'firstname' => 'Simulated',
            'lastname' => 'Learner ' . sprintf('%04d', $learner['number']),
            'email' => $username . '@example.invalid',
            'idnumber' => $idnumber,
        ], false, false);
        $user = $DB->get_record('user', ['id' => $userid], '*', MUST_EXIST);
    }
    if (!is_enrolled($coursecontext, $user)) {
        $manual->enrol_user($enrolinstance, $user->id, $studentrole->id, $learner['enrol_start']);
    }

    // Resume: drop unfinished leftovers, then skip the attempts already present.
    [$insql, $params] = $DB->get_in_or_equal($coursequizids, SQL_PARAMS_NAMED);
    $params['userid'] = $user->id;
    $existing = $DB->get_records_select('quiz_attempts', "quiz $insql AND userid = :userid", $params, 'id');
    $done = 0;
    foreach ($existing as $attempt) {
        if ($attempt->state === \mod_quiz\quiz_attempt::FINISHED) {
            $done++;
        } else {
            quiz_delete_attempt($attempt, $DB->get_record('quiz', ['id' => $attempt->quiz], '*', MUST_EXIST));
        }
    }
    $todo = array_slice($learner['attempts'], $done);
    if (!$todo) {
        continue;
    }

    \core\session\manager::set_user($user);
    foreach ($todo as $planned) {
        $quizid = (int) $planned['quiz_id'];
        $start = (int) $planned['start'];
        $finish = (int) $planned['finish'];
        $answers = array_column($planned['answers'], null, 'entry_id');
        if (!isset($quizcms[$quizid])) {
            cli_error("Quiz {$quizid} from the plan is not in this course; run simulate-learners again.");
        }
        $quizcm = $quizcms[$quizid];

        // ---- page views before the attempt (overview, KC lessons)
        foreach ($planned['views_before'] ?? [] as $view) {
            $cmid = (int) $view['cmid'];
            if (!isset($pagecms[$cmid])) {
                cli_error("Page {$cmid} from the plan is not in this course; run simulate-learners again.");
            }
            $mark = step_begin();
            page_view($pages[$pagecms[$cmid]->instance], $course, $pagecms[$cmid], $pagecms[$cmid]->context);
            step_end($mark, (int) $user->id, (int) $view['time'], (int) $view['time']);
            $viewsadded++;
        }

        // ---- the quiz attempt
        $mark = step_begin();
        $quizobj = \mod_quiz\quiz_settings::create($quizid, $user->id);
        quiz_view($quizobj->get_quiz(), $course, $quizcm, $quizobj->get_context());
        $quba = \question_engine::make_questions_usage_by_activity('mod_quiz', $quizobj->get_context());
        $quba->set_preferred_behaviour($quizobj->get_quiz()->preferredbehaviour);
        $previous = quiz_get_user_attempts($quizid, $user->id, 'all', true);
        $attemptnumber = count($previous) + 1;
        $lastattempt = $previous ? end($previous) : false;
        $attempt = quiz_create_attempt($quizobj, $attemptnumber, $lastattempt, $start, false, $user->id);
        quiz_start_new_attempt($quizobj, $quba, $attempt, $attemptnumber, $start);
        quiz_attempt_save_started($quizobj, $quba, $attempt, $start);

        $attemptobj = \mod_quiz\quiz_attempt::create($attempt->id);
        foreach ($attemptobj->get_slots() as $slot) {
            $qa = $attemptobj->get_question_attempt($slot);
            $question = $qa->get_question();
            $entryid = (int) $question->questionbankentryid;
            if (!isset($answers[$entryid])) {
                cli_error("The plan has no answer for question-bank entry {$entryid} in quiz {$quizid}. "
                    . "The course changed since the plan was made; run simulate-learners again.");
            }
            // Pick the right option, or the first wrong one in this attempt's
            // shuffled order. The engine's simulated-response format wants the
            // option text, which it maps back to the shuffled position.
            $wantcorrect = !empty($answers[$entryid]['correct']);
            $choice = null;
            foreach ($question->get_order($qa) as $answerid) {
                $iscorrect = $question->answers[$answerid]->fraction > 0.999;
                if ($iscorrect === $wantcorrect) {
                    $choice = clean_param($question->answers[$answerid]->answer, PARAM_NOTAGS);
                    break;
                }
            }
            $answertime = (int) $answers[$entryid]['time'];
            $attemptobj->process_submitted_actions($answertime, false, [$slot => ['answer' => $choice, '-submit' => 1]]);
            // Moodle stamps a question's first step when it is first shown on
            // screen. Nothing is shown here, so set that time from the plan.
            $DB->set_field('question_attempt_steps', 'timecreated',
                $answertime - (int) $answers[$entryid]['seconds'],
                ['questionattemptid' => $qa->get_database_id(), 'sequencenumber' => 0]);
            $answersadded++;
        }
        $attemptobj = \mod_quiz\quiz_attempt::create($attempt->id);
        $attemptobj->process_submit($finish, false);
        $attemptobj->process_grade_submission($finish);

        // Fail loudly if Moodle did not grade the attempt as planned.
        $expected = array_sum(array_column($planned['answers'], 'correct'));
        $graded = (float) $DB->get_field('quiz_attempts', 'sumgrades', ['id' => $attempt->id]);
        if (abs($graded - $expected) > 0.001) {
            cli_error("Attempt {$attempt->id} was graded {$graded} but the plan expects {$expected}.");
        }
        // Pass-grade completion follows the learner's quiz grade (highest
        // attempt). Recompute it explicitly and check it against the grade.
        $completion->update_state($quizcm, COMPLETION_UNKNOWN, $user->id);
        $state = (int) $completion->get_data($quizcm, false, $user->id)->completionstate;
        $quizgrade = $DB->get_field('quiz_grades', 'grade', ['quiz' => $quizid, 'userid' => $user->id]);
        $gradeitem = grade_item::fetch(['itemtype' => 'mod', 'itemmodule' => 'quiz',
            'iteminstance' => $quizid, 'itemnumber' => 0, 'courseid' => $course->id]);
        $passed = $quizgrade !== false && (float) $quizgrade >= (float) $gradeitem->gradepass - 0.00001;
        $wantstate = $passed ? COMPLETION_COMPLETE_PASS : COMPLETION_COMPLETE_FAIL;
        if ($state !== $wantstate) {
            cli_error("Quiz {$quizid} completion for {$idnumber} is {$state}, expected {$wantstate} "
                . "(grade {$quizgrade}, pass {$gradeitem->gradepass}).");
        }
        step_end($mark, (int) $user->id, $start, $finish);
        $attemptsadded++;

        // ---- optional review of the finished attempt
        if (!empty($planned['review_at'])) {
            $mark = step_begin();
            \mod_quiz\quiz_attempt::create($attempt->id)->fire_attempt_reviewed_event();
            step_end($mark, (int) $user->id, (int) $planned['review_at'], (int) $planned['review_at']);
            $reviewsadded++;
        }
    }
    \core\session\manager::set_user($admin);
    if (($position + 1) % 10 === 0 || $position + 1 === $total) {
        cli_writeln(sprintf('  %d / %d learners processed, %d attempts added so far',
            $position + 1, $total, $attemptsadded));
    }
}
// Completion state is cached per user; the cache must not hold wall-clock times.
\cache::make('core', 'completion')->purge();

setup_result([
    'simulated_users' => count(setup_simulated_users()),
    'attempts_added' => $attemptsadded,
    'answers_added' => $answersadded,
    'views_added' => $viewsadded,
    'reviews_added' => $reviewsadded,
]);
