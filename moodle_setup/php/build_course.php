<?php
// Build the English microlearning course from a course specification.
//
// Usage: php build_course.php <course_spec.json> [--rebuild]
//
// Creates: the course with one section per unit, one course question bank
// whose categories are  top > dimension > KC  (KC category ID number = kc_id),
// the questions, one overview page per unit, one lesson page per KC
// (course-module ID number = kc_id) and one practice quiz per unit and
// dimension. The category layout and the lesson ID numbers are the
// conventions the collector reads to derive the KC catalog, the
// question-to-KC mapping and the KC-to-activity map.
//
// Activity completion: pages are complete when viewed; a quiz is complete
// with a passing grade (pass_fraction of the marks), so a failed quiz is
// recorded as complete_fail.

require(__DIR__ . '/lib.php');

$spec = setup_read_json($argv[1] ?? '');
$rebuild = in_array('--rebuild', $argv, true);

if (empty($CFG->enablecompletion)) {
    cli_error('Completion tracking is switched off for the whole site. Turn on "Enable completion '
        . 'tracking" in Site administration > Advanced features, then run this again.');
}

$existing = $DB->get_record('course', ['shortname' => $spec['shortname']]);
if ($existing) {
    if (!$rebuild) {
        cli_error("Course {$spec['shortname']} already exists (id {$existing->id}). "
            . "Use --rebuild to delete and recreate it, including all attempts in it.");
    }
    cli_writeln("Deleting existing course {$spec['shortname']} (id {$existing->id})...");
    delete_course($existing, false);
}

$generator = \core\test\phpunit\phpunit_util::get_data_generator();
$questiongenerator = $generator->get_plugin_generator('core_question');

// ---- course and sections ---------------------------------------------------
// The start date lies in the past so simulated learners can have a history.
$startdate = usergetmidnight(time() - 150 * DAYSECS);
$course = create_course((object) [
    'fullname' => $spec['fullname'],
    'shortname' => $spec['shortname'],
    'category' => \core_course_category::get_default()->id,
    'format' => 'topics',
    'numsections' => count($spec['units']),
    'startdate' => $startdate,
    'enablecompletion' => 1,
    'summary' => 'Short practice activities for everyday English: grammar, vocabulary, reading and writing.',
    'summaryformat' => FORMAT_HTML,
]);
course_create_sections_if_missing($course, range(0, count($spec['units'])));
foreach ($spec['units'] as $unit) {
    $section = $DB->get_record('course_sections', ['course' => $course->id, 'section' => $unit['number']], '*', MUST_EXIST);
    course_update_section($course, $section, ['name' => "Unit {$unit['number']}: {$unit['name']}"]);
}
cli_writeln("Created course {$course->shortname} (id {$course->id}) with " . count($spec['units']) . " units.");

// ---- question bank: top > dimension > KC ------------------------------------
$qbank = $generator->create_module('qbank', [
    'course' => $course->id,
    'name' => 'English Microlearning question bank',
    'section' => 0,
]);
$bankcontext = \context_module::instance($qbank->cmid);
$top = question_get_top_category($bankcontext->id, true);

$dimensioncategories = [];
foreach ($spec['dimensions'] as $dimension) {
    $dimensioncategories[$dimension] = $questiongenerator->create_question_category([
        'contextid' => $bankcontext->id,
        'parent' => $top->id,
        'name' => $dimension,
        'info' => "Knowledge Components of the {$dimension} dimension.",
    ]);
}
$kccategories = [];
foreach ($spec['kcs'] as $kc) {
    $kccategories[$kc['kc_id']] = $questiongenerator->create_question_category([
        'contextid' => $bankcontext->id,
        'parent' => $dimensioncategories[$kc['dimension']]->id,
        'name' => $kc['name'],
        'info' => $kc['objective'],
        'idnumber' => $kc['kc_id'],
    ]);
}
cli_writeln('Created ' . count($kccategories) . ' KC categories under ' . count($dimensioncategories) . ' dimensions.');

// ---- questions --------------------------------------------------------------
$kcs = array_column($spec['kcs'], null, 'kc_id');
$questionsbykc = [];
foreach ($spec['questions'] as $item) {
    $answers = $fractions = $feedback = [];
    foreach ($item['options'] as $index => $option) {
        $answers[] = ['text' => s($option), 'format' => FORMAT_HTML];
        $fractions[] = $index === 0 ? '1.0' : '0.0';   // The first option is the correct one.
        $feedback[] = ['text' => '', 'format' => FORMAT_HTML];
    }
    // Saved through the question type itself (the core question generator
    // needs PHPUnit, which a normal site does not have).
    $form = (object) [
        'category' => $kccategories[$item['kc_id']]->id . ',' . $bankcontext->id,
        'name' => $item['name'],
        'questiontext' => ['text' => '<p>' . s($item['stem']) . '</p>', 'format' => FORMAT_HTML],
        'generalfeedback' => [
            'text' => '<p>Focus of this question: ' . s($kcs[$item['kc_id']]['objective']) . '</p>',
            'format' => FORMAT_HTML,
        ],
        'defaultmark' => 1,
        'penalty' => 0.3333333,
        'status' => \core_question\local\bank\question_version_status::QUESTION_STATUS_READY,
        'idnumber' => null,
        'single' => '1',
        'shuffleanswers' => 1,
        'answernumbering' => 'abc',
        'showstandardinstruction' => 0,
        'shownumcorrect' => 0,
        'correctfeedback' => ['text' => 'Your answer is correct.', 'format' => FORMAT_HTML],
        'partiallycorrectfeedback' => ['text' => 'Your answer is partially correct.', 'format' => FORMAT_HTML],
        'incorrectfeedback' => ['text' => 'Your answer is incorrect.', 'format' => FORMAT_HTML],
        'noanswers' => count($answers),
        'answer' => $answers,
        'fraction' => $fractions,
        'feedback' => $feedback,
    ];
    $question = (object) [
        'qtype' => 'multichoice',
        'createdby' => $USER->id,
        'idnumber' => null,
        'status' => $form->status,
    ];
    $question = question_bank::get_qtype('multichoice')->save_question($question, $form);
    $questionsbykc[$item['kc_id']][] = $question->id;
}
cli_writeln('Created ' . count($spec['questions']) . ' questions.');

// ---- per unit: overview page, then per dimension: lessons + practice quiz ----
// Page completion: complete once viewed. Quiz completion: complete with a
// passing grade, so Moodle records complete_pass / complete_fail.
$viewcompletion = ['completion' => COMPLETION_TRACKING_AUTOMATIC, 'completionview' => COMPLETION_VIEW_REQUIRED];
$quizgrade = 10;
$gradepass = round($quizgrade * (float) $spec['pass_fraction'], 2);
$lessons = array_column($spec['lessons'], null, 'kc_id');
$quizcount = $lessoncount = 0;
foreach ($spec['units'] as $unit) {
    $unitkcs = array_filter($spec['kcs'], fn($kc) => $kc['unit'] === $unit['number']);
    $objectives = '';
    foreach ($spec['dimensions'] as $dimension) {
        $items = array_filter($unitkcs, fn($kc) => $kc['dimension'] === $dimension);
        if (!$items) {
            continue;
        }
        $objectives .= '<h4>' . s($dimension) . '</h4><ul>';
        foreach ($items as $kc) {
            $objectives .= '<li><strong>' . s($kc['name']) . '</strong>: ' . s($kc['objective']) . '</li>';
        }
        $objectives .= '</ul>';
    }
    $generator->create_module('page', [
        'course' => $course->id,
        'section' => $unit['number'],
        'name' => "Unit {$unit['number']} overview: {$unit['name']}",
        'intro' => 'What you will practise in this unit.',
        'content' => '<p>In this unit you will practise:</p>' . $objectives
            . '<p>Each point has a short lesson, followed by a practice quiz for its skill area. '
            . 'A practice quiz counts as complete when you reach ' . s((string) $gradepass) . '/' . $quizgrade . '.</p>',
        'contentformat' => FORMAT_HTML,
    ] + $viewcompletion);

    foreach ($spec['dimensions'] as $dimension) {
        $items = array_filter($unitkcs, fn($kc) => $kc['dimension'] === $dimension);
        if (!$items) {
            continue;
        }
        foreach ($items as $kc) {
            $generator->create_module('page', [
                'course' => $course->id,
                'section' => $unit['number'],
                'name' => "Lesson: {$kc['name']}",
                'idnumber' => $kc['kc_id'],   // The KC this lesson teaches (collector convention).
                'intro' => s($kc['objective']),
                'content' => $lessons[$kc['kc_id']]['html'],
                'contentformat' => FORMAT_HTML,
            ] + $viewcompletion);
            $lessoncount++;
        }
        $quiz = $generator->create_module('quiz', [
            'course' => $course->id,
            'section' => $unit['number'],
            'name' => "Unit {$unit['number']} {$dimension} practice",
            'intro' => "Short {$dimension} practice for {$unit['name']}. You can try as often as you like; "
                . "it counts as complete when you reach {$gradepass}/{$quizgrade}.",
            'preferredbehaviour' => 'immediatefeedback',
            'attempts' => 0,
            'questionsperpage' => 1,
            'shuffleanswers' => 1,
            'grade' => $quizgrade,
            'gradepass' => $gradepass,
            'timeopen' => 0,
            'timeclose' => 0,
            'timelimit' => 0,
            'completion' => COMPLETION_TRACKING_AUTOMATIC,
            'completionusegrade' => 1,
            'completionpassgrade' => 1,
        ]);
        $quizrecord = $DB->get_record('quiz', ['id' => $quiz->id], '*', MUST_EXIST);
        foreach ($items as $kc) {
            foreach ($questionsbykc[$kc['kc_id']] as $questionid) {
                quiz_add_quiz_question($questionid, $quizrecord, 0, 1);
            }
        }
        \mod_quiz\quiz_settings::create($quiz->id)->get_grade_calculator()->recompute_quiz_sumgrades();
        $quizcount++;
    }
}
rebuild_course_cache($course->id, true);
cli_writeln("Created {$lessoncount} KC lessons, {$quizcount} practice quizzes and "
    . count($spec['units']) . ' overview pages, all with completion tracking.');

setup_result([
    'course_id' => (int) $course->id,
    'shortname' => $course->shortname,
    'startdate' => (int) $course->startdate,
    'kcs' => count($kccategories),
    'questions' => count($spec['questions']),
    'quizzes' => $quizcount,
    'lessons' => $lessoncount,
    'gradepass' => $gradepass,
]);
