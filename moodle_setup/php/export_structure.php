<?php
// Export the course structure the learner simulator needs.
//
// Usage: php export_structure.php <course shortname>
//
// quizzes   every quiz in course order: its cmid, pass mark, slots, and each
//           slot's question-bank entry and KC (ID number of its category)
// lessons   kc_id -> cmid of the page whose course-module ID number is the KC
// overviews section number -> cmid of the first other page in the section
//           (the unit overview)

require(__DIR__ . '/lib.php');

$course = setup_get_course($argv[1] ?? '');
$modinfo = get_fast_modinfo($course);
$quizzes = $lessons = $overviews = [];
foreach ($modinfo->get_sections() as $sectionnumber => $cmids) {
    foreach ($cmids as $cmid) {
        $cm = $modinfo->get_cm($cmid);
        if ($cm->modname === 'page') {
            if ($cm->idnumber !== null && $cm->idnumber !== '') {
                $lessons[$cm->idnumber] = (int) $cm->id;
            } else if (!isset($overviews[$sectionnumber])) {
                $overviews[$sectionnumber] = (int) $cm->id;
            }
            continue;
        }
        if ($cm->modname !== 'quiz') {
            continue;
        }
        $slots = $DB->get_records_sql(
            "SELECT qs.slot, qr.questionbankentryid, qc.idnumber AS kc_id
               FROM {quiz_slots} qs
               JOIN {question_references} qr
                 ON qr.component = 'mod_quiz' AND qr.questionarea = 'slot' AND qr.itemid = qs.id
               JOIN {question_bank_entries} qbe ON qbe.id = qr.questionbankentryid
               JOIN {question_categories} qc ON qc.id = qbe.questioncategoryid
              WHERE qs.quizid = ?
           ORDER BY qs.slot",
            [$cm->instance]
        );
        $entries = [];
        foreach ($slots as $slot) {
            if ($slot->kc_id === null || $slot->kc_id === '') {
                cli_error("Quiz {$cm->name} uses a question whose category has no ID number (no KC).");
            }
            $entries[] = ['entry_id' => (int) $slot->questionbankentryid, 'kc_id' => $slot->kc_id];
        }
        if (!$entries) {
            continue;
        }
        $quiz = $DB->get_record('quiz', ['id' => $cm->instance], 'id, grade, sumgrades', MUST_EXIST);
        $gradeitem = grade_item::fetch([
            'itemtype' => 'mod', 'itemmodule' => 'quiz', 'iteminstance' => $cm->instance,
            'itemnumber' => 0, 'courseid' => $course->id,
        ]);
        $quizzes[] = [
            'quiz_id' => (int) $cm->instance,
            'cmid' => (int) $cm->id,
            'name' => $cm->name,
            'section' => (int) $sectionnumber,
            'grade' => (float) $quiz->grade,
            'gradepass' => $gradeitem ? (float) $gradeitem->gradepass : 0.0,
            'completion_pass_required' => (int) $cm->completionpassgrade,
            'entries' => $entries,
        ];
    }
}
setup_result([
    'course_id' => (int) $course->id,
    'startdate' => (int) $course->startdate,
    'quizzes' => $quizzes,
    'lessons' => (object) $lessons,
    'overviews' => (object) $overviews,
]);
