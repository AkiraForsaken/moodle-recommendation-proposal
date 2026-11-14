<?php
// Delete every simulated learner (ID number SIM-...) with their quiz attempts,
// log rows and completion records. Real users and the course itself are not touched.
//
// Usage: php remove_simulated.php

require(__DIR__ . '/lib.php');

$users = setup_simulated_users();
$attemptsremoved = 0;
$quizzes = [];
foreach ($users as $user) {
    foreach ($DB->get_records('quiz_attempts', ['userid' => $user->id]) as $attempt) {
        $quizzes[$attempt->quiz] ??= $DB->get_record('quiz', ['id' => $attempt->quiz], '*', MUST_EXIST);
        quiz_delete_attempt($attempt, $quizzes[$attempt->quiz]);
        $attemptsremoved++;
    }
    $DB->delete_records('logstore_standard_log', ['userid' => $user->id]);
    foreach (['course_modules_completion', 'course_modules_viewed', 'course_completions',
            'course_completion_crit_compl'] as $table) {
        $DB->delete_records($table, ['userid' => $user->id]);
    }
    delete_user($user);
}
\cache::make('core', 'completion')->purge();
cli_writeln('Removed ' . count($users) . " simulated learners and {$attemptsremoved} attempts.");
setup_result(['users_removed' => count($users), 'attempts_removed' => $attemptsremoved]);
