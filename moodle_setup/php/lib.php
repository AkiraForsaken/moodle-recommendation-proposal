<?php
// Shared bootstrap for the moodle_setup CLI scripts.
//
// These scripts run inside the Moodle web container (see ../docker.py). They
// only use Moodle's own APIs, so everything they create is ordinary Moodle
// data: visible in the UI, exported by the collector, removable in the UI.

define('CLI_SCRIPT', true);
require('/var/www/html/config.php');
require_once($CFG->libdir . '/clilib.php');
require_once($CFG->libdir . '/questionlib.php');
require_once($CFG->dirroot . '/course/lib.php');
require_once($CFG->dirroot . '/user/lib.php');
require_once($CFG->dirroot . '/mod/quiz/locallib.php');
require_once($CFG->dirroot . '/mod/page/lib.php');
require_once($CFG->libdir . '/completionlib.php');
require_once($CFG->libdir . '/gradelib.php');

// ID-number prefix that marks an account as a simulated learner. The collector
// uses the same prefix to label exports as simulated.
const SIM_IDNUMBER_PREFIX = 'SIM-';
const RESULT_MARKER = 'MOODLE_SETUP_RESULT:';

\core\session\manager::set_user(get_admin());

/** Report the script result to the Python driver. */
function setup_result(array $data): void {
    echo RESULT_MARKER . json_encode($data) . "\n";
}

/** Read a JSON input file passed on the command line. */
function setup_read_json(string $path): array {
    if (!is_readable($path)) {
        cli_error("Cannot read input file: {$path}");
    }
    $data = json_decode(file_get_contents($path), true);
    if (!is_array($data)) {
        cli_error("Input file is not valid JSON: {$path}");
    }
    return $data;
}

/** The course built by build_course.php, or an error if it does not exist. */
function setup_get_course(string $shortname): stdClass {
    global $DB;
    $course = $DB->get_record('course', ['shortname' => $shortname]);
    if (!$course) {
        cli_error("Course {$shortname} does not exist. Run: python -m moodle_setup build-course");
    }
    return $course;
}

/** All simulated learner accounts. */
function setup_simulated_users(): array {
    global $DB;
    return $DB->get_records_select(
        'user',
        'deleted = 0 AND ' . $DB->sql_like('idnumber', ':prefix'),
        ['prefix' => SIM_IDNUMBER_PREFIX . '%'],
        'idnumber'
    );
}
