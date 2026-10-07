"""
Event whitelist and behavioural classification.

This is the documented taxonomy the thesis methodology refers to: which Moodle
log events are collected, and which behavioural category each one maps to.
Pure logic - no database access, no configuration.
"""

# --------------------------------------------------------------------------
# Event selection and classification
#
# Keep this list explicit and documented: the build plan requires a "small,
# documented set of meaningful events", not everything the log store holds.
# Names that do not exist in a given Moodle version simply match nothing.
# --------------------------------------------------------------------------

QUESTION_EVENTS = [
    r"\core\event\question_viewed",
    r"\core\event\question_answered",
    r"\mod_quiz\event\question_manually_graded",
    r"\mod_lesson\event\question_viewed",
    r"\mod_lesson\event\question_answered",
]

CONTENT_VIEW_EVENTS = [
    r"\mod_resource\event\course_module_viewed",
    r"\mod_page\event\course_module_viewed",
    r"\mod_url\event\course_module_viewed",
    r"\mod_book\event\course_module_viewed",
    r"\mod_lesson\event\course_module_viewed",
    r"\mod_quiz\event\course_module_viewed",
    r"\mod_assign\event\course_module_viewed",
    r"\mod_forum\event\course_module_viewed",
]

LESSON_EVENTS = [
    r"\mod_lesson\event\lesson_started",
    r"\mod_lesson\event\lesson_resumed",
    r"\mod_lesson\event\lesson_ended",
    r"\mod_lesson\event\content_page_viewed",
    r"\mod_lesson\event\highscore_added",
]

QUIZ_ATTEMPT_EVENTS = [
    r"\mod_quiz\event\attempt_started",
    r"\mod_quiz\event\attempt_updated",
    r"\mod_quiz\event\attempt_submitted",
    r"\mod_quiz\event\attempt_graded",
    r"\mod_quiz\event\attempt_summary_viewed",
    r"\mod_quiz\event\attempt_viewed",
    r"\mod_quiz\event\attempt_reviewed",
    r"\mod_quiz\event\attempt_abandoned",
    r"\mod_quiz\event\attempt_becameoverdue",
]

COMPLETION_EVENTS = [
    r"\core\event\course_module_completion_updated",
]

EVENT_NAME_WHITELIST = (
    QUESTION_EVENTS
    + CONTENT_VIEW_EVENTS
    + LESSON_EVENTS
    + QUIZ_ATTEMPT_EVENTS
    + COMPLETION_EVENTS
)

# Safety net for version-specific question events we may not have listed.
# NOTE: these run through SQL LIKE, so they must not contain backslashes
# (MariaDB treats "\" as an escape character inside LIKE patterns).
EVENT_NAME_PATTERNS = [
    "%question_viewed",
    "%question_answered",
    "%question_manually_graded",
    "%question_flagged",
]

# Explicitly excluded: authoring/teaching traffic such as slot_created,
# edit_page_viewed, report_viewed, page_created/updated/deleted,
# section_shuffle_updated, attempt_deleted, attempt_preview_started.
# These are course-construction events, not learning behaviour.

QUESTION_EVENT_SET = set(QUESTION_EVENTS)
CONTENT_VIEW_SET = set(CONTENT_VIEW_EVENTS)
LESSON_EVENT_SET = set(LESSON_EVENTS)
QUIZ_ATTEMPT_SET = set(QUIZ_ATTEMPT_EVENTS)
COMPLETION_EVENT_SET = set(COMPLETION_EVENTS)

# Which categories count as evidence of actual learning behaviour.
LEARNING_CATEGORIES = {
    "content_view",
    "lesson_interaction",
    "question_interaction",
    "quiz_attempt",
}


def classify_event(event_name, target, action):
    """
    Map a raw Moodle event name to one behavioural category.

    Returns one of:
        question_interaction | lesson_interaction | content_view
        quiz_attempt | quiz_review | completion | other
    """
    name = event_name or ""
    lowered = name.lower()

    if name in QUESTION_EVENT_SET or "question_" in lowered.rsplit("\\", 1)[-1]:
        return "question_interaction"
    if name in COMPLETION_EVENT_SET:
        return "completion"
    if name in LESSON_EVENT_SET:
        return "lesson_interaction"
    if name in QUIZ_ATTEMPT_SET:
        # Reviewing a graded attempt is feedback-seeking, not assessment.
        if lowered.endswith(("attempt_reviewed", "attempt_viewed")):
            return "quiz_review"
        return "quiz_attempt"
    if name in CONTENT_VIEW_SET:
        return "content_view"
    if (target or "") == "course_module" and (action or "") == "viewed":
        return "content_view"
    return "other"
