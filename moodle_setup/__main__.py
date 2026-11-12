"""Set up the local Moodle test instance.

python -m moodle_setup build-course       create the English course
python -m moodle_setup simulate-learners  add simulated learners and attempts
python -m moodle_setup remove-simulated   delete the simulated learners
python -m moodle_setup check-content      validate the course content only
"""

import argparse
import json
import tempfile
from pathlib import Path

from collector.collect_question_answers import PLAN_MANIFEST_NAME
from collector.config import OUTPUT_DIR as COLLECTOR_OUTPUT_DIR

from knowledge_engine.simulator import DEFAULT_LESSON_GAIN

from . import content, docker


def _write_temp_json(payload, name):
    path = Path(tempfile.mkdtemp(prefix="moodle_setup_")) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def build_course(args):
    spec = content.build_spec()
    remote = docker.copy_in(_write_temp_json(spec, "course_spec.json"))
    result = docker.run_php(
        "build_course.php", remote, *(["--rebuild"] if args.rebuild else [])
    )
    print(
        f"\nCourse {result['shortname']} is ready: {result['kcs']} KCs, "
        f"{result['questions']} questions, {result['lessons']} lessons, "
        f"{result['quizzes']} quizzes (pass mark {result['gradepass']}/10).\n"
        f"For the collector:  export MOODLE_COURSE_ID={result['course_id']}"
    )


def simulate_learners(args):
    from knowledge_engine.simulator import simulate_plan

    structure = docker.run_php("export_structure.php", content.COURSE_SHORTNAME)
    lessons = structure.get("lessons") or {}
    if not lessons:
        print(
            "NOTE: the course has no KC lesson pages (pages with a KC ID number); "
            "learners will only take quizzes. Rebuild the course to add them."
        )
    plan, manifest = simulate_plan(
        structure["quizzes"],
        n_learners=args.learners,
        seed=args.seed,
        start_unix=structure["startdate"],
        lessons=lessons,
        overviews=structure.get("overviews") or {},
        lesson_gain=args.lesson_gain,
    )
    answers = sum(len(a["answers"]) for learner in plan for a in learner["attempts"])
    attempts = sum(len(learner["attempts"]) for learner in plan)
    print(
        f"Plan: {len(plan)} learners, {attempts:,} quiz attempts, {answers:,} answers "
        f"(seed {args.seed}). Generator ceiling AUC: {manifest['ceiling_auc']}."
    )
    pages = manifest.get("page_behaviour")
    if pages:
        print(
            f"Pages: {pages['lesson_views']:,} lesson views, {pages['overview_views']:,} overview "
            f"views, {pages['revisions']:,} revision sessions, {pages['reviews']:,} attempt "
            f"reviews (lesson gain {pages['lesson_gain']})."
        )
    if args.plan_out:
        Path(args.plan_out).write_text(
            json.dumps({"manifest": manifest, "learners": plan}), encoding="utf-8"
        )
        print(f"Plan written to {args.plan_out}")
    if args.dry_run:
        return
    remote = docker.copy_in(_write_temp_json({"learners": plan}, "attempt_plan.json"))
    result = docker.run_php("inject_attempts.php", content.COURSE_SHORTNAME, remote)
    print(
        f"\nMoodle now holds {result['simulated_users']} simulated learners; this run added "
        f"{result['attempts_added']:,} attempts ({result['answers_added']:,} answers)."
    )
    save_plan_manifest(manifest, result["simulated_users"], args.data_dir)


def save_plan_manifest(manifest, simulated_users, data_dir):
    """Keep the plan's manifest (with the ceiling AUC) next to the collector export.

    ``python -m knowledge_engine train`` reads ``ceiling_auc`` from it. The
    plan for N learners covers every simulated learner only if Moodle holds no
    more than N of them; otherwise the existing manifest is left alone.
    """
    path = Path(data_dir) / PLAN_MANIFEST_NAME
    if simulated_users > manifest["learners"]:
        print(
            f"Moodle holds {simulated_users} simulated learners but this plan covers "
            f"{manifest['learners']}; {path} was not updated."
        )
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Plan manifest (ceiling AUC {manifest['ceiling_auc']}) saved to {path}")
    return path


def remove_simulated(args):
    result = docker.run_php("remove_simulated.php")
    print(
        f"Removed {result['users_removed']} simulated learners and {result['attempts_removed']:,} attempts."
    )
    (Path(COLLECTOR_OUTPUT_DIR) / PLAN_MANIFEST_NAME).unlink(missing_ok=True)
    print("Run the collector again so the export no longer contains them.")


def check_content(args):
    spec = content.build_spec()
    print(
        f"Content is valid: {len(spec['units'])} units, {len(spec['kcs'])} KCs, "
        f"{len(spec['questions'])} questions, {len(spec['lessons'])} lessons."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m moodle_setup",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build-course", help="create the English course in Moodle"
    )
    build.add_argument(
        "--rebuild",
        action="store_true",
        help="delete the existing course (and every attempt in it) first",
    )
    build.set_defaults(run=build_course)

    sim = commands.add_parser(
        "simulate-learners", help="inject simulated learners and quiz attempts"
    )
    sim.add_argument(
        "--learners",
        type=int,
        default=300,
        help="total simulated learners wanted; existing ones are kept and skipped",
    )
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument(
        "--plan-out", help="also save the attempt plan (with true answers) to this file"
    )
    sim.add_argument(
        "--dry-run", action="store_true", help="make the plan but do not touch Moodle"
    )
    sim.add_argument(
        "--lesson-gain",
        type=float,
        default=DEFAULT_LESSON_GAIN,
        help="knowledge gain from reading a KC lesson, as a share of one practice "
        "answer's gain (generator assumption; 0 = lessons have no effect)",
    )
    sim.add_argument(
        "--data-dir",
        type=Path,
        default=COLLECTOR_OUTPUT_DIR,
        help="collector output directory that receives the plan manifest "
        "(default: the collector's MOODLE_OUTPUT_DIR)",
    )
    sim.set_defaults(run=simulate_learners)

    commands.add_parser(
        "remove-simulated", help="delete simulated learners and their attempts"
    ).set_defaults(run=remove_simulated)
    commands.add_parser(
        "check-content", help="validate the course content"
    ).set_defaults(run=check_content)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except ValueError as error:
        raise SystemExit(f"moodle_setup: {error}")


if __name__ == "__main__":
    main()
