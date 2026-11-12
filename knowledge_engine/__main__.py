"""Command line: python -m knowledge_engine <simulate|status|estimate|train>."""

import argparse
import json
from pathlib import Path

from . import config


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m knowledge_engine", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sim = commands.add_parser(
        "simulate", help="write simulated learners in collector format"
    )
    sim.add_argument("--output-dir", type=Path, default=config.DEFAULT_SIMULATED_DIR)
    sim.add_argument(
        "--course-dir",
        type=Path,
        help="collector export whose KCs and questions to reuse (default: a synthetic course)",
    )
    sim.add_argument("--learners", type=int, default=300)
    sim.add_argument("--seed", type=int, default=7)

    stat = commands.add_parser("status", help="show readiness and the active estimator")
    stat.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)

    est = commands.add_parser("estimate", help="write learner-by-KC knowledge states")
    est.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    est.add_argument("--output-dir", type=Path, default=config.DEFAULT_STATES_DIR)
    est.add_argument(
        "--as-of", help="inclusive cutoff: Unix seconds or timezone-aware ISO datetime"
    )

    trn = commands.add_parser(
        "train", help="train a deep model and apply the promotion gate"
    )
    trn.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    trn.add_argument(
        "--epochs", type=int, help="override the configured number of epochs"
    )
    trn.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="one model per seed; the best by validation AUC is the candidate",
    )
    trn.add_argument(
        "--force",
        action="store_true",
        help="train even if the readiness thresholds are not met",
    )

    args = parser.parse_args(argv)
    if args.command == "simulate":
        from .simulator import simulate

        result = simulate(args.output_dir, args.course_dir, args.learners, args.seed)
        print(json.dumps(result, indent=2))
        print(f"Simulated data written to {args.output_dir}")
    elif args.command == "status":
        from .phase import status

        print(json.dumps(status(args.data_dir), indent=2))
    elif args.command == "estimate":
        from .estimate import estimate

        states = estimate(args.data_dir, args.output_dir, args.as_of)
        manifest = json.loads((args.output_dir / config.MANIFEST_FILE_NAME).read_text())
        print(
            f"{len(states):,} learner-KC states written to "
            f"{args.output_dir / config.STATE_FILE_NAME}\n"
            f"Estimator: {manifest['estimator']} (phase {manifest['estimator_phase']}) - "
            f"{manifest['estimator_reason']}"
        )
    elif args.command == "train":
        from .phase import train

        record = train(
            args.data_dir, epochs=args.epochs, seeds=tuple(args.seeds), force=args.force
        )
        print(json.dumps(record, indent=2))
        print(f"Phase {record['phase']}: {record['estimator']} - {record['reason']}")
        if record.get("simulator_ceiling_auc") is not None:
            print(
                f"Simulator ceiling AUC: {record['simulator_ceiling_auc']} "
                "(what a model knowing every hidden parameter would score)"
            )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        # Missing or inconsistent inputs are user-facing conditions, not crashes.
        raise SystemExit(f"knowledge_engine: {error}")
