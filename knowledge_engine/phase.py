"""Which estimator is in charge: readiness, training, and the promotion gate.

    Phase 1  not enough answers for a deep model       -> BKT
    Phase 2  enough answers; a deep model is trained,   -> BKT (still)
             but has not (yet) beaten BKT
    Phase 3  the deep model beat BKT on held-out        -> deep model
             learners of this course

The decision is recorded in ``artifacts/active_model.json``. Retraining is a
full retrain through the same gate: new questions change the model's
vocabulary, so a promoted model is never patched in place.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .bkt import DEFAULT_PARAMS, MODEL_NAME as BKT_MODEL_NAME, next_answer_predictions
from .data import (
    ANSWER_EVENTS_FILE,
    QUESTION_KC_MAP_FILE,
    current_mapping_version,
    detect_population,
    load_answer_events,
    load_kc_catalog,
    sha256_file,
    write_json,
)

KT_DIR = config.PROJECT_ROOT / "knowledge_tracing"
PREPARED_DIR = config.PROJECT_ROOT / "data" / "processed" / config.DEEP_DATASET_NAME
DEEP_MODEL = "akt"  # the only model the inference path supports


def default_thresholds():
    return {
        "min_learners_with_answers": config.MIN_LEARNERS_WITH_ANSWERS,
        "min_answer_events": config.MIN_ANSWER_EVENTS,
        "min_answers_per_kc": config.MIN_ANSWERS_PER_KC,
        "min_kc_coverage": config.MIN_KC_COVERAGE,
    }


def readiness(events, kc_catalog, thresholds=None):
    """Is there enough answer data to attempt deep-model training?"""
    limits = {**default_thresholds(), **(thresholds or {})}
    per_kc = events.groupby("kc_id").size() if len(events) else {}
    catalog_kcs = kc_catalog["kc_id"].astype(str).tolist()
    covered = sum(
        1
        for kc in catalog_kcs
        if int(dict(per_kc).get(kc, 0)) >= limits["min_answers_per_kc"]
    )
    counts = {
        "learners_with_answers": (
            int(events["learner_id"].nunique()) if len(events) else 0
        ),
        "answer_events": int(len(events)),
        "catalog_kcs": len(catalog_kcs),
        "kcs_with_enough_answers": covered,
        "kc_coverage": round(covered / len(catalog_kcs), 4) if catalog_kcs else 0.0,
    }
    unmet = []
    if counts["learners_with_answers"] < limits["min_learners_with_answers"]:
        unmet.append(
            f"{counts['learners_with_answers']} learners with answers "
            f"(need {limits['min_learners_with_answers']})"
        )
    if counts["answer_events"] < limits["min_answer_events"]:
        unmet.append(
            f"{counts['answer_events']} answer events (need {limits['min_answer_events']})"
        )
    if counts["kc_coverage"] < limits["min_kc_coverage"]:
        unmet.append(
            f"{counts['kc_coverage']:.0%} of KCs have {limits['min_answers_per_kc']}+ "
            f"answers (need {limits['min_kc_coverage']:.0%})"
        )
    return {"ready": not unmet, "counts": counts, "thresholds": limits, "unmet": unmet}


def decide_promotion(bkt_auc, deep_auc, margin=config.GATE_MIN_AUC_MARGIN):
    """The deep model must beat BKT by ``margin`` AUC on the same targets."""
    if deep_auc is None or bkt_auc is None:
        return False, "gate metrics are unavailable"
    if deep_auc >= bkt_auc + margin:
        return True, (
            f"deep model AUC {deep_auc:.4f} exceeds BKT AUC {bkt_auc:.4f} "
            f"by at least {margin:.3f}"
        )
    return False, (
        f"deep model AUC {deep_auc:.4f} does not exceed BKT AUC {bkt_auc:.4f} "
        f"by {margin:.3f}; BKT stays in charge"
    )


def load_active_model(path=config.ACTIVE_MODEL_FILE):
    """The current decision; a missing record means Phase 1 with BKT."""
    path = Path(path)
    if not path.is_file():
        return {
            "phase": 1,
            "estimator": BKT_MODEL_NAME,
            "reason": "no training has been run",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def status(
    data_dir=config.DEFAULT_DATA_DIR, active_model_file=config.ACTIVE_MODEL_FILE
):
    """Live readiness of ``data_dir`` next to the recorded decision."""
    events = load_answer_events(data_dir)
    return {
        "data_dir": str(data_dir),
        "population": detect_population(data_dir),
        "readiness": readiness(events, load_kc_catalog(data_dir)),
        "active_model": load_active_model(active_model_file),
    }


def bkt_metrics_on_deep_test_split(data_config, params=DEFAULT_PARAMS):
    """Score BKT on exactly the targets the deep model is evaluated on.

    Reads the pyKT test file (held-out learners, one row per window) and, like
    ``knowledge_tracing.evaluate.baseline_scores``, skips each row's first
    answer and padded positions. BKT starts from its prior in every window, as
    the deep model starts without earlier context.
    """
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(KT_DIR))
    from evaluate import metrics_for, seq_lists

    test = pd.read_csv(Path(data_config["dpath"]) / data_config["test_file"])
    truth, predicted = [], []
    for _, row in test.iterrows():
        concepts = seq_lists(row, "concepts")
        responses = seq_lists(row, "responses")
        masks = seq_lists(row, "selectmasks")
        valid = [i for i, mask in enumerate(masks) if mask != -1]
        window = [(concepts[i], responses[i]) for i in valid]
        scores = next_answer_predictions(
            [c for c, _ in window], [r for _, r in window], params
        )
        truth.extend(r for _, r in window[1:])
        predicted.extend(scores[1:])
    metrics = metrics_for(np.asarray(truth), np.asarray(predicted))
    return {**metrics, "n_targets": len(truth), "n_test_rows": int(len(test))}


def _simulator_ceiling_auc(data_dir):
    """The generator's own AUC, when the simulator recorded one next to the data."""
    for name in (config.SIMULATION_MANIFEST_NAME, config.PLAN_MANIFEST_NAME):
        path = Path(data_dir) / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8")).get("ceiling_auc")
            if value is not None:
                return value
    return None


def _run(script, *args):
    command = [sys.executable, str(KT_DIR / script), *map(str, args)]
    print(f"\n[knowledge-engine] $ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=config.PROJECT_ROOT)


def _selected_deep_run(evaluation, selection):
    """Test metrics of the checkpoint chosen on validation AUC."""
    for run in evaluation["runs"]:
        if str(run.get("model")).lower() == DEEP_MODEL and int(
            run.get("seed", -1)
        ) == int(selection["seed"]):
            return run
    raise RuntimeError("The selected checkpoint has no test evaluation")


def train(
    data_dir=config.DEFAULT_DATA_DIR,
    active_model_file=config.ACTIVE_MODEL_FILE,
    epochs=None,
    seeds=(42, 43, 44),
    split_seed=20240116,
    force=False,
):
    """Phase 2: train a deep model on the course's answers, then apply the gate.

    Returns the record written to ``active_model.json``. If the data is not
    ready (and ``force`` is false), nothing is trained and Phase 1 is recorded.
    """
    data_dir = Path(data_dir)
    events = load_answer_events(data_dir)
    ready = readiness(events, load_kc_catalog(data_dir))
    record = {
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "training_data_dir": str(data_dir),
        "training_population": detect_population(data_dir),
        "mapping_version": current_mapping_version(data_dir, events),
        "readiness": ready,
        "gate_min_auc_margin": config.GATE_MIN_AUC_MARGIN,
        "simulator_ceiling_auc": _simulator_ceiling_auc(data_dir),
    }
    if not ready["ready"] and not force:
        record.update(
            phase=1,
            estimator=BKT_MODEL_NAME,
            reason="not enough answer data to train a deep model: "
            + "; ".join(ready["unmet"]),
        )
        write_json(active_model_file, record)
        return record

    events_file = data_dir / ANSWER_EVENTS_FILE
    mapping_file = data_dir / QUESTION_KC_MAP_FILE
    _run(
        "prepare_data.py",
        "--dataset-name",
        config.DEEP_DATASET_NAME,
        "--input",
        events_file,
        "--question-mapping",
        mapping_file,
        "--output-dir",
        PREPARED_DIR,
        "--seed",
        split_seed,
    )
    _run(
        "init_pykt_dataset.py",
        "--dataset-name",
        config.DEEP_DATASET_NAME,
        "--input",
        PREPARED_DIR / "pykt_formatted_sequences.csv",
    )
    train_args = [
        "--dataset-name",
        config.DEEP_DATASET_NAME,
        "--models",
        DEEP_MODEL,
        "--seeds",
        *seeds,
    ]
    if epochs is not None:
        train_args += ["--epochs", epochs]
    _run("train.py", *train_args)
    _run(
        "evaluate.py",
        "--dataset-name",
        config.DEEP_DATASET_NAME,
        "--models",
        DEEP_MODEL,
    )

    sys.path.insert(0, str(KT_DIR))
    from kt_common import dataset_artifacts_dir, load_data_config, selected_model_path

    selection = json.loads(
        selected_model_path(config.DEEP_DATASET_NAME).read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (
            dataset_artifacts_dir(config.DEEP_DATASET_NAME)
            / "evaluation"
            / "test_comparison.json"
        ).read_text()
    )
    deep = _selected_deep_run(evaluation, selection)
    bkt = bkt_metrics_on_deep_test_split(load_data_config(config.DEEP_DATASET_NAME))
    if bkt["n_targets"] != evaluation["n_targets"]:
        raise RuntimeError(
            f"BKT was scored on {bkt['n_targets']} targets but the deep model on "
            f"{evaluation['n_targets']}; the gate needs identical targets"
        )
    promoted, reason = decide_promotion(bkt["auc"], deep["auc"])
    record.update(
        phase=3 if promoted else 2,
        estimator=DEEP_MODEL if promoted else BKT_MODEL_NAME,
        reason=reason,
        training_data_sha256=sha256_file(events_file),
        gate={
            "n_test_learners": evaluation["n_test_learners"],
            "n_targets": evaluation["n_targets"],
            "bkt": {k: bkt[k] for k in ("auc", "accuracy", "log_loss")},
            "deep": {k: deep[k] for k in ("auc", "accuracy", "log_loss")},
            "other_baselines": evaluation.get("baselines", []),
        },
        deep_model={
            "model_name": selection["model_name"],
            "checkpoint_id": selection["checkpoint_id"],
            "checkpoint": selection["checkpoint"],
            "validation_metrics": selection["validation_metrics"],
        },
    )
    write_json(active_model_file, record)
    return record
