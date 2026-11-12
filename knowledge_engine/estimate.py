"""Produce the learner-by-KC knowledge states for one data directory."""

from datetime import datetime, timezone
from pathlib import Path

from . import bkt, config, deep
from .data import (
    current_mapping_version,
    detect_population,
    load_answer_events,
    load_kc_catalog,
    load_roster_ids,
    normalize_as_of,
    write_json,
)
from .phase import load_active_model


def choose_estimator(active_model, population, mapping_version):
    """Return ``("deep" | "bkt", reason)``.

    The promoted deep model is used only for the population and question
    mapping it was trained on. Anything else falls back to BKT, so a model
    trained on simulated learners is never applied to real ones, and a changed
    question bank forces a retrain instead of silent mis-scoring.
    """
    if active_model.get("phase") != 3:
        return "bkt", active_model.get("reason", "no promoted deep model")
    if active_model.get("training_population") != population:
        return "bkt", (
            f"the promoted model was trained on {active_model.get('training_population')!r} "
            f"data, not {population!r}"
        )
    if active_model.get("mapping_version") != mapping_version:
        return (
            "bkt",
            "the question-to-KC mapping changed since training; retrain required",
        )
    return "deep", active_model.get("reason", "promoted deep model")


def estimate(
    data_dir=config.DEFAULT_DATA_DIR,
    output_dir=config.DEFAULT_STATES_DIR,
    as_of=None,
    active_model_file=config.ACTIVE_MODEL_FILE,
    inference_timestamp=None,
):
    """Write ``learner_kc_state.csv`` and its manifest; return the states."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    cutoff = normalize_as_of(as_of)
    as_of_iso = None if cutoff is None else cutoff.isoformat()
    inference_timestamp = inference_timestamp or datetime.now(timezone.utc).isoformat()
    population = detect_population(data_dir)
    events = load_answer_events(data_dir, as_of_iso)
    mapping_version = current_mapping_version(data_dir, events)
    active = load_active_model(active_model_file)
    estimator, reason = choose_estimator(active, population, mapping_version)

    if estimator == "deep":
        states = deep.estimate_states(
            data_dir,
            output_dir / "deep_export",
            as_of=as_of_iso,
            mapping_version=mapping_version,
            source_population=population,
        )
    else:
        states = bkt.estimate_states(
            events,
            load_kc_catalog(data_dir),
            load_roster_ids(data_dir, as_of_iso),
            inference_timestamp,
            as_of=as_of_iso,
            mapping_version=mapping_version,
            source_population=population,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    states.to_csv(output_dir / config.STATE_FILE_NAME, index=False)
    manifest = {
        "generated_at": inference_timestamp,
        "data_dir": str(data_dir),
        "source_population": population,
        "as_of": as_of_iso,
        "mapping_version": mapping_version,
        "estimator": estimator,
        "estimator_reason": reason,
        "recorded_phase": active.get("phase", 1),
        "estimator_phase": (
            int(states["estimator_phase"].iloc[0]) if len(states) else None
        ),
        "model_names": sorted(states["model_name"].dropna().unique().tolist()),
        "learners": int(states["learner_id"].nunique()),
        "kcs": int(states["kc_id"].nunique()),
        "rows": int(len(states)),
        "state_status_counts": {
            str(k): int(v) for k, v in states["state_status"].value_counts().items()
        },
        "bkt_parameters": (
            {
                "p_init": bkt.DEFAULT_PARAMS.p_init,
                "p_learn": bkt.DEFAULT_PARAMS.p_learn,
                "p_slip": bkt.DEFAULT_PARAMS.p_slip,
                "p_guess": bkt.DEFAULT_PARAMS.p_guess,
                "p_forget": bkt.DEFAULT_PARAMS.p_forget,
                "note": "fixed, uncalibrated configuration values",
            }
            if estimator == "bkt"
            else None
        ),
        "semantics": {
            "predicted_correctness": "probability of answering the next question on the KC correctly",
            "p_mastery": "BKT latent probability that the KC is known; null for the deep model",
            "cold_start": "prior only: no answer on that KC, not individual evidence",
        },
    }
    write_json(output_dir / config.MANIFEST_FILE_NAME, manifest)
    return states
