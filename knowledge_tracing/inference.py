#!/usr/bin/env python3
"""Load the selected pyKT checkpoint and score AKT target questions."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from kt_common import CONFIG_DIR, EMB_TYPE, PROJECT_ROOT, load_data_config


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_model(device: str | None = None):
    """Load model, checkpoint, and selection metadata without training."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "Inference requires PyTorch. Install it in the active Python environment."
            ) from exc
        raise

    try:
        from pykt.models import init_model
    except ModuleNotFoundError as exc:
        if exc.name in {"pykt", "pykt.models"}:
            raise RuntimeError(
                "Inference requires pykt-toolkit==0.0.38. Install it in the active "
                "Python environment."
            ) from exc
        raise

    selection_path = CONFIG_DIR / "selected_model.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Selected model config not found: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("model_name") != "akt":
        raise ValueError("This inference implementation currently supports AKT only")
    try:
        runtime_pykt_version = version("pykt-toolkit")
    except PackageNotFoundError as exc:
        raise RuntimeError("The installed pyKT package has no version metadata") from exc
    expected_pykt_version = selection.get("pykt_toolkit_version")
    if expected_pykt_version and runtime_pykt_version != expected_pykt_version:
        raise RuntimeError(
            f"Selected checkpoint requires pykt-toolkit=={expected_pykt_version}; "
            f"found {runtime_pykt_version}"
        )

    mapping_paths = {
        "question_map": PROJECT_ROOT / "data/processed/question_map.csv",
        "concept_map": PROJECT_ROOT / "data/processed/concept_map.csv",
    }
    expected_mapping_hashes = selection.get("mapping_sha256", {})
    for mapping_name, mapping_path in mapping_paths.items():
        if not mapping_path.is_file():
            raise FileNotFoundError(f"Training {mapping_name} not found: {mapping_path}")
        expected_hash = expected_mapping_hashes.get(mapping_name)
        if not expected_hash or sha256_file(mapping_path) != expected_hash:
            raise ValueError(
                f"{mapping_name} differs from the mapping used to select this "
                "checkpoint; retrain or restore the matching Milestone 1 maps."
            )

    checkpoint = (PROJECT_ROOT / selection["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint}")
    data_config = load_data_config()
    model = init_model("akt", selection["model_config"], data_config, EMB_TYPE)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state = torch.load(checkpoint, map_location=target_device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=target_device)
    model.load_state_dict(state)
    model.to(target_device)
    model.eval()

    digest = sha256_file(checkpoint)
    selection = dict(selection)
    selection["runtime_versions"] = {
        "pykt_toolkit_version": runtime_pykt_version,
        "torch_version": torch.__version__,
    }
    return model, target_device, selection, digest


def score_akt_questions(
    model,
    device: str,
    concepts: list[int],
    responses: list[int],
    history_questions: list[int],
    target_questions: list[int],
    target_concepts: list[int],
    maxlen: int = 200,
    batch_size: int = 128,
) -> np.ndarray:
    """Score candidate items given a chronological interaction history.

    For AKT, concepts are passed as the response-history sequence and question
    IDs as pid_data. The placeholder response appended for each candidate is
    masked by AKT when producing that position's prediction.
    """
    import torch

    if not (len(concepts) == len(responses) == len(history_questions)):
        raise ValueError("History concept, response, and question sequences must align")
    if len(target_questions) != len(target_concepts):
        raise ValueError("Target question and concept sequences must align")
    if not target_questions:
        return np.asarray([], dtype=float)
    if maxlen < 1 or batch_size < 1:
        raise ValueError("maxlen and batch_size must be positive")

    history_limit = maxlen - 1
    concepts = concepts[-history_limit:] if history_limit else []
    responses = responses[-history_limit:] if history_limit else []
    history_questions = history_questions[-history_limit:] if history_limit else []
    predictions: list[float] = []

    with torch.inference_mode():
        for start in range(0, len(target_questions), batch_size):
            questions = target_questions[start : start + batch_size]
            target_cs = target_concepts[start : start + batch_size]
            count = len(questions)
            concept_tensor = torch.tensor(
                [concepts + [concept] for concept in target_cs],
                dtype=torch.long,
                device=device,
            )
            response_tensor = torch.tensor(
                [responses + [1] for _ in range(count)],
                dtype=torch.long,
                device=device,
            )
            question_tensor = torch.tensor(
                [history_questions + [question] for question in questions],
                dtype=torch.long,
                device=device,
            )
            outputs = model(concept_tensor, response_tensor, question_tensor)
            probabilities = outputs[0] if isinstance(outputs, tuple) else outputs
            predictions.extend(probabilities[:, -1].detach().cpu().tolist())

    result = np.asarray(predictions, dtype=float)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("AKT returned invalid correctness probabilities")
    return result


def score_akt_next_interactions(
    model,
    device: str,
    history_concepts: list[list[int]],
    history_responses: list[list[int]],
    history_questions: list[list[int]],
    target_questions: list[int],
    target_concepts: list[int],
    maxlen: int = 200,
    batch_size: int = 128,
) -> np.ndarray:
    """Score next interactions with a distinct chronological history per row."""
    import torch

    n_targets = len(target_questions)
    if not (
        len(history_concepts)
        == len(history_responses)
        == len(history_questions)
        == len(target_concepts)
        == n_targets
    ):
        raise ValueError("Each test target must have one aligned history and KC")
    if any(
        not (len(concepts) == len(responses) == len(questions))
        for concepts, responses, questions in zip(
            history_concepts, history_responses, history_questions
        )
    ):
        raise ValueError("Each interaction history must contain aligned sequences")
    if not n_targets:
        return np.asarray([], dtype=float)
    if maxlen < 1 or batch_size < 1:
        raise ValueError("maxlen and batch_size must be positive")

    history_limit = maxlen - 1
    groups: dict[int, list[int]] = {}
    trimmed_histories = []
    for index, (concepts, responses, questions) in enumerate(
        zip(history_concepts, history_responses, history_questions)
    ):
        if history_limit:
            concepts = concepts[-history_limit:]
            responses = responses[-history_limit:]
            questions = questions[-history_limit:]
        else:
            concepts, responses, questions = [], [], []
        trimmed_histories.append((concepts, responses, questions))
        groups.setdefault(len(concepts), []).append(index)

    predictions = np.empty(n_targets, dtype=float)
    with torch.inference_mode():
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                concept_tensor = torch.tensor(
                    [
                        trimmed_histories[index][0] + [target_concepts[index]]
                        for index in batch_indices
                    ],
                    dtype=torch.long,
                    device=device,
                )
                response_tensor = torch.tensor(
                    [
                        trimmed_histories[index][1] + [1]
                        for index in batch_indices
                    ],
                    dtype=torch.long,
                    device=device,
                )
                question_tensor = torch.tensor(
                    [
                        trimmed_histories[index][2] + [target_questions[index]]
                        for index in batch_indices
                    ],
                    dtype=torch.long,
                    device=device,
                )
                outputs = model(concept_tensor, response_tensor, question_tensor)
                probabilities = outputs[0] if isinstance(outputs, tuple) else outputs
                batch_scores = (
                    probabilities[:, -1].detach().cpu().numpy().astype(float)
                )
                predictions[batch_indices] = batch_scores

    if not np.isfinite(predictions).all() or (
        (predictions < 0) | (predictions > 1)
    ).any():
        raise ValueError("AKT returned invalid correctness probabilities")
    return predictions