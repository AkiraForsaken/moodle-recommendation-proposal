"""
Shared helpers for Milestone 2 (init_pykt_dataset.py, train.py, evaluate.py).

Everything pyKT-specific that is easy to get wrong lives here once:
  * where files live (relative to the project root, so Colab and local runs agree)
  * how datasets are loaded for each split (train = fold 0, valid = fold 1, test = fold -1)
  * seeding and the CPU shim (see apply_cpu_shim)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def _guard_pykt_turtle_import() -> None:
    """
    pykt-toolkit 0.0.38's qdkt.py contains `from turtle import forward` (an accidental
    IDE auto-import). That needs tkinter, so `import pykt` fails on Pythons built
    without it. Provide an inert stand-in only when the real module is unavailable.
    """
    try:
        import turtle  # noqa: F401
    except Exception:
        import sys
        import types

        stub = types.ModuleType("turtle")
        stub.forward = lambda *a, **k: None
        sys.modules["turtle"] = stub


_guard_pykt_turtle_import()

KT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = KT_DIR.parent
CONFIG_DIR = KT_DIR / "configs"
ARTIFACTS_DIR = KT_DIR / "artifacts"

DATASET_NAME = "english_kt"
EMB_TYPE = "qid"  # pyKT checkpoint is written as "<emb_type>_model.ckpt"

# Folds inside the pyKT-ready files written by init_pykt_dataset.py.
#   train_valid_sequences.csv : fold 0 = train, fold 1 = valid
#   test_sequences.csv        : fold -1 (pyKT's convention for the test set)
TRAIN_FOLD, VALID_FOLD, TEST_FOLD = 0, 1, -1


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_data_config(dataset_name: str = DATASET_NAME) -> dict:
    """Return the *inner* pyKT data config with an absolute dpath."""
    path = CONFIG_DIR / "data_config.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found - run `python knowledge_tracing/init_pykt_dataset.py` first."
        )
    configs = json.loads(path.read_text())
    if dataset_name not in configs:
        raise ValueError(
            f"Dataset {dataset_name!r} is not configured in {path}; "
            "run init_pykt_dataset.py for that dataset first."
        )
    cfg = configs[dataset_name]
    cfg = dict(cfg)
    cfg["dpath"] = str((PROJECT_ROOT / cfg["dpath"]).resolve())
    if "mapping_dir" in cfg:
        mapping_dir = Path(cfg["mapping_dir"])
        cfg["mapping_dir"] = str(
            mapping_dir if mapping_dir.is_absolute() else PROJECT_ROOT / mapping_dir
        )
    return cfg


def dataset_artifacts_dir(dataset_name: str = DATASET_NAME) -> Path:
    """Keep the original English artifact layout; isolate other datasets."""
    return ARTIFACTS_DIR if dataset_name == DATASET_NAME else ARTIFACTS_DIR / dataset_name


def selected_model_path(dataset_name: str = DATASET_NAME) -> Path:
    filename = "selected_model.json" if dataset_name == DATASET_NAME else (
        f"{dataset_name}_selected_model.json"
    )
    return CONFIG_DIR / filename


def load_model_configs() -> dict:
    cfg = json.loads((CONFIG_DIR / "model_configs.json").read_text())
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# Reproducibility / environment
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_cpu_shim() -> bool:
    """
    pyKT's KTDataset does `from torch.cuda import LongTensor, FloatTensor`, so it
    can only build datasets on a CUDA machine. When no GPU is present (local CPU
    smoke tests) point those names at the CPU tensor types instead. This is a
    no-op on a GPU runtime such as Colab. Returns True if the shim was applied.
    """
    import torch

    if torch.cuda.is_available():
        return False
    import pykt.datasets.data_loader as dl

    dl.LongTensor = torch.LongTensor
    dl.FloatTensor = torch.FloatTensor
    return True


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def make_dataset(split: str, data_cfg: dict):
    """
    Build a pyKT KTDataset for exactly one split, reading only the rows whose
    `fold` matches. 'test' is never read by train.py.
    """
    from pykt.datasets.data_loader import KTDataset

    apply_cpu_shim()
    dpath = Path(data_cfg["dpath"])
    if split == "train":
        return KTDataset(
            str(dpath / data_cfg["train_valid_file"]),
            data_cfg["input_type"],
            {TRAIN_FOLD},
        )
    if split == "valid":
        return KTDataset(
            str(dpath / data_cfg["train_valid_file"]),
            data_cfg["input_type"],
            {VALID_FOLD},
        )
    if split == "test":
        return KTDataset(
            str(dpath / data_cfg["test_file"]), data_cfg["input_type"], {TEST_FOLD}
        )
    raise ValueError(f"unknown split {split!r}")
