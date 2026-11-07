#!/usr/bin/env python3
"""
Step 2.2 - train DKT, AKT and SAKT with pykt-toolkit on the learner-level split.

Model selection uses ONLY the validation fold (fold 1). The test split is never
loaded here; it is touched once, in evaluate.py.

Usage (project root):
    python knowledge_tracing/train.py                          # all three models, seed 42
    python knowledge_tracing/train.py --models akt             # one model
    python knowledge_tracing/train.py --seeds 42 43 44         # 3 seeds -> mean/std in evaluate.py
    python knowledge_tracing/train.py --epochs 2               # quick smoke test

Per run, in knowledge_tracing/artifacts/<model>/seed<k>/ :
    qid_model.ckpt   best-validation-AUC state_dict written by pyKT's train_model
    best_model.pth   identical copy under the .pth name
    train.log        full console log of the run
    run_summary.json best epoch, validation AUC/ACC, hyperparameters, versions
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_common import (
    ARTIFACTS_DIR,
    EMB_TYPE,
    load_data_config,
    load_model_configs,
    make_dataset,
    set_seed,
)

SUPPORTED = ["dkt", "akt", "sakt"]


class Tee:
    """Write to the console and a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def train_one(
    model_name: str,
    seed: int,
    epochs_override: int | None,
    data_cfg: dict,
    model_cfgs: dict,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from pykt.models import init_model, train_model

    mcfg = model_cfgs[model_name]["model_config"]
    tcfg = dict(model_cfgs[model_name]["training"])
    if epochs_override:
        tcfg["num_epochs"] = epochs_override

    run_dir = ARTIFACTS_DIR / model_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "train.log", "w") as logf, contextlib.redirect_stdout(
        Tee(sys.stdout, logf)
    ):
        print(f"=== {model_name.upper()} | seed {seed} | {tcfg} ===")
        set_seed(seed)

        train_ds, valid_ds = make_dataset("train", data_cfg), make_dataset(
            "valid", data_cfg
        )
        # pyKT's own loaders do not shuffle the training set; shuffle (seeded) here.
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_ds, batch_size=tcfg["batch_size"], shuffle=True, generator=g
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=tcfg["batch_size"], shuffle=False
        )
        print(f"train learners: {len(train_ds)} | valid learners: {len(valid_ds)}")

        model = init_model(model_name, mcfg, data_cfg, EMB_TYPE)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])

        t0 = time.time()
        # test_loader=None on purpose: model selection must not see the test split.
        out = train_model(
            model,
            train_loader,
            valid_loader,
            tcfg["num_epochs"],
            opt,
            str(run_dir),
            None,
            None,
            True,
        )
        valid_auc, valid_acc, best_epoch = out[4], out[5], out[6]
        elapsed = time.time() - t0

        ckpt = run_dir / f"{EMB_TYPE}_model.ckpt"
        if not ckpt.exists():
            raise RuntimeError(f"pyKT did not write {ckpt}")
        shutil.copy(ckpt, run_dir / "best_model.pth")

        summary = {
            "model": model_name,
            "seed": seed,
            "best_epoch": int(best_epoch),
            "valid_auc": float(valid_auc),
            "valid_acc": float(valid_acc),
            "n_params": int(n_params),
            "train_seconds": round(elapsed, 1),
            "model_config": mcfg,
            "training": tcfg,
            "emb_type": EMB_TYPE,
            "pykt_toolkit_version": metadata.version("pykt-toolkit"),
            "torch_version": torch.__version__,
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"--- done: best epoch {best_epoch}, valid AUC {valid_auc:.4f}, valid ACC {valid_acc:.4f}, "
            f"{elapsed:.0f}s -> {run_dir}"
        )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+", default=SUPPORTED, choices=SUPPORTED)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override num_epochs from model_configs.json",
    )
    args = ap.parse_args()

    data_cfg = load_data_config()
    model_cfgs = load_model_configs()
    rows = [
        train_one(m, s, args.epochs, data_cfg, model_cfgs)
        for m in args.models
        for s in args.seeds
    ]

    print("\n" + "=" * 60 + "\nTRAINING SUMMARY (validation fold only)\n" + "=" * 60)
    for r in rows:
        print(
            f"{r['model']:<5} seed {r['seed']:<4} best epoch {r['best_epoch']:<3} "
            f"valid AUC {r['valid_auc']:.4f}  valid ACC {r['valid_acc']:.4f}  "
            f"params {r['n_params']:,}  {r['train_seconds']}s"
        )
    print("Next: python knowledge_tracing/evaluate.py")


if __name__ == "__main__":
    main()
