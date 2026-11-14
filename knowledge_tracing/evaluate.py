#!/usr/bin/env python3
"""
Step 2.3 - evaluate trained DKT / AKT / SAKT on the held-out TEST split only
(fold 2 in the Milestone 1 CSV; fold -1 in pyKT's test_sequences.csv).

Metrics (pooled over all test prediction targets):
    AUC       ROC-AUC of P(correct) vs. the observed response
    Accuracy  threshold 0.5
    Log loss  binary cross-entropy (probabilities clipped to [1e-7, 1-1e-7])

pyKT's evaluate() returns AUC and accuracy only. It can also dump the per-sequence
predictions, so log loss is computed from that dump and AUC is cross-checked
against pyKT's own number.

A prediction target is every answer except each learner's first one (nothing precedes
it), so the test set has (#test answers - #test learners) targets.

Two non-neural baselines are scored on exactly the same targets:
    constant  : train-set mean correctness for every prediction
    kc_mean   : train-set mean correctness of the target's KC

Usage (project root):
    python knowledge_tracing/evaluate.py [--models dkt akt sakt]

Outputs in knowledge_tracing/artifacts/evaluation/ :
    test_comparison.md / .csv / .json   and   predictions/<model>_seed<k>_test_predictions.txt
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_common import (
    ARTIFACTS_DIR,
    DATASET_NAME,
    EMB_TYPE,
    dataset_artifacts_dir,
    load_data_config,
    make_dataset,
    TRAIN_FOLD,
)

EVAL_DIR = ARTIFACTS_DIR / "evaluation"
EPS = 1e-7


def metrics_for(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob = np.clip(y_prob, EPS, 1 - EPS)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def parse_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read pyKT's per-sequence prediction dump: [qs, rs, ds, ts, ps, prelabels, auc, acc] per line."""
    ts, ps = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        # The per-sequence AUC field is the string `nan` when a learner's targets are all
        # one class; literal_eval cannot read it. That field is unused, so map it to None.
        rec = ast.literal_eval(line.replace("nan", "None"))
        ts.extend(rec[3])
        ps.extend(rec[4])
    p = np.asarray(ps, dtype=float)
    if np.isnan(p).any():
        raise RuntimeError(f"NaN probabilities in {path}")
    return np.asarray(ts, dtype=int), p


def seq_lists(row: pd.Series, field: str) -> list[int]:
    return [int(v) for v in str(row[field]).split(",")]


def baseline_scores(data_cfg: dict) -> tuple[list[dict], int, int]:
    """Constant and per-KC-mean baselines on the same targets as the models."""
    dpath = Path(data_cfg["dpath"])
    tv = pd.read_csv(dpath / data_cfg["train_valid_file"])
    te = pd.read_csv(dpath / data_cfg["test_file"])

    kc_sum, kc_n, tot_sum, tot_n = {}, {}, 0, 0
    for _, row in tv[tv["fold"] == TRAIN_FOLD].iterrows():
        for c, r, m in zip(
            seq_lists(row, "concepts"),
            seq_lists(row, "responses"),
            seq_lists(row, "selectmasks"),
        ):
            if m == -1:
                continue
            kc_sum[c] = kc_sum.get(c, 0) + r
            kc_n[c] = kc_n.get(c, 0) + 1
            tot_sum += r
            tot_n += 1
    global_mean = tot_sum / tot_n

    y, p_const, p_kc = [], [], []
    for _, row in te.iterrows():
        cs, rs, ms = (
            seq_lists(row, "concepts"),
            seq_lists(row, "responses"),
            seq_lists(row, "selectmasks"),
        )
        for c, r, m in list(zip(cs, rs, ms))[1:]:  # skip the first answer
            if m == -1:
                continue
            y.append(r)
            p_const.append(global_mean)
            p_kc.append(kc_sum[c] / kc_n[c] if c in kc_n else global_mean)
    y = np.asarray(y)
    rows = [
        {
            "model": "constant (train mean)",
            "seed": "-",
            **metrics_for(y, np.asarray(p_const)),
        },
        {
            "model": "kc_mean (train, per KC)",
            "seed": "-",
            **metrics_for(y, np.asarray(p_kc)),
        },
    ]
    return rows, len(y), len(te)


def evaluate_run(
    model_name: str, seed_dir: Path, data_cfg: dict, loader, device, eval_dir=EVAL_DIR
) -> dict:
    import torch
    from pykt.models import evaluate, init_model

    summary = json.loads((seed_dir / "run_summary.json").read_text())
    model = init_model(model_name, summary["model_config"], data_cfg, EMB_TYPE)
    model.load_state_dict(
        torch.load(seed_dir / f"{EMB_TYPE}_model.ckpt", map_location=device)
    )
    model.eval()

    pred_dir = eval_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{model_name}_{seed_dir.name}_test_predictions.txt"
    pykt_auc, pykt_acc = evaluate(model, loader, model_name, str(pred_path))

    y, p = parse_predictions(pred_path)
    m = metrics_for(y, p)
    if abs(m["auc"] - pykt_auc) > 1e-6:
        raise RuntimeError(
            f"AUC mismatch for {model_name}/{seed_dir.name}: mine {m['auc']} vs pyKT {pykt_auc}"
        )
    return {
        "model": model_name,
        "seed": summary["seed"],
        "n_targets": int(len(y)),
        "best_epoch": summary["best_epoch"],
        "valid_auc": summary["valid_auc"],
        **m,
    }


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+", default=["dkt", "akt", "sakt"])
    ap.add_argument("--dataset-name", default=DATASET_NAME)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = load_data_config(args.dataset_name)
    eval_dir = dataset_artifacts_dir(args.dataset_name) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    base_rows, n_targets_csv, n_test_learners = baseline_scores(data_cfg)
    test_loader = DataLoader(
        make_dataset("test", data_cfg), batch_size=64, shuffle=False
    )

    per_run = []
    for model_name in args.models:
        seed_dirs = sorted(
            (dataset_artifacts_dir(args.dataset_name) / model_name).glob("seed*/")
        )
        seed_dirs = [d for d in seed_dirs if (d / f"{EMB_TYPE}_model.ckpt").exists()]
        if not seed_dirs:
            print(
                f"WARNING: no trained checkpoints for {model_name}; run train.py first"
            )
        for d in seed_dirs:
            r = evaluate_run(model_name, d, data_cfg, test_loader, device, eval_dir)
            if r["n_targets"] != n_targets_csv:
                raise RuntimeError(
                    f"{model_name}/{d.name}: {r['n_targets']} targets, expected {n_targets_csv}"
                )
            per_run.append(r)
            print(
                f"{model_name:<5} {d.name:<8} AUC {r['auc']:.4f}  ACC {r['accuracy']:.4f}  LogLoss {r['log_loss']:.4f}"
            )

    if not per_run:
        sys.exit("ERROR: nothing to evaluate")

    runs = pd.DataFrame(per_run)
    agg = []
    for name, g in runs.groupby("model", sort=False):
        row = {"model": name.upper(), "seeds": len(g)}
        for k, label in [
            ("auc", "AUC"),
            ("accuracy", "Accuracy"),
            ("log_loss", "Log loss"),
        ]:
            sd = g[k].std(ddof=1) if len(g) > 1 else float("nan")
            row[label] = f"{g[k].mean():.4f}" + (f" ± {sd:.4f}" if len(g) > 1 else "")
        agg.append(row)
    for b in base_rows:
        agg.append(
            {
                "model": b["model"],
                "seeds": "-",
                "AUC": f"{b['auc']:.4f}",
                "Accuracy": f"{b['accuracy']:.4f}",
                "Log loss": f"{b['log_loss']:.4f}",
            }
        )
    summary = pd.DataFrame(agg)

    header = (
        f"# Test-split comparison (held-out learners only)\n\n"
        f"- Test learners: {n_test_learners}; prediction targets: {n_targets_csv:,} "
        f"(every answer except each learner's first)\n"
        f"- Model selection used the validation fold only; the test split was not used in training\n"
        f"- Mean ± sample std over seeds (std shown only when more than one seed)\n"
        + (
            "- Synthetic dataset: absolute numbers are properties of the generator, not of real learners. "
            if args.dataset_name == DATASET_NAME
            else f"- Dataset: `{args.dataset_name}`; conclusions apply to this population and its mapped items.\n"
        )
        + "DKT/SAKT predict from KC ids; AKT also receives item ids, so the models do not see identical inputs.\n"
    )
    md = (
        header
        + "\n## Summary\n\n"
        + to_markdown(summary)
        + "\n\n## Per run\n\n"
        + to_markdown(runs.round(4))
        + "\n"
    )
    (eval_dir / "test_comparison.md").write_text(md)
    pd.concat([runs, pd.DataFrame(base_rows)], ignore_index=True).to_csv(
        eval_dir / "test_comparison.csv", index=False
    )
    (eval_dir / "test_comparison.json").write_text(
        json.dumps(
            {
                "n_test_learners": n_test_learners,
                "n_targets": n_targets_csv,
                "runs": per_run,
                "baselines": base_rows,
            },
            indent=2,
        )
    )

    print("\n" + md)
    print(f"Saved to {eval_dir}")


if __name__ == "__main__":
    main()
