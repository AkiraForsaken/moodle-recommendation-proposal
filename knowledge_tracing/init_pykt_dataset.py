#!/usr/bin/env python3
"""
Step 2.1 - bridge the Milestone 1 CSV into files pyKT's KTDataset can read,
keeping the learner-level folds exactly as assigned.

Why a bridge is needed (checked against pykt-toolkit 0.0.38 source):
  * KTDataset builds tensors with LongTensor(list_of_lists), so every sequence must
    have the same length -> pad to `maxlen` with -1 (pyKT's padding value).
  * KTDataset does int(...) on every `usetimes` entry, so "10.19" would crash ->
    response_time_s is stored here as integer milliseconds (not used by DKT/AKT/SAKT).
  * KTDataset reads one train/valid file where `fold` picks the rows, and a
    separate test file whose rows have fold == -1.
pyKT's own preprocessing scripts are for its built-in datasets and create their own
folds, so they are deliberately NOT used: that would re-split the learners.

Mapping of the Milestone 1 folds  ->  pyKT files:
    fold 0 (train) -> train_valid_sequences.csv  fold 0
    fold 1 (valid) -> train_valid_sequences.csv  fold 1
    fold 2 (test)  -> test_sequences.csv         fold -1

Usage (project root):
    python knowledge_tracing/init_pykt_dataset.py
Outputs:
    data/processed/pykt_english/{train_valid_sequences.csv,test_sequences.csv,bridge_manifest.json}
    knowledge_tracing/configs/data_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_common import CONFIG_DIR, DATASET_NAME, PROJECT_ROOT

SEQ_FIELDS = [
    "questions",
    "concepts",
    "responses",
    "timestamps",
    "usetimes",
    "selectmasks",
]
M1_FOLDS = {0: "train", 1: "valid", 2: "test"}
EXPECTED_LEARNERS = {"train": 840, "valid": 120, "test": 240}
PAD = -1


def to_int_list(row: pd.Series, field: str) -> list[int]:
    parts = str(row[field]).split(",")
    if field == "usetimes":
        return [int(round(float(x) * 1000)) for x in parts]  # seconds -> integer ms
    return [int(x) for x in parts]


def pad(values: list[int], maxlen: int) -> str:
    return ",".join(str(v) for v in values + [PAD] * (maxlen - len(values)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/processed/pykt_formatted_sequences.csv",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/processed/pykt_english"
    )
    ap.add_argument("--maxlen", type=int, default=200)
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="warn instead of failing on split-size mismatch",
    )
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(
            f"ERROR: {args.input} not found - run Milestone 1 (prepare_data.py) first."
        )
    df = pd.read_csv(args.input)
    proc_dir = args.input.parent

    # ---- split / leakage checks ------------------------------------------------
    missing = [
        c for c in ["fold", "uid", "learner_id", *SEQ_FIELDS] if c not in df.columns
    ]
    if missing:
        sys.exit(f"ERROR: input is missing columns {missing}")
    if not set(df["fold"].unique()) <= set(M1_FOLDS):
        sys.exit(f"ERROR: unexpected fold values {sorted(df['fold'].unique())}")
    if df["learner_id"].duplicated().any():
        sys.exit("ERROR: a learner appears in more than one row - splits could leak")
    sizes = {M1_FOLDS[f]: int((df["fold"] == f).sum()) for f in M1_FOLDS}
    print(f"Learners per fold: {sizes}")
    if len(df) == 1200 and sizes != EXPECTED_LEARNERS:
        msg = f"split sizes {sizes} differ from expected {EXPECTED_LEARNERS}"
        if args.no_strict:
            print("WARNING:", msg)
        else:
            sys.exit("ERROR: " + msg)

    # ---- pad + convert -----------------------------------------------------------
    out_rows = []
    for _, row in df.iterrows():
        seqs = {f: to_int_list(row, f) for f in SEQ_FIELDS}
        n = len(seqs["responses"])
        if any(len(v) != n for v in seqs.values()):
            sys.exit(
                f"ERROR: learner {row['learner_id']} has fields of different lengths"
            )
        if n > args.maxlen:
            sys.exit(
                f"ERROR: learner {row['learner_id']} has {n} > maxlen={args.maxlen} interactions"
            )
        if set(seqs["responses"]) - {0, 1}:
            sys.exit(f"ERROR: learner {row['learner_id']} has non-binary responses")
        fold = {0: 0, 1: 1, 2: -1}[int(row["fold"])]
        out_rows.append(
            {
                "fold": fold,
                "uid": int(row["uid"]),
                **{f: pad(seqs[f], args.maxlen) for f in SEQ_FIELDS},
                "learner_id": row["learner_id"],
            }
        )
    out = pd.DataFrame(out_rows)
    cols = ["fold", "uid", *SEQ_FIELDS, "learner_id"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # pyKT caches processed data as "<csv>_<folds>.pkl" next to the CSV and never
    # invalidates it. Remove stale caches so a regenerated CSV is actually used.
    for stale in args.output_dir.glob("*.pkl"):
        stale.unlink()
    out[out["fold"].isin([0, 1])][cols].to_csv(
        args.output_dir / "train_valid_sequences.csv", index=False
    )
    out[out["fold"] == -1][cols].to_csv(
        args.output_dir / "test_sequences.csv", index=False
    )

    # ---- vocabulary sizes from the Milestone 1 maps ------------------------------
    num_q = len(pd.read_csv(proc_dir / "question_map.csv"))
    num_c = len(pd.read_csv(proc_dir / "concept_map.csv"))
    max_q = max(int(v) for s in df["questions"] for v in s.split(","))
    max_c = max(int(v) for s in df["concepts"] for v in s.split(","))
    if max_q >= num_q or max_c >= num_c:
        sys.exit("ERROR: sequence ids exceed the question/concept maps")

    try:
        rel_dpath = str(args.output_dir.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        rel_dpath = str(args.output_dir.resolve())
    data_config = {
        DATASET_NAME: {
            "dpath": rel_dpath,  # resolved against the project root at runtime
            "num_q": num_q,
            "num_c": num_c,
            "max_concepts": 1,  # one KC per item in this dataset
            "input_type": ["questions", "concepts"],
            "maxlen": args.maxlen,
            "emb_path": "",
            "train_valid_file": "train_valid_sequences.csv",
            "test_file": "test_sequences.csv",
            "folds": [0, 1],
            "fold_meaning": {"0": "train", "1": "valid", "-1": "test (separate file)"},
        }
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "data_config.json").write_text(json.dumps(data_config, indent=2))
    manifest = {
        "source": str(args.input),
        "learners_per_split": sizes,
        "maxlen": args.maxlen,
        "pad_value": PAD,
        "usetimes_unit": "milliseconds (integer)",
        "num_q": num_q,
        "num_c": num_c,
    }
    (args.output_dir / "bridge_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print(f"num_q={num_q}  num_c={num_c}  maxlen={args.maxlen}")
    print(
        f"Wrote {args.output_dir}/train_valid_sequences.csv "
        f"({sizes['train'] + sizes['valid']} learners: folds 0=train, 1=valid)"
    )
    print(
        f"Wrote {args.output_dir}/test_sequences.csv ({sizes['test']} learners: fold -1 = test)"
    )
    print(f"Wrote {CONFIG_DIR / 'data_config.json'}")


if __name__ == "__main__":
    main()
