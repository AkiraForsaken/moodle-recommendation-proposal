#!/usr/bin/env python3
"""
Milestone 1 - pyKT data preparation for the English Learning Behaviour Log.

Transforms the raw event log into the per-learner sequence CSV used by pyKT
(fold, uid, questions, concepts, responses, timestamps, usetimes, selectmasks)
and writes the ID maps needed to link items / KCs to the Moodle catalogue later.

Standalone: touches nothing in learner_model/ or recommender/.

Usage (from the project root):
    python knowledge_tracing/prepare_data.py

Options:
    --input        raw CSV            (default: data/raw/english_learning_events.csv)
    --output-dir   output directory   (default: data/processed)
    --seed         split seed         (default: 20240116, the dataset generator seed)
    --timestamps-ms  write timestamps in milliseconds instead of seconds
    --no-strict    downgrade expected-count mismatches from errors to warnings

Outputs (in --output-dir):
    pykt_formatted_sequences.csv   one row per learner
    question_map.csv               item_id -> questions index (+ its KC)
    concept_map.csv                kc_id   -> concepts index
    split_assignments.csv          learner_id -> split / fold (reproducibility record)
    prepare_data_manifest.json     seed, ratios, counts, expected-vs-actual checks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "english_learning_events.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_SEED = 20240116  # seed reported in dataset_summary.json
MAX_SEQ_LEN = 200  # max observed is 185, so truncation should be a no-op
SPLIT_RATIOS = {"train": 0.70, "valid": 0.10, "test": 0.20}
FOLD_CODE = {"train": 0, "valid": 1, "test": 2}  # numeric `fold` column

EXPECTED = {
    "raw_rows": 138_523,
    "answer_rows": 75_006,
    "learners": 1_200,
    "concepts": 96,
    "questions": 2_189,
}

REQUIRED_COLUMNS = [
    "learner_id",
    "event_index",
    "item_id",
    "kc_id",
    "kc_dimension",
    "event_type",
    "timestamp",
    "response_time_s",
    "correct",
]


class Checks:
    """Collects expected-vs-actual comparisons so they can be reported/saved."""

    def __init__(self, strict: bool):
        self.strict = strict
        self.results: dict[str, dict] = {}

    def expect(self, name: str, actual, expected) -> None:
        ok = actual == expected
        self.results[name] = {
            "actual": int(actual),
            "expected": int(expected),
            "ok": bool(ok),
        }
        if not ok:
            msg = f"{name}: expected {expected:,} but found {actual:,}"
            if self.strict:
                sys.exit(f"ERROR: {msg}  (re-run with --no-strict to continue anyway)")
            print(f"WARNING: {msg}")


# --------------------------------------------------------------------------- #
# Step 2.1 - load + filter
# --------------------------------------------------------------------------- #
def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: input file not found: {path}")
    # utf-8-sig: the released CSV starts with a BOM that would otherwise
    # turn the first column into '﻿learner_id'.
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={
            "learner_id": str,
            "session_id": str,
            "item_id": str,
            "kc_id": str,
            "kc_dimension": str,
            "event_type": str,
        },
    )
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing required columns: {missing}")
    return df


def filter_answers(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep answer events only, with a clean binary `correct`."""
    info: dict = {
        "event_type_counts": raw["event_type"].str.strip().value_counts().to_dict()
    }

    df = raw[raw["event_type"].str.strip() == "answer"].copy()
    info["answer_rows_before_cleaning"] = len(df)

    # `correct` is read as float because interaction rows are empty; coerce
    # anything non-numeric to NaN and keep only exact 0 / 1.
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    bad_correct = ~df["correct"].isin([0, 1])
    info["dropped_non_binary_correct"] = int(bad_correct.sum())
    df = df[~bad_correct].copy()
    df["correct"] = df["correct"].astype(int)

    # Fields we need to build sequences must be present; fail loudly rather than
    # silently shortening a learner's history.
    for col in [
        "learner_id",
        "event_index",
        "item_id",
        "kc_id",
        "timestamp",
        "response_time_s",
    ]:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            sys.exit(f"ERROR: {n_missing} answer rows have a missing '{col}'")

    df["event_index"] = df["event_index"].astype(int)
    df["timestamp"] = df["timestamp"].astype("int64")
    df["response_time_s"] = df["response_time_s"].astype(float)
    return df, info


# --------------------------------------------------------------------------- #
# Step 2.2 - chronological ordering
# --------------------------------------------------------------------------- #
def sort_chronologically(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dup = int(df.duplicated(["learner_id", "event_index"]).sum())
    if dup:
        sys.exit(
            f"ERROR: {dup} duplicate (learner_id, event_index) pairs among answer rows"
        )

    # event_index is the documented stable tie-breaker, monotone with timestamp.
    df = df.sort_values(["learner_id", "event_index"], kind="mergesort").reset_index(
        drop=True
    )

    backwards = int((df.groupby("learner_id")["timestamp"].diff() < 0).sum())
    if backwards:
        print(
            f"WARNING: {backwards} rows have a timestamp earlier than the previous "
            f"row of the same learner (event_index order is kept)"
        )
    return df, {"timestamp_order_violations": backwards}


# --------------------------------------------------------------------------- #
# Step 2.3 - contiguous ID maps (0-based; pyKT reserves -1 for padding)
# --------------------------------------------------------------------------- #
def build_maps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Every item must belong to exactly one KC, otherwise the item -> concept
    # mapping used by pyKT (and later the Moodle mapping) is ambiguous.
    kcs_per_item = df.groupby("item_id")["kc_id"].nunique()
    ambiguous = kcs_per_item[kcs_per_item > 1]
    if len(ambiguous):
        sys.exit(
            f"ERROR: {len(ambiguous)} items map to more than one KC, e.g. {ambiguous.index[:5].tolist()}"
        )

    # Sorted order -> deterministic indices across runs (KC001 -> 0, ...).
    concept_map = (
        df[["kc_id", "kc_dimension"]]
        .drop_duplicates()
        .sort_values("kc_id")
        .reset_index(drop=True)
    )
    if concept_map["kc_id"].duplicated().any():
        sys.exit("ERROR: a kc_id appears with more than one kc_dimension")
    concept_map.insert(1, "concepts", np.arange(len(concept_map), dtype=int))

    question_map = (
        df[["item_id", "kc_id"]]
        .drop_duplicates()
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    question_map.insert(1, "questions", np.arange(len(question_map), dtype=int))
    question_map = question_map.merge(concept_map, on="kc_id", how="left")
    question_map = question_map[
        ["item_id", "questions", "kc_id", "concepts", "kc_dimension"]
    ]
    return question_map, concept_map


# --------------------------------------------------------------------------- #
# Step 2.4 - per-learner sequences in pyKT format
# --------------------------------------------------------------------------- #
def build_sequences(
    df: pd.DataFrame,
    question_map: pd.DataFrame,
    concept_map: pd.DataFrame,
    timestamps_ms: bool,
) -> tuple[pd.DataFrame, dict]:
    q_idx = dict(zip(question_map["item_id"], question_map["questions"]))
    c_idx = dict(zip(concept_map["kc_id"], concept_map["concepts"]))
    df = df.assign(
        questions=df["item_id"].map(q_idx),
        concepts=df["kc_id"].map(c_idx),
    )
    assert not df[["questions", "concepts"]].isna().any().any()

    ts_factor = 1000 if timestamps_ms else 1
    join = lambda values: ",".join(str(v) for v in values)

    rows, truncated = [], 0
    learner_ids = sorted(df["learner_id"].unique())
    uid_of = {lid: i for i, lid in enumerate(learner_ids)}

    for lid, g in df.groupby("learner_id", sort=True):
        if len(g) > MAX_SEQ_LEN:
            truncated += 1
            g = g.iloc[:MAX_SEQ_LEN]  # keep the earliest MAX_SEQ_LEN answers
        n = len(g)
        rows.append(
            {
                "uid": uid_of[lid],
                "questions": join(g["questions"].astype(int)),
                "concepts": join(g["concepts"].astype(int)),
                "responses": join(g["correct"].astype(int)),
                "timestamps": join((g["timestamp"] * ts_factor).astype("int64")),
                "usetimes": join(round(float(x), 2) for x in g["response_time_s"]),
                "selectmasks": join(["1"] * n),
                "learner_id": lid,
                "seq_len": n,
            }
        )
    seq = pd.DataFrame(rows)
    return seq, {
        "learners_truncated": truncated,
        "max_seq_len_observed": int(seq["seq_len"].max()),
    }


# --------------------------------------------------------------------------- #
# Section 3 - learner-level split
# --------------------------------------------------------------------------- #
def assign_splits(seq: pd.DataFrame, seed: int) -> pd.DataFrame:
    n = len(seq)
    n_train = int(round(SPLIT_RATIOS["train"] * n))
    n_valid = int(round(SPLIT_RATIOS["valid"] * n))
    n_test = n - n_train - n_valid

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)  # positions into seq (sorted by learner_id)
    labels = np.empty(n, dtype=object)
    labels[perm[:n_train]] = "train"
    labels[perm[n_train : n_train + n_valid]] = "valid"
    labels[perm[n_train + n_valid :]] = "test"

    seq = seq.copy()
    seq["split"] = labels
    seq["fold"] = seq["split"].map(FOLD_CODE)
    return seq


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--timestamps-ms", action="store_true", help="write timestamps in milliseconds"
    )
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="warn instead of failing on count mismatches",
    )
    args = ap.parse_args()

    checks = Checks(strict=not args.no_strict)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2.1 ------------------------------------------------------------------
    raw = load_raw(args.input)
    checks.expect("raw_rows", len(raw), EXPECTED["raw_rows"])
    answers, filter_info = filter_answers(raw)
    checks.expect("answer_rows", len(answers), EXPECTED["answer_rows"])

    # 2.2 ------------------------------------------------------------------
    answers, order_info = sort_chronologically(answers)

    # 2.3 ------------------------------------------------------------------
    question_map, concept_map = build_maps(answers)
    checks.expect("learners", answers["learner_id"].nunique(), EXPECTED["learners"])
    checks.expect("concepts", len(concept_map), EXPECTED["concepts"])
    checks.expect("questions", len(question_map), EXPECTED["questions"])

    # 2.4 + 3 --------------------------------------------------------------
    seq, seq_info = build_sequences(
        answers, question_map, concept_map, args.timestamps_ms
    )
    seq = assign_splits(seq, args.seed)

    # Anti-leakage verification: each learner in exactly one split.
    assert seq["learner_id"].is_unique
    sizes = seq["split"].value_counts().to_dict()
    expected_sizes = {"train": 840, "valid": 120, "test": 240}
    if len(seq) == EXPECTED["learners"]:
        checks.expect("split_train", sizes.get("train", 0), expected_sizes["train"])
        checks.expect("split_valid", sizes.get("valid", 0), expected_sizes["valid"])
        checks.expect("split_test", sizes.get("test", 0), expected_sizes["test"])

    # Write outputs ----------------------------------------------------------
    out_cols = [
        "fold",
        "uid",
        "questions",
        "concepts",
        "responses",
        "timestamps",
        "usetimes",
        "selectmasks",
        "learner_id",
        "split",
        "seq_len",
    ]
    seq[out_cols].sort_values("uid").to_csv(
        args.output_dir / "pykt_formatted_sequences.csv", index=False
    )
    question_map.to_csv(args.output_dir / "question_map.csv", index=False)
    concept_map.to_csv(args.output_dir / "concept_map.csv", index=False)
    seq[["learner_id", "uid", "split", "fold", "seq_len"]].sort_values("uid").to_csv(
        args.output_dir / "split_assignments.csv", index=False
    )

    # Per-split descriptive stats (class balance, length) for the thesis audit.
    ans_split = answers.merge(seq[["learner_id", "split"]], on="learner_id")
    split_stats = {}
    for s in ["train", "valid", "test"]:
        g = ans_split[ans_split["split"] == s]
        split_stats[s] = {
            "learners": int((seq["split"] == s).sum()),
            "answers": int(len(g)),
            "positive_rate": round(float(g["correct"].mean()), 4) if len(g) else None,
            "mean_seq_len": round(
                float(seq.loc[seq["split"] == s, "seq_len"].mean()), 2
            ),
        }

    manifest = {
        "input": str(args.input),
        "seed": args.seed,
        "split_ratios": SPLIT_RATIOS,
        "fold_code": FOLD_CODE,
        "max_seq_len": MAX_SEQ_LEN,
        "timestamp_unit": "ms" if args.timestamps_ms else "s",
        "checks": checks.results,
        "filter_info": filter_info,
        "order_info": order_info,
        "sequence_info": seq_info,
        "split_stats": split_stats,
    }
    (args.output_dir / "prepare_data_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    # Console summary --------------------------------------------------------
    print("=" * 64)
    print("pyKT DATA PREPARATION SUMMARY")
    print("=" * 64)
    print(f"Raw rows loaded               : {len(raw):,}")
    print(f"Event types                   : {filter_info['event_type_counts']}")
    print(f"Retained answer rows          : {len(answers):,}")
    print(
        f"Dropped (non-binary correct)  : {filter_info['dropped_non_binary_correct']}"
    )
    print(f"Unique learners               : {answers['learner_id'].nunique():,}")
    print(f"Unique KCs                    : {len(concept_map):,}")
    print(f"Unique items                  : {len(question_map):,}")
    print(f"Overall positive rate         : {answers['correct'].mean():.4f}")
    print(
        f"Sequence length (min/med/max) : {seq['seq_len'].min()} / "
        f"{int(seq['seq_len'].median())} / {seq['seq_len'].max()}"
    )
    print(
        f"Learners truncated at {MAX_SEQ_LEN}      : {seq_info['learners_truncated']}"
    )
    print("-" * 64)
    print(f"Learner split (seed={args.seed}):")
    for s in ["train", "valid", "test"]:
        st = split_stats[s]
        print(
            f"  {s:<5}: {st['learners']:>4} learners | {st['answers']:>6,} answers | "
            f"pos rate {st['positive_rate']} | mean len {st['mean_seq_len']}"
        )
    total = sum(v["learners"] for v in split_stats.values())
    print(f"  total: {total} learners (each learner in exactly one split)")
    ok_all = all(c["ok"] for c in checks.results.values())
    print("-" * 64)
    print(
        "Expected-count checks         : "
        + ("ALL PASSED" if ok_all else "SEE WARNINGS ABOVE")
    )
    print(f"Outputs written to            : {args.output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
