"""Draft skill taxonomy and aggregation for English-dataset KC states."""

from pathlib import Path

import pandas as pd

from .config import MAPPING_VERSION
from .schemas import SKILL_SUMMARY_COLUMNS


def build_taxonomy(concept_map):
    """Generate provisional dimension skills and KC relationships.

    The source provides KC IDs and dimensions but no reviewed sub-skill names.
    Therefore dimension groups are explicitly marked needs_review rather than
    presented as a validated content taxonomy.
    """
    required = {"kc_id", "kc_dimension"}
    missing = required.difference(concept_map.columns)
    if missing:
        raise ValueError(f"Concept map is missing required columns: {sorted(missing)}")
    mapping = concept_map[["kc_id", "kc_dimension"]].copy()
    mapping["kc_id"] = mapping["kc_id"].astype("string")
    mapping["kc_dimension"] = (
        mapping["kc_dimension"].astype("string").str.strip().str.lower()
    )
    if mapping["kc_id"].isna().any() or mapping["kc_id"].duplicated().any():
        raise ValueError("Concept map must have one non-empty row per kc_id")
    if mapping["kc_dimension"].isna().any() or mapping["kc_dimension"].eq("").any():
        raise ValueError("Concept map contains a KC without a dimension")

    mapping["skill_id"] = "dimension:" + mapping["kc_dimension"]
    mapping["skill_name"] = mapping["kc_dimension"].str.replace("_", " ").str.title()
    mapping["topic"] = pd.NA
    mapping["mapping_status"] = "needs_review"
    mapping["mapping_source"] = "KC dimension metadata; English Moodle mapping pending"
    mapping["mapping_version"] = MAPPING_VERSION

    taxonomy = (
        mapping[["skill_id", "skill_name", "kc_dimension"]]
        .drop_duplicates()
        .sort_values("skill_id")
        .reset_index(drop=True)
    )
    taxonomy["taxonomy_status"] = "needs_review"
    taxonomy["taxonomy_source"] = "KC dimension metadata"
    taxonomy["mapping_version"] = MAPPING_VERSION
    return taxonomy, mapping


def summarize_skills(states, kc_mapping, mapping_version=MAPPING_VERSION):
    if states.empty:
        return pd.DataFrame(columns=SKILL_SUMMARY_COLUMNS)
    joined = states.merge(
        kc_mapping[
            ["kc_id", "skill_id", "skill_name", "kc_dimension", "mapping_status"]
        ],
        on=["kc_id", "kc_dimension"],
        how="left",
        validate="many_to_one",
    )
    mapped = joined[
        joined["skill_id"].notna() & joined["mapping_status"].ne("unmapped")
    ].copy()
    if mapped.empty:
        return pd.DataFrame(columns=SKILL_SUMMARY_COLUMNS)
    summary = mapped.groupby(
        ["learner_id", "skill_id", "skill_name", "kc_dimension"], as_index=False
    ).agg(
        kc_count=("kc_id", "nunique"),
        estimated_kc_count=(
            "state_status",
            lambda values: values.eq("estimated").sum(),
        ),
        cold_start_kc_count=(
            "state_status",
            lambda values: values.eq("cold_start").sum(),
        ),
        mean_predicted_correctness=("predicted_correctness", "mean"),
        mean_observed_accuracy=("observed_accuracy", "mean"),
        total_attempt_count=("attempt_count", "sum"),
    )
    # A skill is only "mapped" when every KC relationship in it has been
    # reviewed; one draft relationship keeps the whole skill "needs_review".
    review = (
        mapped.assign(_pending=mapped["mapping_status"].ne("mapped"))
        .groupby(["learner_id", "skill_id"])["_pending"]
        .any()
        .rename("_pending")
        .reset_index()
    )
    summary = summary.merge(review, on=["learner_id", "skill_id"], how="left")
    summary["mapping_version"] = mapping_version
    summary["state_status"] = summary["_pending"].map(
        {True: "needs_review", False: "mapped"}
    )
    return summary[SKILL_SUMMARY_COLUMNS]


def ensure_kc_coverage(states, kc_mapping, mapping_version=MAPPING_VERSION):
    """Include explicit unmapped rows when the source has KCs absent from metadata."""
    if states.empty:
        return kc_mapping.copy()
    state_dimensions = states[["kc_id", "kc_dimension"]].drop_duplicates()
    conflicts = state_dimensions.groupby("kc_id")["kc_dimension"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(
            "Knowledge states assign multiple dimensions to KC IDs: "
            f"{conflicts[conflicts.gt(1)].index.tolist()[:10]}"
        )
    dimension_check = state_dimensions.merge(
        kc_mapping[["kc_id", "kc_dimension"]],
        on="kc_id",
        how="inner",
        suffixes=("_state", "_metadata"),
    )
    inconsistent = dimension_check[
        dimension_check["kc_dimension_state"]
        != dimension_check["kc_dimension_metadata"]
    ]
    if not inconsistent.empty:
        raise ValueError(
            "KC dimensions disagree between state output and concept map: "
            f"{inconsistent['kc_id'].tolist()[:10]}"
        )
    existing = set(kc_mapping["kc_id"].astype(str))
    missing = state_dimensions[
        ~state_dimensions["kc_id"].astype(str).isin(existing)
    ].copy()
    if missing.empty:
        return kc_mapping
    missing["skill_id"] = pd.NA
    missing["skill_name"] = pd.NA
    missing["topic"] = pd.NA
    missing["mapping_status"] = "unmapped"
    missing["mapping_source"] = "KC is absent from the source concept map"
    missing["mapping_version"] = mapping_version
    return pd.concat([kc_mapping, missing[kc_mapping.columns]], ignore_index=True)


MAPPING_COLUMNS = [
    "kc_id",
    "kc_dimension",
    "skill_id",
    "skill_name",
    "topic",
    "mapping_status",
    "mapping_source",
    "mapping_version",
]
VALID_MAPPING_STATUSES = {"mapped", "unmapped", "needs_review"}


def load_kc_mapping(path):
    """Load a (possibly human-reviewed) KC mapping and validate its contract.

    Returns ``(mapping, mapping_version)``. The version is taken from the file
    so a reviewed mapping is never reported under the draft version.
    """
    path = Path(path)
    mapping = pd.read_csv(
        path,
        dtype={
            "kc_id": "string",
            "kc_dimension": "string",
            "skill_id": "string",
            "skill_name": "string",
            "topic": "string",
            "mapping_status": "string",
            "mapping_source": "string",
            "mapping_version": "string",
        },
    )
    missing = set(MAPPING_COLUMNS).difference(mapping.columns)
    if missing:
        raise ValueError(f"KC mapping {path} is missing columns: {sorted(missing)}")
    if mapping["kc_id"].isna().any() or mapping["kc_id"].duplicated().any():
        raise ValueError(f"KC mapping {path} must have one non-empty row per kc_id")
    mapping["kc_dimension"] = mapping["kc_dimension"].str.strip().str.lower()
    bad = set(mapping["mapping_status"].dropna()) - VALID_MAPPING_STATUSES
    if mapping["mapping_status"].isna().any() or bad:
        raise ValueError(
            f"KC mapping {path} has invalid mapping_status values: {sorted(bad)}"
        )
    needs_skill = mapping["mapping_status"].ne("unmapped")
    if mapping.loc[needs_skill, "skill_id"].isna().any():
        raise ValueError(f"KC mapping {path}: mapped/needs_review rows need a skill_id")
    versions = mapping["mapping_version"].dropna().unique()
    if len(versions) != 1:
        raise ValueError(f"KC mapping {path} must carry exactly one mapping_version")
    return mapping[MAPPING_COLUMNS], str(versions[0])


def write_mapping_artifacts(taxonomy, kc_mapping, output_dir):
    """Write draft mapping files only if absent; never overwrite reviewed files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_file = output_dir / "skill_taxonomy.csv"
    topic_map_file = output_dir / "kc_topic_map.csv"
    if not taxonomy_file.exists():
        taxonomy.to_csv(taxonomy_file, index=False)
    if not topic_map_file.exists():
        kc_mapping.to_csv(topic_map_file, index=False)
