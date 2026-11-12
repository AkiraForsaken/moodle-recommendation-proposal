"""Skills: KC states aggregated through the course's own KC catalog.

A skill is one KC dimension within one course topic (for example
"Grammar: Unit 1"), built from the collector's ``kc_catalog.csv`` and
``kc_topic_map.csv``. A KC with no topic link falls back to its dimension
alone. Nothing here comes from a benchmark dataset.

KC rows are never dropped: a KC that the knowledge states mention but the
catalog does not is kept with ``mapping_status = unmapped`` and simply takes
part in no skill.
"""

from pathlib import Path

import pandas as pd

from .config import KC_CATALOG_FILE, KC_TOPIC_MAP_FILE, UNVERSIONED_MAPPING
from .schemas import KC_SKILL_MAP_COLUMNS, SKILL_SUMMARY_COLUMNS

MAPPING_SOURCE = "course KC catalog (Moodle question categories)"


def load_kc_catalog(data_dir):
    """Return the course KC catalog, or an empty frame when none was exported."""
    path = Path(data_dir) / KC_CATALOG_FILE
    columns = ["kc_id", "kc_name", "kc_dimension", "learning_objective"]
    if not path.is_file():
        return pd.DataFrame(columns=columns, dtype="string")
    catalog = pd.read_csv(path, dtype="string")
    missing = {"kc_id", "kc_dimension"}.difference(catalog.columns)
    if missing:
        raise ValueError(f"KC catalog {path} is missing columns: {sorted(missing)}")
    if catalog["kc_id"].isna().any() or catalog["kc_id"].duplicated().any():
        raise ValueError(f"KC catalog {path} must have one non-empty row per kc_id")
    catalog["kc_dimension"] = catalog["kc_dimension"].str.strip().str.lower()
    if catalog["kc_dimension"].isna().any() or catalog["kc_dimension"].eq("").any():
        raise ValueError(f"KC catalog {path} contains a KC without a dimension")
    for column in columns:
        if column not in catalog:
            catalog[column] = pd.NA
    catalog["kc_name"] = catalog["kc_name"].fillna(catalog["kc_id"])
    return catalog[columns]


def load_kc_topic_map(data_dir):
    """Return ``(kc_topic_map, mapping_version)``; the map may be empty."""
    path = Path(data_dir) / KC_TOPIC_MAP_FILE
    if not path.is_file():
        return pd.DataFrame(columns=["kc_id", "topic"], dtype="string"), None
    topic_map = pd.read_csv(path, dtype="string")
    missing = {"kc_id", "topic"}.difference(topic_map.columns)
    if missing:
        raise ValueError(f"KC topic map {path} is missing columns: {sorted(missing)}")
    topic_map = topic_map.dropna(subset=["kc_id", "topic"]).drop_duplicates(
        ["kc_id", "topic"]
    )
    version = None
    if "mapping_version" in topic_map:
        versions = topic_map["mapping_version"].dropna().unique()
        if len(versions) > 1:
            raise ValueError(
                f"KC topic map {path} must carry exactly one mapping_version"
            )
        version = str(versions[0]) if len(versions) else None
    return topic_map[["kc_id", "topic"]], version


def build_kc_skill_map(kc_catalog, kc_topic_map, mapping_version=UNVERSIONED_MAPPING):
    """One row per KC and skill. A KC used in two topics belongs to two skills."""
    if kc_catalog.empty:
        return pd.DataFrame(columns=KC_SKILL_MAP_COLUMNS)
    mapping = kc_catalog[["kc_id", "kc_name", "kc_dimension"]].merge(
        kc_topic_map, on="kc_id", how="left"
    )
    dimension_label = mapping["kc_dimension"].str.replace("_", " ").str.title()
    has_topic = mapping["topic"].notna()
    mapping["skill_id"] = ("dimension:" + mapping["kc_dimension"]).where(
        ~has_topic, mapping["kc_dimension"] + "|" + mapping["topic"].fillna("")
    )
    mapping["skill_name"] = dimension_label.where(
        ~has_topic, dimension_label + ": " + mapping["topic"].fillna("")
    )
    mapping["mapping_status"] = "mapped"
    mapping["mapping_source"] = MAPPING_SOURCE
    mapping["mapping_version"] = mapping_version
    return (
        mapping[KC_SKILL_MAP_COLUMNS]
        .sort_values(["kc_id", "skill_id"])
        .reset_index(drop=True)
    )


def ensure_kc_coverage(states, kc_skill_map, mapping_version=UNVERSIONED_MAPPING):
    """Add explicit unmapped rows for state KCs the course catalog does not list."""
    if states.empty:
        return kc_skill_map.copy()
    state_dimensions = states[["kc_id", "kc_dimension"]].drop_duplicates()
    conflicts = state_dimensions.groupby("kc_id")["kc_dimension"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(
            "Knowledge states assign multiple dimensions to KC IDs: "
            f"{conflicts[conflicts.gt(1)].index.tolist()[:10]}"
        )
    dimension_check = state_dimensions.merge(
        kc_skill_map[["kc_id", "kc_dimension"]].drop_duplicates(),
        on="kc_id",
        how="inner",
        suffixes=("_state", "_catalog"),
    )
    inconsistent = dimension_check[
        dimension_check["kc_dimension_state"] != dimension_check["kc_dimension_catalog"]
    ]
    if not inconsistent.empty:
        raise ValueError(
            "KC dimensions disagree between knowledge states and the KC catalog: "
            f"{inconsistent['kc_id'].tolist()[:10]}"
        )
    existing = set(kc_skill_map["kc_id"].astype(str))
    missing = state_dimensions[
        ~state_dimensions["kc_id"].astype(str).isin(existing)
    ].copy()
    if missing.empty:
        return kc_skill_map
    missing["kc_name"] = missing["kc_id"]
    missing["skill_id"] = pd.NA
    missing["skill_name"] = pd.NA
    missing["topic"] = pd.NA
    missing["mapping_status"] = "unmapped"
    missing["mapping_source"] = "KC is absent from the course KC catalog"
    missing["mapping_version"] = mapping_version
    return pd.concat([kc_skill_map, missing[KC_SKILL_MAP_COLUMNS]], ignore_index=True)


def summarize_skills(states, kc_skill_map, mapping_version=UNVERSIONED_MAPPING):
    """Aggregate KC states per learner and skill; every row traces to its KCs."""
    if states.empty:
        return pd.DataFrame(columns=SKILL_SUMMARY_COLUMNS)
    joined = states.merge(
        kc_skill_map[
            ["kc_id", "skill_id", "skill_name", "kc_dimension", "mapping_status"]
        ],
        on=["kc_id", "kc_dimension"],
        how="left",
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
    summary["mapping_version"] = mapping_version
    summary["state_status"] = "mapped"
    return summary[SKILL_SUMMARY_COLUMNS]
