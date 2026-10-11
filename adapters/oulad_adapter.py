"""
OULAD ETL Adapter
Transforms raw OULAD CSVs into the standardized Moodle Collector schema.
Ensures point-in-time timestamp_unix columns exist for temporal.py.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Fixed reference epoch so day offsets convert to valid timestamps
BASE_TIMESTAMP = pd.Timestamp("2024-01-01 00:00:00")

# OULAD activity types mapped to standard roles
OULAD_ROLE_MAP = {
    "oucontent": "lesson",
    "page": "page",
    "subpage": "page",
    "url": "url",
    "resource": "resource",
    "forumng": "forum",
    "quiz": "quiz",
    "externalquiz": "quiz",
    "questionnaire": "quiz",
    "repeatactivity": "lesson",
    "glossary": "resource",
    "dataplus": "resource",
    "htmlactivity": "page",
}


def clean_columns(df):
    """Strips hidden whitespaces and quotes from CSV column headers to prevent KeyErrors."""
    df.columns = df.columns.str.strip().str.replace('"', "").str.replace("'", "")
    return df


def transform_oulad_to_collector_schema(raw_oulad_dir: Path, output_dir: Path):
    """Converts raw OULAD files into activity_catalog, learning_events, and quiz_attempts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw OULAD files...")

    # Load and immediately clean column headers
    vle = clean_columns(pd.read_csv(raw_oulad_dir / "vle.csv"))
    student_vle = clean_columns(pd.read_csv(raw_oulad_dir / "studentVle.csv"))
    assessments = clean_columns(pd.read_csv(raw_oulad_dir / "assessments.csv"))
    student_assessment = clean_columns(
        pd.read_csv(raw_oulad_dir / "studentAssessment.csv")
    )

    # ---------------------------------------------------------
    # 1. Build activity_catalog.csv
    # ---------------------------------------------------------
    vle_catalog = vle.copy()
    vle_catalog["course_module_id"] = vle_catalog["id_site"].astype(int)
    vle_catalog["activity_id"] = vle_catalog["id_site"].astype(int)
    vle_catalog["activity_type"] = (
        vle_catalog["activity_type"].map(OULAD_ROLE_MAP).fillna("resource")
    )
    vle_catalog["topic"] = (
        vle_catalog["code_module"] + "_" + vle_catalog["code_presentation"]
    )
    vle_catalog["sequence"] = pd.to_numeric(
        vle_catalog["week_from"], errors="coerce"
    ).fillna(0)
    vle_catalog["activity_name"] = (
        vle_catalog["activity_type"] + "_" + vle_catalog["course_module_id"].astype(str)
    )
    vle_catalog["completion_tracking"] = 1
    vle_catalog["section_number"] = 1

    catalog_cols = [
        "course_module_id",
        "activity_id",
        "activity_name",
        "activity_type",
        "topic",
        "sequence",
        "completion_tracking",
        "section_number",
    ]
    catalog = vle_catalog[catalog_cols].drop_duplicates("course_module_id")
    catalog.to_csv(output_dir / "activity_catalog.csv", index=False)
    print(f"Exported activity_catalog.csv ({len(catalog)} items)")

    # ---------------------------------------------------------
    # 2. Build learning_events.csv
    # ---------------------------------------------------------
    # Safeguard check to ensure the column is found after cleaning
    if "date" not in student_vle.columns:
        raise KeyError(
            f"Column 'date' still missing after cleaning. Found columns: {list(student_vle.columns)}"
        )

    events = student_vle.rename(
        columns={"id_student": "learner_id", "id_site": "course_module_id"}
    ).copy()

    # Convert day integer (date) to Timestamp and Unix
    events["event_dt"] = BASE_TIMESTAMP + pd.to_timedelta(events["date"], unit="D")
    events["timestamp"] = events["event_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    events["timestamp_unix"] = events["event_dt"].astype("int64") // 10**9

    events["event_category"] = "content_view"
    events["is_learning_event"] = 1
    events["seconds_since_prev_event"] = 300
    events["session_id"] = (
        events["learner_id"].astype(str) + "_" + events["date"].astype(str)
    )
    events["log_id"] = np.arange(len(events))

    event_cols = [
        "log_id",
        "learner_id",
        "course_module_id",
        "timestamp",
        "timestamp_unix",
        "event_category",
        "is_learning_event",
        "seconds_since_prev_event",
        "session_id",
    ]
    events[event_cols].to_csv(output_dir / "learning_events.csv", index=False)
    print(f"Exported learning_events.csv ({len(events)} interaction records)")

    # ---------------------------------------------------------
    # 3. Build quiz_attempts.csv
    # ---------------------------------------------------------
    # Only merge the columns we actually need from assessments to prevent column collision
    asm_merged = student_assessment.merge(
        assessments[["id_assessment", "assessment_type"]],
        on="id_assessment",
        how="inner",
    )

    quizzes = asm_merged.rename(
        columns={
            "id_student": "learner_id",
            "id_assessment": "quiz_id",
            "date_submitted": "date",
        }
    ).copy()

    quizzes["date"] = pd.to_numeric(quizzes["date"], errors="coerce").fillna(0)
    quizzes["finish_dt"] = BASE_TIMESTAMP + pd.to_timedelta(quizzes["date"], unit="D")
    quizzes["finish_time_unix"] = quizzes["finish_dt"].astype("int64") // 10**9
    quizzes["quiz_name"] = (
        quizzes["assessment_type"] + "_" + quizzes["quiz_id"].astype(str)
    )
    quizzes["normalized_score"] = (
        pd.to_numeric(quizzes["score"], errors="coerce") / 100.0
    ).clip(0, 1)
    quizzes["attempt"] = 1
    quizzes["duration_seconds"] = 1800

    quiz_cols = [
        "learner_id",
        "quiz_id",
        "quiz_name",
        "attempt",
        "normalized_score",
        "duration_seconds",
        "finish_time_unix",
    ]
    quizzes[quiz_cols].to_csv(output_dir / "quiz_attempts.csv", index=False)
    print(f"Exported quiz_attempts.csv ({len(quizzes)} assessment attempts)")


if __name__ == "__main__":
    import sys

    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/oulad_raw")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/oulad_processed")
    transform_oulad_to_collector_schema(raw_dir, out_dir)
