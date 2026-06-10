from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "student_data.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "student_data_clean.csv"

EXPECTED_COLUMNS = [
    "cgpa",
    "backlogs",
    "internships",
    "projects",
    "certifications",
    "aptitude_score",
    "communication_score",
    "coding_skill",
    "placed",
    "salary_lpa",
]


def validate_dataset(df):
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains null values")

    checks = {
        "cgpa": df["cgpa"].between(5.0, 10.0),
        "backlogs": df["backlogs"].between(0, 5),
        "internships": df["internships"].between(0, 4),
        "projects": df["projects"].between(0, 7),
        "certifications": df["certifications"].between(0, 6),
        "aptitude_score": df["aptitude_score"].between(0, 100),
        "communication_score": df["communication_score"].between(0, 100),
        "coding_skill": df["coding_skill"].between(0, 100),
        "placed": df["placed"].isin([0, 1]),
    }

    failed = [column for column, mask in checks.items() if not mask.all()]
    if failed:
        raise ValueError(f"Out-of-range values found in: {failed}")

    if not (df.loc[df["placed"] == 0, "salary_lpa"] == 0).all():
        raise ValueError("Unplaced students must have salary_lpa equal to 0")


def load_and_process():
    df = pd.read_csv(RAW_DATA_PATH)
    df = df[EXPECTED_COLUMNS].copy()
    validate_dataset(df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df


def main():
    df = load_and_process()
    print("Shape:", df.shape)
    print("\nPlacement Distribution:")
    print(df["placed"].value_counts())
    print("\nCorrelation with placement:")
    print(df.corr(numeric_only=True)["placed"].sort_values(ascending=False))
    print(f"\nProcessed data saved to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
