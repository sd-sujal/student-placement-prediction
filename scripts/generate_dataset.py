from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "student_data.csv"
RANDOM_SEED = 42
N_STUDENTS = 800


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def build_dataset(n_students=N_STUDENTS, random_seed=RANDOM_SEED):
    rng = np.random.default_rng(random_seed)

    cgpa = np.round(np.clip(rng.normal(7.2, 0.85, n_students), 5.0, 10.0), 2)
    backlogs = rng.choice([0, 1, 2, 3, 4, 5], n_students, p=[0.46, 0.24, 0.15, 0.08, 0.05, 0.02])
    internships = rng.choice([0, 1, 2, 3, 4], n_students, p=[0.34, 0.30, 0.20, 0.11, 0.05])
    projects = rng.integers(1, 8, n_students)
    certifications = rng.integers(0, 7, n_students)
    aptitude_score = rng.integers(40, 101, n_students)
    communication_score = rng.integers(40, 101, n_students)
    coding_skill = rng.integers(40, 101, n_students)

    placement_signal = (
        0.85 * (cgpa - 7.0)
        + 0.55 * internships
        + 0.22 * projects
        + 0.015 * (aptitude_score - 65)
        + 0.018 * (communication_score - 65)
        + 0.024 * (coding_skill - 65)
        + 0.08 * certifications
        - 0.55 * backlogs
        - 0.55
    )

    placement_probability = sigmoid(placement_signal)
    placed = rng.binomial(1, placement_probability)

    salary_lpa = (
        1.8
        + 0.65 * cgpa
        + 0.45 * internships
        + 0.20 * projects
        + 0.035 * coding_skill
        + 0.018 * aptitude_score
        + 0.12 * certifications
        - 0.35 * backlogs
        + rng.normal(0, 0.75, n_students)
    )
    salary_lpa = np.round(np.clip(salary_lpa, 3.0, 18.0), 2)
    salary_lpa[placed == 0] = 0.0

    return pd.DataFrame(
        {
            "cgpa": cgpa,
            "backlogs": backlogs,
            "internships": internships,
            "projects": projects,
            "certifications": certifications,
            "aptitude_score": aptitude_score,
            "communication_score": communication_score,
            "coding_skill": coding_skill,
            "placed": placed,
            "salary_lpa": salary_lpa,
        }
    )


def main():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset generated successfully: {RAW_DATA_PATH}")
    print(f"Rows: {len(df)} | Placement rate: {df['placed'].mean():.1%}")


if __name__ == "__main__":
    main()
