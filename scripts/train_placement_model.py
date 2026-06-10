import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "student_data_clean.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "placement_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "cgpa",
    "backlogs",
    "internships",
    "projects",
    "certifications",
    "aptitude_score",
    "communication_score",
    "coding_skill",
]


def train_models(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    x = df[FEATURE_COLUMNS]
    y = df["placed"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    y_probability = classifier.predict_proba(x_test)[:, 1]

    placed_df = df[df["placed"] == 1].copy()
    salary_x = placed_df[FEATURE_COLUMNS]
    salary_y = placed_df["salary_lpa"]
    sx_train, sx_test, sy_train, sy_test = train_test_split(
        salary_x,
        salary_y,
        test_size=0.2,
        random_state=42,
    )

    salary_regressor = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=3,
        random_state=42,
    )
    salary_regressor.fit(sx_train, sy_train)
    salary_pred = salary_regressor.predict(sx_test)

    feature_importance = dict(
        sorted(
            zip(FEATURE_COLUMNS, salary_regressor.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    metrics = {
        "classification": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_probability)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report": classification_report(y_test, y_pred, output_dict=True),
        },
        "regression": {
            "mae_lpa": round(float(mean_absolute_error(sy_test, salary_pred)), 4),
            "r2_score": round(float(r2_score(sy_test, salary_pred)), 4),
        },
        "dataset": {
            "rows": int(len(df)),
            "placement_rate": round(float(df["placed"].mean()), 4),
            "average_salary_lpa": round(float(df.loc[df["placed"] == 1, "salary_lpa"].mean()), 2),
        },
        "feature_importance": {key: round(float(value), 4) for key, value in feature_importance.items()},
    }

    bundle = {
        "classifier": classifier,
        "salary_regressor": salary_regressor,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
    }
    return bundle, metrics


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle, metrics = train_models()
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Keep backward compatibility with the original app notes.
    joblib.dump(bundle["classifier"], MODEL_DIR / "placement_model.pkl")

    print(f"Model bundle saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print("\nClassification metrics:")
    print(json.dumps(metrics["classification"], indent=2))
    print("\nRegression metrics:")
    print(json.dumps(metrics["regression"], indent=2))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
