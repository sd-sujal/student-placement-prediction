import csv
import io
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from database import get_prediction_summary, get_recent_predictions, init_db, save_prediction


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "placement_pipeline.pkl"

FEATURES = {
    "cgpa": {"label": "CGPA", "min": 5.0, "max": 10.0, "step": 0.01, "type": float},
    "backlogs": {"label": "Backlogs", "min": 0, "max": 5, "step": 1, "type": int},
    "internships": {"label": "Internships", "min": 0, "max": 4, "step": 1, "type": int},
    "projects": {"label": "Projects", "min": 0, "max": 7, "step": 1, "type": int},
    "certifications": {"label": "Certifications", "min": 0, "max": 6, "step": 1, "type": int},
    "aptitude_score": {"label": "Aptitude Score", "min": 0, "max": 100, "step": 1, "type": int},
    "communication_score": {"label": "Communication Score", "min": 0, "max": 100, "step": 1, "type": int},
    "coding_skill": {"label": "Coding Skill", "min": 0, "max": 100, "step": 1, "type": int},
}

DEFAULT_FORM = {
    "student_name": "",
    "cgpa": 7.5,
    "backlogs": 0,
    "internships": 1,
    "projects": 3,
    "certifications": 2,
    "aptitude_score": 75,
    "communication_score": 75,
    "coding_skill": 75,
}


def create_app():
    app = Flask(__name__, static_folder="statics")
    app.config["SECRET_KEY"] = "student-placement-demo"
    init_db()

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        errors = {}
        form_data = DEFAULT_FORM.copy()

        if request.method == "POST":
            form_data.update(request.form.to_dict())
            features, errors = validate_features(request.form)
            student_name = request.form.get("student_name", "").strip()

            if not errors:
                result = predict(features)
                result["id"] = save_prediction(student_name, features, result)
                form_data.update(features)
                form_data["student_name"] = student_name

        return render_template(
            "index.html",
            features=FEATURES,
            form_data=form_data,
            result=result,
            errors=errors,
            metrics=load_metrics(),
            recent_predictions=get_recent_predictions(5),
            summary=get_prediction_summary(),
        )

    @app.route("/history")
    def history():
        return render_template(
            "history.html",
            predictions=get_recent_predictions(100),
            summary=get_prediction_summary(),
        )

    @app.route("/history/export")
    def export_history():
        predictions = get_recent_predictions(500)
        output = io.StringIO()
        fieldnames = list(predictions[0].keys()) if predictions else ["message"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        if predictions:
            writer.writerows(predictions)
        else:
            writer.writerow({"message": "No predictions available"})
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=placement_prediction_history.csv"},
        )

    @app.route("/about")
    def about():
        return render_template("about.html", metrics=load_metrics())

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        payload = request.get_json(silent=True) or {}
        features, errors = validate_features(payload)
        if errors:
            return jsonify({"errors": errors}), 400
        return jsonify(predict(features))

    return app


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. Run `python scripts/generate_dataset.py`, "
            "`python training/eda_and_process.py`, and `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def load_metrics():
    try:
        return load_model_bundle().get("metrics", {})
    except FileNotFoundError:
        return {}


def validate_features(source):
    features = {}
    errors = {}

    for name, config in FEATURES.items():
        raw_value = source.get(name)
        try:
            value = config["type"](raw_value)
        except (TypeError, ValueError):
            errors[name] = f"{config['label']} is required."
            continue

        if value < config["min"] or value > config["max"]:
            errors[name] = f"{config['label']} must be between {config['min']} and {config['max']}."
        else:
            features[name] = value

    return features, errors


def predict(features):
    bundle = load_model_bundle()
    feature_columns = bundle["feature_columns"]
    frame = pd.DataFrame([{column: features[column] for column in feature_columns}])

    probability = float(bundle["classifier"].predict_proba(frame)[0][1])
    label = "Likely Placed" if probability >= 0.5 else "Needs Improvement"
    salary = float(bundle["salary_regressor"].predict(frame)[0]) if probability >= 0.5 else 0.0

    return {
        "placement_probability": round(probability * 100, 2),
        "prediction_label": label,
        "expected_salary_lpa": round(max(salary, 0.0), 2),
        "recommendation": build_recommendation(features, probability),
        "strengths": build_strengths(features),
    }


def build_strengths(features):
    strengths = []
    if features["cgpa"] >= 8:
        strengths.append("Strong academic profile")
    if features["coding_skill"] >= 80:
        strengths.append("High coding readiness")
    if features["projects"] >= 4:
        strengths.append("Good project portfolio")
    if features["internships"] >= 2:
        strengths.append("Practical internship exposure")
    if not strengths:
        strengths.append("Profile has room for targeted improvement")
    return strengths


def build_recommendation(features, probability):
    actions = []
    if features["coding_skill"] < 75:
        actions.append("raise coding score with DSA and project practice")
    if features["aptitude_score"] < 70:
        actions.append("improve aptitude test performance")
    if features["communication_score"] < 70:
        actions.append("practice interview communication")
    if features["projects"] < 3:
        actions.append("add two resume-ready projects")
    if features["internships"] < 1:
        actions.append("seek internship or live project exposure")
    if features["backlogs"] > 0:
        actions.append("clear active backlogs")

    if probability >= 0.75 and not actions:
        return "Profile is placement-ready. Focus on mock interviews and company-specific preparation."
    if not actions:
        return "Profile is balanced. Strengthen interview practice and keep applying consistently."
    return "Priority actions: " + ", ".join(actions[:3]) + "."


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
