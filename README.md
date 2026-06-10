# Student Placement Prediction System

A resume-ready machine learning web application that predicts student placement probability and expected salary using academic, skill, and experience data.

## Features

- Placement classification with a balanced Logistic Regression model
- Salary prediction for likely placed students using Random Forest Regression
- Clean Flask dashboard with form validation and responsive UI
- Prediction history stored in SQLite
- CSV export for prediction records
- Model metrics page with accuracy, F1 score, ROC AUC, MAE, R2 score, and feature importance
- JSON API endpoint for programmatic predictions
- Reproducible dataset generation, processing, and training scripts

## Tech Stack

- Python
- Flask
- Pandas and NumPy
- Scikit-learn
- SQLite
- HTML and CSS

## Project Structure

```text
student_placement_prediction/
  app.py
  database.py
  train_model.py
  requirements.txt
  data/
    raw/student_data.csv
    processed/student_data_clean.csv
  models/
    placement_pipeline.pkl
    metrics.json
  scripts/
    generate_dataset.py
    train_placement_model.py
  training/
    eda_and_process.py
  templates/
  statics/css/style.css
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Rebuild Dataset and Model

```bash
python scripts/generate_dataset.py
python training/eda_and_process.py
python train_model.py
```

## Run the App

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## API Usage

```bash
curl -X POST http://127.0.0.1:5000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"cgpa\":8.2,\"backlogs\":0,\"internships\":2,\"projects\":4,\"certifications\":3,\"aptitude_score\":82,\"communication_score\":78,\"coding_skill\":86}"
```

## Resume Bullet

Built an end-to-end Flask and Scikit-learn student placement prediction system with reproducible data generation, ML training pipeline, placement probability scoring, salary estimation, SQLite prediction history, CSV export, and model explainability dashboard.
