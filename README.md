# Student Placement Prediction System

A production-style machine learning web application that predicts:
- Whether a student will be placed (classification)
- Expected salary for placed students (regression)

Built using Python, Flask, Scikit-learn, and MySQL with a clean ML pipeline.

---

## Phase 1 — Project Setup & Architecture

- Defined problem statement and prediction objectives
- Selected tech stack:
  - Backend: Flask
  - ML: Scikit-learn
  - Database: MySQL
- Designed modular project structure
- Set up Git repository and version control
- Created base application files (`app.py`, database connectivity)

---

## Phase 2 — Dataset Generation & Validation

- Synthetic student dataset generated with controlled statistical distributions
- Realistic correlations between academic performance, skills, placement, and salary
- Dataset stored under `data/raw/student_data.csv`
- Data generation script: `scripts/generate_dataset.py`
- Raw vs Processed data folder separation followed (industry standard)
- Initial sanity checks and validation performed before model training

---

## Upcoming Phases

- Phase 3: Model training (Logistic & Linear Regression)
- Phase 4: Model serialization and Flask integration
- Phase 5: Database integration and deployment readiness
