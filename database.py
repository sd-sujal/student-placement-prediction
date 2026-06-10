import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = ROOT_DIR / "placement_predictions.db"


def get_connection():
    database_uri = f"file:{DATABASE_PATH.as_posix()}?mode=rwc&cache=shared"
    connection = sqlite3.connect(database_uri, uri=True, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def init_db():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT,
                cgpa REAL NOT NULL,
                backlogs INTEGER NOT NULL,
                internships INTEGER NOT NULL,
                projects INTEGER NOT NULL,
                certifications INTEGER NOT NULL,
                aptitude_score INTEGER NOT NULL,
                communication_score INTEGER NOT NULL,
                coding_skill INTEGER NOT NULL,
                placement_probability REAL NOT NULL,
                prediction_label TEXT NOT NULL,
                expected_salary_lpa REAL NOT NULL,
                recommendation TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def save_prediction(student_name, features, result):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                student_name,
                cgpa,
                backlogs,
                internships,
                projects,
                certifications,
                aptitude_score,
                communication_score,
                coding_skill,
                placement_probability,
                prediction_label,
                expected_salary_lpa,
                recommendation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_name,
                features["cgpa"],
                features["backlogs"],
                features["internships"],
                features["projects"],
                features["certifications"],
                features["aptitude_score"],
                features["communication_score"],
                features["coding_skill"],
                result["placement_probability"],
                result["prediction_label"],
                result["expected_salary_lpa"],
                result["recommendation"],
            ),
        )
        return cursor.lastrowid


def get_recent_predictions(limit=12):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_prediction_summary():
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                AVG(placement_probability) AS avg_probability,
                AVG(expected_salary_lpa) AS avg_salary,
                SUM(CASE WHEN prediction_label = 'Likely Placed' THEN 1 ELSE 0 END) AS likely_placed
            FROM predictions
            """
        ).fetchone()
    return dict(row)
