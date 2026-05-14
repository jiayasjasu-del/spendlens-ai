"""
database/db.py — SQLite database utilities for SpendLens AI
"""
import sqlite3
import pandas as pd
import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "database" / "spendlens.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    """Return a SQLite connection."""
    return sqlite3.connect(str(DB_PATH))


def init_db():
    """Initialize all tables."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            row_count INTEGER
        );

        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER,
            date TEXT,
            description TEXT,
            amount REAL,
            category TEXT,
            predicted_category TEXT,
            is_anomaly INTEGER DEFAULT 0,
            FOREIGN KEY(upload_id) REFERENCES uploads(id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER,
            prediction_date TEXT,
            model_name TEXT,
            predicted_amount REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(upload_id) REFERENCES uploads(id)
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER,
            health_score REAL,
            report_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(upload_id) REFERENCES uploads(id)
        );
    """)
    conn.commit()
    conn.close()


def save_upload(filename: str, row_count: int) -> int:
    """Save upload metadata and return upload_id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO uploads (filename, row_count) VALUES (?, ?)",
        (filename, row_count)
    )
    upload_id = cur.lastrowid
    conn.commit()
    conn.close()
    return upload_id


def save_expenses(df: pd.DataFrame, upload_id: int):
    """Persist processed expense rows."""
    conn = get_connection()
    rows = []
    for _, r in df.iterrows():
        rows.append((
            upload_id,
            str(r.get("Date", "")),
            str(r.get("Description", "")),
            float(r.get("Amount", 0)),
            str(r.get("Category", "")),
            str(r.get("PredictedCategory", r.get("Category", ""))),
            int(r.get("IsAnomaly", 0)),
        ))
    conn.executemany(
        """INSERT INTO expenses
           (upload_id, date, description, amount, category, predicted_category, is_anomaly)
           VALUES (?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    conn.close()


def save_predictions(records: list, upload_id: int):
    """Save forecast records."""
    conn = get_connection()
    for rec in records:
        conn.execute(
            """INSERT INTO predictions (upload_id, prediction_date, model_name, predicted_amount)
               VALUES (?,?,?,?)""",
            (upload_id, rec["date"], rec["model"], rec["amount"]),
        )
    conn.commit()
    conn.close()


def save_report(upload_id: int, health_score: float, report_dict: dict):
    """Persist AI advisor report."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO reports (upload_id, health_score, report_json) VALUES (?,?,?)",
        (upload_id, health_score, json.dumps(report_dict)),
    )
    conn.commit()
    conn.close()


def load_expenses(upload_id: int) -> pd.DataFrame:
    """Load expenses for a given upload."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM expenses WHERE upload_id=?", conn, params=(upload_id,)
    )
    conn.close()
    return df


def load_uploads() -> pd.DataFrame:
    """Load all upload records."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM uploads ORDER BY uploaded_at DESC", conn)
    conn.close()
    return df


# Initialize on import
init_db()
