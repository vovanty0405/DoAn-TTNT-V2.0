"""SQLite connection manager, schema initialization, and model version registry."""

import sqlite3
from pathlib import Path
from typing import Optional

from src.config import DB_PATH, MODEL_FILES, MODELS_DIR
from src.prediction import load_experiment_metadata
from src.utils import sha256_file


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys and Row factory enabled."""
    target = Path(db_path) if db_path else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def init_db(db_path: Optional[Path] = None) -> None:
    """Create or migrate HeartCare-AI SQLite tables inside a transaction."""
    with get_connection(db_path) as conn:
        # If an older v0 schema exists (without is_active or model_version), upgrade cleanly.
        p_cols = _table_columns(conn, "patients")
        if p_cols and "is_active" not in p_cols:
            conn.execute("PRAGMA foreign_keys = OFF;")
            conn.execute("DROP TABLE IF EXISTS predictions;")
            conn.execute("DROP TABLE IF EXISTS patients;")
            conn.execute("PRAGMA foreign_keys = ON;")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_code TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                birth_year INTEGER,
                gender TEXT NOT NULL DEFAULT 'Khác',
                phone TEXT,
                address TEXT,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_sha256 TEXT,
                sklearn_version TEXT,
                notes TEXT,
                registered_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                UNIQUE(model_name, model_version)
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                input_features TEXT NOT NULL,
                predicted_label INTEGER NOT NULL,
                label_text TEXT NOT NULL,
                model_score REAL NOT NULL,
                predicted_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_patients_code ON patients(patient_code);
            CREATE INDEX IF NOT EXISTS idx_patients_active ON patients(is_active);
            CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(predicted_at);
            CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);
            """
        )


def register_current_models(db_path: Optional[Path] = None) -> None:
    """Register the copied .joblib models in model_versions for provenance."""
    try:
        exp = load_experiment_metadata()
        sklearn_ver = exp.get("versions", {}).get("sklearn", "unknown")
    except Exception:
        sklearn_ver = "unknown"

    with get_connection(db_path) as conn:
        for model_name, filename in MODEL_FILES.items():
            path = MODELS_DIR / filename
            if not path.exists():
                continue
            digest = sha256_file(path)
            version_tag = f"v1.0-{digest[:8]}"
            conn.execute(
                """
                INSERT OR IGNORE INTO model_versions
                    (model_name, model_version, model_sha256, sklearn_version, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    version_tag,
                    digest,
                    sklearn_ver,
                    "Copied verbatim from verified research experiment",
                ),
            )


def get_model_version_tag(model_name: str, db_path: Optional[Path] = None) -> str:
    """Return the registered version tag for a model name."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT model_version FROM model_versions
            WHERE model_name = ?
            ORDER BY id DESC LIMIT 1
            """,
            (model_name,),
        ).fetchone()
        if row:
            return str(row["model_version"])
    filename = MODEL_FILES.get(model_name)
    if filename and (MODELS_DIR / filename).exists():
        return f"v1.0-{sha256_file(MODELS_DIR / filename)[:8]}"
    return "v1.0"
