"""Repository for immutable prediction history, filtering, pagination, and dashboard stats."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from database.db import get_connection, get_model_version_tag


def save_prediction(
    model_name: str,
    predicted_label: int,
    label_text: str,
    model_score: float,
    input_features: Dict[str, Any],
    patient_id: Optional[int] = None,
    model_version: Optional[str] = None,
    predicted_at: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Persist an immutable prediction record and return its row ID."""
    version = model_version or get_model_version_tag(model_name, db_path=db_path)
    ts = predicted_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (patient_id, model_name, model_version, input_features, predicted_label, label_text, model_score, predicted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                model_name,
                version,
                json.dumps(input_features, ensure_ascii=False),
                int(predicted_label),
                label_text,
                float(model_score),
                ts,
            ),
        )
        return int(cur.lastrowid)


def get_prediction_by_id(
    prediction_id: int, db_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Fetch a single prediction joined with patient info."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                p.*,
                pt.patient_code,
                pt.full_name AS patient_name,
                pt.is_active AS patient_is_active
            FROM predictions p
            LEFT JOIN patients pt ON p.patient_id = pt.id
            WHERE p.id = ?
            """,
            (prediction_id,),
        ).fetchone()
        return dict(row) if row else None


def get_predictions_for_patient(
    patient_id: int, db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Return all predictions for a specific patient, newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            WHERE patient_id = ?
            ORDER BY predicted_at DESC, id DESC
            """,
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def query_predictions(
    search: str = "",
    model_name: Optional[str] = None,
    predicted_label: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    db_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Query prediction history with SQL-level filtering and pagination."""
    where_clauses: List[str] = []
    params: List[Any] = []

    if search and search.strip():
        like = f"%{search.strip()}%"
        where_clauses.append(
            "(COALESCE(pt.full_name, '') LIKE ? OR COALESCE(pt.patient_code, '') LIKE ?)"
        )
        params.extend([like, like])

    if model_name and model_name != "Tất cả":
        where_clauses.append("p.model_name = ?")
        params.append(model_name)

    if predicted_label in (0, 1):
        where_clauses.append("p.predicted_label = ?")
        params.append(int(predicted_label))

    if date_from:
        where_clauses.append("date(p.predicted_at) >= date(?)")
        params.append(date_from)

    if date_to:
        where_clauses.append("date(p.predicted_at) <= date(?)")
        params.append(date_to)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    offset = max(0, (int(page) - 1) * int(page_size))

    with get_connection(db_path) as conn:
        total = int(
            conn.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM predictions p
                LEFT JOIN patients pt ON p.patient_id = pt.id
                {where_sql}
                """,
                params,
            ).fetchone()["cnt"]
        )
        rows = conn.execute(
            f"""
            SELECT
                p.*,
                pt.patient_code,
                pt.full_name AS patient_name,
                pt.is_active AS patient_is_active
            FROM predictions p
            LEFT JOIN patients pt ON p.patient_id = pt.id
            {where_sql}
            ORDER BY p.predicted_at DESC, p.id DESC
            LIMIT ? OFFSET ?
            """,
            [*params, int(page_size), offset],
        ).fetchall()
        return [dict(r) for r in rows], total


def get_dashboard_prediction_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Aggregate all prediction statistics for the Dashboard strictly from SQLite."""
    current_month = datetime.now().strftime("%Y-%m")
    with get_connection(db_path) as conn:
        summary = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN predicted_label = 0 THEN 1 ELSE 0 END) AS negative,
                SUM(CASE WHEN strftime('%Y-%m', predicted_at) = ? THEN 1 ELSE 0 END) AS this_month
            FROM predictions
            """,
            (current_month,),
        ).fetchone()

        by_month = [
            {"month": r["month"], "count": int(r["cnt"])}
            for r in conn.execute(
                """
                SELECT strftime('%Y-%m', predicted_at) AS month, COUNT(*) AS cnt
                FROM predictions
                GROUP BY month
                ORDER BY month ASC
                """
            ).fetchall()
        ]

        by_model = [
            {"model_name": r["model_name"], "count": int(r["cnt"])}
            for r in conn.execute(
                """
                SELECT model_name, COUNT(*) AS cnt
                FROM predictions
                GROUP BY model_name
                ORDER BY cnt DESC
                """
            ).fetchall()
        ]

        recent = [
            dict(r)
            for r in conn.execute(
                """
                SELECT
                    p.id,
                    p.predicted_at,
                    pt.patient_code,
                    pt.full_name AS patient_name,
                    p.model_name,
                    p.label_text,
                    p.model_score
                FROM predictions p
                LEFT JOIN patients pt ON p.patient_id = pt.id
                ORDER BY p.predicted_at DESC, p.id DESC
                LIMIT 8
                """
            ).fetchall()
        ]

        return {
            "total_predictions": int(summary["total"] or 0),
            "positive_predictions": int(summary["positive"] or 0),
            "negative_predictions": int(summary["negative"] or 0),
            "this_month_predictions": int(summary["this_month"] or 0),
            "by_month": by_month,
            "by_model": by_model,
            "recent_activity": recent,
        }
