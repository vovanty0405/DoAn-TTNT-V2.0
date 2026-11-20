"""Repository for patient CRUD, search, filtering, sorting, pagination, and soft delete."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from database.db import get_connection

ALLOWED_SORT_COLUMNS = {
    "id": "id",
    "patient_code": "patient_code",
    "full_name": "full_name",
    "birth_year": "birth_year",
    "gender": "gender",
    "created_at": "created_at",
    "is_active": "is_active",
}


def generate_patient_code(db_path: Optional[Path] = None) -> str:
    """Generate next sequential patient code like BN-0001."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM patients").fetchone()
        next_num = int(row["max_id"]) + 1
        return f"BN-{next_num:04d}"


def create_patient(
    full_name: str,
    patient_code: Optional[str] = None,
    birth_year: Optional[int] = None,
    gender: str = "Khác",
    phone: Optional[str] = None,
    address: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert a new patient record and return its ID."""
    code = (patient_code or "").strip() or generate_patient_code(db_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO patients
                (patient_code, full_name, birth_year, gender, phone, address, notes, created_at, updated_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                code,
                full_name.strip(),
                int(birth_year) if birth_year is not None else None,
                gender,
                (phone or "").strip() or None,
                (address or "").strip() or None,
                (notes or "").strip() or None,
                now,
                now,
            ),
        )
        return int(cur.lastrowid)


def get_patient_by_id(
    patient_id: int, db_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Fetch a single patient by primary key."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        return dict(row) if row else None


def update_patient(
    patient_id: int,
    full_name: str,
    patient_code: str,
    birth_year: Optional[int] = None,
    gender: str = "Khác",
    phone: Optional[str] = None,
    address: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """Update an existing patient record."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE patients
            SET patient_code = ?,
                full_name = ?,
                birth_year = ?,
                gender = ?,
                phone = ?,
                address = ?,
                notes = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                patient_code.strip(),
                full_name.strip(),
                int(birth_year) if birth_year is not None else None,
                gender,
                (phone or "").strip() or None,
                (address or "").strip() or None,
                (notes or "").strip() or None,
                now,
                patient_id,
            ),
        )
        return cur.rowcount > 0


def soft_delete_patient(patient_id: int, db_path: Optional[Path] = None) -> bool:
    """Soft-delete a patient by setting is_active = 0."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "UPDATE patients SET is_active = 0, updated_at = ? WHERE id = ?",
            (now, patient_id),
        )
        return cur.rowcount > 0


def restore_patient(patient_id: int, db_path: Optional[Path] = None) -> bool:
    """Restore a soft-deleted patient by setting is_active = 1."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "UPDATE patients SET is_active = 1, updated_at = ? WHERE id = ?",
            (now, patient_id),
        )
        return cur.rowcount > 0


def query_patients(
    search: str = "",
    gender: Optional[str] = None,
    status: str = "all",  # "all", "active", "inactive"
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = "id",
    sort_dir: str = "DESC",
    page: int = 1,
    page_size: int = 10,
    db_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Query patients with SQL-level filtering, sorting, and pagination."""
    where_clauses: List[str] = []
    params: List[Any] = []

    if search and search.strip():
        like = f"%{search.strip()}%"
        where_clauses.append(
            "(patient_code LIKE ? OR full_name LIKE ? OR COALESCE(phone, '') LIKE ?)"
        )
        params.extend([like, like, like])

    if gender and gender != "Tất cả":
        where_clauses.append("gender = ?")
        params.append(gender)

    if status == "active":
        where_clauses.append("is_active = 1")
    elif status == "inactive":
        where_clauses.append("is_active = 0")

    if date_from:
        where_clauses.append("date(created_at) >= date(?)")
        params.append(date_from)

    if date_to:
        where_clauses.append("date(created_at) <= date(?)")
        params.append(date_to)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sort_col = ALLOWED_SORT_COLUMNS.get(sort_by, "id")
    direction = "ASC" if str(sort_dir).upper() == "ASC" else "DESC"
    offset = max(0, (int(page) - 1) * int(page_size))

    with get_connection(db_path) as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(*) AS cnt FROM patients {where_sql}", params
            ).fetchone()["cnt"]
        )
        rows = conn.execute(
            f"""
            SELECT * FROM patients
            {where_sql}
            ORDER BY {sort_col} {direction}, id DESC
            LIMIT ? OFFSET ?
            """,
            [*params, int(page_size), offset],
        ).fetchall()
        return [dict(r) for r in rows], total


def list_active_patients(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all active patients for dropdown selectors."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM patients WHERE is_active = 1 ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_patient_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Return aggregate patient counts and birth years for Dashboard."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active
            FROM patients
            """
        ).fetchone()
        birth_years = [
            int(r["birth_year"])
            for r in conn.execute(
                "SELECT birth_year FROM patients WHERE birth_year IS NOT NULL AND is_active = 1"
            ).fetchall()
        ]
        return {
            "total_patients": int(row["total"] or 0),
            "active_patients": int(row["active"] or 0),
            "birth_years": birth_years,
        }
