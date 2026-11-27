"""Excel import service for patients with template generation, validation preview, and confirmed DB commit."""

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

from database.db import get_connection, get_model_version_tag
from database.patient_repository import generate_patient_code
from src.config import CODES, FEATURE_LABELS, FEATURES
from src.data_validation import validate_input_dict
from src.prediction import predict_single

HEADER_ALIASES = {
    "patient_code": {"mã bệnh nhân", "mã bn", "ma benh nhan", "ma bn", "patient_code", "code"},
    "full_name": {"họ và tên", "họ tên", "ho va ten", "ho ten", "full_name", "name"},
    "birth_year": {"năm sinh", "nam sinh", "birth_year", "year_of_birth"},
    "gender": {"giới tính", "gioi tinh", "gender", "sex"},
    "phone": {"điện thoại", "số điện thoại", "dien thoai", "so dien thoai", "sdt", "phone"},
    "address": {"địa chỉ", "dia chi", "address"},
    "notes": {"ghi chú", "ghi chu", "notes", "note"},
}

GENDER_MAP = {
    "nam": "Nam",
    "male": "Nam",
    "m": "Nam",
    "1": "Nam",
    "nữ": "Nữ",
    "nu": "Nữ",
    "female": "Nữ",
    "f": "Nữ",
    "0": "Nữ",
    "khác": "Khác",
    "khac": "Khác",
    "other": "Khác",
}


def build_patient_import_template() -> bytes:
    """Generate a sample .xlsx template for importing patients."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Mau_Nhap_Benh_Nhan"

    headers = ["Mã bệnh nhân", "Họ và tên", "Năm sinh", "Giới tính", "Điện thoại", "Địa chỉ", "Ghi chú"]
    ws.append(headers)

    navy_fill = PatternFill(start_color="0B3C5D", end_color="0B3C5D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    for idx in range(1, len(headers) + 1):
        c = ws.cell(row=1, column=idx)
        c.fill = navy_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[c.column_letter].width = 20

    ws.append(["BN-IMPORT-01", "Nguyễn Văn Mẫu 1", 1968, "Nam", "0901234567", "Quận 1, TP.HCM", "Hồ sơ mẫu nhập Excel"])
    ws.append(["", "Trần Thị Mẫu 2", 1974, "Nữ", "0912345678", "Quận 3, TP.HCM", "Để trống mã để hệ thống tự sinh"])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _normalize_header(val: Any) -> str:
    return str(val or "").strip().lower()


def _find_header_row(ws) -> Tuple[Optional[int], Dict[str, int]]:
    """Scan the first 10 rows to locate the column header row (works for row 1 or exported row 4)."""
    for row_idx in range(1, min(11, ws.max_row + 1)):
        col_map: Dict[str, int] = {}
        for col_idx in range(1, ws.max_column + 1):
            norm = _normalize_header(ws.cell(row=row_idx, column=col_idx).value)
            if not norm:
                continue
            for field, aliases in HEADER_ALIASES.items():
                if norm in aliases and field not in col_map:
                    col_map[field] = col_idx
        if "full_name" in col_map:
            return row_idx, col_map
    return None, {}


def _get_existing_codes(db_path: Optional[Path] = None) -> set[str]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT patient_code FROM patients").fetchall()
        return {str(r["patient_code"]).strip().upper() for r in rows if r["patient_code"]}


def parse_patients_excel(
    file_bytes: bytes, db_path: Optional[Path] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse and validate an uploaded .xlsx file without writing to SQLite.

    Returns:
        (valid_rows, error_rows)
    """
    wb = load_workbook(io.BytesIO(file_bytes), data_only=True)
    ws = wb.active

    header_row, col_map = _find_header_row(ws)
    if header_row is None:
        raise ValueError(
            "Không tìm thấy cột 'Họ và tên' (hoặc 'Họ tên' / 'full_name') trong 10 dòng đầu của file Excel."
        )

    existing_codes = _get_existing_codes(db_path)
    seen_codes: set[str] = set()
    curr_year = datetime.now().year

    valid_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for row_idx in range(header_row + 1, ws.max_row + 1):
        raw = {
            field: ws.cell(row=row_idx, column=c_idx).value
            for field, c_idx in col_map.items()
        }
        # Skip completely empty rows
        if all(v is None or str(v).strip() == "" for v in raw.values()):
            continue

        reasons: List[str] = []

        full_name = str(raw.get("full_name") or "").strip()
        if not full_name:
            reasons.append("Thiếu Họ và tên")

        patient_code = str(raw.get("patient_code") or "").strip()
        if patient_code:
            upper_code = patient_code.upper()
            if upper_code in existing_codes:
                reasons.append(f"Mã bệnh nhân '{patient_code}' đã tồn tại trong CSDL")
            elif upper_code in seen_codes:
                reasons.append(f"Mã bệnh nhân '{patient_code}' bị trùng trong file Excel")
            else:
                seen_codes.add(upper_code)

        birth_year_raw = raw.get("birth_year")
        birth_year: Optional[int] = None
        if birth_year_raw is not None and str(birth_year_raw).strip() != "":
            try:
                by = int(float(str(birth_year_raw).strip()))
                if not (1900 <= by <= curr_year):
                    reasons.append(f"Năm sinh ({by}) phải nằm trong [1900, {curr_year}]")
                else:
                    birth_year = by
            except ValueError:
                reasons.append(f"Năm sinh không hợp lệ: '{birth_year_raw}'")

        gender_raw = str(raw.get("gender") or "").strip().lower()
        gender = GENDER_MAP.get(gender_raw, "Khác") if gender_raw else "Khác"

        phone = str(raw.get("phone") or "").strip()
        if phone.endswith(".0"):
            phone = phone[:-2]
        address = str(raw.get("address") or "").strip()
        notes = str(raw.get("notes") or "").strip()

        record = {
            "excel_row": row_idx,
            "patient_code": patient_code,
            "full_name": full_name,
            "birth_year": birth_year,
            "gender": gender,
            "phone": phone or None,
            "address": address or None,
            "notes": notes or None,
        }

        if reasons:
            error_rows.append({**record, "reason": "; ".join(reasons)})
        else:
            valid_rows.append(record)

    return valid_rows, error_rows


def commit_imported_patients(
    valid_rows: List[Dict[str, Any]], db_path: Optional[Path] = None
) -> int:
    """Persist validated patient rows into SQLite inside a single transaction."""
    if not valid_rows:
        return 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inserted = 0
    with get_connection(db_path) as conn:
        max_id = int(
            conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM patients").fetchone()["max_id"]
        )
        for rec in valid_rows:
            code = (rec.get("patient_code") or "").strip()
            if not code:
                max_id += 1
                code = f"BN-{max_id:04d}"
            conn.execute(
                """
                INSERT INTO patients
                    (patient_code, full_name, birth_year, gender, phone, address, notes, created_at, updated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    code,
                    rec["full_name"].strip(),
                    rec.get("birth_year"),
                    rec.get("gender") or "Khác",
                    rec.get("phone"),
                    rec.get("address"),
                    rec.get("notes"),
                    now,
                    now,
                ),
            )
            inserted += 1
    return inserted


# =============================================================================
# Batch Prediction Excel Import (Template -> Parse/Validate -> Preview -> Commit)
# =============================================================================

PREDICTION_CODE_ALIASES = {"patient_code", "mã bệnh nhân", "mã bn", "ma benh nhan", "ma bn"}


def build_prediction_import_template() -> bytes:
    """Generate a sample .xlsx template with optional patient_code + 13 clinical features."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Mau_Du_Doan_Benh_Tim"

    headers = ["patient_code"] + FEATURES
    ws.append(headers)

    navy_fill = PatternFill(start_color="0B3C5D", end_color="0B3C5D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    for idx in range(1, len(headers) + 1):
        c = ws.cell(row=1, column=idx)
        c.fill = navy_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[c.column_letter].width = 14

    ws.append(["", 63, 0, 0, 108, 269, 0, 1, 169, 1, 1.8, 1, 2, 2])
    ws.append(["", 67, 0, 2, 115, 564, 0, 0, 160, 0, 1.6, 1, 0, 3])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _find_prediction_header_row(ws) -> Tuple[Optional[int], Dict[str, int]]:
    """Locate header row containing the 13 FEATURES (by raw code or Vietnamese label)."""
    label_to_col = {v.strip().lower(): k for k, v in FEATURE_LABELS.items()}
    for row_idx in range(1, min(11, ws.max_row + 1)):
        col_map: Dict[str, int] = {}
        for col_idx in range(1, ws.max_column + 1):
            norm = _normalize_header(ws.cell(row=row_idx, column=col_idx).value)
            if not norm:
                continue
            if norm in PREDICTION_CODE_ALIASES and "patient_code" not in col_map:
                col_map["patient_code"] = col_idx
            elif norm in FEATURES and norm not in col_map:
                col_map[norm] = col_idx
            elif norm in label_to_col and label_to_col[norm] not in col_map:
                col_map[label_to_col[norm]] = col_idx
        if all(f in col_map for f in FEATURES):
            return row_idx, col_map
    return None, {}


def _load_patient_lookup(db_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT id, patient_code, full_name FROM patients").fetchall()
        return {
            str(r["patient_code"]).strip().upper(): {
                "id": int(r["id"]),
                "patient_code": r["patient_code"],
                "full_name": r["full_name"],
            }
            for r in rows
            if r["patient_code"]
        }


def parse_predictions_excel(
    file_bytes: bytes, db_path: Optional[Path] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse and validate an uploaded .xlsx file of 13-feature rows without DB writes."""
    wb = load_workbook(io.BytesIO(file_bytes), data_only=True)
    ws = wb.active

    header_row, col_map = _find_prediction_header_row(ws)
    if header_row is None:
        raise ValueError(
            "Không tìm thấy đủ 13 cột đặc trưng (age, sex, cp, trestbps, chol, fbs, restecg, "
            "thalach, exang, oldpeak, slope, ca, thal) trong 10 dòng đầu của file Excel."
        )

    patient_lookup = _load_patient_lookup(db_path)
    valid_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for row_idx in range(header_row + 1, ws.max_row + 1):
        raw_vals = {
            f: ws.cell(row=row_idx, column=col_map[f]).value for f in FEATURES
        }
        p_code_raw = (
            ws.cell(row=row_idx, column=col_map["patient_code"]).value
            if "patient_code" in col_map
            else None
        )
        if all(v is None or str(v).strip() == "" for v in raw_vals.values()) and not p_code_raw:
            continue

        features: Dict[str, Any] = {}
        parse_errors: List[str] = []
        for col in FEATURES:
            val = raw_vals[col]
            if val is None or str(val).strip() == "":
                features[col] = None
                continue
            s_val = str(val).strip()
            # Handle strings like "0 – Nữ" if copied from exported Excel
            if col in CODES and "–" in s_val:
                s_val = s_val.split("–", 1)[0].strip()
            try:
                features[col] = int(float(s_val)) if col in CODES else float(s_val)
            except ValueError:
                parse_errors.append(f"'{col}' không phải số hợp lệ ({val})")
                features[col] = val

        ok, val_errors = validate_input_dict(features)
        all_errors = parse_errors + (val_errors if not parse_errors else [])

        p_code = str(p_code_raw or "").strip()
        p_info = patient_lookup.get(p_code.upper()) if p_code else None

        item = {
            "excel_row": row_idx,
            "patient_code": p_info["patient_code"] if p_info else (p_code or None),
            "patient_id": p_info["id"] if p_info else None,
            "patient_name": p_info["full_name"] if p_info else ("Không liên kết" if not p_code else f"{p_code} (chưa có trong CSDL)"),
            "features": features,
        }

        if all_errors or not ok:
            error_rows.append({**item, "reason": "; ".join(all_errors)})
        else:
            valid_rows.append(item)

    return valid_rows, error_rows


def run_batch_excel_predictions(
    model,
    model_name: str,
    valid_rows: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run ML inference on validated Excel rows for preview & export (does NOT write to DB)."""
    version = get_model_version_tag(model_name, db_path=db_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(valid_rows, start=1):
        pred, score, label_text = predict_single(model, row["features"])
        results.append(
            {
                "id": idx,
                "excel_row": row["excel_row"],
                "patient_id": row["patient_id"],
                "patient_code": row["patient_code"] or "—",
                "patient_name": row["patient_name"],
                "model_name": model_name,
                "model_version": version,
                "predicted_label": pred,
                "label_text": label_text,
                "model_score": score,
                "input_features": row["features"],
                "predicted_at": ts,
            }
        )
    return results


def commit_batch_predictions(
    preview_results: List[Dict[str, Any]], db_path: Optional[Path] = None
) -> int:
    """Persist confirmed batch prediction results into SQLite inside one transaction."""
    if not preview_results:
        return 0

    import json

    inserted = 0
    with get_connection(db_path) as conn:
        for r in preview_results:
            conn.execute(
                """
                INSERT INTO predictions
                    (patient_id, model_name, model_version, input_features, predicted_label, label_text, model_score, predicted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("patient_id"),
                    r["model_name"],
                    r["model_version"],
                    json.dumps(r["input_features"], ensure_ascii=False),
                    int(r["predicted_label"]),
                    r["label_text"],
                    float(r["model_score"]),
                    r["predicted_at"],
                ),
            )
            inserted += 1
    return inserted
