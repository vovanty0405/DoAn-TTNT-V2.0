"""Excel export service using openpyxl with Vietnamese support and clean styling."""

import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from src.config import CODES, FEATURE_LABELS, FEATURES

NAVY_FILL = PatternFill(start_color="0B3C5D", end_color="0B3C5D", fill_type="solid")
SUBHEADER_FILL = PatternFill(start_color="EDF5FA", end_color="EDF5FA", fill_type="solid")
ZEBRA_FILL = PatternFill(start_color="F8FBFD", end_color="F8FBFD", fill_type="solid")

TITLE_FONT = Font(name="Calibri", size=14, bold=True, color="0B3C5D")
META_FONT = Font(name="Calibri", size=10, italic=True, color="5A7184")
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
BODY_FONT = Font(name="Calibri", size=10, color="162B3A")

THIN_BORDER = Border(
    left=Side(style="thin", color="D0DCE5"),
    right=Side(style="thin", color="D0DCE5"),
    top=Side(style="thin", color="D0DCE5"),
    bottom=Side(style="thin", color="D0DCE5"),
)


def _auto_fit_columns(ws, min_width: int = 10, max_width: int = 42) -> None:
    """Adjust column widths based on cell content."""
    for col_idx in range(1, ws.max_column + 1):
        max_len = 0
        col_letter = get_column_letter(col_idx)
        for row_idx in range(3, ws.max_row + 1):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val is not None:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[col_letter].width = max(min_width, min(max_width, max_len + 4))


def _style_header_row(ws, row_idx: int, num_cols: int) -> None:
    ws.row_dimensions[row_idx].height = 24
    for col_idx in range(1, num_cols + 1):
        cell = ws.cell(row=row_idx, column=col_idx)
        cell.fill = NAVY_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN_BORDER


def _format_feature(col: str, val: Any) -> str:
    if val is None:
        return "Thiếu / không xác định"
    if col in CODES:
        try:
            code = int(val)
            return f"{code} – {CODES[col].get(code, val)}"
        except (TypeError, ValueError):
            return str(val)
    return str(val)


def export_patients_to_excel(
    patients: List[Dict[str, Any]],
    title: str = "Danh sách bệnh nhân — HeartCare AI",
    filter_summary: str = "Tất cả hồ sơ",
) -> bytes:
    """Export a list of patient dictionaries to a formatted .xlsx file in memory."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Danh sách bệnh nhân"

    ws["A1"] = title
    ws["A1"].font = TITLE_FONT
    ws["A2"] = (
        f"Ngày xuất: {datetime.now().strftime('%d/%m/%Y %H:%M')} | "
        f"Phạm vi: {filter_summary} | Tổng số: {len(patients)}"
    )
    ws["A2"].font = META_FONT

    headers = [
        "STT",
        "Mã bệnh nhân",
        "Họ và tên",
        "Năm sinh",
        "Tuổi",
        "Giới tính",
        "Điện thoại",
        "Địa chỉ",
        "Trạng thái",
        "Ngày tạo",
        "Ghi chú",
    ]
    header_row = 4
    ws.append([])  # row 3 spacer
    ws.append(headers)
    _style_header_row(ws, header_row, len(headers))

    curr_year = datetime.now().year
    for idx, p in enumerate(patients, start=1):
        birth_year = p.get("birth_year")
        age = max(0, curr_year - int(birth_year)) if birth_year is not None else ""
        created = str(p.get("created_at") or "")
        row_vals = [
            idx,
            p.get("patient_code") or "",
            p.get("full_name") or "",
            birth_year if birth_year is not None else "",
            age,
            p.get("gender") or "",
            p.get("phone") or "",
            p.get("address") or "",
            "Đang hoạt động" if p.get("is_active") else "Đã lưu trữ",
            created[:10],
            p.get("notes") or "",
        ]
        ws.append(row_vals)
        r = ws.max_row
        for c_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=r, column=c_idx)
            cell.font = BODY_FONT
            cell.border = THIN_BORDER
            if idx % 2 == 0:
                cell.fill = ZEBRA_FILL
            if c_idx in (1, 4, 5, 6, 9, 10):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

    ws.freeze_panes = "A5"
    _auto_fit_columns(ws)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export_predictions_to_excel(
    predictions: List[Dict[str, Any]],
    title: str = "Lịch sử dự đoán — HeartCare AI",
    filter_summary: str = "Tất cả lượt dự đoán",
    patient: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Export prediction history to a multi-sheet .xlsx workbook."""
    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = "Lịch sử dự đoán"

    ws_summary["A1"] = title
    ws_summary["A1"].font = TITLE_FONT
    meta = f"Ngày xuất: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Phạm vi: {filter_summary} | Tổng lượt: {len(predictions)}"
    if patient:
        meta = f"Bệnh nhân: {patient['patient_code']} – {patient['full_name']} | " + meta
    ws_summary["A2"] = meta
    ws_summary["A2"].font = META_FONT

    headers = [
        "STT",
        "Mã lượt",
        "Thời gian dự đoán",
        "Mã bệnh nhân",
        "Họ tên bệnh nhân",
        "Mô hình",
        "Phiên bản mô hình",
        "Kết quả phân loại",
        "Điểm mô hình (lớp 1)",
    ]
    ws_summary.append([])
    ws_summary.append(headers)
    _style_header_row(ws_summary, 4, len(headers))

    # Sheet 2: 13 detailed clinical features per prediction
    ws_feats = wb.create_sheet(title="Chi tiết 13 chỉ số")
    ws_feats["A1"] = "Chi tiết 13 chỉ số đầu vào theo từng lượt dự đoán"
    ws_feats["A1"].font = TITLE_FONT
    ws_feats["A2"] = "Giá trị lưu bất biến tại thời điểm thực hiện phân loại."
    ws_feats["A2"].font = META_FONT

    feat_headers = ["Mã lượt", "Thời gian", "Mã BN"] + [FEATURE_LABELS[c] for c in FEATURES]
    ws_feats.append([])
    ws_feats.append(feat_headers)
    _style_header_row(ws_feats, 4, len(feat_headers))

    for idx, rec in enumerate(predictions, start=1):
        p_code = rec.get("patient_code") or (patient["patient_code"] if patient else "—")
        p_name = rec.get("patient_name") or (patient["full_name"] if patient else "Không liên kết")
        score = float(rec.get("model_score", 0.0))

        ws_summary.append(
            [
                idx,
                rec.get("id"),
                str(rec.get("predicted_at") or ""),
                p_code,
                p_name,
                rec.get("model_name") or "",
                rec.get("model_version") or "",
                rec.get("label_text") or "",
                score,
            ]
        )
        r = ws_summary.max_row
        for c_idx in range(1, len(headers) + 1):
            cell = ws_summary.cell(row=r, column=c_idx)
            cell.font = BODY_FONT
            cell.border = THIN_BORDER
            if idx % 2 == 0:
                cell.fill = ZEBRA_FILL
            if c_idx == 9:
                cell.number_format = "0.0%"
                cell.alignment = Alignment(horizontal="right", vertical="center")
            elif c_idx in (1, 2, 3, 4, 7):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

        # Populate Sheet 2
        raw_feats = rec.get("input_features", "{}")
        feats = json.loads(raw_feats) if isinstance(raw_feats, str) else raw_feats
        ws_feats.append(
            [rec.get("id"), str(rec.get("predicted_at") or ""), p_code]
            + [_format_feature(c, feats.get(c)) for c in FEATURES]
        )
        rf = ws_feats.max_row
        for c_idx in range(1, len(feat_headers) + 1):
            cell = ws_feats.cell(row=rf, column=c_idx)
            cell.font = BODY_FONT
            cell.border = THIN_BORDER
            if idx % 2 == 0:
                cell.fill = ZEBRA_FILL

    ws_summary.freeze_panes = "A5"
    ws_feats.freeze_panes = "D5"
    _auto_fit_columns(ws_summary)
    _auto_fit_columns(ws_feats)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
