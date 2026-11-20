"""PDF report export service for patient clinical summary and model comparison."""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import CODES, FEATURE_LABELS, FEATURES


def _register_unicode_font() -> str:
    """Register a Unicode TTF font on Windows so Vietnamese renders cleanly."""
    candidates = [
        ("ArialUnicode", Path("C:/Windows/Fonts/arial.ttf")),
        ("SegoeUI", Path("C:/Windows/Fonts/segoeui.ttf")),
        ("Tahoma", Path("C:/Windows/Fonts/tahoma.ttf")),
    ]
    for name, path in candidates:
        if path.exists():
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(path)))
            return name
    return "Helvetica"


def _build_styles(font_name: str):
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleVN",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=15.5,
            textColor=colors.HexColor("#0B3C5D"),
            spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "H2VN",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=11.5,
            textColor=colors.HexColor("#1D6FA5"),
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyVN",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=9.5,
            leading=13.5,
        ),
        "note": ParagraphStyle(
            "NoteVN",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=8.5,
            textColor=colors.HexColor("#555555"),
            leading=12,
        ),
    }


def build_patient_pdf(
    patient: Dict[str, Any],
    prediction: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Generate a PDF summary for a patient and optional selected prediction.

    Raises ValueError if `prediction` belongs to a different patient_id.
    """
    if (
        prediction is not None
        and prediction.get("patient_id") is not None
        and patient.get("id") is not None
        and int(prediction["patient_id"]) != int(patient["id"])
    ):
        raise ValueError(
            "Dữ liệu dự đoán không khớp với hồ sơ bệnh nhân được chọn (ngăn xuất nhầm bệnh nhân)."
        )

    font_name = _register_unicode_font()
    st_map = _build_styles(font_name)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    story = [
        Paragraph("HeartCare AI — Phiếu Tóm Tắt Hồ Sơ & Kết Quả Phân Loại", st_map["title"]),
        Paragraph(
            f"Ngày xuất phiếu: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            st_map["note"],
        ),
        Spacer(1, 6),
        Paragraph("1. Thông tin bệnh nhân", st_map["h2"]),
    ]

    age_str = (
        f"{datetime.now().year - int(patient['birth_year'])} tuổi (sinh năm {patient['birth_year']})"
        if patient.get("birth_year")
        else "Chưa cập nhật"
    )
    info_data = [
        ["Mã bệnh nhân", str(patient.get("patient_code") or ""), "Họ và tên", str(patient.get("full_name") or "")],
        ["Giới tính", str(patient.get("gender") or ""), "Tuổi", age_str],
        ["Điện thoại", str(patient.get("phone") or "—"), "Địa chỉ", str(patient.get("address") or "—")],
        ["Ghi chú", str(patient.get("notes") or "—"), "Trạng thái", "Đang hoạt động" if patient.get("is_active") else "Đã lưu trữ"],
    ]
    info_table = Table(info_data, colWidths=[30 * mm, 55 * mm, 28 * mm, 61 * mm])
    info_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9.2),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F6FA")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F0F6FA")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0DCE5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(info_table)

    story.append(Paragraph("2. Kết quả phân loại của mô hình", st_map["h2"]))
    if not prediction:
        story.append(
            Paragraph("Chưa có lượt dự đoán nào được ghi nhận cho hồ sơ này.", st_map["body"])
        )
    else:
        pred_data = [
            ["Mã lượt dự đoán", f"#{prediction['id']}", "Ngày thực hiện", str(prediction["predicted_at"])],
            ["Mô hình sử dụng", f"{prediction['model_name']} ({prediction['model_version']})", "Kết quả phân loại", str(prediction["label_text"])],
            ["Điểm đầu ra mô hình", f"{float(prediction['model_score']):.1%} ({float(prediction['model_score']):.6f})", "Nhãn chuẩn hóa", f"{prediction['predicted_label']}"],
        ]
        pred_table = Table(pred_data, colWidths=[35 * mm, 52 * mm, 32 * mm, 55 * mm])
        pred_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.2),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F6FA")),
                    ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F0F6FA")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0DCE5")),
                    ("PADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(pred_table)
        story.append(Spacer(1, 6))
        story.append(Paragraph("3. Bảng 13 chỉ số đầu vào", st_map["h2"]))

        features = (
            json.loads(prediction["input_features"])
            if isinstance(prediction["input_features"], str)
            else prediction["input_features"]
        )
        feat_rows = [["STT", "Chỉ số", "Giá trị"]]
        for idx, col in enumerate(FEATURES, start=1):
            raw_val = features.get(col)
            if col in CODES and raw_val is not None:
                display_val = f"{int(raw_val)} – {CODES[col].get(int(raw_val), raw_val)}"
            else:
                display_val = str(raw_val)
            feat_rows.append([str(idx), FEATURE_LABELS[col], display_val])

        feat_table = Table(feat_rows, colWidths=[14 * mm, 75 * mm, 85 * mm])
        feat_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3C5D")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0DCE5")),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(feat_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Giới hạn của mô hình", st_map["h2"]))
    story.append(
        Paragraph(
            "• Kết quả chỉ phục vụ mục đích học tập và nghiên cứu, không thay thế thăm khám hoặc chẩn đoán của bác sĩ.<br/>"
            "• Điểm đầu ra của mô hình chưa được hiệu chỉnh thành xác suất mắc bệnh thực tế hay nguy cơ lâm sàng.<br/>"
            "• Mô hình được đánh giá trên tập Test độc lập gồm 91 bản ghi duy nhất, chưa có kiểm định ngoài lâm sàng.",
            st_map["note"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


def build_model_comparison_pdf(
    experiment: Dict[str, Any],
    metrics_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    figure_paths: Optional[List[Path]] = None,
) -> bytes:
    """Generate a PDF report comparing the 3 trained models from copied metrics and charts."""
    font_name = _register_unicode_font()
    st_map = _build_styles(font_name)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    selected = experiment.get("selected_model", "Random Forest")
    story = [
        Paragraph("HeartCare AI — Báo Cáo Đánh Giá & So Sánh Mô Hình", st_map["title"]),
        Paragraph(
            f"Ngày xuất báo cáo: {datetime.now().strftime('%d/%m/%Y %H:%M')} | "
            f"Mô hình được chọn theo F1 CV: {selected}",
            st_map["note"],
        ),
        Spacer(1, 6),
        Paragraph("1. Kết quả trên tập Test độc lập (91 bản ghi)", st_map["h2"]),
    ]

    m_rows = [["Mô hình", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]
    for _, row in metrics_df.iterrows():
        m_rows.append(
            [
                str(row["model"]),
                f"{float(row['accuracy']):.3f}",
                f"{float(row['precision']):.3f}",
                f"{float(row['recall']):.3f}",
                f"{float(row['f1']):.3f}",
                f"{float(row['roc_auc']):.3f}",
            ]
        )
    m_table = Table(m_rows, colWidths=[44 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm])
    m_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9.2),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3C5D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0DCE5")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(m_table)

    story.append(Paragraph("2. So sánh Stratified 5-Fold CV trên Train (Baseline vs Tuned)", st_map["h2"]))
    cv_rows = [["Mô hình", "Biến thể", "F1 Mean", "F1 Std", "Recall Mean", "ROC-AUC Mean"]]
    for _, row in cv_df.iterrows():
        cv_rows.append(
            [
                str(row["model"]),
                str(row["variant"]),
                f"{float(row['f1_mean']):.3f}",
                f"{float(row['f1_std']):.3f}",
                f"{float(row['recall_mean']):.3f}",
                f"{float(row['roc_auc_mean']):.3f}",
            ]
        )
    cv_table = Table(cv_rows, colWidths=[40 * mm, 26 * mm, 27 * mm, 25 * mm, 28 * mm, 28 * mm])
    cv_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D6FA5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0DCE5")),
                ("PADDING", (0, 0), (-1, -1), 4.5),
            ]
        )
    )
    story.append(cv_table)

    if figure_paths:
        story.append(Paragraph("3. Biểu đồ đánh giá từ thực nghiệm gốc", st_map["h2"]))
        for fig_path in figure_paths:
            if fig_path.exists():
                img = Image(str(fig_path))
                max_w = 168 * mm
                scale = min(1.0, max_w / img.drawWidth)
                img.drawWidth *= scale
                img.drawHeight *= scale
                story.append(img)
                story.append(Spacer(1, 5))

    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"Quy tắc chọn mô hình: {experiment.get('selection_rule', 'N/A')}<br/>"
            f"Lưu ý CV: {experiment.get('cv_caveat', '')}",
            st_map["note"],
        )
    )

    doc.build(story)
    return buffer.getvalue()
