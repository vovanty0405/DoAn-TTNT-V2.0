"""Reusable Patient Detail view — rendered when user clicks 'Xem chi tiết' in Patients or History."""

import json
from datetime import datetime

import streamlit as st

from components.ui import (
    format_feature_value,
    render_disclaimer_banner,
    render_health_card,
    render_page_header,
    render_patient_profile_card,
    render_styled_table,
)
from database.patient_repository import get_patient_by_id, update_patient
from database.prediction_repository import get_predictions_for_patient
from services.excel_export_service import export_predictions_to_excel
from services.pdf_export_service import build_patient_pdf
from src.config import FEATURE_LABELS, FEATURES


def render_patient_detail_view(patient_id: int) -> None:
    """Render the complete Patient Detail screen for a given patient_id."""
    patient = get_patient_by_id(int(patient_id))
    if not patient:
        st.error("Không tìm thấy hồ sơ bệnh nhân.")
        if st.button("⬅️ Quay lại danh sách bệnh nhân"):
            st.session_state.pop("selected_patient_id", None)
            st.rerun()
        return

    render_page_header(
        "📋 Chi tiết hồ sơ bệnh nhân",
        f"Hồ sơ {patient['patient_code']} · {patient['full_name']}",
    )

    predictions = get_predictions_for_patient(patient["id"])
    selected_pred = None
    if predictions:
        pred_map = {
            p["id"]: f"Lần #{p['id']} · {str(p['predicted_at'])[:16]} · {p['model_name']} ({p['label_text']})"
            for p in predictions
        }
        pred_ids = list(pred_map.keys())
        selected_pred_id = st.selectbox(
            "Chọn lần dự đoán để xem thông số (mặc định mới nhất)",
            pred_ids,
            format_func=lambda pid: pred_map[pid],
        )
        selected_pred = next(p for p in predictions if p["id"] == selected_pred_id)

    # Compact Action Bar
    b1, b2, b3, b4, _ = st.columns([1.4, 1.3, 1.2, 1.5, 2.2])
    if b1.button("⬅️ Quay lại danh sách"):
        st.session_state.pop("selected_patient_id", None)
        st.rerun()

    if b2.button("🩺 Dự đoán mới", type="primary"):
        st.session_state["prediction_patient_id"] = patient["id"]
        st.session_state.pop("selected_patient_id", None)
        st.switch_page("pages/prediction.py")

    b3.download_button(
        "📄 Xuất PDF",
        data=build_patient_pdf(patient, selected_pred),
        file_name=f"{patient['patient_code']}_heartcare.pdf",
        mime="application/pdf",
    )

    b4.download_button(
        "📥 Xuất Excel",
        data=export_predictions_to_excel(
            predictions,
            title=f"Lịch sử dự đoán — {patient['patient_code']} ({patient['full_name']})",
            filter_summary=f"Hồ sơ {patient['patient_code']}",
            patient=patient,
        ),
        file_name=f"{patient['patient_code']}_lich_su.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Compact Modern Profile Card
    render_patient_profile_card(patient)

    # Inline Edit Expander
    with st.expander("✏️ Sửa thông tin hồ sơ"):
        with st.form("detail_edit_patient"):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                e_code = st.text_input("Mã bệnh nhân *", value=patient["patient_code"])
                e_name = st.text_input("Họ và tên *", value=patient["full_name"])
            with ec2:
                e_birth = st.number_input(
                    "Năm sinh",
                    min_value=1900,
                    max_value=datetime.now().year,
                    value=int(patient["birth_year"] or 1970),
                    step=1,
                )
                g_opts = ["Nam", "Nữ", "Khác"]
                e_gender = st.selectbox(
                    "Giới tính",
                    g_opts,
                    index=g_opts.index(patient["gender"]) if patient["gender"] in g_opts else 2,
                )
            with ec3:
                e_phone = st.text_input("Điện thoại", value=patient["phone"] or "")
                e_addr = st.text_input("Địa chỉ", value=patient["address"] or "")
            e_notes = st.text_area("Ghi chú", value=patient["notes"] or "", height=70)
            if st.form_submit_button("Lưu cập nhật", type="primary"):
                update_patient(
                    patient["id"],
                    full_name=e_name,
                    patient_code=e_code,
                    birth_year=e_birth,
                    gender=e_gender,
                    phone=e_phone,
                    address=e_addr,
                    notes=e_notes,
                )
                st.success("Đã cập nhật hồ sơ.")
                st.rerun()

    # Health Parameters & 13 Input Features
    st.subheader("Thông số sức khỏe & Kết quả phân loại")
    if not selected_pred:
        st.info("Chưa có dữ liệu dự đoán cho hồ sơ này. Nhấn **🩺 Dự đoán mới** để thực hiện.")
        return

    render_disclaimer_banner()
    features = (
        json.loads(selected_pred["input_features"])
        if isinstance(selected_pred["input_features"], str)
        else selected_pred["input_features"]
    )

    border_color = "#D9534F" if selected_pred["predicted_label"] == 1 else "#0E9F9F"
    st.markdown(
        f"""
        <div class="hc-result-box" style="border-left-color: {border_color};">
            <strong>Kết quả mô hình ({selected_pred['model_name']} · {selected_pred['model_version']}):</strong>
            {selected_pred['label_text']} &nbsp;|&nbsp;
            <strong>Điểm mô hình (lớp 1):</strong> {float(selected_pred['model_score']):.1%}
            <br><span style="font-size:0.82rem; color:#5A7184;">Thời gian ghi nhận: {selected_pred['predicted_at']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hc1, hc2, hc3, hc4 = st.columns(4)
    with hc1:
        render_health_card("🩺", "Huyết áp nghỉ", f"{features.get('trestbps', '—')} mmHg", "#1D6FA5")
    with hc2:
        render_health_card("🧪", "Cholesterol", f"{features.get('chol', '—')} mg/dL", "#0E9F9F")
    with hc3:
        render_health_card("💓", "Nhịp tim tối đa", f"{features.get('thalach', '—')} bpm", "#0B3C5D")
    with hc4:
        render_health_card("📉", "ST chênh xuống", f"{features.get('oldpeak', '—')}", "#3B82F6")

    st.markdown("#### Bảng đầy đủ 13 chỉ số đầu vào")
    render_styled_table(
        ["STT", "Mã biến", "Chỉ số sức khỏe", "Giá trị"],
        [
            [i, col, FEATURE_LABELS[col], format_feature_value(col, features.get(col))]
            for i, col in enumerate(FEATURES, start=1)
        ],
    )

    st.subheader("Lịch sử các lần dự đoán của bệnh nhân")
    render_styled_table(
        ["Mã", "Thời gian", "Mô hình", "Phiên bản", "Kết quả phân loại", "Điểm mô hình"],
        [
            [
                f"#{p['id']}",
                str(p["predicted_at"])[:16],
                p["model_name"],
                p["model_version"],
                p["label_text"],
                f"{float(p['model_score']):.1%}",
            ]
            for p in predictions
        ],
        badge_col_indices={4},
    )
