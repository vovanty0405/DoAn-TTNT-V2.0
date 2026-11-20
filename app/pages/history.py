"""Prediction History page — search, filter, pagination, detail view, and link to patient profile."""

import json
import math
from datetime import date

import pandas as pd
import streamlit as st

try:
    from app.components.ui import format_feature_value, render_page_header, render_styled_table
except ModuleNotFoundError:
    from components.ui import format_feature_value, render_page_header, render_styled_table
from database.prediction_repository import get_prediction_by_id, query_predictions
from services.excel_export_service import export_predictions_to_excel
from src.config import FEATURE_LABELS, FEATURES, MODEL_FILES

render_page_header(
    "🕒 Lịch sử dự đoán",
    "Tra cứu các lượt phân loại đã lưu (bất biến theo thời điểm thực hiện), xem lại 13 chỉ số và liên kết về hồ sơ bệnh nhân.",
)

# Filters
f1, f2, f3, f4 = st.columns([2.2, 1.5, 1.5, 1.3])
with f1:
    search = st.text_input("Tìm kiếm bệnh nhân (mã hoặc họ tên)", placeholder="VD: DEMO-001, Nguyễn...")
with f2:
    model_filter = st.selectbox("Mô hình", ["Tất cả"] + list(MODEL_FILES.keys()))
with f3:
    label_choice = st.selectbox("Kết quả phân loại", ["Tất cả", "Có bệnh (1)", "Không có bệnh (0)"])
    label_map = {"Tất cả": None, "Có bệnh (1)": 1, "Không có bệnh (0)": 0}
with f4:
    page_size = st.selectbox("Bản ghi / trang", [10, 20, 50], index=0)

d1, d2, d3 = st.columns([1.4, 1.6, 1.6])
with d1:
    use_date = st.checkbox("Lọc theo ngày dự đoán", value=False)
with d2:
    date_from = st.date_input("Từ ngày", value=date(2025, 1, 1)).isoformat() if use_date else None
with d3:
    date_to = st.date_input("Đến ngày", value=date.today()).isoformat() if use_date else None

if "history_page" not in st.session_state:
    st.session_state["history_page"] = 1

rows, total_records = query_predictions(
    search=search,
    model_name=model_filter,
    predicted_label=label_map[label_choice],
    date_from=date_from,
    date_to=date_to,
    page=st.session_state["history_page"],
    page_size=page_size,
)

total_pages = max(1, math.ceil(total_records / page_size))
if st.session_state["history_page"] > total_pages:
    st.session_state["history_page"] = total_pages
    rows, total_records = query_predictions(
        search=search,
        model_name=model_filter,
        predicted_label=label_map[label_choice],
        date_from=date_from,
        date_to=date_to,
        page=st.session_state["history_page"],
        page_size=page_size,
    )

st.caption(
    f"Hiển thị **{len(rows)}** / **{total_records}** lượt dự đoán · Trang **{st.session_state['history_page']} / {total_pages}**"
)

hx1, hx2, _ = st.columns([1.8, 1.8, 4])
filtered_preds, _ = query_predictions(
    search=search,
    model_name=model_filter,
    predicted_label=label_map[label_choice],
    date_from=date_from,
    date_to=date_to,
    page=1,
    page_size=max(1, total_records),
)
hx1.download_button(
    "📥 Xuất Excel (theo bộ lọc)",
    data=export_predictions_to_excel(
        filtered_preds,
        title="Lịch sử dự đoán (theo bộ lọc) — HeartCare AI",
        filter_summary=f"Mô hình: {model_filter}, Kết quả: {label_choice}",
    ),
    file_name="heartcare_lich_su_loc.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
all_preds_export, _ = query_predictions(page=1, page_size=10000)
hx2.download_button(
    "📥 Xuất Excel (toàn bộ)",
    data=export_predictions_to_excel(
        all_preds_export,
        title="Toàn bộ lịch sử dự đoán — HeartCare AI",
        filter_summary="Toàn bộ lịch sử trong SQLite",
    ),
    file_name="heartcare_lich_su_tat_ca.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

if not rows:
    st.info("Chưa có lịch sử dự đoán nào phù hợp.")
else:
    hcols = st.columns([0.7, 1.6, 1.3, 1.8, 1.5, 1.4, 1.1, 1.6])
    for hc, title in zip(
        hcols,
        ["Mã", "Thời gian", "Mã BN", "Bệnh nhân", "Mô hình", "Kết quả", "Điểm", "Thao tác"],
    ):
        hc.markdown(f"**{title}**")
    st.divider()

    for r in rows:
        cols = st.columns([0.7, 1.6, 1.3, 1.8, 1.5, 1.4, 1.1, 1.6])
        cols[0].write(f"#{r['id']}")
        cols[1].write(str(r["predicted_at"])[:16])
        cols[2].write(r["patient_code"] or "—")
        cols[3].write(r["patient_name"] or "Không liên kết")
        cols[4].write(r["model_name"])
        cols[5].write(r["label_text"])
        cols[6].write(f"{float(r['model_score']):.1%}")
        b1, b2 = cols[7].columns(2)
        if b1.button("🔍", key=f"hist_detail_{r['id']}", help="Xem chi tiết 13 thông số"):
            st.session_state["selected_prediction_id"] = r["id"]
        if r["patient_id"] and b2.button("👤", key=f"hist_pat_{r['id']}", help="Mở hồ sơ bệnh nhân"):
            st.session_state["selected_patient_id"] = r["patient_id"]
            st.switch_page("pages/patients.py")

    # Pagination
    p1, p2, p3 = st.columns([1, 2, 1])
    with p1:
        if st.button("⬅️ Trang trước", disabled=st.session_state["history_page"] <= 1):
            st.session_state["history_page"] -= 1
            st.rerun()
    with p2:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px;'>Trang {st.session_state['history_page']} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with p3:
        if st.button("Trang sau ➡️", disabled=st.session_state["history_page"] >= total_pages):
            st.session_state["history_page"] += 1
            st.rerun()

# Detail viewer for selected prediction
sel_pred_id = st.session_state.get("selected_prediction_id")
if sel_pred_id:
    detail = get_prediction_by_id(int(sel_pred_id))
    if detail:
        st.divider()
        st.subheader(f"🔍 Chi tiết lượt dự đoán #{detail['id']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mô hình", detail["model_name"])
        c2.metric("Phiên bản", detail["model_version"])
        c3.metric("Kết quả", detail["label_text"])
        c4.metric("Điểm mô hình", f"{float(detail['model_score']):.1%}")

        st.write(
            f"**Thời gian:** {detail['predicted_at']}  |  "
            f"**Bệnh nhân:** {detail['patient_name'] or 'Không liên kết'} "
            f"({detail['patient_code'] or '—'})"
        )
        if detail["patient_id"]:
            if st.button("📋 Mở hồ sơ bệnh nhân này"):
                st.session_state["selected_patient_id"] = detail["patient_id"]
                st.switch_page("pages/patients.py")

        feats = json.loads(detail["input_features"])
        render_styled_table(
            ["STT", "Mã biến", "Chỉ số sức khỏe", "Giá trị tại thời điểm dự đoán"],
            [
                [i, col, FEATURE_LABELS[col], format_feature_value(col, feats.get(col))]
                for i, col in enumerate(FEATURES, start=1)
            ],
        )
