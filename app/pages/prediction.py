"""Heart disease prediction page — single prediction + batch Excel import/export with preview & confirm."""

from datetime import datetime

import streamlit as st

from components.state import get_evaluation_artifacts, get_experiment_and_models
from components.ui import render_disclaimer_banner, render_page_header, render_styled_table
from database.patient_repository import list_active_patients
from services.excel_export_service import export_predictions_to_excel
from services.excel_import_service import (
    build_prediction_import_template,
    commit_batch_predictions,
    parse_predictions_excel,
    run_batch_excel_predictions,
)
from services.prediction_service import run_prediction
from src.config import CODES, FEATURE_LABELS, FEATURE_LIMITS, FEATURES, LABELS

render_page_header(
    "🩺 Dự đoán bệnh tim",
    "Dự đoán từng hồ sơ hoặc nhập/xuất hàng loạt từ Excel (xem trước kết quả trước khi xác nhận lưu SQLite).",
)
render_disclaimer_banner()

experiment, models = get_experiment_and_models()
artifacts = get_evaluation_artifacts()
samples = artifacts["demo_samples"]

model_names = list(models.keys())
default_idx = (
    model_names.index(experiment["selected_model"])
    if experiment["selected_model"] in model_names
    else 0
)

tab_single, tab_batch = st.tabs(
    ["🩺 Dự đoán từng hồ sơ", "📊 Nhập / Xuất Excel hàng loạt"]
)

# =============================================================================
# TAB 1: Single Patient Prediction
# =============================================================================
with tab_single:
    patients = list_active_patients()
    patient_options = [None] + [p["id"] for p in patients]
    patient_labels = {None: "Không liên kết bệnh nhân (chế độ thử nghiệm)"}
    patient_by_id = {}
    for p in patients:
        patient_labels[p["id"]] = f"{p['patient_code']} – {p['full_name']}"
        patient_by_id[p["id"]] = p

    preselected_pid = st.session_state.get("prediction_patient_id")
    default_p_idx = (
        patient_options.index(preselected_pid) if preselected_pid in patient_options else 0
    )

    c_top1, c_top2, c_top3 = st.columns([2, 2, 2.2])
    with c_top1:
        model_name = st.selectbox("Mô hình phân loại", model_names, index=default_idx, key="single_model")
    with c_top2:
        demo_choice = st.selectbox(
            "Điền nhanh từ mẫu kiểm thử",
            ["Có bệnh theo nhãn thật (Test)", "Không có bệnh theo nhãn thật (Test)"],
        )
    with c_top3:
        patient_id = st.selectbox(
            "Hồ sơ bệnh nhân",
            patient_options,
            index=default_p_idx,
            format_func=lambda pid: patient_labels[pid],
        )

    sample = samples[0 if demo_choice.startswith("Có bệnh") else 1]
    st.caption(
        f"Giá trị mặc định trên biểu mẫu lấy từ mẫu Test dòng nguồn {sample['source_row']} "
        f"(nhãn thật: {LABELS[sample['true_label']]}). Bạn có thể chỉnh sửa trực tiếp các chỉ số bên dưới."
    )

    with st.form("patient_prediction_form"):
        columns = st.columns(3)
        values = {}
        for i, col in enumerate(FEATURES):
            with columns[i % 3]:
                val = sample["features"][col]
                if col in CODES:
                    options = list(CODES[col].keys())
                    code = (4 if col == "ca" else 0) if val is None else int(val)
                    values[col] = st.selectbox(
                        FEATURE_LABELS[col],
                        options,
                        index=options.index(code),
                        format_func=lambda v, c=col: f"{v} – {CODES[c][v]}",
                        key=f"pred_{sample['source_row']}_{col}",
                    )
                else:
                    lo, hi = FEATURE_LIMITS[col]
                    values[col] = st.number_input(
                        FEATURE_LABELS[col],
                        min_value=lo,
                        max_value=hi,
                        value=float(val),
                        step=0.1 if col == "oldpeak" else 1.0,
                        key=f"pred_{sample['source_row']}_{col}",
                    )

        save_to_history = st.checkbox("Lưu kết quả vào lịch sử SQLite", value=True)
        submitted = st.form_submit_button("Thực hiện phân loại", type="primary")

    if submitted:
        try:
            result = run_prediction(
                model=models[model_name],
                model_name=model_name,
                features=values,
                patient_id=patient_id,
                persist=save_to_history,
            )
            pred = result["predicted_label"]
            score = result["model_score"]
            border_color = "#D9534F" if pred == 1 else "#0E9F9F"

            st.markdown(
                f"""
                <div class="hc-result-box" style="border-left-color: {border_color};">
                    <h3 style="margin:0 0 0.4rem 0; color:#0B3C5D;">
                        Kết quả phân loại: Thuộc nhóm {'có bệnh' if pred else 'không có bệnh'} theo dữ liệu nghiên cứu
                    </h3>
                    <p style="margin:0.2rem 0; font-size:1.05rem;">
                        <strong>Điểm mô hình ước lượng cho lớp có bệnh (lớp 1):</strong> {score:.1%}
                        <span style="color:#5A7184; font-size:0.88rem;">(giá trị thô: {score:.6f})</span>
                    </p>
                    <p style="margin:0.35rem 0 0 0; font-size:0.84rem; color:#5A7184;">
                        Mô hình: <strong>{model_name}</strong> ({result['model_version']}) ·
                        Nhãn chuẩn hóa: <strong>{pred} ({LABELS[pred]})</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Lưu ý: Điểm mô hình chưa được hiệu chỉnh thành xác suất mắc bệnh thực tế hay nguy cơ lâm sàng trong tương lai. "
                "Kết quả phân loại có thể sai ở cả hai nhóm."
            )

            p_obj = patient_by_id.get(patient_id)
            single_export_row = [
                {
                    "id": result["record_id"] or 1,
                    "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "patient_code": p_obj["patient_code"] if p_obj else "—",
                    "patient_name": p_obj["full_name"] if p_obj else "Không liên kết",
                    "model_name": model_name,
                    "model_version": result["model_version"],
                    "label_text": result["label_text"],
                    "model_score": score,
                    "input_features": values,
                }
            ]
            if save_to_history and result["record_id"]:
                st.success(f"Đã lưu kết quả vào lịch sử SQLite (Mã lượt dự đoán: #{result['record_id']}).")

            c_btn1, c_btn2, c_btn3, _ = st.columns([1.8, 1.5, 1.8, 2])
            c_btn1.download_button(
                "📥 Xuất Excel kết quả này",
                data=export_predictions_to_excel(
                    single_export_row,
                    title="Kết quả dự đoán bệnh tim — HeartCare AI",
                    filter_summary=f"Mô hình: {model_name}",
                    patient=p_obj,
                ),
                file_name="heartcare_ket_qua_du_doan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="btn_export_single_pred",
            )
            if save_to_history and result["record_id"]:
                if c_btn2.button("🕒 Xem lịch sử"):
                    st.switch_page("pages/history.py")
                if patient_id and c_btn3.button("📋 Xem hồ sơ BN"):
                    st.session_state["selected_patient_id"] = patient_id
                    st.switch_page("pages/patients.py")
        except Exception as exc:
            st.error(f"Lỗi khi thực hiện dự đoán: {exc}")

# =============================================================================
# TAB 2: Batch Excel Import (Preview -> Export -> Confirm Save to SQLite)
# =============================================================================
with tab_batch:
    st.caption(
        "Tải file `.xlsx` chứa 13 chỉ số lâm sàng (có thể kèm cột `patient_code` để tự động liên kết bệnh nhân trong SQLite). "
        "Hệ thống sẽ kiểm tra, chạy dự đoán **xem trước** và cho phép bạn xuất Excel hoặc bấm **Xác nhận lưu vào SQLite**."
    )

    bc1, bc2 = st.columns([2, 2])
    with bc1:
        batch_model_name = st.selectbox(
            "Chọn mô hình dự đoán hàng loạt",
            model_names,
            index=default_idx,
            key="batch_model_select",
        )
    with bc2:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        st.download_button(
            "📄 Tải file Excel mẫu (13 đặc trưng)",
            data=build_prediction_import_template(),
            file_name="heartcare_mau_du_doan_hang_loat.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="btn_download_pred_template",
        )

    if "pred_excel_nonce" not in st.session_state:
        st.session_state["pred_excel_nonce"] = 0

    uploaded_pred_file = st.file_uploader(
        "Chọn file Excel (.xlsx) chứa danh sách hồ sơ cần dự đoán",
        type=["xlsx"],
        key=f"pred_excel_uploader_{st.session_state['pred_excel_nonce']}",
    )

    if uploaded_pred_file is not None:
        try:
            valid_rows, error_rows = parse_predictions_excel(uploaded_pred_file.getvalue())
            preview_results = (
                run_batch_excel_predictions(models[batch_model_name], batch_model_name, valid_rows)
                if valid_rows
                else []
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng dòng đọc được", len(valid_rows) + len(error_rows))
            m2.metric("Hợp lệ (đã dự đoán)", len(preview_results))
            m3.metric(
                "Phân loại: Có bệnh",
                sum(1 for r in preview_results if r["predicted_label"] == 1),
            )
            m4.metric("Dòng lỗi (bỏ qua)", len(error_rows))

            if preview_results:
                st.markdown("**Xem trước kết quả dự đoán (chưa lưu vào SQLite):**")
                render_styled_table(
                    ["Dòng Excel", "Mã BN", "Bệnh nhân", "Tuổi", "Mô hình", "Kết quả phân loại", "Điểm mô hình"],
                    [
                        [
                            r["excel_row"],
                            r["patient_code"],
                            r["patient_name"],
                            r["input_features"]["age"],
                            r["model_name"],
                            r["label_text"],
                            f"{r['model_score']:.1%}",
                        ]
                        for r in preview_results
                    ],
                    badge_col_indices={5},
                )

                act1, act2, act3, _ = st.columns([2.0, 2.0, 1.2, 2])
                act1.download_button(
                    "📥 Xuất Excel kết quả xem trước",
                    data=export_predictions_to_excel(
                        preview_results,
                        title=f"Kết quả dự đoán hàng loạt ({batch_model_name}) — HeartCare AI",
                        filter_summary=f"Nhập từ {uploaded_pred_file.name}",
                    ),
                    file_name="heartcare_du_doan_hang_loat.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="btn_export_batch_preview",
                )
                if act2.button(
                    f"✅ Xác nhận lưu {len(preview_results)} kết quả vào SQLite",
                    type="primary",
                    key="btn_confirm_batch_save",
                ):
                    saved_cnt = commit_batch_predictions(preview_results)
                    st.session_state["pred_excel_nonce"] += 1
                    st.session_state["batch_pred_success_msg"] = (
                        f"Đã lưu thành công **{saved_cnt}** lượt dự đoán hàng loạt vào lịch sử SQLite!"
                    )
                    st.rerun()
                if act3.button("❌ Hủy", key="btn_cancel_batch_pred"):
                    st.session_state["pred_excel_nonce"] += 1
                    st.rerun()
            else:
                st.warning("Không có dòng hợp lệ nào để dự đoán.")

            if error_rows:
                with st.expander(f"⚠️ Xem {len(error_rows)} dòng lỗi bị bỏ qua"):
                    render_styled_table(
                        ["Dòng Excel", "Mã BN", "Lý do lỗi"],
                        [
                            [e["excel_row"], e["patient_code"] or "—", e["reason"]]
                            for e in error_rows
                        ],
                    )
        except Exception as exc:
            st.error(f"Không thể xử lý file Excel: {exc}")

    if st.session_state.get("batch_pred_success_msg"):
        st.success(st.session_state.pop("batch_pred_success_msg"))
