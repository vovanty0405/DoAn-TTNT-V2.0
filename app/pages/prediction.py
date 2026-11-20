"""Heart disease prediction page using the 3 copied scikit-learn pipelines."""

import streamlit as st

try:
    from app.components.state import get_evaluation_artifacts, get_experiment_and_models
    from app.components.ui import render_disclaimer_banner, render_page_header
except ModuleNotFoundError:
    from components.state import get_evaluation_artifacts, get_experiment_and_models
    from components.ui import render_disclaimer_banner, render_page_header
from database.patient_repository import list_active_patients
from services.prediction_service import run_prediction
from src.config import CODES, FEATURE_LABELS, FEATURE_LIMITS, FEATURES, LABELS

render_page_header(
    "🩺 Dự đoán bệnh tim",
    "Nhập 13 chỉ số đầu vào, chọn mô hình đã huấn luyện và lưu kết quả vào lịch sử SQLite.",
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

patients = list_active_patients()
patient_options = [None] + [p["id"] for p in patients]
patient_labels = {None: "Không liên kết bệnh nhân (chế độ thử nghiệm)"}
for p in patients:
    patient_labels[p["id"]] = f"{p['patient_code']} – {p['full_name']}"

preselected_pid = st.session_state.get("prediction_patient_id")
default_p_idx = (
    patient_options.index(preselected_pid) if preselected_pid in patient_options else 0
)

c_top1, c_top2, c_top3 = st.columns([2, 2, 2.2])
with c_top1:
    model_name = st.selectbox("Mô hình phân loại", model_names, index=default_idx)
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

        if save_to_history and result["record_id"]:
            st.success(f"Đã lưu kết quả vào lịch sử SQLite (Mã lượt dự đoán: #{result['record_id']}).")
            c_btn1, c_btn2, _ = st.columns([1.6, 1.8, 4])
            if c_btn1.button("🕒 Xem trong lịch sử"):
                st.switch_page("pages/history.py")
            if patient_id and c_btn2.button("📋 Xem hồ sơ bệnh nhân"):
                st.session_state["selected_patient_id"] = patient_id
                st.switch_page("pages/patients.py")
    except Exception as exc:
        st.error(f"Lỗi khi thực hiện dự đoán: {exc}")
