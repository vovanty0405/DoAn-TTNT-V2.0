"""Model Evaluation page — displays copied research artifacts with interactive Plotly charts."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from app.components.state import get_evaluation_artifacts, get_experiment_and_models
    from app.components.ui import render_kpi_card, render_page_header
except ModuleNotFoundError:
    from components.state import get_evaluation_artifacts, get_experiment_and_models
    from components.ui import render_kpi_card, render_page_header
from services.pdf_export_service import build_model_comparison_pdf
from src.evaluate_models import get_figure_path, list_available_figures

render_page_header(
    "📈 Đánh giá mô hình Machine Learning",
    "Toàn bộ chỉ số và biểu đồ được đọc trực tiếp từ kết quả thực nghiệm gốc đã sao chép (không huấn luyện lại).",
)

experiment, _ = get_experiment_and_models()
artifacts = get_evaluation_artifacts()
metrics = artifacts["metrics"]
cv = artifacts["cv"]
audit = artifacts["audit"]
importance = artifacts["importance"]
selected = experiment["selected_model"]

# Highlight KPIs for selected model
sel_test = experiment["models"][selected]["test"]
hdr_c1, hdr_c2 = st.columns([3.5, 1.5])
hdr_c1.subheader(f"Mô hình được chọn theo Stratified 5-Fold CV (F1): {selected}")
report_figs = [
    get_figure_path("performance.png"),
    get_figure_path(f"cm_{selected.replace(' ', '_')}.png"),
    get_figure_path("importance.png"),
]
hdr_c2.download_button(
    "📄 Xuất PDF so sánh mô hình",
    data=build_model_comparison_pdf(experiment, metrics, cv, figure_paths=report_figs),
    file_name="heartcare_bao_cao_mo_hinh.pdf",
    mime="application/pdf",
)
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card("Accuracy", f"{sel_test['accuracy']:.3f}", "🎯", "#0B3C5D")
with k2:
    render_kpi_card("Precision", f"{sel_test['precision']:.3f}", "📌", "#1D6FA5")
with k3:
    render_kpi_card("Recall", f"{sel_test['recall']:.3f}", "🔎", "#0E9F9F")
with k4:
    render_kpi_card("F1-score", f"{sel_test['f1']:.3f}", "⚖️", "#2E9E6B")
with k5:
    render_kpi_card("ROC-AUC", f"{sel_test['roc_auc']:.3f}", "📈", "#3B82F6")

# Test metrics table + grouped bar chart
st.subheader("1. So sánh 3 mô hình trên tập Test độc lập (91 bản ghi)")
st.dataframe(
    metrics.style.format(
        {c: "{:.3f}" for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    ),
    hide_index=True,
    width="stretch",
)

melted = metrics.melt(
    id_vars="model",
    value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
    var_name="Metric",
    value_name="Score",
)
fig_perf = px.bar(
    melted,
    x="Metric",
    y="Score",
    color="model",
    barmode="group",
    text_auto=".3f",
    color_discrete_sequence=["#0B3C5D", "#1D6FA5", "#0E9F9F"],
)
fig_perf.update_layout(
    yaxis=dict(range=[0, 1.05]),
    margin=dict(t=20, b=20, l=10, r=10),
    height=360,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_perf, use_container_width=True, key="eval_bar_perf")

# Confusion Matrix & Permutation Importance
st.subheader("2. Confusion Matrix & Permutation Importance")
c_left, c_right = st.columns(2)

with c_left:
    cm_model = st.selectbox(
        "Chọn mô hình xem Confusion Matrix (Test)",
        list(experiment["models"].keys()),
        index=list(experiment["models"].keys()).index(selected),
    )
    cm = experiment["models"][cm_model]["test"]["confusion_matrix"]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Dự đoán: 0 (Không bệnh)", "Dự đoán: 1 (Có bệnh)"],
            y=["Thực tế: 0 (Không bệnh)", "Thực tế: 1 (Có bệnh)"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig_cm.update_layout(
        title=f"Confusion Matrix — {cm_model}",
        margin=dict(t=40, b=20, l=10, r=10),
        height=340,
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cm, use_container_width=True, key="eval_heatmap_cm")

with c_right:
    fig_imp = px.bar(
        importance.sort_values("mean", ascending=True),
        x="mean",
        y="feature",
        error_x="std",
        orientation="h",
        labels={"mean": "Mức giảm F1 trung bình (30 lần hoán vị)", "feature": "Đặc trưng"},
        color_discrete_sequence=["#1D6FA5"],
        title=f"Permutation Importance trên Test ({selected})",
    )
    fig_imp.update_layout(
        margin=dict(t=40, b=20, l=10, r=10),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_imp, use_container_width=True, key="eval_bar_imp")
    st.caption(
        "Phân tích sau đánh giá trên tập Test, không dùng để chọn đặc trưng hay huấn luyện lại, không suy ra quan hệ nhân quả."
    )

# Baseline vs Tuned comparison + GridSearchCV info
st.subheader("3. So sánh Baseline vs Tuned (GridSearchCV) trên tập Test")
bt_rows = []
for m_name, info in experiment["models"].items():
    b_t = info["baseline_test"]
    t_t = info["test"]
    bt_rows.append(
        {
            "Mô hình": m_name,
            "Số cấu hình GridSearchCV": info["search_candidates"],
            "Baseline F1": b_t["f1"],
            "Tuned F1": t_t["f1"],
            "Baseline Recall": b_t["recall"],
            "Tuned Recall": t_t["recall"],
            "Baseline ROC-AUC": b_t["roc_auc"],
            "Tuned ROC-AUC": t_t["roc_auc"],
            "Tham số tốt nhất": str(info["best_params"]),
        }
    )
st.dataframe(
    pd.DataFrame(bt_rows).style.format(
        {
            "Baseline F1": "{:.3f}",
            "Tuned F1": "{:.3f}",
            "Baseline Recall": "{:.3f}",
            "Tuned Recall": "{:.3f}",
            "Baseline ROC-AUC": "{:.3f}",
            "Tuned ROC-AUC": "{:.3f}",
        }
    ),
    hide_index=True,
    width="stretch",
)

# Cross-Validation table
st.subheader("4. Stratified 5-Fold Cross-Validation trên tập Train (211 bản ghi)")
st.dataframe(cv, hide_index=True, width="stretch")
st.caption(experiment.get("cv_caveat", ""))

with st.expander("🖼️ Xem các biểu đồ PNG gốc đã sao chép từ thực nghiệm"):
    figures = list_available_figures()
    if figures:
        chosen = st.selectbox("Chọn biểu đồ", [f.name for f in figures])
        st.image(str(get_figure_path(chosen)), width=720)

with st.expander("📋 Chi tiết kiểm tra dữ liệu (data_audit.json)"):
    st.json(audit, expanded=False)
