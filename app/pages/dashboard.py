"""Healthcare Management Dashboard — 100% backed by SQLite."""

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from app.components.ui import (
        render_disclaimer_banner,
        render_kpi_card,
        render_page_header,
        render_styled_table,
    )
except ModuleNotFoundError:
    from components.ui import (
        render_disclaimer_banner,
        render_kpi_card,
        render_page_header,
        render_styled_table,
    )
from database.patient_repository import get_patient_stats
from database.prediction_repository import get_dashboard_prediction_stats

render_page_header(
    "📊 Dashboard Tổng Quan",
    "Thống kê hồ sơ bệnh nhân và hoạt động phân loại bệnh tim trực tiếp từ cơ sở dữ liệu SQLite.",
)
render_disclaimer_banner()

p_stats = get_patient_stats()
pred_stats = get_dashboard_prediction_stats()

# Row 1: 6 KPI Cards (2 rows of 3 for clean readability)
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    render_kpi_card(
        "Tổng số bệnh nhân",
        p_stats["total_patients"],
        icon="👥",
        accent="#0B3C5D",
        subtitle="Tất cả hồ sơ trong SQLite",
    )
with r1c2:
    render_kpi_card(
        "Hồ sơ đang hoạt động",
        p_stats["active_patients"],
        icon="✅",
        accent="#0E9F9F",
        subtitle="Chưa bị xóa mềm",
    )
with r1c3:
    render_kpi_card(
        "Tổng số lượt dự đoán",
        pred_stats["total_predictions"],
        icon="🩺",
        accent="#1D6FA5",
        subtitle="Lịch sử đã lưu trữ",
    )

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    render_kpi_card(
        "Phân loại: Có bệnh (1)",
        pred_stats["positive_predictions"],
        icon="❤️‍🩹",
        accent="#D9534F",
        subtitle="Theo nhãn chuẩn hóa của mô hình",
    )
with r2c2:
    render_kpi_card(
        "Phân loại: Không bệnh (0)",
        pred_stats["negative_predictions"],
        icon="💚",
        accent="#2E9E6B",
        subtitle="Theo nhãn chuẩn hóa của mô hình",
    )
with r2c3:
    render_kpi_card(
        "Lượt dự đoán trong tháng",
        pred_stats["this_month_predictions"],
        icon="📅",
        accent="#3B82F6",
        subtitle=f"Tháng {datetime.now().strftime('%m/%Y')}",
    )

st.divider()

# Row 2: Donut (Label distribution) & Bar (Predictions by month)
ch1, ch2 = st.columns(2)
with ch1:
    st.subheader("Phân bố nhãn dự đoán")
    if pred_stats["total_predictions"] == 0:
        st.info("Chưa có dữ liệu dự đoán trong SQLite.")
    else:
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Có bệnh (1)", "Không có bệnh (0)"],
                    values=[
                        pred_stats["positive_predictions"],
                        pred_stats["negative_predictions"],
                    ],
                    hole=0.55,
                    marker_colors=["#D9534F", "#0E9F9F"],
                    textinfo="label+percent",
                )
            ]
        )
        fig_donut.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True, key="dash_donut_labels")

with ch2:
    st.subheader("Số lượt dự đoán theo tháng")
    if not pred_stats["by_month"]:
        st.info("Chưa có dữ liệu dự đoán theo tháng.")
    else:
        df_month = pd.DataFrame(pred_stats["by_month"])
        fig_month = px.bar(
            df_month,
            x="month",
            y="count",
            text_auto=True,
            labels={"month": "Tháng", "count": "Số lượt dự đoán"},
            color_discrete_sequence=["#1D6FA5"],
        )
        fig_month.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_month, use_container_width=True, key="dash_bar_month")

# Row 3: Age distribution & Predictions by model
ch3, ch4 = st.columns(2)
with ch3:
    st.subheader("Phân bố độ tuổi bệnh nhân")
    if not p_stats["birth_years"]:
        st.info("Chưa có dữ liệu năm sinh bệnh nhân đang hoạt động.")
    else:
        curr_year = datetime.now().year
        ages = [max(0, curr_year - y) for y in p_stats["birth_years"]]
        fig_age = px.histogram(
            x=ages,
            nbins=10,
            labels={"x": "Tuổi", "y": "Số bệnh nhân"},
            color_discrete_sequence=["#0E9F9F"],
        )
        fig_age.update_layout(
            yaxis_title="Số bệnh nhân",
            margin=dict(t=20, b=20, l=10, r=10),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_age, use_container_width=True, key="dash_hist_age")

with ch4:
    st.subheader("Số lượt dự đoán theo mô hình")
    if not pred_stats["by_model"]:
        st.info("Chưa có thống kê theo mô hình.")
    else:
        df_model = pd.DataFrame(pred_stats["by_model"])
        fig_model = px.bar(
            df_model,
            x="model_name",
            y="count",
            text_auto=True,
            labels={"model_name": "Mô hình", "count": "Số lượt"},
            color="model_name",
            color_discrete_sequence=["#0B3C5D", "#1D6FA5", "#0E9F9F"],
        )
        fig_model.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=10, r=10),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_model, use_container_width=True, key="dash_bar_model")

# Recent activity
st.subheader("Hoạt động dự đoán gần đây")
recent = pred_stats["recent_activity"]
if not recent:
    st.info(
        "Chưa có hoạt động gần đây. Bạn có thể tạo bệnh nhân mới hoặc chạy `py -3.13 scripts/seed_demo_data.py` để sinh dữ liệu DEMO kiểm thử."
    )
else:
    render_styled_table(
        ["Mã", "Thời gian", "Mã BN", "Họ tên", "Mô hình", "Kết quả phân loại", "Điểm mô hình"],
        [
            [
                f"#{r['id']}",
                str(r["predicted_at"])[:16],
                r["patient_code"] or "—",
                r["patient_name"] or "Không liên kết",
                r["model_name"],
                r["label_text"],
                f"{float(r['model_score']):.1%}",
            ]
            for r in recent
        ],
        badge_col_indices={5},
    )
