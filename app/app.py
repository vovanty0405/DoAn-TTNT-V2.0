"""HeartCare AI Streamlit entrypoint using st.Page and st.navigation."""

import sys
from pathlib import Path

# Ensure project root and app directory are in sys.path (PROJECT_ROOT first, APP_DIR second)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

for p in (APP_DIR, PROJECT_ROOT):
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)
    sys.path.insert(0, p_str)

import streamlit as st

from components.state import artifacts_ready, get_experiment_and_models
from database.db import init_db, register_current_models

st.set_page_config(
    page_title="HeartCare AI – Quản lý bệnh nhân & Hỗ trợ phân loại bệnh tim",
    page_icon="🫀",
    layout="wide",
)

# Load custom Healthcare Dashboard CSS
css_path = APP_DIR / "styles.css"
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# Initialize SQLite schema & register current model versions
init_db()
register_current_models()

if not artifacts_ready():
    st.error(
        "Chưa tìm thấy reports/model_results/experiment.json. Vui lòng kiểm tra lại artifact."
    )
    st.stop()

experiment, _ = get_experiment_and_models()

with st.sidebar:
    st.markdown(
        """
        <div class="hc-brand">
            <div class="hc-brand-title">🫀 HeartCare AI</div>
            <div class="hc-brand-sub">Healthcare Management & ML Classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Mô hình mặc định: **{experiment['selected_model']}**")

pages = [
    st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True),
    st.Page("pages/patients.py", title="Quản lý bệnh nhân", icon="👥"),
    st.Page("pages/prediction.py", title="Dự đoán bệnh tim", icon="🩺"),
    st.Page("pages/history.py", title="Lịch sử dự đoán", icon="🕒"),
    st.Page("pages/evaluation.py", title="Đánh giá mô hình", icon="📈"),
    st.Page("pages/settings.py", title="Cài đặt", icon="⚙️"),
]

nav = st.navigation(pages, position="sidebar")
nav.run()
