"""Settings and environment diagnostics page."""

import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

try:
    from app.components.state import get_experiment_and_models
    from app.components.ui import render_page_header
except ModuleNotFoundError:
    from components.state import get_experiment_and_models
    from components.ui import render_page_header
from src.config import DB_PATH, FIGURES_DIR, MODEL_RESULTS_DIR, MODELS_DIR

render_page_header(
    "⚙️ Cài đặt & Thông tin hệ thống",
    "Thông tin phiên bản mô hình, đối chiếu môi trường Python/Scikit-learn và giới hạn bản demo local.",
)

experiment, models = get_experiment_and_models()

st.subheader("Thông tin mô hình & thực nghiệm")
st.write(f"- **Mô hình mặc định (experiment.json):** `{experiment['selected_model']}`")
st.write(f"- **Quy tắc chọn mô hình:** {experiment.get('selection_rule', 'N/A')}")
st.write(f"- **Các mô hình đã nạp:** {', '.join(models.keys())}")

st.subheader("So sánh phiên bản môi trường")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Môi trường huấn luyện gốc (`experiment.json`)**")
    st.json(experiment.get("versions", {}))
with c2:
    st.markdown("**Môi trường hiện tại**")
    st.json(
        {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "streamlit": st.__version__,
        }
    )

st.subheader("Đường dẫn tài nguyên")
st.code(
    f"MODELS_DIR        = {MODELS_DIR}\n"
    f"MODEL_RESULTS_DIR = {MODEL_RESULTS_DIR}\n"
    f"FIGURES_DIR       = {FIGURES_DIR}\n"
    f"DB_PATH           = {DB_PATH}",
    language="text",
)

