"""Reusable UI components for HeartCare AI Healthcare Dashboard."""

from datetime import datetime
import html
from typing import Any, Dict, List, Optional

import streamlit as st

from src.config import CODES


def compute_age(birth_year: Optional[int]) -> Optional[int]:
    """Compute approximate age from birth year."""
    if birth_year is None:
        return None
    return max(0, datetime.now().year - int(birth_year))


def format_feature_value(col: str, val: Any) -> str:
    """Format a single clinical feature value with its Vietnamese code label if applicable."""
    if val is None:
        return "Thiếu / không xác định"
    if col in CODES:
        try:
            code = int(val)
            return f"{code} – {CODES[col].get(code, val)}"
        except (TypeError, ValueError):
            return str(val)
    return str(val)


def render_page_header(title: str, subtitle: str = "") -> None:
    """Render a consistent Healthcare Dashboard page header."""
    st.markdown(
        f"""
        <div class="hc-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: Any,
    icon: str = "📊",
    accent: str = "#1D6FA5",
    subtitle: str = "",
) -> None:
    """Render a rounded KPI card with colored top/left accent."""
    sub_html = f'<div class="hc-kpi-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="hc-kpi-card" style="border-left: 4px solid {accent};">
            <div class="hc-kpi-top">
                <span class="hc-kpi-label">{label}</span>
                <span class="hc-kpi-icon">{icon}</span>
            </div>
            <div class="hc-kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_health_card(
    icon: str,
    label: str,
    value: Any,
    accent: str = "#1D6FA5",
) -> None:
    """Render an informative clinical parameter card (neutral, non-diagnostic)."""
    st.markdown(
        f"""
        <div class="hc-health-card" style="border-top: 3px solid {accent};">
            <div class="hc-health-label">{icon} {label}</div>
            <div class="hc-health-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer_banner() -> None:
    """Render the mandatory research/academic disclaimer."""
    st.info(
        "Ứng dụng phục vụ mục đích học tập và nghiên cứu, không thay thế việc thăm khám "
        "hoặc chẩn đoán của bác sĩ. Điểm mô hình chưa được hiệu chỉnh thành xác suất mắc bệnh thực tế."
    )


def render_patient_profile_card(patient: Dict[str, Any]) -> None:
    """Render a compact, modern Healthcare profile card for a patient."""
    age = compute_age(patient.get("birth_year"))
    age_str = f"{age} tuổi (sinh năm {patient['birth_year']})" if age is not None else "Chưa cập nhật"
    is_active = bool(patient.get("is_active"))
    badge_cls = "hc-badge-green" if is_active else "hc-badge-gray"
    badge_txt = "Đang hoạt động" if is_active else "Đã lưu trữ"
    name = html.escape(str(patient.get("full_name") or ""))
    code = html.escape(str(patient.get("patient_code") or ""))
    gender = html.escape(str(patient.get("gender") or "—"))
    phone = html.escape(str(patient.get("phone") or "—"))
    address = html.escape(str(patient.get("address") or "—"))
    created = html.escape(str(patient.get("created_at") or "—")[:16])
    notes = html.escape(str(patient.get("notes") or "Không có ghi chú"))

    st.markdown(
        f"""
        <div class="hc-profile-card">
            <div class="hc-profile-top">
                <div class="hc-profile-identity">
                    <div class="hc-avatar">👤</div>
                    <div>
                        <div class="hc-profile-name">{name}</div>
                        <div class="hc-profile-code">Mã hồ sơ: <strong>{code}</strong></div>
                    </div>
                </div>
                <span class="hc-badge {badge_cls}">{badge_txt}</span>
            </div>
            <div class="hc-profile-grid">
                <div>
                    <div class="hc-profile-item-label">Tuổi & Giới tính</div>
                    <div class="hc-profile-item-val">{age_str} · {gender}</div>
                </div>
                <div>
                    <div class="hc-profile-item-label">Điện thoại</div>
                    <div class="hc-profile-item-val">{phone}</div>
                </div>
                <div>
                    <div class="hc-profile-item-label">Địa chỉ</div>
                    <div class="hc-profile-item-val">{address}</div>
                </div>
                <div>
                    <div class="hc-profile-item-label">Ngày tạo hồ sơ</div>
                    <div class="hc-profile-item-val">{created}</div>
                </div>
            </div>
            <div style="margin-top:0.75rem; padding-top:0.6rem; border-top:1px dashed #E2ECF4; font-size:0.84rem; color:#5A7184;">
                <strong>Ghi chú:</strong> {notes}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_cell_badge(val: Any) -> str:
    txt = str(val)
    escaped = html.escape(txt)
    if txt == "Có bệnh":
        return f'<span class="hc-badge hc-badge-red">{escaped}</span>'
    if txt == "Không có bệnh":
        return f'<span class="hc-badge hc-badge-green">{escaped}</span>'
    if txt == "Đang hoạt động":
        return f'<span class="hc-badge hc-badge-green">{escaped}</span>'
    if txt == "Đã lưu trữ":
        return f'<span class="hc-badge hc-badge-gray">{escaped}</span>'
    return escaped


def render_styled_table(
    headers: List[str],
    rows: List[List[Any]],
    badge_col_indices: Optional[set[int]] = None,
) -> None:
    """Render a clean, styled HTML table with navy header, zebra rows, and pill badges."""
    badge_cols = badge_col_indices or set()
    th_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        tds = []
        for idx, val in enumerate(row):
            cell_html = _format_cell_badge(val) if idx in badge_cols else html.escape(str(val))
            tds.append(f"<td>{cell_html}</td>")
        body_rows.append(f"<tr>{''.join(tds)}</tr>")

    st.markdown(
        f"""
        <div class="hc-table-wrap">
            <table class="hc-table">
                <thead><tr>{th_html}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

