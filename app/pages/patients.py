"""Patient Management page — full CRUD, search, filter, sort, soft delete, and SQLite pagination."""

import math
from datetime import date, datetime

import streamlit as st

try:
    from app.components.ui import compute_age, render_page_header
except ModuleNotFoundError:
    from components.ui import compute_age, render_page_header
from database.patient_repository import (
    create_patient,
    generate_patient_code,
    get_patient_by_id,
    query_patients,
    restore_patient,
    soft_delete_patient,
    update_patient,
)
from services.excel_export_service import export_patients_to_excel

# If a patient was selected via "Xem chi tiết", render the Detail view inline
if st.session_state.get("selected_patient_id"):
    try:
        from app.pages.patient_detail import render_patient_detail_view
    except ModuleNotFoundError:
        from pages.patient_detail import render_patient_detail_view
    render_patient_detail_view(int(st.session_state["selected_patient_id"]))
    st.stop()

render_page_header(
    "👥 Quản lý bệnh nhân",
    "Thêm, sửa, xem chi tiết, xóa mềm có xác nhận, khôi phục hồ sơ và phân trang trực tiếp trên SQLite.",
)

# --- Add Patient Form ---
with st.expander("➕ Thêm hồ sơ bệnh nhân mới", expanded=False):
    with st.form("create_patient_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            new_code = st.text_input("Mã bệnh nhân (để trống để tự sinh)", placeholder=generate_patient_code())
            new_name = st.text_input("Họ và tên *")
        with c2:
            new_birth_year = st.number_input(
                "Năm sinh",
                min_value=1900,
                max_value=datetime.now().year,
                value=1970,
                step=1,
            )
            new_gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])
        with c3:
            new_phone = st.text_input("Số điện thoại")
            new_address = st.text_input("Địa chỉ")
        new_notes = st.text_area("Ghi chú", height=70)
        if st.form_submit_button("Lưu hồ sơ mới", type="primary"):
            if not new_name.strip():
                st.error("Vui lòng nhập họ và tên bệnh nhân.")
            else:
                try:
                    create_patient(
                        full_name=new_name,
                        patient_code=new_code,
                        birth_year=new_birth_year,
                        gender=new_gender,
                        phone=new_phone,
                        address=new_address,
                        notes=new_notes,
                    )
                    st.success(f"Đã thêm hồ sơ bệnh nhân: {new_name.strip()}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Không thể thêm bệnh nhân: {exc}")

# --- Edit Patient Panel (if triggered) ---
edit_id = st.session_state.get("edit_patient_id")
if edit_id:
    edit_rec = get_patient_by_id(int(edit_id))
    if edit_rec:
        st.subheader(f"✏️ Chỉnh sửa hồ sơ: {edit_rec['patient_code']} – {edit_rec['full_name']}")
        with st.form("edit_patient_form"):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                e_code = st.text_input("Mã bệnh nhân *", value=edit_rec["patient_code"])
                e_name = st.text_input("Họ và tên *", value=edit_rec["full_name"])
            with ec2:
                e_birth = st.number_input(
                    "Năm sinh",
                    min_value=1900,
                    max_value=datetime.now().year,
                    value=int(edit_rec["birth_year"] or 1970),
                    step=1,
                )
                g_opts = ["Nam", "Nữ", "Khác"]
                e_gender = st.selectbox(
                    "Giới tính",
                    g_opts,
                    index=g_opts.index(edit_rec["gender"]) if edit_rec["gender"] in g_opts else 2,
                )
            with ec3:
                e_phone = st.text_input("Số điện thoại", value=edit_rec["phone"] or "")
                e_address = st.text_input("Địa chỉ", value=edit_rec["address"] or "")
            e_notes = st.text_area("Ghi chú", value=edit_rec["notes"] or "", height=70)
            b1, b2 = st.columns([1, 5])
            save_edit = b1.form_submit_button("Lưu thay đổi", type="primary")
            cancel_edit = b2.form_submit_button("Hủy")
            if save_edit:
                if not e_name.strip() or not e_code.strip():
                    st.error("Mã bệnh nhân và họ tên không được để trống.")
                else:
                    try:
                        update_patient(
                            int(edit_id),
                            full_name=e_name,
                            patient_code=e_code,
                            birth_year=e_birth,
                            gender=e_gender,
                            phone=e_phone,
                            address=e_address,
                            notes=e_notes,
                        )
                        st.session_state.pop("edit_patient_id", None)
                        st.success("Đã cập nhật hồ sơ bệnh nhân.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Lỗi cập nhật: {exc}")
            elif cancel_edit:
                st.session_state.pop("edit_patient_id", None)
                st.rerun()

# --- Soft Delete Confirmation ---
del_id = st.session_state.get("confirm_delete_patient_id")
if del_id:
    del_rec = get_patient_by_id(int(del_id))
    if del_rec:
        st.warning(
            f"Xác nhận xóa mềm hồ sơ **{del_rec['patient_code']} – {del_rec['full_name']}**? "
            "(Hồ sơ sẽ chuyển sang trạng thái Đã lưu trữ và có thể khôi phục lại bất cứ lúc nào.)"
        )
        dc1, dc2, _ = st.columns([1.2, 1, 5])
        if dc1.button("Xác nhận xóa mềm", type="primary"):
            soft_delete_patient(int(del_id))
            st.session_state.pop("confirm_delete_patient_id", None)
            st.success("Đã xóa mềm hồ sơ.")
            st.rerun()
        if dc2.button("Hủy bỏ"):
            st.session_state.pop("confirm_delete_patient_id", None)
            st.rerun()

# --- Search, Filter, Sort & Pagination Controls ---
st.subheader("Danh sách bệnh nhân")
f1, f2, f3, f4 = st.columns([2.5, 1.2, 1.3, 1.5])
with f1:
    search = st.text_input("Tìm kiếm (mã, họ tên, số điện thoại)", placeholder="VD: DEMO-001, Nguyễn...")
with f2:
    gender_filter = st.selectbox("Giới tính", ["Tất cả", "Nam", "Nữ", "Khác"])
with f3:
    status_label = st.selectbox("Trạng thái", ["Đang hoạt động", "Tất cả", "Đã xóa mềm"])
    status_map = {"Đang hoạt động": "active", "Tất cả": "all", "Đã xóa mềm": "inactive"}
with f4:
    sort_choice = st.selectbox(
        "Sắp xếp",
        ["Mới nhất", "Cũ nhất", "Tên A → Z", "Tên Z → A", "Mã BN A → Z"],
    )
    sort_map = {
        "Mới nhất": ("created_at", "DESC"),
        "Cũ nhất": ("created_at", "ASC"),
        "Tên A → Z": ("full_name", "ASC"),
        "Tên Z → A": ("full_name", "DESC"),
        "Mã BN A → Z": ("patient_code", "ASC"),
    }

d1, d2, d3, d4 = st.columns([1.5, 1.5, 1.2, 1.8])
with d1:
    use_date_filter = st.checkbox("Lọc theo ngày tạo", value=False)
with d2:
    date_from = (
        st.date_input("Từ ngày", value=date(2025, 1, 1)).isoformat()
        if use_date_filter
        else None
    )
with d3:
    date_to = (
        st.date_input("Đến ngày", value=date.today()).isoformat()
        if use_date_filter
        else None
    )
with d4:
    page_size = st.selectbox("Số bản ghi / trang", [10, 20, 50], index=0)

sort_by, sort_dir = sort_map[sort_choice]
if "patients_page" not in st.session_state:
    st.session_state["patients_page"] = 1

rows, total_records = query_patients(
    search=search,
    gender=gender_filter,
    status=status_map[status_label],
    date_from=date_from,
    date_to=date_to,
    sort_by=sort_by,
    sort_dir=sort_dir,
    page=st.session_state["patients_page"],
    page_size=page_size,
)

total_pages = max(1, math.ceil(total_records / page_size))
if st.session_state["patients_page"] > total_pages:
    st.session_state["patients_page"] = total_pages
    rows, total_records = query_patients(
        search=search,
        gender=gender_filter,
        status=status_map[status_label],
        date_from=date_from,
        date_to=date_to,
        sort_by=sort_by,
        sort_dir=sort_dir,
        page=st.session_state["patients_page"],
        page_size=page_size,
    )

st.caption(
    f"Hiển thị **{len(rows)}** / **{total_records}** hồ sơ · Trang **{st.session_state['patients_page']} / {total_pages}**"
)

# Export Excel buttons (filtered vs all)
ex1, ex2, _ = st.columns([1.8, 1.8, 4])
filtered_all, _ = query_patients(
    search=search,
    gender=gender_filter,
    status=status_map[status_label],
    date_from=date_from,
    date_to=date_to,
    sort_by=sort_by,
    sort_dir=sort_dir,
    page=1,
    page_size=max(1, total_records),
)
ex1.download_button(
    "📥 Xuất Excel (theo bộ lọc)",
    data=export_patients_to_excel(
        filtered_all,
        title="Danh sách bệnh nhân (theo bộ lọc) — HeartCare AI",
        filter_summary=f"Trạng thái: {status_label}, Giới tính: {gender_filter}",
    ),
    file_name="heartcare_benh_nhan_loc.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
all_patients_export, _ = query_patients(status="all", page=1, page_size=10000)
ex2.download_button(
    "📥 Xuất Excel (toàn bộ)",
    data=export_patients_to_excel(
        all_patients_export,
        title="Toàn bộ danh sách bệnh nhân — HeartCare AI",
        filter_summary="Toàn bộ hồ sơ trong SQLite",
    ),
    file_name="heartcare_benh_nhan_tat_ca.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

if not rows:
    st.info("Không tìm thấy hồ sơ bệnh nhân nào phù hợp với bộ lọc hiện tại.")
else:
    # Header row
    hcols = st.columns([0.5, 1.2, 2.0, 0.7, 0.9, 1.3, 1.4, 1.2, 2.2])
    headers = ["STT", "Mã BN", "Họ tên", "Tuổi", "Giới tính", "Điện thoại", "Ngày tạo", "Trạng thái", "Thao tác"]
    for hc, title in zip(hcols, headers):
        hc.markdown(f"**{title}**")
    st.divider()

    start_idx = (st.session_state["patients_page"] - 1) * page_size
    for idx, p in enumerate(rows, start=start_idx + 1):
        cols = st.columns([0.5, 1.2, 2.0, 0.7, 0.9, 1.3, 1.4, 1.2, 2.2])
        age = compute_age(p["birth_year"])
        status_text = "🟢 Hoạt động" if p["is_active"] else "⚪ Đã lưu trữ"
        cols[0].write(idx)
        cols[1].write(p["patient_code"])
        cols[2].write(p["full_name"])
        cols[3].write(age if age is not None else "—")
        cols[4].write(p["gender"])
        cols[5].write(p["phone"] or "—")
        cols[6].write(str(p["created_at"])[:10])
        cols[7].write(status_text)

        act1, act2, act3 = cols[8].columns(3)
        if act1.button("👁️", key=f"view_{p['id']}", help="Xem chi tiết"):
            st.session_state["selected_patient_id"] = p["id"]
            st.rerun()
        if act2.button("✏️", key=f"edit_{p['id']}", help="Sửa hồ sơ"):
            st.session_state["edit_patient_id"] = p["id"]
            st.rerun()
        if p["is_active"]:
            if act3.button("🗑️", key=f"del_{p['id']}", help="Xóa mềm"):
                st.session_state["confirm_delete_patient_id"] = p["id"]
                st.rerun()
        else:
            if act3.button("♻️", key=f"res_{p['id']}", help="Khôi phục"):
                restore_patient(p["id"])
                st.success(f"Đã khôi phục hồ sơ {p['patient_code']}.")
                st.rerun()

# Pagination controls
p1, p2, p3 = st.columns([1, 2, 1])
with p1:
    if st.button("⬅️ Trang trước", disabled=st.session_state["patients_page"] <= 1):
        st.session_state["patients_page"] -= 1
        st.rerun()
with p2:
    st.markdown(
        f"<div style='text-align:center; padding-top:6px;'>Trang {st.session_state['patients_page']} / {total_pages}</div>",
        unsafe_allow_html=True,
    )
with p3:
    if st.button("Trang sau ➡️", disabled=st.session_state["patients_page"] >= total_pages):
        st.session_state["patients_page"] += 1
        st.rerun()
