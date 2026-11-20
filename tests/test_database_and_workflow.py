"""Tests for SQLite schema, patient CRUD, soft delete, pagination, prediction persistence, and PDF export."""

import json
from pathlib import Path

from database.db import init_db, register_current_models
from database.patient_repository import (
    create_patient,
    get_patient_by_id,
    get_patient_stats,
    query_patients,
    restore_patient,
    soft_delete_patient,
    update_patient,
)
from database.prediction_repository import (
    get_dashboard_prediction_stats,
    get_prediction_by_id,
    get_predictions_for_patient,
    query_predictions,
)
from services.pdf_export_service import build_patient_pdf
from services.prediction_service import run_prediction
from src.data_loader import load_demo_samples
from src.prediction import load_model


def test_patient_crud_soft_delete_and_pagination(tmp_path: Path):
    db = tmp_path / "test_heartcare.db"
    init_db(db)

    # Create 15 patients
    created_ids = []
    for i in range(1, 16):
        pid = create_patient(
            full_name=f"Bệnh nhân {i:02d}",
            patient_code=f"BN-{i:04d}",
            birth_year=1960 + i,
            gender="Nam" if i % 2 == 1 else "Nữ",
            phone=f"0901234{i:03d}",
            address="TP.HCM",
            notes="Ghi chú kiểm thử",
            db_path=db,
        )
        created_ids.append(pid)

    # Pagination: page 1 & page 2 with page_size=10
    page1, total = query_patients(page=1, page_size=10, db_path=db)
    page2, total2 = query_patients(page=2, page_size=10, db_path=db)
    assert total == 15 and total2 == 15
    assert len(page1) == 10
    assert len(page2) == 5

    # Search & gender filter
    found, cnt = query_patients(search="BN-0005", db_path=db)
    assert cnt == 1 and found[0]["full_name"] == "Bệnh nhân 05"

    female_rows, female_cnt = query_patients(gender="Nữ", page_size=50, db_path=db)
    assert female_cnt == 7 and all(r["gender"] == "Nữ" for r in female_rows)

    # Update
    target_id = created_ids[0]
    assert update_patient(
        target_id,
        full_name="Bệnh nhân 01 (Đã sửa)",
        patient_code="BN-0001",
        birth_year=1961,
        gender="Nam",
        phone="0999888777",
        address="Hà Nội",
        notes="Cập nhật",
        db_path=db,
    )
    updated = get_patient_by_id(target_id, db_path=db)
    assert updated["full_name"] == "Bệnh nhân 01 (Đã sửa)"
    assert updated["address"] == "Hà Nội"

    # Soft delete & restore
    assert soft_delete_patient(target_id, db_path=db)
    assert get_patient_by_id(target_id, db_path=db)["is_active"] == 0
    stats = get_patient_stats(db_path=db)
    assert stats["total_patients"] == 15
    assert stats["active_patients"] == 14

    assert restore_patient(target_id, db_path=db)
    assert get_patient_by_id(target_id, db_path=db)["is_active"] == 1


def test_end_to_end_patient_prediction_history_and_pdf(tmp_path: Path):
    db = tmp_path / "test_e2e.db"
    init_db(db)
    register_current_models(db)

    pid = create_patient(
        full_name="Nguyễn Văn Kiểm Thử",
        patient_code="BN-TEST-01",
        birth_year=1963,
        gender="Nam",
        phone="0912345678",
        address="Quận 1, TP.HCM",
        db_path=db,
    )

    sample = load_demo_samples()[0]
    model = load_model("Random Forest")
    res = run_prediction(
        model=model,
        model_name="Random Forest",
        features=sample["features"],
        patient_id=pid,
        persist=True,
        db_path=db,
    )

    assert res["record_id"] is not None
    assert res["predicted_label"] == sample["prediction"]
    assert res["model_version"].startswith("v1.0-")

    # Verify history & immutability
    saved = get_prediction_by_id(res["record_id"], db_path=db)
    assert saved is not None
    assert saved["patient_id"] == pid
    assert saved["model_name"] == "Random Forest"
    saved_features = json.loads(saved["input_features"])
    assert saved_features["age"] == sample["features"]["age"]

    # Query predictions & dashboard stats
    rows, total = query_predictions(search="BN-TEST-01", model_name="Random Forest", db_path=db)
    assert total == 1 and len(rows) == 1
    p_preds = get_predictions_for_patient(pid, db_path=db)
    assert len(p_preds) == 1

    dash = get_dashboard_prediction_stats(db_path=db)
    assert dash["total_predictions"] == 1
    assert dash["positive_predictions"] == 1

    # PDF export smoke test
    patient = get_patient_by_id(pid, db_path=db)
    pdf_bytes = build_patient_pdf(patient, saved)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000

