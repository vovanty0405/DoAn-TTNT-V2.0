"""Comprehensive tests for Excel/PDF exports, edge cases, empty DB, validation, and idempotency."""

import io
from pathlib import Path

import pytest
from openpyxl import load_workbook

from database.db import get_connection, init_db, register_current_models
from database.patient_repository import (
    create_patient,
    get_patient_by_id,
    get_patient_stats,
    query_patients,
)
from database.prediction_repository import (
    get_dashboard_prediction_stats,
    get_prediction_by_id,
    get_predictions_for_patient,
    query_predictions,
)
from services.excel_export_service import (
    export_patients_to_excel,
    export_predictions_to_excel,
)
from services.pdf_export_service import build_model_comparison_pdf, build_patient_pdf
from services.prediction_service import run_prediction
from src.data_loader import load_demo_samples
from src.data_validation import validate_input_dict
from src.evaluate_models import get_figure_path, load_cross_validation, load_metrics
from src.prediction import load_experiment_metadata, load_model


def test_empty_database_and_exports(tmp_path: Path):
    db = tmp_path / "empty.db"
    init_db(db)

    assert get_patient_stats(db)["total_patients"] == 0
    assert get_dashboard_prediction_stats(db)["total_predictions"] == 0
    rows, total = query_patients(db_path=db)
    assert rows == [] and total == 0

    # Excel export on empty lists must still produce valid readable workbooks
    wb_p = load_workbook(io.BytesIO(export_patients_to_excel([])))
    assert "Danh sách bệnh nhân" in wb_p.sheetnames

    wb_h = load_workbook(io.BytesIO(export_predictions_to_excel([])))
    assert wb_h.sheetnames == ["Lịch sử dự đoán", "Chi tiết 13 chỉ số"]


def test_input_validation_missing_and_invalid_values():
    sample = dict(load_demo_samples()[0]["features"])

    # Missing ca/thal (None) is allowed because pipeline imputes them
    sample_missing_ca = dict(sample, ca=None, thal=None)
    ok, errs = validate_input_dict(sample_missing_ca)
    assert ok and errs == []

    # Invalid numeric range & invalid categorical code must be rejected
    bad = dict(sample, age=-10, cp=99)
    ok_bad, errs_bad = validate_input_dict(bad)
    assert not ok_bad and len(errs_bad) >= 2

    # Missing required key must be rejected
    incomplete = {k: v for k, v in sample.items() if k != "age"}
    ok_inc, errs_inc = validate_input_dict(incomplete)
    assert not ok_inc and "age" in errs_inc[0]


def test_idempotent_init_and_model_registration(tmp_path: Path):
    db = tmp_path / "rerun.db"
    for _ in range(3):
        init_db(db)
        register_current_models(db)

    with get_connection(db) as conn:
        cnt = conn.execute("SELECT COUNT(*) AS c FROM model_versions").fetchone()["c"]
    assert cnt == 3


def test_excel_and_pdf_exports_with_wrong_patient_guard(tmp_path: Path):
    db = tmp_path / "exports.db"
    init_db(db)
    register_current_models(db)

    p1 = create_patient("Nguyễn Văn A", "BN-001", 1965, "Nam", "0901111111", "TP.HCM", db_path=db)
    p2 = create_patient("Trần Thị B", "BN-002", 1972, "Nữ", "0902222222", "Hà Nội", db_path=db)

    sample = load_demo_samples()[0]
    model = load_model("Random Forest")
    res = run_prediction(model, "Random Forest", sample["features"], patient_id=p1, db_path=db)
    pred_rec = get_prediction_by_id(res["record_id"], db_path=db)

    patient1 = get_patient_by_id(p1, db_path=db)
    patient2 = get_patient_by_id(p2, db_path=db)

    # Guard: exporting p1's prediction under p2's profile must raise ValueError
    with pytest.raises(ValueError):
        build_patient_pdf(patient2, pred_rec)

    # Valid patient PDF
    pdf1 = build_patient_pdf(patient1, pred_rec)
    assert pdf1.startswith(b"%PDF") and len(pdf1) > 1500

    # Excel exports: patients & predictions
    all_pts, _ = query_patients(db_path=db)
    wb_pts = load_workbook(io.BytesIO(export_patients_to_excel(all_pts)))
    assert wb_pts.active.max_row == 4 + 2  # 4 header/meta rows + 2 patients

    p1_preds = get_predictions_for_patient(p1, db_path=db)
    wb_hist = load_workbook(io.BytesIO(export_predictions_to_excel(p1_preds, patient=patient1)))
    assert wb_hist["Lịch sử dự đoán"].max_row == 5
    assert wb_hist["Chi tiết 13 chỉ số"].max_row == 5

    # Model comparison PDF
    exp = load_experiment_metadata()
    metrics = load_metrics()
    cv = load_cross_validation()
    figs = [get_figure_path("performance.png"), get_figure_path("importance.png")]
    pdf_cmp = build_model_comparison_pdf(exp, metrics, cv, figure_paths=figs)
    assert pdf_cmp.startswith(b"%PDF") and len(pdf_cmp) > 5000
