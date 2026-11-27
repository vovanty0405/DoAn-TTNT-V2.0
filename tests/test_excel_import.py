"""Tests for Excel patient import (template, validation, duplicate check, and confirmed SQLite commit)."""

from pathlib import Path

from database.db import init_db, register_current_models
from database.patient_repository import create_patient, query_patients
from database.prediction_repository import query_predictions
from services.excel_export_service import export_patients_to_excel, export_predictions_to_excel
from services.excel_import_service import (
    build_patient_import_template,
    build_prediction_import_template,
    commit_batch_predictions,
    commit_imported_patients,
    parse_patients_excel,
    parse_predictions_excel,
    run_batch_excel_predictions,
)
from src.prediction import load_model


def test_template_parse_and_confirmed_commit(tmp_path: Path):
    db = tmp_path / "import_test.db"
    init_db(db)

    # 1. Parse template without committing yet -> DB must stay empty
    template_bytes = build_patient_import_template()
    valid, errors = parse_patients_excel(template_bytes, db_path=db)
    assert len(valid) == 2
    assert len(errors) == 0
    _, total_before = query_patients(db_path=db)
    assert total_before == 0

    # 2. Confirm import -> 2 rows written
    inserted = commit_imported_patients(valid, db_path=db)
    assert inserted == 2
    rows, total_after = query_patients(db_path=db)
    assert total_after == 2
    codes = {r["patient_code"] for r in rows}
    assert "BN-IMPORT-01" in codes

    # 3. Re-parsing same template must flag BN-IMPORT-01 as duplicate in DB
    valid2, errors2 = parse_patients_excel(template_bytes, db_path=db)
    assert len(valid2) == 1  # The blank-code row is still valid
    assert len(errors2) == 1
    assert "đã tồn tại" in errors2[0]["reason"]


def test_reimport_exported_excel_headers_on_row_4(tmp_path: Path):
    db1 = tmp_path / "src.db"
    db2 = tmp_path / "dst.db"
    init_db(db1)
    init_db(db2)

    create_patient("Lê Văn Export", "BN-EXP-01", 1966, "Nam", "0909998888", "TP.HCM", db_path=db1)
    pts, _ = query_patients(db_path=db1)
    exported_bytes = export_patients_to_excel(pts)

    # Parse into fresh db2 (auto-detects header on row 4!)
    valid, errors = parse_patients_excel(exported_bytes, db_path=db2)
    assert len(valid) == 1 and len(errors) == 0
    assert valid[0]["patient_code"] == "BN-EXP-01"
    assert valid[0]["full_name"] == "Lê Văn Export"
    assert commit_imported_patients(valid, db_path=db2) == 1


def test_batch_prediction_excel_preview_export_and_commit(tmp_path: Path):
    db = tmp_path / "batch_pred.db"
    init_db(db)
    register_current_models(db)

    tpl = build_prediction_import_template()
    valid, errors = parse_predictions_excel(tpl, db_path=db)
    assert len(valid) == 2 and len(errors) == 0

    model = load_model("Random Forest")
    preview = run_batch_excel_predictions(model, "Random Forest", valid, db_path=db)
    assert len(preview) == 2
    assert preview[0]["predicted_label"] == 1
    assert preview[1]["predicted_label"] == 0

    # Before confirmation, SQLite must still have 0 predictions
    _, total_before = query_predictions(db_path=db)
    assert total_before == 0

    # Export preview to Excel works
    xlsx_out = export_predictions_to_excel(preview)
    assert len(xlsx_out) > 1000

    # After confirmation, 2 predictions saved to SQLite
    saved = commit_batch_predictions(preview, db_path=db)
    assert saved == 2
    _, total_after = query_predictions(db_path=db)
    assert total_after == 2
