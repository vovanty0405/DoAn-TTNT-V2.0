"""Prediction orchestration service combining validation, ML inference, and persistence."""

from pathlib import Path
from typing import Any, Dict, Optional

from database.db import get_model_version_tag
from database.prediction_repository import save_prediction
from src.data_validation import validate_input_dict
from src.prediction import predict_single


def run_prediction(
    model,
    model_name: str,
    features: Dict[str, Any],
    patient_id: Optional[int] = None,
    persist: bool = True,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate input, run ML prediction, and optionally save to SQLite."""
    is_valid, errors = validate_input_dict(features)
    if not is_valid:
        raise ValueError("; ".join(errors))

    pred, score, label_text = predict_single(model, features)
    model_version = get_model_version_tag(model_name, db_path=db_path)
    record_id = None
    if persist:
        record_id = save_prediction(
            model_name=model_name,
            model_version=model_version,
            predicted_label=pred,
            label_text=label_text,
            model_score=score,
            input_features=features,
            patient_id=patient_id,
            db_path=db_path,
        )

    return {
        "record_id": record_id,
        "model_name": model_name,
        "model_version": model_version,
        "predicted_label": pred,
        "label_text": label_text,
        "model_score": score,
    }
