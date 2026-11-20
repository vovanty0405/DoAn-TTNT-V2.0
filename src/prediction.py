"""Model loading and inference module for HeartCare-AI.

Loads the persisted scikit-learn pipelines copied from the original research project
without retraining or modifying the .joblib files.
"""

import sys
import types
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import CODES, FEATURES, LABELS, MODEL_FILES, MODEL_RESULTS_DIR, MODELS_DIR
from src.preprocessing import SpecialMissing, normalize_input
from src.utils import read_json


def ensure_legacy_pickle_compat() -> None:
    """Register a lightweight shim for `heart_pipeline` so joblib can unpickle
    `heart_pipeline.SpecialMissing` without importing the legacy project folder.
    """
    if "heart_pipeline" not in sys.modules:
        shim = types.ModuleType("heart_pipeline")
        shim.SpecialMissing = SpecialMissing
        shim.normalize_input = normalize_input
        shim.FEATURES = FEATURES
        shim.CODES = CODES
        shim.LABELS = LABELS
        sys.modules["heart_pipeline"] = shim


def load_experiment_metadata() -> Dict[str, Any]:
    """Load experiment.json metadata from reports/model_results/."""
    return read_json(MODEL_RESULTS_DIR / "experiment.json")


def get_default_model_name() -> str:
    """Return the selected default model name recorded in experiment.json."""
    return load_experiment_metadata()["selected_model"]


def resolve_model_path(model_name: str) -> Path:
    """Resolve the .joblib path for a given model display name."""
    filename = MODEL_FILES.get(model_name, f"{model_name.replace(' ', '_')}.joblib")
    return MODELS_DIR / filename


def load_model(model_name: str):
    """Load a single persisted model pipeline by name."""
    ensure_legacy_pickle_compat()
    path = resolve_model_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp mô hình: {path}")
    return joblib.load(path)


def load_all_models() -> Dict[str, Any]:
    """Load all models defined in experiment.json."""
    ensure_legacy_pickle_compat()
    experiment = load_experiment_metadata()
    return {name: load_model(name) for name in experiment["models"]}


def disease_score(model, frame: pd.DataFrame) -> np.ndarray:
    """Return class-1 probability score using the model's learned class order."""
    return model.predict_proba(normalize_input(frame))[
        :, list(model.classes_).index(1)
    ]


def predict_single(model, features: Dict[str, Any]) -> Tuple[int, float, str]:
    """Run inference on a single patient dictionary.

    Returns:
        (predicted_label, disease_score_value, human_readable_label)
    """
    frame = normalize_input(pd.DataFrame([features], columns=FEATURES))
    pred = int(model.predict(frame)[0])
    score = float(disease_score(model, frame)[0])
    return pred, score, LABELS[pred]


def predict_batch(model, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataFrame of patient records."""
    normalized = normalize_input(frame)
    preds = model.predict(normalized).astype(int)
    scores = disease_score(model, normalized)
    return preds, scores

