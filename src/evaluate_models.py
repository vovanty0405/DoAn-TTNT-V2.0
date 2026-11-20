"""Evaluation artifact reader and metric computation helpers."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURES_DIR, MODEL_RESULTS_DIR
from src.prediction import disease_score
from src.utils import read_json


def load_metrics() -> pd.DataFrame:
    """Load test-set metrics from reports/model_results/metrics.csv."""
    return pd.read_csv(MODEL_RESULTS_DIR / "metrics.csv")


def load_cross_validation() -> pd.DataFrame:
    """Load 5-fold CV results from reports/model_results/cross_validation.csv."""
    return pd.read_csv(MODEL_RESULTS_DIR / "cross_validation.csv")


def load_data_audit() -> Dict[str, Any]:
    """Load data audit metadata from reports/model_results/data_audit.json."""
    return read_json(MODEL_RESULTS_DIR / "data_audit.json")


def load_permutation_importance() -> pd.DataFrame:
    """Load permutation importance table from reports/model_results/permutation_importance.csv."""
    return pd.read_csv(MODEL_RESULTS_DIR / "permutation_importance.csv")


def get_figure_path(filename: str) -> Path:
    """Return absolute path to a figure inside reports/figures/."""
    return FIGURES_DIR / filename


def list_available_figures() -> List[Path]:
    """List all copied PNG figures in reports/figures/."""
    if not FIGURES_DIR.exists():
        return []
    return sorted(FIGURES_DIR.glob("*.png"))


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Evaluate a loaded model against ground-truth labels (read-only helper)."""
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, disease_score(model, X))),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }

