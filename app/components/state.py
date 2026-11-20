"""Cached artifact and model loaders for Streamlit pages."""

import streamlit as st

from src.config import MODEL_RESULTS_DIR
from src.data_loader import load_demo_samples
from src.evaluate_models import (
    load_cross_validation,
    load_data_audit,
    load_metrics,
    load_permutation_importance,
)
from src.prediction import load_all_models, load_experiment_metadata


@st.cache_resource
def get_experiment_and_models():
    """Load experiment metadata and all persisted sklearn pipelines once."""
    experiment = load_experiment_metadata()
    models = load_all_models()
    return experiment, models


@st.cache_data
def get_evaluation_artifacts():
    """Load copied evaluation tables and metadata."""
    return {
        "metrics": load_metrics(),
        "cv": load_cross_validation(),
        "audit": load_data_audit(),
        "demo_samples": load_demo_samples(),
        "importance": load_permutation_importance(),
    }


def artifacts_ready() -> bool:
    """Check whether required experiment metadata exists."""
    return (MODEL_RESULTS_DIR / "experiment.json").exists()

