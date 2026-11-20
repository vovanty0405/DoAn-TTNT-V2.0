"""Data loading helpers for raw, processed, and demo datasets."""

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODEL_RESULTS_DIR
from src.utils import read_json


def load_raw_heart() -> pd.DataFrame:
    """Load the copied raw heart.csv dataset."""
    return pd.read_csv(
        DATA_RAW_DIR / "heart.csv",
        na_values=["?", "", " ", "-9", "-9.0"],
    )


def load_clean_heart() -> pd.DataFrame:
    """Load the copied cleaned heart dataset."""
    return pd.read_csv(DATA_PROCESSED_DIR / "heart_clean.csv", index_col="source_row")


def load_train_test_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the preserved train and test splits without regenerating them."""
    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv", index_col="source_row")
    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv", index_col="source_row")
    return train_df, test_df


def load_demo_samples() -> List[Dict[str, Any]]:
    """Load the preserved demo samples from reports/model_results/demo_samples.json."""
    return read_json(MODEL_RESULTS_DIR / "demo_samples.json")

