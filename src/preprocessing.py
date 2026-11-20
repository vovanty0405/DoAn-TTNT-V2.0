"""Preprocessing logic preserved identically from the original heart_pipeline.py."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BINARY, CATEGORICAL, FEATURES, NUMERIC


def normalize_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to FEATURES and convert special missing codes on a copy."""
    frame = frame.loc[:, FEATURES].copy()
    frame["ca"] = frame["ca"].replace(4, np.nan)
    frame["thal"] = frame["thal"].replace(0, np.nan)
    return frame


class SpecialMissing(TransformerMixin, BaseEstimator):
    """Custom transformer embedded inside the persisted sklearn Pipelines."""

    def fit(self, X, y=None):
        """Record input feature names and count; no parameters are estimated."""
        self.n_features_in_ = len(FEATURES)
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        return self

    def transform(self, X):
        """Apply special missing-code normalization to training or inference data."""
        return normalize_input(X)

    def get_feature_names_out(self, input_features=None):
        """Return ordered feature names."""
        return np.array(FEATURES, dtype=object)


def build_pipeline(model, scale: bool = False) -> Pipeline:
    """Compose SpecialMissing, ColumnTransformer preprocessing, and classifier."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer(
        [
            ("numeric", Pipeline(steps), NUMERIC),
            ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline([("codes", SpecialMissing()), ("preprocess", pre), ("model", model)])

