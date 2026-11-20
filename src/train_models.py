"""Refactored training pipeline reference (from legacy heart_pipeline.py / benchmark.py).

NOTE: Do NOT run this module during normal operation of HeartCare-AI.
The production models in `models/` and evaluation artifacts in `reports/` were
copied verbatim from the verified research experiment and must not be overwritten
or retrained without explicit intent.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_pipeline


def get_model_configs():
    """Return the exact baseline pipelines and GridSearchCV parameter grids
    used in the original research experiment.
    """
    return {
        "Decision Tree": (
            build_pipeline(
                DecisionTreeClassifier(
                    criterion="entropy", max_depth=4, random_state=42
                )
            ),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [2, 4, 6, None],
                "model__min_samples_split": [2, 6],
                "model__min_samples_leaf": [1, 4],
            },
        ),
        "Random Forest": (
            build_pipeline(
                RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42, n_jobs=1
                )
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, None],
                "model__min_samples_split": [2, 6],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "Gaussian NB": (
            build_pipeline(GaussianNB(), scale=True),
            {"model__var_smoothing": [1e-11, 1e-09, 1e-07, 1e-05, 0.001, 0.1]},
        ),
    }

