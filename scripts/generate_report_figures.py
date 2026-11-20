"""Generate presentation-ready model evaluation figures from existing HeartCare-AI artifacts.

Outputs (saved to reports/figures/ at 300 DPI):
1. roc_curves.png                    — ROC curves of all 3 models on held-out Test (91 rows)
2. precision_recall_curves.png       — Precision-Recall curves + Average Precision on Test
3. cv_f1_before_after.png            — Stratified 5-Fold CV F1 (Baseline vs Tuned, with +/- std)
4. test_accuracy_f1_before_after.png — Held-out Test Accuracy & F1 (Baseline vs Tuned),
                                       highlighting Random Forest's drop after tuning

Scientific guarantees:
- Never retrains any model and never modifies .joblib files.
- Never re-splits data; loads the preserved held-out Test set (data/processed/test.csv).
- Cross-checks computed Test ROC-AUC against reports/model_results/metrics.csv.
"""

import logging
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, FEATURES, FIGURES_DIR, MODEL_RESULTS_DIR
from src.data_loader import load_train_test_splits
from src.evaluate_models import load_cross_validation, load_metrics
from src.prediction import disease_score, load_all_models, load_experiment_metadata

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ORDER = ["Decision Tree", "Random Forest", "Gaussian NB"]
MODEL_DISPLAY = {
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "Gaussian NB": "Gaussian Naive Bayes",
}
MODEL_COLORS = {
    "Decision Tree": "#E67E22",   # Warm amber
    "Random Forest": "#0B3C5D",   # Deep navy
    "Gaussian NB": "#0E9F9F",     # Medical teal
}

DPI = 300


def _apply_slide_style() -> None:
    """Configure crisp, slide-friendly matplotlib typography and layout."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 1.0,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
        }
    )


def _check_required_artifacts() -> None:
    """Verify all required artifacts exist before plotting; never fabricate data."""
    required = [
        DATA_PROCESSED_DIR / "test.csv",
        MODEL_RESULTS_DIR / "cross_validation.csv",
        MODEL_RESULTS_DIR / "metrics.csv",
        MODEL_RESULTS_DIR / "experiment.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Thiếu artifact bắt buộc để sinh biểu đồ:\n- " + "\n- ".join(missing)
        )


def compute_test_scores():
    """Load persisted .joblib models and compute class-1 scores on preserved Test set."""
    logger.info("Loading preserved Test split from data/processed/test.csv ...")
    _, test_df = load_train_test_splits()
    X_test = test_df[FEATURES]
    y_test = test_df["target"].astype(int).to_numpy()
    logger.info(
        "Test set loaded: %d rows (class 0: %d, class 1: %d)",
        len(y_test),
        int((y_test == 0).sum()),
        int((y_test == 1).sum()),
    )

    logger.info("Loading persisted .joblib pipelines from models/ (no retraining) ...")
    models = load_all_models()
    metrics_df = load_metrics().set_index("model")

    scores = {}
    for name in MODEL_ORDER:
        mdl = models[name]
        y_score = disease_score(mdl, X_test)
        computed_auc = float(roc_auc_score(y_test, y_score))
        recorded_auc = float(metrics_df.loc[name, "roc_auc"])
        if not math.isclose(computed_auc, recorded_auc, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(
                f"ROC-AUC mismatch for {name}: computed={computed_auc:.12f} vs metrics.csv={recorded_auc:.12f}"
            )
        logger.info(
            "  Verified %s on Test: ROC-AUC = %.4f (matches metrics.csv)",
            name,
            computed_auc,
        )
        scores[name] = y_score

    return y_test, scores


def plot_roc_curves(y_test: np.ndarray, scores: dict, out_path: Path) -> None:
    """1. Plot ROC Curves of all 3 models on held-out Test."""
    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    for name in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y_test, scores[name])
        auc_val = roc_auc_score(y_test, scores[name])
        ax.plot(
            fpr,
            tpr,
            label=f"{MODEL_DISPLAY[name]} (AUC = {auc_val:.3f})",
            color=MODEL_COLORS[name],
            linewidth=2.6,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#94A3B8",
        linewidth=1.5,
        label="Random Classifier (AUC = 0.500)",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("ROC Curves on Held-out Test Set (n = 91)", pad=12)
    ax.grid(True, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_precision_recall_curves(y_test: np.ndarray, scores: dict, out_path: Path) -> None:
    """2. Plot Precision-Recall Curves + Average Precision on held-out Test."""
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    pos_rate = float(np.mean(y_test))

    for name in MODEL_ORDER:
        precision, recall, _ = precision_recall_curve(y_test, scores[name])
        ap_val = average_precision_score(y_test, scores[name])
        ax.plot(
            recall,
            precision,
            label=f"{MODEL_DISPLAY[name]} (AP = {ap_val:.3f})",
            color=MODEL_COLORS[name],
            linewidth=2.6,
        )

    ax.axhline(
        pos_rate,
        linestyle="--",
        color="#94A3B8",
        linewidth=1.5,
        label=f"No-Skill Baseline (Prevalence = {pos_rate:.3f})",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision–Recall Curves on Held-out Test Set (n = 91)", pad=12)
    ax.grid(True, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_cv_f1_before_after(out_path: Path) -> None:
    """3. Grouped bar chart of Stratified 5-Fold CV F1 (Baseline vs Tuned) with std error bars."""
    cv_df = load_cross_validation()
    base_means, base_stds = [], []
    tuned_means, tuned_stds = [], []

    for name in MODEL_ORDER:
        b = cv_df[(cv_df["model"] == name) & (cv_df["variant"] == "baseline")].iloc[0]
        t = cv_df[(cv_df["model"] == name) & (cv_df["variant"] == "tuned")].iloc[0]
        base_means.append(float(b["f1_mean"]))
        base_stds.append(float(b["f1_std"]))
        tuned_means.append(float(t["f1_mean"]))
        tuned_stds.append(float(t["f1_std"]))

    x = np.arange(len(MODEL_ORDER))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    bars_b = ax.bar(
        x - width / 2,
        base_means,
        width,
        yerr=base_stds,
        capsize=5,
        label="Baseline (Before Tuning)",
        color="#93C5FD",
        edgecolor="#1D6FA5",
        linewidth=1.0,
        error_kw={"ecolor": "#334155", "elinewidth": 1.3},
    )
    bars_t = ax.bar(
        x + width / 2,
        tuned_means,
        width,
        yerr=tuned_stds,
        capsize=5,
        label="Tuned (GridSearchCV)",
        color="#0B3C5D",
        edgecolor="#0B3C5D",
        linewidth=1.0,
        error_kw={"ecolor": "#0F172A", "elinewidth": 1.3},
    )

    for bars, stds in [(bars_b, base_stds), (bars_t, tuned_stds)]:
        for bar, sd in zip(bars, stds):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + sd + 0.015,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.8,
                fontweight="bold",
                color="#1E293B",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER])
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("F1-score (Mean ± Std)")
    ax.set_title("Stratified 5-Fold CV F1-score on Train: Baseline vs Tuned", pad=12)
    ax.grid(True, axis="y", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_test_accuracy_f1_before_after(out_path: Path) -> None:
    """4. Two-panel grouped bar chart of Held-out Test Accuracy & F1 (Baseline vs Tuned),
    highlighting Random Forest's drop on Test despite improving in CV.
    """
    exp = load_experiment_metadata()
    metrics_df = load_metrics().set_index("model")

    panels = [
        ("accuracy", "Accuracy on Held-out Test (n = 91)"),
        ("f1", "F1-score on Held-out Test (n = 91)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.34

    for ax, (metric_key, panel_title) in zip(axes, panels):
        base_vals = []
        tuned_vals = []
        for name in MODEL_ORDER:
            b_val = float(exp["models"][name]["baseline_test"][metric_key])
            t_val = float(exp["models"][name]["test"][metric_key])
            rec_val = float(metrics_df.loc[name, metric_key])
            if not math.isclose(t_val, rec_val, rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError(f"Metric mismatch for {name} {metric_key}")
            base_vals.append(b_val)
            tuned_vals.append(t_val)

        tuned_colors = [
            "#D9534F" if name == "Random Forest" else "#0B3C5D"
            for name in MODEL_ORDER
        ]

        bars_b = ax.bar(
            x - width / 2,
            base_vals,
            width,
            label="Baseline (Before Tuning)",
            color="#93C5FD",
            edgecolor="#1D6FA5",
            linewidth=1.0,
        )
        bars_t = ax.bar(
            x + width / 2,
            tuned_vals,
            width,
            label="Tuned (After GridSearchCV)",
            color=tuned_colors,
            edgecolor=tuned_colors,
            linewidth=1.0,
        )

        # Value labels
        for bar in bars_b:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color="#334155",
                fontweight="bold",
            )
        for idx, bar in enumerate(bars_t):
            h = bar.get_height()
            color = "#B42318" if MODEL_ORDER[idx] == "Random Forest" else "#0B3C5D"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=color,
                fontweight="bold",
            )

        # Highlight Random Forest drop with delta callout
        rf_idx = MODEL_ORDER.index("Random Forest")
        rf_diff = tuned_vals[rf_idx] - base_vals[rf_idx]
        ax.annotate(
            f"RF drop: {rf_diff:+.3f}",
            xy=(rf_idx + width / 2, tuned_vals[rf_idx] + 0.048),
            xytext=(rf_idx + 0.02, 0.945),
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color="#B42318",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEC", edgecolor="#D9534F", lw=1.0),
            arrowprops=dict(arrowstyle="->", color="#D9534F", lw=1.4),
        )

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=8)
        ax.set_ylim(0.0, 1.04)
        ax.set_title(panel_title, pad=10)
        ax.grid(True, axis="y", alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Score")
    # Unified legend showing baseline, tuned, and highlighted RF drop
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#93C5FD", edgecolor="#1D6FA5", label="Baseline (Before Tuning)"),
        Patch(facecolor="#0B3C5D", edgecolor="#0B3C5D", label="Tuned (Decision Tree / Gaussian NB)"),
        Patch(facecolor="#D9534F", edgecolor="#D9534F", label="Tuned Random Forest (Drops on Test)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
    )

    fig.suptitle(
        "Held-out Test Performance: Baseline vs Tuned Models",
        fontsize=15,
        fontweight="bold",
        color="#0B3C5D",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "Note: Random Forest improves in CV (F1: 0.805 → 0.831) but drops on held-out Test (Acc: 0.846 → 0.813, F1: 0.825 → 0.795).",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.87])
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    _apply_slide_style()
    _check_required_artifacts()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Generating 4 report figures into %s ===", FIGURES_DIR)
    y_test, scores = compute_test_scores()

    plot_roc_curves(y_test, scores, FIGURES_DIR / "roc_curves.png")
    plot_precision_recall_curves(y_test, scores, FIGURES_DIR / "precision_recall_curves.png")
    plot_cv_f1_before_after(FIGURES_DIR / "cv_f1_before_after.png")
    plot_test_accuracy_f1_before_after(FIGURES_DIR / "test_accuracy_f1_before_after.png")

    logger.info("=== Done! All 4 report figures generated successfully. ===")


if __name__ == "__main__":
    main()
