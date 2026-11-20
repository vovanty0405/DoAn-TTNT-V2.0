"""Chạy toàn bộ quy trình huấn luyện, GridSearchCV và đánh giá 3 mô hình phân loại bệnh tim.

Lệnh chạy:
    py -3.13 scripts/train.py
"""

from typing import Any
import hashlib
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CODES, DATA_PROCESSED_DIR, DATA_RAW_DIR, FEATURES, MODEL_RESULTS_DIR, MODELS_DIR
from src.evaluate_models import evaluate_model
from src.prediction import disease_score
from src.train_models import get_model_configs
from src.utils import print_terminal_table, write_json


def load_and_audit_data():
    """Đọc dữ liệu thô, loại bỏ bản ghi trùng và chuẩn hóa nhãn."""
    raw_path = DATA_RAW_DIR / "heart.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp {raw_path}")

    raw = pd.read_csv(raw_path, na_values=["?", "", " ", "-9", "-9.0"])
    if list(raw.columns) != FEATURES + ["target"]:
        raise ValueError("Lược đồ cột của tệp heart.csv không khớp FEATURES + target")

    data = raw.drop_duplicates().copy()
    data["target"] = 1 - data.target  # 0: Không bệnh, 1: Có bệnh

    conflicts = data.groupby(FEATURES, dropna=False).target.nunique().gt(1).sum()
    if conflicts:
        raise ValueError(f"Phát hiện {conflicts} nhóm dữ liệu có cùng đặc trưng nhưng khác nhãn!")

    audit = {
        "raw_rows": len(raw),
        "columns": len(raw.columns),
        "duplicates": int(raw.duplicated().sum()),
        "unique_rows": len(data),
        "conflicting_groups": conflicts,
        "raw_target_counts": raw.target.value_counts().to_dict(),
        "target_counts": data.target.value_counts().to_dict(),
        "dtypes": raw.dtypes.astype(str).to_dict(),
        "raw_missing": raw.isna().sum().to_dict(),
        "special_missing_unique": {
            "ca=4": data.ca.eq(4).sum(),
            "thal=0": data.thal.eq(0).sum(),
        },
        "ranges": raw.agg(["min", "max"]).to_dict(),
        "label_rule": "normalized_target = 1 - raw_target",
        "outlier_policy": "Retain plausible extremes including chol=564; no outcome-driven deletion.",
    }

    # Chuyển mã thiếu thành NaN trước khi đưa vào pipeline
    data["ca"] = data.ca.replace(4, np.nan)
    data["thal"] = data.thal.replace(0, np.nan)

    return raw, data, audit


def train() -> None:
    print("=" * 76)
    print("       HEARTCARE-AI — HUẤN LUYỆN & GRIDSEARCHCV 3 MÔ HÌNH PHÂN LOẠI")
    print("=" * 76)

    # 1. Đọc và chia tập phân tầng 70/30
    print("\n[1/4] Đọc dữ liệu từ data/raw/heart.csv và chuẩn hóa...", flush=True)
    raw, data, audit = load_and_audit_data()
    print(f"  • {audit['raw_rows']} dòng gốc - {audit['duplicates']} dòng trùng = {len(data)} dòng duy nhất")
    print("  • Nhãn chuẩn hóa: 0 = Không có bệnh, 1 = Có bệnh (target = 1 - target gốc)")

    X, y = data[FEATURES], data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    audit.update(
        train_rows=len(X_train),
        test_rows=len(X_test),
        train_target_counts=y_train.value_counts().to_dict(),
        test_target_counts=y_test.value_counts().to_dict(),
    )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(DATA_PROCESSED_DIR / "heart_clean.csv", index_label="source_row")
    data.loc[X_train.index].to_csv(DATA_PROCESSED_DIR / "train.csv", index_label="source_row")
    data.loc[X_test.index].to_csv(DATA_PROCESSED_DIR / "test.csv", index_label="source_row")
    write_json(MODEL_RESULTS_DIR / "data_audit.json", audit)

    print(f"  • Tập Train: {len(X_train)} dòng (0: {y_train.value_counts().get(0, 0)}, 1: {y_train.value_counts().get(1, 0)})")
    print(f"  • Tập Test:  {len(X_test)} dòng (0: {y_test.value_counts().get(0, 0)}, 1: {y_test.value_counts().get(1, 0)})")

    # 2. Cấu hình kiểm định chéo và siêu tham số
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scoring = {m: m for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    configs = get_model_configs()

    details = {}
    fitted = {}
    cv_rows = []
    baselines = {}

    print("\n[2/4] Đang chạy GridSearchCV trên tập Train (5-fold phân tầng, tối ưu F1)...", flush=True)
    for name, (base, grid) in configs.items():
        print(f"  --> Đang xử lý: {name}...", flush=True)
        scores = cross_validate(base, X_train, y_train, cv=cv, scoring=scoring)
        base.fit(X_train, y_train)
        baselines[name] = base

        search = GridSearchCV(
            base,
            grid,
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_

        details[name] = {
            "best_params": search.best_params_,
            "baseline_params": base.named_steps["model"].get_params(),
            "search_candidates": len(search.cv_results_["params"]),
        }

        for variant in ["baseline", "tuned"]:
            row: dict[str, Any] = {"model": name, "variant": variant}
            for metric in scoring:
                if variant == "baseline":
                    values = scores["test_" + metric]
                else:
                    values = np.array([
                        search.cv_results_[f"split{i}_test_{metric}"][search.best_index_]
                        for i in range(5)
                    ])
                row[metric + "_mean"] = float(values.mean())
                row[metric + "_std"] = float(values.std())
            cv_rows.append(row)

        print(f"      F1 CV tốt nhất = {search.best_score_:.3f} | Params: {search.best_params_}")

    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(MODEL_RESULTS_DIR / "cross_validation.csv", index=False)

    selected = cv_table[cv_table.variant.eq("tuned")].sort_values(
        ["f1_mean", "model"], ascending=[False, True]
    ).iloc[0]["model"]
    print(f"\n  ==> Mô hình được chọn tự động theo F1 CV: {selected}")

    # 3. Đánh giá trên tập Test độc lập và lưu file .joblib
    print(f"\n[3/4] Đánh giá trên tập Test độc lập ({len(y_test)} mẫu) & lưu models...", flush=True)
    rows = []
    for name, model in fitted.items():
        res = evaluate_model(model, X_test, y_test)
        details[name]["test"] = res
        details[name]["baseline_test"] = evaluate_model(baselines[name], X_test, y_test)
        rows.append({"model": name, **{k: v for k, v in res.items() if k != "confusion_matrix"}})

        joblib_path = MODELS_DIR / f"{name.replace(' ', '_')}.joblib"
        joblib.dump(model, joblib_path)

    test_table = pd.DataFrame(rows)
    test_table.to_csv(MODEL_RESULTS_DIR / "metrics.csv", index=False)

    # 4. Đo Permutation Importance & lưu metadata
    print("\n[4/4] Tính độ quan trọng đặc trưng (Permutation Importance)...", flush=True)
    imp = permutation_importance(
        fitted[selected],
        X_test,
        y_test,
        scoring="f1",
        n_repeats=30,
        random_state=42,
        n_jobs=1,
    )
    pd.DataFrame({
        "feature": FEATURES,
        "mean": imp.importances_mean,
        "std": imp.importances_std,
    }).sort_values("mean", ascending=False).to_csv(
        MODEL_RESULTS_DIR / "permutation_importance.csv", index=False
    )

    write_json(
        MODEL_RESULTS_DIR / "experiment.json",
        {
            "selected_model": selected,
            "selection_rule": "Highest mean F1 in Stratified 5-fold CV on Train; alphabetical tie-break. Default classification threshold.",
            "cv_caveat": "Tuned CV reuses search folds and may be optimistic. Test is independent. No nested CV.",
            "random_state": 42,
            "models": details,
            "dataset_sha256": hashlib.sha256((DATA_RAW_DIR / "heart.csv").read_bytes()).hexdigest(),
            "versions": {
                "python": platform.python_version(),
                "sklearn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        },
    )

    print("\n" + "=" * 76)
    print("                   KẾT QUẢ TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 76)
    fmt_test = test_table.copy()
    for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        fmt_test[c] = fmt_test[c].map("{:.3f}".format)
    print_terminal_table(fmt_test)
    print("\nHoàn tất huấn luyện! Các tệp kết quả đã được lưu vào models/ và reports/model_results/.")


if __name__ == "__main__":
    train()
