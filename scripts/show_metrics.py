"""In toàn bộ các bảng so sánh mô hình, kết quả kiểm định chéo và hiệu năng trên terminal."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_models import (
    load_cross_validation,
    load_data_audit,
    load_metrics,
    load_permutation_importance,
)
from src.prediction import load_experiment_metadata
from src.utils import print_terminal_table


def main() -> None:
    audit = load_data_audit()
    exp = load_experiment_metadata()
    metrics = load_metrics()
    cv = load_cross_validation()
    imp = load_permutation_importance()

    print("=" * 78)
    print("           HEARTCARE-AI — BÁO CÁO KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 78)

    print("\n[1] THÔNG TIN DỮ LIỆU:")
    print(f"  • Số bản ghi gốc (Data/heart.csv): {audit['raw_rows']}")
    print(f"  • Số dòng trùng lặp:              {audit['duplicates']} (-70.5%)")
    print(f"  • Số dòng duy nhất sau lọc trùng:  {audit['unique_rows']}")
    print(f"  • Phân chia phân tầng (70/30):     Train = {audit['train_rows']} | Test = {audit['test_rows']}")
    print(f"  • Mô hình được chọn theo F1 CV:    {exp['selected_model']}")

    print("\n[2] BẢNG SO SÁNH HIỆU NĂNG 3 MÔ HÌNH TRÊN TẬP TEST ĐỘC LẬP (n = 91):")
    formatted_metrics = metrics.copy()
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        formatted_metrics[col] = formatted_metrics[col].map("{:.3f}".format)
    formatted_metrics.columns = ["Mô hình", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    print_terminal_table(formatted_metrics)

    print("\n[3] BẢNG KIỂM ĐỊNH CHÉO STRATIFIED 5-FOLD CV TRÊN TRAIN (BASELINE vs TUNED):")
    cv_summary = cv.assign(
        f1_cv=cv["f1_mean"].map("{:.3f}".format) + " +/- " + cv["f1_std"].map("{:.3f}".format)
    ).pivot(index="model", columns="variant", values="f1_cv").reset_index()
    cv_summary.columns = ["Mô hình", "F1 Baseline", "F1 Tuned (GridSearchCV)"]
    print_terminal_table(cv_summary)

    print("\n[4] SO SÁNH HIỆU NĂNG TEST TRƯỚC VÀ SAU TINH CHỈNH (BASELINE -> TUNED):")
    test_comp = []
    for name, info in exp["models"].items():
        b_test = info["baseline_test"]
        t_test = info["test"]
        missed, detected = t_test["confusion_matrix"][1]
        test_comp.append({
            "Mô hình": name,
            "Accuracy": f"{b_test['accuracy']:.3f} -> {t_test['accuracy']:.3f}",
            "Recall": f"{b_test['recall']:.3f} -> {t_test['recall']:.3f}",
            "F1-score": f"{b_test['f1']:.3f} -> {t_test['f1']:.3f}",
            "ROC-AUC": f"{b_test['roc_auc']:.3f} -> {t_test['roc_auc']:.3f}",
            "Bỏ sót ca bệnh": f"{missed} / {detected + missed}",
        })
    print_terminal_table(pd.DataFrame(test_comp))

    print("\n[5] SIÊU THAM SỐ TỐI ƯU CHỌN BỞI GRIDSEARCHCV:")
    for name, info in exp["models"].items():
        print(f"  • {name} ({info['search_candidates']} cấu hình thử):")
        print(f"    {info['best_params']}")

    print("\n[6] TOP ĐẶC TRƯNG QUAN TRỌNG NHẤT (PERMUTATION IMPORTANCE TRÊN TEST):")
    imp_top = imp.head(7).copy()
    imp_top["mean"] = imp_top["mean"].map("{:.4f}".format)
    imp_top["std"] = imp_top["std"].map("{:.4f}".format)
    imp_top.columns = ["Đặc trưng", "Mức giảm F1", "Độ lệch chuẩn"]
    print_terminal_table(imp_top)
    print("=" * 78)


if __name__ == "__main__":
    main()
