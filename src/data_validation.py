"""Input validation for the 13 HeartCare-AI clinical features."""

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config import CODES, FEATURE_LIMITS, FEATURES


def validate_input_dict(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single patient feature dictionary against schema and ranges."""
    errors: List[str] = []

    missing_keys = [col for col in FEATURES if col not in record]
    if missing_keys:
        errors.append(f"Thiếu đặc trưng đầu vào: {', '.join(missing_keys)}")
        return False, errors

    for col in FEATURES:
        val = record[col]
        if val is None or pd.isna(val):
            # ca and thal may legitimately be missing (converted to NaN by pipeline).
            if col not in ("ca", "thal"):
                errors.append(f"Giá trị '{col}' không được để trống.")
            continue

        if col in CODES:
            try:
                int_val = int(val)
            except (TypeError, ValueError):
                errors.append(f"Đặc trưng '{col}' phải là mã số nguyên hợp lệ.")
                continue
            if int_val not in CODES[col]:
                errors.append(
                    f"Đặc trưng '{col}' có mã không hợp lệ: {val}. "
                    f"Giá trị hợp lệ: {sorted(CODES[col].keys())}"
                )
        elif col in FEATURE_LIMITS:
            try:
                num_val = float(val)
            except (TypeError, ValueError):
                errors.append(f"Đặc trưng '{col}' phải là số hợp lệ.")
                continue
            lo, hi = FEATURE_LIMITS[col]
            if not (lo <= num_val <= hi):
                errors.append(
                    f"Đặc trưng '{col}' ({num_val}) nằm ngoài khoảng hợp lệ [{lo}, {hi}]."
                )

    return len(errors) == 0, errors


def validate_input_dataframe(frame: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that a DataFrame contains all 13 required features."""
    missing = [col for col in FEATURES if col not in frame.columns]
    if missing:
        return False, [f"DataFrame thiếu các cột: {', '.join(missing)}"]
    if frame.empty:
        return False, ["DataFrame đầu vào rỗng."]
    return True, []

