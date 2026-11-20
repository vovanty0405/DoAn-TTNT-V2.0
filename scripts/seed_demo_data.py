"""Explicit script to seed 18 synthetic [DEMO] patient records and predictions.

Run manually when you want sample data to test CRUD, pagination, and Dashboard charts:
    py -3.13 scripts/seed_demo_data.py

NOTE:
- All generated profiles are clearly marked with `[DEMO]` and synthetic codes `DEMO-001..DEMO-018`.
- Never uses real patient data and never converts Train/Test rows into real patient profiles.
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db import init_db, register_current_models
from database.patient_repository import create_patient, query_patients
from database.prediction_repository import save_prediction
from src.prediction import load_all_models, predict_single

DEMO_NAMES = [
    ("Nguyễn Minh An", "Nam", 1962),
    ("Trần Thị Bích", "Nữ", 1971),
    ("Lê Hoàng Cường", "Nam", 1958),
    ("Phạm Thu Dung", "Nữ", 1968),
    ("Hoàng Văn Đức", "Nam", 1975),
    ("Vũ Thị Hạnh", "Nữ", 1964),
    ("Đặng Quốc Huy", "Nam", 1955),
    ("Bùi Ngọc Lan", "Nữ", 1979),
    ("Đỗ Thành Long", "Nam", 1960),
    ("Hồ Thị Mai", "Nữ", 1966),
    ("Ngô Quang Minh", "Nam", 1973),
    ("Dương Mỹ Ngọc", "Nữ", 1959),
    ("Lý Gia Phúc", "Nam", 1981),
    ("Trương Thị Quỳnh", "Nữ", 1969),
    ("Võ Thanh Sơn", "Nam", 1957),
    ("Phan Thị Thảo", "Nữ", 1974),
    ("Tạ Hữu Trung", "Nam", 1963),
    ("Đinh Thị Vân", "Nữ", 1977),
]

DEMO_ADDRESSES = [
    "Quận 1, TP.HCM",
    "Quận 3, TP.HCM",
    "Quận 7, TP.HCM",
    "TP. Thủ Đức, TP.HCM",
    "Quận Bình Thạnh, TP.HCM",
    "Quận Tân Bình, TP.HCM",
]


def _synthetic_features(rng: random.Random, age: int, gender: str) -> dict:
    """Generate plausible synthetic 13-feature values purely for UI demo."""
    return {
        "age": float(age),
        "sex": 1 if gender == "Nam" else 0,
        "cp": rng.choice([0, 1, 2, 3]),
        "trestbps": float(rng.randint(105, 165)),
        "chol": float(rng.randint(175, 320)),
        "fbs": rng.choice([0, 0, 0, 1]),
        "restecg": rng.choice([0, 1, 2]),
        "thalach": float(rng.randint(110, 185)),
        "exang": rng.choice([0, 1]),
        "oldpeak": round(rng.uniform(0.0, 3.6), 1),
        "slope": rng.choice([0, 1, 2]),
        "ca": rng.choice([0, 1, 2, 3]),
        "thal": rng.choice([1, 2, 3]),
    }


def seed_demo_data() -> None:
    init_db()
    register_current_models()
    models = load_all_models()
    model_names = list(models.keys())

    existing, total_existing = query_patients(search="DEMO-", page=1, page_size=50)
    if total_existing >= len(DEMO_NAMES):
        print(f"Đã có sẵn {total_existing} hồ sơ DEMO trong SQLite. Bỏ qua sinh trùng.")
        return

    rng = random.Random(2026)
    now = datetime.now()
    created_patients = 0
    created_preds = 0

    for idx, (name, gender, birth_year) in enumerate(DEMO_NAMES, start=1):
        code = f"DEMO-{idx:03d}"
        pid = create_patient(
            full_name=f"[DEMO] {name}",
            patient_code=code,
            birth_year=birth_year,
            gender=gender,
            phone=f"090{rng.randint(1000000, 9999999)}",
            address=rng.choice(DEMO_ADDRESSES),
            notes="Hồ sơ giả lập DEMO phục vụ kiểm thử giao diện và phân trang.",
        )
        created_patients += 1

        # Create 1–2 synthetic predictions spread across recent months
        num_preds = 2 if idx % 2 == 0 else 1
        age = now.year - birth_year
        for _ in range(num_preds):
            m_name = rng.choice(model_names)
            feats = _synthetic_features(rng, age, gender)
            pred, score, label_text = predict_single(models[m_name], feats)
            days_ago = rng.randint(0, 140)
            ts = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
            save_prediction(
                model_name=m_name,
                predicted_label=pred,
                label_text=label_text,
                model_score=score,
                input_features=feats,
                patient_id=pid,
                predicted_at=ts,
            )
            created_preds += 1

    print(
        f"Hoàn tất sinh dữ liệu DEMO: {created_patients} hồ sơ bệnh nhân [DEMO] "
        f"và {created_preds} lượt dự đoán."
    )


if __name__ == "__main__":
    seed_demo_data()

