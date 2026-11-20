"""Configuration, paths, feature definitions, and label/code mappings for HeartCare-AI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_RESULTS_DIR = REPORTS_DIR / "model_results"
FIGURES_DIR = REPORTS_DIR / "figures"

DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "heartcare.db"

# Exact 13-feature order preserved from the original research pipeline.
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal"]

# Normalized label meaning: 0 = no disease, 1 = disease (target = 1 - raw_target).
LABELS = {0: "Không có bệnh", 1: "Có bệnh"}

CODES = {
    "sex": {0: "Nữ", 1: "Nam"},
    "fbs": {0: "≤ 120 mg/dL", 1: "> 120 mg/dL"},
    "exang": {0: "Không", 1: "Có"},
    "cp": {
        0: "Không triệu chứng",
        1: "Đau thắt ngực không điển hình",
        2: "Đau ngực không do thắt ngực",
        3: "Đau thắt ngực điển hình",
    },
    "restecg": {0: "Phì đại thất trái", 1: "Bình thường", 2: "Bất thường ST-T"},
    "slope": {0: "Dốc xuống", 1: "Ngang", 2: "Dốc lên"},
    "ca": {
        0: "0 mạch",
        1: "1 mạch",
        2: "2 mạch",
        3: "3 mạch",
        4: "Thiếu / không xác định",
    },
    "thal": {
        0: "Thiếu / không xác định",
        1: "Khiếm khuyết cố định",
        2: "Bình thường",
        3: "Khiếm khuyết hồi phục",
    },
}

FEATURE_LABELS = {
    "age": "Tuổi (năm)",
    "trestbps": "Huyết áp nghỉ (mmHg)",
    "chol": "Cholesterol (mg/dL)",
    "thalach": "Nhịp tim tối đa (nhịp/phút)",
    "oldpeak": "ST chênh xuống (oldpeak)",
    "sex": "Giới tính",
    "cp": "Loại đau ngực",
    "fbs": "Đường huyết lúc đói",
    "restecg": "Điện tâm đồ nghỉ",
    "exang": "Đau ngực khi gắng sức",
    "slope": "Độ dốc ST",
    "ca": "Số mạch lớn",
    "thal": "Kết quả thal",
}

FEATURE_LIMITS = {
    "age": (1.0, 120.0),
    "trestbps": (1.0, 300.0),
    "chol": (1.0, 1000.0),
    "thalach": (1.0, 250.0),
    "oldpeak": (0.0, 10.0),
}

MODEL_FILES = {
    "Decision Tree": "Decision_Tree.joblib",
    "Random Forest": "Random_Forest.joblib",
    "Gaussian NB": "Gaussian_NB.joblib",
}

