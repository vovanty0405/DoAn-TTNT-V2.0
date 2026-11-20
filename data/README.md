# 📂 HeartCare AI — Dataset Documentation (`data/`)

Tài liệu này mô tả chi tiết nguồn dữ liệu, quy trình làm sạch, kiểm tra chất lượng (Data Audit), từ điển đặc trưng lâm sàng và cách phân chia tập Train/Test được bảo toàn nguyên vẹn từ nghiên cứu gốc.

---

## 1. Cấu trúc thư mục `data/`

```text
data/
├── README.md                  # Tài liệu mô tả bộ dữ liệu
├── raw/
│   └── heart.csv              # Dữ liệu gốc từ Kaggle (1,025 bản ghi × 14 cột)
└── processed/
    ├── heart_clean.csv        # Sau loại trùng & chuẩn hóa nhãn (302 bản ghi duy nhất)
    ├── train.csv              # Tập huấn luyện 70% phân tầng (211 bản ghi)
    └── test.csv               # Tập kiểm tra độc lập 30% phân tầng (91 bản ghi)
```

> **Nguyên tắc bảo toàn khoa học:** Các tệp trong `data/processed/` được sao chép nguyên vẹn từ thực nghiệm gốc (`random_state=42`). **Không tự ý chia lại split** và **không sử dụng tập Test** để huấn luyện hay tinh chỉnh siêu tham số.

---

## 2. Kiểm tra chất lượng dữ liệu (Data Audit Summary)

Số liệu được trích xuất trực tiếp từ `reports/model_results/data_audit.json`:

| Hạng mục | Giá trị thực tế | Ghi chú |
| :--- | :---: | :--- |
| Số dòng gốc (`raw/heart.csv`) | **1,025** | 14 cột (13 đặc trưng + 1 `target`) |
| Số dòng trùng lặp hoàn toàn | **723** | Bắt buộc loại bỏ trước khi chia Train/Test để chống rò rỉ dữ liệu (data leakage) |
| Số dòng duy nhất (`heart_clean.csv`) | **302** | 0 nhóm xung đột nhãn (`conflicting_groups = 0`) |
| Nhãn sau chuẩn hóa (`0` / `1`) | **164 (54.3%)** / **138 (45.7%)** | Quy tắc: `normalized_target = 1 - raw_target` (`0`: Không bệnh, `1`: Có bệnh) |
| Giá trị thiếu dạng `NaN` thô | **0** | Không có ô trống hoặc ký tự `?` trong file gốc |
| Mã thiếu đặc biệt (Special Missing) | **`ca=4` (4 dòng), `thal=0` (2 dòng)** | Được transformer `SpecialMissing` đổi thành `NaN` bên trong Pipeline |
| Tập Train (`processed/train.csv`) | **211 dòng** | `0`: 115 dòng · `1`: 96 dòng (70% phân tầng) |
| Tập Test (`processed/test.csv`) | **91 dòng** | `0`: 49 dòng · `1`: 42 dòng (30% phân tầng, hoàn toàn tách biệt) |
| Chính sách ngoại lai (Outliers) | **Giữ nguyên** | Giữ lại các cực trị lâm sàng hợp lệ (ví dụ `chol = 564 mg/dL`), không xóa theo nhãn |

---

## 3. Clinical Feature Dictionary (Từ điển 13 Đặc trưng & Nhãn)

| Cột | Nhóm biến | Kiểu dữ liệu | Miền giá trị thực tế | Giải thích mã & Ý nghĩa lâm sàng |
| :--- | :--- | :---: | :---: | :--- |
| `age` | Số (Numeric) | `int64` | `29 – 77` | Tuổi bệnh nhân (năm) |
| `sex` | Nhị phân (Binary) | `int64` | `0, 1` | Giới tính: `0 = Nữ`, `1 = Nam` |
| `cp` | Danh mục (Categorical) | `int64` | `0, 1, 2, 3` | Loại đau ngực: `0 = Không triệu chứng`, `1 = Đau thắt ngực không điển hình`, `2 = Đau ngực không do thắt ngực`, `3 = Đau thắt ngực điển hình` |
| `trestbps` | Số (Numeric) | `int64` | `94 – 200` | Huyết áp lúc nghỉ (mmHg) khi nhập viện |
| `chol` | Số (Numeric) | `int64` | `126 – 564` | Cholesterol huyết thanh (mg/dL) |
| `fbs` | Nhị phân (Binary) | `int64` | `0, 1` | Đường huyết lúc đói: `0 = ≤ 120 mg/dL`, `1 = > 120 mg/dL` |
| `restecg` | Danh mục (Categorical) | `int64` | `0, 1, 2` | Điện tâm đồ nghỉ: `0 = Phì đại thất trái`, `1 = Bình thường`, `2 = Bất thường ST-T` |
| `thalach` | Số (Numeric) | `int64` | `71 – 202` | Nhịp tim tối đa đạt được khi gắng sức (nhịp/phút) |
| `exang` | Nhị phân (Binary) | `int64` | `0, 1` | Đau thắt ngực khi gắng sức: `0 = Không`, `1 = Có` |
| `oldpeak` | Số (Numeric) | `float64` | `0.0 – 6.2` | Mức chênh xuống đoạn ST do gắng sức so với nghỉ |
| `slope` | Danh mục (Categorical) | `int64` | `0, 1, 2` | Độ dốc đoạn ST: `0 = Dốc xuống`, `1 = Ngang`, `2 = Dốc lên` |
| `ca` | Danh mục (Categorical) | `int64` | `0 – 4` | Số mạch vành chính (0–3 mạch); `4 = Thiếu / không xác định` (đổi sang `NaN` khi tiền xử lý) |
| `thal` | Danh mục (Categorical) | `int64` | `0 – 3` | Xạ hình tưới máu cơ tim: `0 = Thiếu / không xác định` (đổi sang `NaN`), `1 = Khiếm khuyết cố định`, `2 = Bình thường`, `3 = Khiếm khuyết hồi phục` |
| `target` | **Biến mục tiêu** | `int64` | `0, 1` | Nhãn chuẩn hóa: `0 = Không có bệnh`, `1 = Có bệnh` |

---

## 4. Quy trình Tiền xử lý trong Pipeline (`src/preprocessing.py`)

Toàn bộ bước tiền xử lý chỉ được `fit` trên tập Train trong từng fold của Stratified 5-Fold CV và đóng gói chung trong tệp `.joblib`:

1. **`SpecialMissing` Transformer:** Sắp xếp đúng thứ tự 13 cột `FEATURES`, chuyển `ca = 4` và `thal = 0` thành `NaN` trên bản sao dữ liệu.
2. **5 biến số (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`):** Điền thiếu bằng **trung vị (`median`)**. Riêng **Gaussian Naive Bayes** áp dụng thêm **`StandardScaler`**.
3. **3 biến nhị phân (`sex`, `fbs`, `exang`):** Điền thiếu bằng giá trị phổ biến nhất (**`most_frequent`**), giữ nguyên mã `0/1`.
4. **5 biến danh mục (`cp`, `restecg`, `slope`, `ca`, `thal`):** Điền thiếu bằng **`most_frequent`**, sau đó mã hóa **`OneHotEncoder(handle_unknown="ignore")`**.

---

## 5. Lưu ý tách biệt dữ liệu nghiên cứu và dữ liệu ứng dụng

- Dữ liệu trong `data/` **chỉ phục vụ nghiên cứu, đánh giá và đối chiếu mô hình**.
- Tuyệt đối **không biến các dòng trong `train.csv` hoặc `test.csv` thành hồ sơ bệnh nhân thật** trong SQLite. Khi cần kiểm thử giao diện CRUD/phân trang, sử dụng script riêng `scripts/seed_demo_data.py` sinh hồ sơ giả lập được gắn nhãn rõ `[DEMO]`.
