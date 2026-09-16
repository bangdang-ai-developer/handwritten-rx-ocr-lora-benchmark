# Phase 6 — Chuẩn bị Fine-tuning (kết quả điều tra + đóng băng test set)

## 1. Điều tra "780 vs 936 ảnh" — KHÔNG phải lỗi path-resolution

Kế hoạch Q2 (`docs/05-ke-hoach-Q2.md`) trước đó suy đoán số ảnh test thực tế (780) thấp hơn lý thuyết (936 = 20%×4680) là do lỗi khớp đường dẫn trong `build_manifest_kaggle_rx()`. **Đã tải dataset về máy (`data/kaggle_rx/`) và đếm trực tiếp để kiểm chứng — suy đoán này SAI.**

| Split | Số dòng CSV | Số file ảnh | Khớp? |
|---|---|---|---|
| Training | 3.120 | 3.120 | ✅ |
| Validation | 780 | 780 | ✅ |
| Testing | 780 | 780 | ✅ |
| **Tổng** | **4.680** | **4.680** | ✅ khớp đúng tổng dataset |

**Kết luận:** Code đã resolve đúng **100%** số ảnh trong CSV từ đầu (780/780, không mất ảnh nào). Vấn đề chỉ là mô tả "60% training, 20% validation, 20% testing" trên trang Kaggle **không khớp chính xác** với tỷ lệ chia thật (thực tế ≈ 66,7% / 16,65% / 16,65%). Không có gì cần sửa trong code — chỉ cần **sửa lại toàn bộ tài liệu kế hoạch**: train set thật là **3.120 ảnh**, không phải 2.808 như đã viết trước đó. Đã thêm ghi chú cảnh báo ở đầu `docs/05-ke-hoach-Q2.md`.

## 2. Đóng băng test manifest (frozen test set)

Đã xác minh: cả 7 model đã benchmark ở Phase 1-5 (Tesseract, EasyOCR, TrOCR, Donut×2, GOT-OCR2.0, Qwen2.5-VL) dùng **chính xác cùng một bộ 780 ảnh kaggle_rx và 400 ảnh IAM** (kiểm tra set-equality trên toàn bộ image filename — khớp 100%, seed=42 hoạt động đúng như thiết kế). Đã trích xuất và lưu thành 2 file bất biến để mọi lần đánh giá fine-tune sau này (Phase 8) bắt buộc dùng lại đúng bộ này (điều kiện cần cho Wilcoxon signed-rank paired test):

- `results/frozen_test_manifest_kaggle_rx.csv` (780 dòng: `image_filename`, `label`)
- `results/frozen_test_manifest_iam.csv` (400 dòng: `image_filename`, `label`)

## 3. Pipeline augmentation

Đã viết `src/augmentation.py` (Albumentations): Affine (rotation ±5°, shear ±8°, scale 0.95-1.05), ElasticTransform (tuỳ chọn, dùng cho A/B ablation), GaussNoise, GaussianBlur, RandomBrightnessContrast, Morphological erosion/dilation (mô phỏng độ đậm nét bút) — đúng danh sách đã đề xuất ở kế hoạch Q2 Mục 3.2. Self-test chạy trên ảnh training thật, output preview lưu tại `results/augmentation_preview/`.

## Việc còn lại trước khi vào Phase 7 (LoRA fine-tune)

- Đồng bộ `src/augmentation.py` vào `notebooks/kaggle_benchmark.py` (bản chạy trên Kaggle) — tương tự cách `src/metrics.py` đã được đồng bộ.
- Viết script fine-tune (`peft` + `LoraConfig`), dùng đúng 3.120 ảnh Training (không phải 2.808).
