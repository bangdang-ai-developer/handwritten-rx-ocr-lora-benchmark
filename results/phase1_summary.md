# Phase 1 — Kết quả thật (OCR cổ điển: Tesseract, EasyOCR)

Chạy trên Kaggle (kernel `dangbang1/ocr-med-benchmark-phase1`, version 5, GPU T4 x2, 15/09/2026 → 16/09/2026), tổng thời gian thực thi ~220s. Dữ liệu: **780 ảnh** test split của Kaggle "Doctor's Handwritten Prescription BD dataset" (đơn thuốc viết tay, 78 lớp thuốc), **400 ảnh** mẫu ngẫu nhiên (seed=42) từ IAM Handwriting Word Database (đối chứng domain-shift).

## Bảng tổng hợp

| Model | Dataset | n | CER (mean) | 95% CI (bootstrap) | WER (mean) | Exact-match | Top-1 classification acc (78 lớp) | Tỷ lệ output "degenerate" |
|---|---|---|---|---|---|---|---|---|
| Tesseract | kaggle_rx | 780 | 0.625 | [0.601, 0.648] | 1.304 | 7.3% | 49.2% | 1.4% |
| Tesseract | iam | 400 | 0.836 | [0.787, 0.886] | 1.248 | 6.3% | – | 1.8% |
| EasyOCR | kaggle_rx | 780 | 0.552 | [0.529, 0.574] | 1.117 | 10.4% | 54.9% | 3.3% |
| EasyOCR | iam | 400 | 0.736 | [0.703, 0.766] | 1.105 | 4.8% | – | **33.3%** |

*(Top-1 classification acc chỉ tính cho kaggle_rx vì IAM là từ vựng mở, không có 78 nhãn cố định. CI = khoảng tin cậy 95% bằng bootstrap 1000 lần resample, seed=0.)*

## Nhận định ban đầu (chưa phải kết luận cuối, cần thêm model)

1. **Cả 2 engine cổ điển đều gặp khó khăn đáng kể với chữ viết tay** — CER 0.55–0.84 (tức trung bình sai hơn một nửa số ký tự) — khớp với các phát hiện trong tài liệu đã tổng hợp trước đó (mục 4.2 `docs/04-dataset-y-khoa-cong-khai.md`).
2. **EasyOCR vượt trội Tesseract rõ rệt** trên cả 2 dataset (CER thấp hơn ~0.07–0.10, CI không chồng lấn) — đúng như kỳ vọng (CRNN deep learning vs. engine cổ điển).
3. **Bất ngờ:** cả 2 engine đều có CER THẤP HƠN trên kaggle_rx (đơn thuốc) so với IAM (chữ viết tay tổng quát) — ngược với giả thuyết ban đầu "domain y khoa khó hơn". Cần điều tra thêm ở Phase 5 (phân tích lỗi định tính) — có thể do ảnh kaggle_rx đã được crop/chuẩn hoá sẵn (128×128 hay tương tự) trong khi IAM có nhiều biến thể viết tay hơn (657 người viết khác nhau).
4. **Top-1 classification accuracy** (ánh xạ về 78 nhãn thuốc gần nhất) cao hơn nhiều so với exact-match (54.9% vs 10.4% cho EasyOCR) — cho thấy nhiều lỗi OCR là "gần đúng" và có thể sửa được nếu có bước hậu xử lý dùng từ điển đóng.
5. **EasyOCR có 33.3% output "degenerate" trên IAM** (rỗng hoặc lặp token) — đáng chú ý, có thể do EasyOCR's decoder bị nhầm khi gặp chữ viết tay quá khó đọc/không đúng phân bố huấn luyện; Tesseract hiếm khi "bỏ cuộc" hoàn toàn (degenerate thấp) nhưng vẫn cho output sai (CER cao) — hai kiểu thất bại khác nhau, đáng đưa vào phần Discussion.

## File dữ liệu thô

`results_master_phase1.csv` (2360 dòng: model × dataset × per-image reference/hypothesis/metrics) — dùng cho phân tích lỗi định tính ở Phase 5 (Ngày 11 theo kế hoạch).
