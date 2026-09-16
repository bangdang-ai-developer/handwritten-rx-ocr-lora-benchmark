# Phase 3 — Kết quả thật (GOT-OCR2.0)

Chạy trên Kaggle (kernel `dangbang1/ocr-med-benchmark-phase1`, version 11, GPU T4 x2, 16/09/2026, ~32 phút). Cùng 780 ảnh test Kaggle-Rx + 400 ảnh IAM.

## Sự cố kỹ thuật đã xử lý (quan trọng, ảnh hưởng độ tin cậy số liệu)

`pip install transformers` (không ghim version) tự chọn bản **5.0.0** (major version rất mới, phát hành gần đây) — gây ra output hoàn toàn vô nghĩa, hỗn loạn đa ngôn ngữ (CER 55-120!) dù prompt/tensor đầu vào đã xác nhận đúng định dạng. Nguyên nhân: GOT-OCR2.0 được merge vào `transformers` từ 31/01/2025 ([PR #34721](https://github.com/huggingface/transformers/pull/34721)), ổn định trên các bản 4.x nhưng chưa chắc tương thích với bước nhảy major 5.0.0. **Đã sửa bằng cách ghim `transformers==4.57.0`** — debug 5 ảnh đầu cho kết quả hợp lý ngay ("Acek", "Acd", "Ach" cho từ "Aceta"), full run cho CER hợp lý (~0.38-0.48). Bài học áp dụng cho Phase 4-5 (PaddleOCR-VL, Qwen-VL): luôn ghim version `transformers`/thư viện liên quan và chạy debug 5 ảnh + có cơ chế tự-dừng-sớm trước khi chạy full 1180 ảnh.

## Bảng tổng hợp đầy đủ (4 phase, 6 model)

| Model | Dataset | n | CER | WER | Exact-match | Top-1 acc (78 lớp) | Degenerate |
|---|---|---|---|---|---|---|---|
| **got-ocr2.0** | **kaggle_rx** | 780 | **0.479** | 1.238 | **15.8%** | 67.2% | 0.0% |
| easyocr | kaggle_rx | 780 | 0.552 | 1.117 | 10.4% | 54.9% | 3.3% |
| trocr-large-handwritten | kaggle_rx | 780 | 0.580 | 1.587 | 8.1% | **79.5%** | 0.0% |
| tesseract | kaggle_rx | 780 | 0.625 | 1.304 | 7.3% | 49.2% | 1.4% |
| donut-base-synthdog (raw/padded) | kaggle_rx | 780 | 1.00 / 4.20 | – | 0.0% | 1.3% / 3.6% | ~89-100% |
| **got-ocr2.0** | **iam** | 400 | **0.386** | 0.518 | 53.8% | – | 0.3% |
| trocr-large-handwritten | iam | 400 | 0.441 | 0.570 | **57.3%** | – | 0.0% |
| easyocr | iam | 400 | 0.736 | 1.105 | 4.8% | – | 33.3% |
| tesseract | iam | 400 | 0.836 | 1.248 | 6.3% | – | 1.8% |
| donut-base-synthdog (raw/padded) | iam | 400 | 1.00 / 1.06 | – | 0.0% | – | ~96-100% |

## Nhận định

1. **GOT-OCR2.0 là model tốt nhất tính đến Phase 3** trên CẢ 2 dataset — đúng như kỳ vọng cho một VLM-OCR chuyên biệt thế hệ mới ("OCR-2.0"), vượt cả TrOCR (vốn có lợi thế "sân nhà" trên IAM) và các engine cổ điển.
2. Trên đơn thuốc, GOT-OCR2.0 dẫn đầu cả CER và exact-match, nhưng TrOCR vẫn có Top-1 classification accuracy cao nhất (79.5%) — khẳng định lại nhận định Phase 2: các metric khác nhau kể câu chuyện khác nhau, cần báo cáo đầy đủ cả 2 loại trong bài báo.
3. Donut-base (cả 2 cách) vẫn là ngoại lệ thất bại rõ ràng, đã xác nhận là phát hiện hợp lệ (Phase 2 summary) — càng nổi bật khi đối chiếu với GOT-OCR2.0 (cũng là kiến trúc "OCR-free"/VLM nhưng có task-tuning phù hợp) thành công vượt trội. Đây là điểm đối lập tốt cho Discussion: không phải mọi kiến trúc VLM-OCR đều như nhau, sự khác biệt về pretraining/task-alignment quan trọng hơn kiến trúc.

## File dữ liệu
- `results_master_phase3_got_ocr2.csv`
- `results_master_combined.csv` (gộp Phase 1+2+2b+3, 7080 dòng — dùng cho phân tích thống kê/viết bài)
