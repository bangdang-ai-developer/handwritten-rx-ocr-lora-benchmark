# Phase 2 — Kết quả thật (TrOCR-large-handwritten, Donut-base zero-shot)

Chạy trên Kaggle (kernel `dangbang1/ocr-med-benchmark-phase1`, version 6, GPU T4 x2, 16/09/2026, ~29 phút). Cùng 780 ảnh test Kaggle-Rx + 400 ảnh IAM như Phase 1.

## Bảng tổng hợp (kết hợp Phase 1 + Phase 2)

| Model | Dataset | n | CER | WER | Exact-match | Top-1 acc (78 lớp) | Degenerate |
|---|---|---|---|---|---|---|---|
| **trocr-large-handwritten** | **iam** | 400 | **0.441** | 0.570 | **57.3%** | – | 0.0% |
| easyocr | iam | 400 | 0.736 | 1.105 | 4.8% | – | 33.3% |
| tesseract | iam | 400 | 0.836 | 1.248 | 6.3% | – | 1.8% |
| donut-base-synthdog (raw crop) | iam | 400 | 1.000 | 1.000 | 0.0% | – | **99.75%** |
| easyocr | kaggle_rx | 780 | **0.552** | 1.117 | 10.4% | 54.9% | 3.3% |
| trocr-large-handwritten | kaggle_rx | 780 | 0.580 | 1.587 | 8.1% | **79.5%** | 0.0% |
| tesseract | kaggle_rx | 780 | 0.625 | 1.304 | 7.3% | 49.2% | 1.4% |
| donut-base-synthdog (raw crop) | kaggle_rx | 780 | 1.000 | 1.000 | 0.0% | 1.3% | **100%** |

## Nhận định

1. **TrOCR-large-handwritten vượt trội tuyệt đối trên IAM** (CER 0.44, exact-match 57.3%) — dễ hiểu vì đây chính là domain nó được huấn luyện chuyên biệt (chữ viết tay tiếng Anh dạng IAM). Đây **không phải phép so sánh zero-shot công bằng cho IAM**, cần ghi rõ caveat này trong bài báo — TrOCR có lợi thế "sân nhà".
2. **Trên đơn thuốc (kaggle_rx) — domain thực sự mới với TrOCR** — EasyOCR vẫn thắng về CER thô (0.552 vs 0.580), nhưng TrOCR có **Top-1 classification accuracy cao nhất (79.5%)** — nghĩa là output của TrOCR tuy có CER hơi cao hơn nhưng "gần đúng" nhiều hơn, dễ ánh xạ về đúng tên thuốc trong từ điển 78 lớp. **Đây là phát hiện quan trọng cho Discussion**: CER thô và độ chính xác phân loại closed-vocabulary có thể cho kết luận khác nhau — chọn metric nào phụ thuộc mục đích triển khai thực tế (đọc tự do vs. đối chiếu danh mục thuốc có sẵn).
3. **Quan sát kỹ thuật**: TrOCR có xu hướng thêm dấu "." ở cuối hầu hết output (vd "Aceta ." thay vì "Aceta") — do được train trên toàn dòng câu (IAM) nên "học" thói quen kết thúc câu bằng dấu câu, dù áp dụng lên ảnh 1-từ. Ghi nhận như một đặc điểm cần thảo luận, không sửa hậu kỳ (giữ nguyên để đo đúng hiệu năng zero-shot thật).
4. **Donut-base (chỉ pretrain SynthDoG, chưa fine-tune) THẤT BẠI GẦN NHƯ HOÀN TOÀN** khi áp trực tiếp lên ảnh crop 1 từ nhỏ — degenerate rate ~100% trên cả 2 dataset (output rỗng). **Giả thuyết đang kiểm chứng**: SynthDoG pretrain trên ảnh "trang tài liệu" đầy đủ (page-shaped), khi đưa ảnh crop nhỏ gần vuông vào, processor resize/pad thành một "trang" gần như trống → model dự đoán EOS ngay lập tức. Đang chạy thử nghiệm bổ sung: dán crop vào giữa 1 "trang trắng" mô phỏng tỷ lệ khung hình gốc mà model được train, xem có phục hồi được output có nghĩa hay không (kernel v7, đang chạy). **Kết quả này CHƯA nên coi là kết luận cuối** — sẽ cập nhật sau khi có kết quả v7.

## File dữ liệu thô
- `results_master_phase1.csv` (Tesseract, EasyOCR)
- `results_master_phase2.csv` (TrOCR-large-handwritten, Donut-base raw-crop)
- `results_master_combined.csv` (gộp cả 2, dùng cho phân tích)
