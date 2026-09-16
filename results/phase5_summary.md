# Phase 5 — Kết quả thật (Qwen2.5-VL-3B-Instruct)

Chạy trên Kaggle (kernel v13, GPU T4 x2, 16/09/2026, ~7 phút — nhanh nhất trong các VLM đã test). Cùng 780 ảnh Kaggle-Rx + 400 ảnh IAM.

## Kết quả

| Dataset | CER (mean) | CER (median) | WER | Exact-match | Top-1 acc (78 lớp) |
|---|---|---|---|---|---|
| kaggle_rx | **0.434 (TỐT NHẤT)** | – | 0.912 | **28.2% (TỐT NHẤT, gấp 2x model nhì)** | **81.5% (TỐT NHẤT)** |
| iam | 1.100 (TỆ NHẤT) | **0.0 (đa số đúng hoàn toàn!)** | 0.613 | 59.3% (nhì, sau TrOCR) | – |

## Phát hiện quan trọng nhất của Phase 5: mean CER có thể gây hiểu lầm

Trên IAM, **median CER = 0.0** (quá nửa số ảnh model đọc đúng 100%) nhưng **mean CER = 1.10** (tệ nhất trong toàn bộ 6 model đã test) — mâu thuẫn tưởng như vô lý. Nguyên nhân: một số ít trường hợp (~3%) model **không tuân theo chỉ dẫn "chỉ xuất văn bản"**, mà trả lời kiểu chatbot đầy đủ câu, ví dụ:
- Tham chiếu `"."` → model trả lời *"The text is not visible in the image provided. Please upload an image..."* (CER = 109!)
- Tham chiếu `"a"` → model trả lời *"The text in the image is: \"1234567890\"..."* (CER = 78)

Vì nhiều từ/ký tự trong IAM rất ngắn (dấu câu, từ nối 1-2 chữ), một câu trả lời dài "từ chối/giải thích" tạo ra khoảng cách chỉnh sửa khổng lồ so với tham chiếu ngắn, kéo mean CER lên rất cao dù đa số dự đoán hoàn hảo.

**Ý nghĩa cho bài báo:** đây là bằng chứng cụ thể cho luận điểm "mean CER có thể đánh lừa khi đánh giá VLM tổng quát — cần báo cáo cả median CER hoặc tỷ lệ 'instruction-following failure' riêng, không chỉ mean". Sẽ áp dụng chỉ số median CER bổ sung ở Phase 9 (phân tích tổng hợp) cho tất cả model, không chỉ Qwen.

## Phát hiện chính khác

**Trên đơn thuốc (kaggle_rx) — domain mục tiêu của bài báo — Qwen2.5-VL-3B (VLM tổng quát, CHỈ 3B tham số) vượt qua TẤT CẢ model OCR chuyên biệt đã test (GOT-OCR2.0 580M, TrOCR-large 558M, EasyOCR, Tesseract)** ở cả CER, exact-match, và top-1 accuracy. Đây là kết quả bất ngờ và rất đáng chú ý: một VLM tổng quát nhỏ, dùng prompt tiếng Anh đơn giản, không cần huấn luyện chuyên biệt, lại đọc đơn thuốc viết tay tốt hơn các engine OCR chuyên dụng — gợi ý khả năng suy luận ngữ cảnh/kiến thức thế giới (biết "Aceta" là tên thuốc hợp lý) giúp Qwen "đoán đúng" tốt hơn dù không được huấn luyện riêng cho OCR.

## File dữ liệu
`results_master_phase5_qwen.csv`, gộp vào `results_master_combined.csv` (8.260 dòng, 6 model × 2 dataset, đầy đủ cho Phase 9).
