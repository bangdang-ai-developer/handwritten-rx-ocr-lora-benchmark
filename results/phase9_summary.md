# Phase 9 — Phân tích định lượng tổng hợp (toàn bộ model, zero-shot + fine-tuned)

Script tái lập: `src/analyze_aggregate.py` (đọc `results/results_master_combined.csv`, xuất
`results/phase9_summary_table.csv`, `results/phase9_domain_gap.csv`, `results/phase9_wilcoxon_pairs.csv`).
Bootstrap CI 95% (1.000 resample, seed=0), Wilcoxon signed-rank paired trên đúng từng ảnh (mọi model
đều chạy trên cùng manifest, seed=42 — đã xác minh set-equality 100% giữa các lần chạy, xem `phase6_summary.md`).

## 1. Bảng tổng hợp đầy đủ — 8 model × 2 dataset

### Kaggle-Rx Testing (780 ảnh, in-domain — đơn thuốc), xếp theo CER tăng dần

| Model | CER (95% CI) | WER | Exact-match | Degenerate | Top-1 acc (78 lớp) |
|---|---|---|---|---|---|
| **trocr-lora-finetuned** | **0,149 [0,132–0,166]** | 0,402 | **60,0%** | 0,0% | **89,9%** |
| qwen2.5-vl-3b | 0,434 [0,372–0,501] | 0,912 | 28,2% | 0,0% | 81,5% |
| got-ocr2.0 | 0,479 [0,446–0,513] | 1,238 | 15,8% | 0,0% | 67,2% |
| easyocr | 0,552 [0,529–0,574] | 1,117 | 10,4% | 3,3% | 54,9% |
| trocr-large-handwritten (zero-shot) | 0,580 [0,555–0,606] | 1,587 | 8,1% | 0,0% | 79,5% |
| tesseract | 0,625 [0,601–0,648] | 1,304 | 7,3% | 1,4% | 49,2% |
| donut-base-synthdog | 1,000 [1,000–1,000] | 1,000 | 0,0% | 100,0% | 1,3% |
| donut-base-synthdog-padded | 4,201 [3,310–5,171] | 1,580 | 1,3% | 77,8% | 3,6% |

### IAM (400 ảnh, out-of-domain — chữ viết tay tổng quát), xếp theo CER tăng dần

| Model | CER (95% CI) | WER | Exact-match | Degenerate |
|---|---|---|---|---|
| got-ocr2.0 | 0,386 [0,275–0,581] | 0,518 | 53,8% | 0,3% |
| trocr-large-handwritten (zero-shot) | 0,441 [0,364–0,515] | 0,570 | 57,3% | 0,0% |
| trocr-lora-finetuned | 0,524 [0,472–0,579] | 0,745 | 25,8% | 0,0% |
| easyocr | 0,736 [0,703–0,766] | 1,105 | 4,8% | 33,3% |
| tesseract | 0,836 [0,787–0,886] | 1,248 | 6,3% | 1,8% |
| donut-base-synthdog | 1,000 [1,000–1,000] | 1,000 | 0,0% | 99,8% |
| donut-base-synthdog-padded | 1,058 [0,993–1,184] | 0,998 | 0,3% | 96,5% |
| qwen2.5-vl-3b | 1,100 [0,471–1,960] | 0,613 | 59,3% | 3,5% |

*(Top-1 accuracy chỉ có ý nghĩa trên kaggle_rx — IAM là từ vựng mở, không có 78 nhãn cố định.)*

## 2. Phát hiện đầu tiên (headline): fine-tuned TrOCR vượt qua MỌI model zero-shot trên domain mục tiêu

Ở Phase 5, model tốt nhất zero-shot trên kaggle_rx là **Qwen2.5-VL-3B** (CER 0,434), vượt qua cả GOT-OCR2.0
vốn chuyên OCR. Sau khi fine-tune LoRA, **TrOCR-large-handwritten (558M) vượt qua Qwen2.5-VL-3B (3B, gấp ~5,4
lần tham số)**:

| So sánh (paired, kaggle_rx) | CER model A | CER model B | Wilcoxon p |
|---|---|---|---|
| trocr-lora-finetuned vs qwen2.5-vl-3b | 0,149 | 0,434 | **p = 1,81×10⁻⁴⁸** |
| trocr-lora-finetuned vs got-ocr2.0 | 0,149 | 0,479 | **p = 3,08×10⁻⁸⁶** |

Cả hai đều cực kỳ có ý nghĩa thống kê — đây là bằng chứng mạnh nhất cho luận điểm trung tâm của bài báo:
**một model chuyên biệt nhỏ, fine-tune nhẹ (LoRA, <1% tham số cập nhật) trên phần cứng miễn phí (Kaggle T4)
có thể vượt qua VLM tổng quát lớn hơn nhiều trên domain mục tiêu**, không cần compute của các phòng lab lớn.

## 3. Domain-shift gap (CER trên IAM trừ CER trên kaggle_rx) — mọi model, xếp tăng dần

| Model | CER kaggle_rx | CER iam | Gap (iam − kaggle_rx) |
|---|---|---|---|
| donut-base-synthdog-padded | 4,201 | 1,058 | −3,143 (kaggle_rx tệ hơn hẳn — hallucination) |
| trocr-large-handwritten (zero-shot) | 0,580 | 0,441 | −0,139 |
| got-ocr2.0 | 0,479 | 0,386 | −0,093 |
| donut-base-synthdog | 1,000 | 1,000 | 0,000 (fail đều cả 2, xem phase2) |
| easyocr | 0,552 | 0,736 | +0,184 |
| tesseract | 0,625 | 0,836 | +0,211 |
| **trocr-lora-finetuned** | 0,149 | 0,524 | **+0,376** |
| qwen2.5-vl-3b | 0,434 | 1,100 | +0,667 (xem lưu ý Mục 4) |

**Diễn giải:** các model zero-shot **chuyên OCR/HTR nói chung** (TrOCR zero-shot, GOT-OCR2.0) thực ra làm
**tốt hơn một chút** trên IAM so với kaggle_rx (gap âm) — hợp lý vì IAM (chữ viết tay tiếng Anh phổ thông)
gần với phân bố pretraining của chúng hơn là từ vựng dược phẩm hiếm gặp. Sau khi fine-tune trên kaggle_rx,
gap của TrOCR đảo chiều mạnh (từ −0,139 thành +0,376) — bằng chứng số hoá trực tiếp cho hiện tượng
**đặc biệt-hoá (specialization) đánh đổi lấy tổng quát-hoá**, đúng như phát hiện catastrophic forgetting
đã nêu ở Phase 7-8.

## 4. Lưu ý quan trọng về phương pháp: mean CER có thể gây hiểu lầm với phân phối đuôi dài (heavy-tailed)

Kiểm tra thêm median CER trên IAM cho thấy một chi tiết quan trọng, không thấy được nếu chỉ nhìn mean:

| Model (IAM) | Mean CER | Median CER |
|---|---|---|
| qwen2.5-vl-3b | 1,100 | **0,000** |
| got-ocr2.0 | 0,386 | **0,000** |
| trocr-large-handwritten (zero-shot) | 0,441 | **0,000** |
| trocr-lora-finetuned | 0,524 | **0,429** |

Ba model zero-shot có **median = 0** trên IAM — tức là **đa số ảnh được nhận dạng hoàn hảo**, mean bị kéo lệch
hẳn lên bởi một nhóm nhỏ ảnh thất bại nặng (đuôi phân phối dài, đặc biệt rõ với Qwen2.5-VL: mean 1,10 nhưng
median 0). Đây là lý do phép kiểm định **Wilcoxon signed-rank (dựa trên rank, bền với outlier)** giữa
Qwen2.5-VL và GOT-OCR2.0 trên IAM cho **p = 0,330 — KHÔNG có ý nghĩa thống kê**, dù chênh lệch mean (1,10 vs
0,39) trông rất lớn. **Kết luận đúng đắn**: không thể khẳng định GOT-OCR2.0 "tốt hơn" Qwen2.5-VL trên IAM chỉ
dựa vào so sánh mean — cần báo cáo cả median và kết quả kiểm định trong bài báo, không chỉ mean ± CI.

Ngược lại, model **fine-tuned có median CER = 0,429** trên IAM (không phải 0) — nghĩa là sự suy giảm không
đến từ vài outlier, mà là **suy giảm thật, lan rộng trên đa số ảnh** → củng cố thêm (không phải làm yếu đi)
kết luận catastrophic forgetting ở Phase 7-8: đây là hiệu ứng hệ thống, không phải nhiễu thống kê.

## 5. Toàn bộ số liệu Wilcoxon đã chạy

| model_a | model_b | dataset | n | CER a | CER b | p-value | Ý nghĩa (α=0,05) |
|---|---|---|---|---|---|---|---|
| trocr-large-handwritten | trocr-lora-finetuned | kaggle_rx | 780 | 0,580 | 0,149 | 6,44×10⁻¹¹⁰ | Có ý nghĩa |
| trocr-large-handwritten | trocr-lora-finetuned | iam | 400 | 0,441 | 0,524 | 4,27×10⁻⁵ | Có ý nghĩa |
| qwen2.5-vl-3b | got-ocr2.0 | kaggle_rx | 780 | 0,434 | 0,479 | 9,57×10⁻²⁵ | Có ý nghĩa |
| qwen2.5-vl-3b | got-ocr2.0 | iam | 400 | 1,100 | 0,386 | 0,330 | **Không** có ý nghĩa |
| trocr-lora-finetuned | qwen2.5-vl-3b | kaggle_rx | 780 | 0,149 | 0,434 | 1,81×10⁻⁴⁸ | Có ý nghĩa |
| trocr-lora-finetuned | got-ocr2.0 | kaggle_rx | 780 | 0,149 | 0,479 | 3,08×10⁻⁸⁶ | Có ý nghĩa |
| trocr-lora-finetuned | got-ocr2.0 | iam | 400 | 0,524 | 0,386 | 7,67×10⁻¹⁴ | Có ý nghĩa |

## 6. Tóm tắt cho phần Kết quả của bài báo

1. **Đóng góp chính**: LoRA fine-tune TrOCR-large-handwritten (558M) trên 3.120 ảnh giúp nó vượt qua mọi
   model zero-shot kể cả VLM tổng quát lớn hơn (Qwen2.5-VL-3B, GOT-OCR2.0) trên domain mục tiêu, với ý nghĩa
   thống kê rất mạnh (p<10⁻⁴⁸ cho mọi so sánh liên quan).
2. **Giới hạn cần nêu trung thực**: cải thiện này đi kèm suy giảm tổng quát hoá có ý nghĩa thống kê và mang
   tính hệ thống (không phải outlier) trên chữ viết tay ngoài domain (IAM).
3. **Lưu ý phương pháp cho Discussion**: một số model (đặc biệt VLM tổng quát như Qwen2.5-VL) có phân phối
   CER đuôi dài — nên báo cáo median song song với mean, và ưu tiên kiểm định phi tham số (Wilcoxon) khi so
   sánh model thay vì chỉ nhìn khoảng tin cậy của mean.

## 7. Việc còn lại

- Phase 10: phân tích lỗi định tính (fine-tune sửa được loại lỗi nào — nhầm tên gần giống, hallucination,
  degenerate — và loại lỗi nào mới xuất hiện trên IAM sau fine-tune).
- Cân nhắc thêm phụ lục "outlier-robustness" (mục 4 ở trên) vào bài báo chính thay vì chỉ phụ lục, vì đây là
  một đóng góp phương pháp luận nhỏ nhưng thật (nhiều bài OCR benchmark khác chỉ báo cáo mean, có thể đã bỏ
  sót hiện tượng tương tự).
