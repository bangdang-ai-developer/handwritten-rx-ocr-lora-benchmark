# Phase 9 — Phân tích định lượng tổng hợp (toàn bộ model, zero-shot + fine-tuned)

> **⚠️ Cập nhật (17/09/2026):** Đã chạy lại toàn bộ Phase 9 sau khi ablation (Phase 7b) chọn **r=32** làm
> cấu hình fine-tune chính thức (thay r=16 ban đầu) — mọi số liệu "trocr-lora-finetuned" dưới đây là **r=32**.
> Không cần GPU để làm lại (chỉ đổi tên model trong `results_master_combined.csv` rồi chạy lại
> `src/analyze_aggregate.py`) — kết luận tổng thể không đổi, chỉ các con số cụ thể mạnh hơn.

Script tái lập: `src/analyze_aggregate.py`. Bootstrap CI 95% (1.000 resample, seed=0), Wilcoxon signed-rank
paired trên đúng từng ảnh (mọi model đều chạy trên cùng manifest, seed=42).

## 1. Bảng tổng hợp đầy đủ — 8 model × 2 dataset

### Kaggle-Rx Testing (780 ảnh, in-domain — đơn thuốc), xếp theo CER tăng dần

| Model | CER (95% CI) | WER | Exact-match | Degenerate | Top-1 acc (78 lớp) |
|---|---|---|---|---|---|
| **trocr-lora-finetuned (r=32)** | **0,114 [0,100–0,130]** | 0,319 | **68,1%** | 0,0% | **92,4%** |
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
| **trocr-lora-finetuned (r=32)** | **0,501 [0,453–0,552]** | 0,733 | 26,8% | 0,0% |
| easyocr | 0,736 [0,703–0,766] | 1,105 | 4,8% | 33,3% |
| tesseract | 0,836 [0,787–0,886] | 1,248 | 6,3% | 1,8% |
| donut-base-synthdog | 1,000 [1,000–1,000] | 1,000 | 0,0% | 99,8% |
| donut-base-synthdog-padded | 1,058 [0,993–1,184] | 0,998 | 0,3% | 96,5% |
| qwen2.5-vl-3b | 1,100 [0,471–1,960] | 0,613 | 59,3% | 3,5% |

*(3 cấu hình ablation — r=8, r=16, r=16-không-elastic — không đưa vào bảng chính này để giữ so sánh
"model chuyên biệt vs VLM tổng quát" gọn; xem đầy đủ ở `results/phase7b_ablation_summary.md`.)*

## 2. Phát hiện đầu tiên (headline): fine-tuned TrOCR vượt qua MỌI model zero-shot trên domain mục tiêu

Ở Phase 5, model tốt nhất zero-shot trên kaggle_rx là **Qwen2.5-VL-3B** (CER 0,434), vượt qua cả GOT-OCR2.0.
Sau khi fine-tune LoRA (r=32), **TrOCR-large-handwritten (558M) vượt qua Qwen2.5-VL-3B (3B, gấp ~5,4 lần
tham số) với khoảng cách còn LỚN HƠN so với kết quả r=16 trước đây**:

| So sánh (paired, kaggle_rx) | CER model A | CER model B | Wilcoxon p |
|---|---|---|---|
| trocr-lora-finetuned (r=32) vs qwen2.5-vl-3b | 0,114 | 0,434 | **p = 4,43×10⁻⁵⁹** |
| trocr-lora-finetuned (r=32) vs got-ocr2.0 | 0,114 | 0,479 | **p = 8,00×10⁻⁹³** |

Cả hai đều cực kỳ có ý nghĩa thống kê — bằng chứng mạnh nhất cho luận điểm trung tâm bài báo: **một model
chuyên biệt nhỏ, fine-tune nhẹ (LoRA r=32, ~1-2% tham số cập nhật) trên phần cứng miễn phí (Kaggle T4) có
thể vượt qua VLM tổng quát lớn hơn nhiều trên domain mục tiêu**.

## 3. Domain-shift gap (CER trên IAM trừ CER trên kaggle_rx) — mọi model, xếp tăng dần

| Model | CER kaggle_rx | CER iam | Gap (iam − kaggle_rx) |
|---|---|---|---|
| donut-base-synthdog-padded | 4,201 | 1,058 | −3,143 |
| trocr-large-handwritten (zero-shot) | 0,580 | 0,441 | −0,139 |
| got-ocr2.0 | 0,479 | 0,386 | −0,093 |
| donut-base-synthdog | 1,000 | 1,000 | 0,000 |
| easyocr | 0,552 | 0,736 | +0,184 |
| tesseract | 0,625 | 0,836 | +0,211 |
| **trocr-lora-finetuned (r=32)** | 0,114 | 0,501 | **+0,387** |
| qwen2.5-vl-3b | 0,434 | 1,100 | +0,667 (xem lưu ý Mục 4) |

**Diễn giải không đổi so với r=16**: các model zero-shot chuyên OCR/HTR (TrOCR zero-shot, GOT-OCR2.0) làm
tốt hơn một chút trên IAM so với kaggle_rx (gap âm). Sau fine-tune, gap của TrOCR đảo chiều mạnh (từ −0,139
thành +0,387) — bằng chứng số hoá cho đặc biệt-hoá đánh đổi lấy tổng quát-hoá.

**Lưu ý (tránh nhầm lẫn với Mục 4/`phase7_summary.md`)**: gap tuyệt đối của r=32 (+0,387) thực ra **nhích cao
hơn** gap của r=16 (+0,376) — vì r=32 cải thiện kaggle_rx (CER giảm từ 0,149→0,114) MẠNH HƠN mức cải thiện
trên IAM. Điều này KHÔNG mâu thuẫn với kết luận "r=32 tốt hơn r=16 trên cả 2 tiêu chí" ở `phase7_summary.md`
— r=32 có CER TUYỆT ĐỐI thấp hơn r=16 trên CẢ 2 dataset riêng lẻ (0,114<0,149 và 0,501<0,524), chỉ là *hiệu
số* giữa 2 dataset (gap) hơi lớn hơn vì mẫu số 2 phía cải thiện không đều nhau. Hai cách đọc số liệu (CER
tuyệt đối vs. gap tương đối) đo 2 câu hỏi khác nhau — bài báo nên ưu tiên báo cáo CER tuyệt đối từng dataset
(Mục 1) làm kết luận chính, dùng gap chỉ để minh hoạ xu hướng "đặc biệt hoá" chứ không dùng để so sánh giữa
các rank.

## 4. Lưu ý quan trọng về phương pháp: mean CER có thể gây hiểu lầm với phân phối đuôi dài (heavy-tailed)

*(Không đổi so với bản trước — không phụ thuộc vào lựa chọn rank, giữ nguyên các model không liên quan
LoRA fine-tune.)*

| Model (IAM) | Mean CER | Median CER |
|---|---|---|
| qwen2.5-vl-3b | 1,100 | **0,000** |
| got-ocr2.0 | 0,386 | **0,000** |
| trocr-large-handwritten (zero-shot) | 0,441 | **0,000** |
| trocr-lora-finetuned (r=32) | 0,501 | **0,400** |

Ba model zero-shot có **median = 0** trên IAM — đa số ảnh nhận dạng hoàn hảo, mean bị kéo lệch bởi đuôi
phân phối dài (đặc biệt Qwen2.5-VL). Wilcoxon giữa Qwen2.5-VL và GOT-OCR2.0 trên IAM: **p = 0,330 — KHÔNG có
ý nghĩa** dù chênh mean trông rất lớn (1,10 vs 0,39) — không thể khẳng định GOT-OCR2.0 "tốt hơn" chỉ dựa
vào mean.

Model **fine-tuned (r=32) có median CER = 0,400** trên IAM (không phải 0, và thực tế **thấp hơn** median
0,429 của r=16) — suy giảm tổng quát hoá vẫn là hiệu ứng **hệ thống, lan rộng** (không chỉ outlier), nhưng
**nhẹ hơn một chút** so với r=16 — nhất quán với r=32 có CER mean thấp hơn r=16 trên IAM (0,501 vs 0,524).

## 5. Toàn bộ số liệu Wilcoxon đã chạy

| model_a | model_b | dataset | n | CER a | CER b | p-value | Ý nghĩa (α=0,05) |
|---|---|---|---|---|---|---|---|
| trocr-large-handwritten | trocr-lora-finetuned (r=32) | kaggle_rx | 780 | 0,580 | 0,114 | 6,97×10⁻¹¹¹ | Có ý nghĩa |
| trocr-large-handwritten | trocr-lora-finetuned (r=32) | iam | 400 | 0,441 | 0,501 | 3,18×10⁻⁴ | Có ý nghĩa |
| qwen2.5-vl-3b | got-ocr2.0 | kaggle_rx | 780 | 0,434 | 0,479 | 9,57×10⁻²⁵ | Có ý nghĩa |
| qwen2.5-vl-3b | got-ocr2.0 | iam | 400 | 1,100 | 0,386 | 0,330 | **Không** có ý nghĩa |
| trocr-lora-finetuned (r=32) | qwen2.5-vl-3b | kaggle_rx | 780 | 0,114 | 0,434 | 4,43×10⁻⁵⁹ | Có ý nghĩa |
| trocr-lora-finetuned (r=32) | got-ocr2.0 | kaggle_rx | 780 | 0,114 | 0,479 | 8,00×10⁻⁹³ | Có ý nghĩa |
| trocr-lora-finetuned (r=32) | got-ocr2.0 | iam | 400 | 0,501 | 0,386 | 2,34×10⁻¹³ | Có ý nghĩa |

## 6. Tóm tắt cho phần Kết quả của bài báo

1. **Đóng góp chính**: LoRA fine-tune (r=32) TrOCR-large-handwritten (558M) trên 3.120 ảnh giúp nó vượt qua
   mọi model zero-shot kể cả VLM tổng quát lớn hơn (Qwen2.5-VL-3B, GOT-OCR2.0) trên domain mục tiêu, với ý
   nghĩa thống kê rất mạnh (p<10⁻⁵⁸ cho mọi so sánh liên quan) — CER giảm 80% so với zero-shot.
2. **Giới hạn cần nêu trung thực**: cải thiện này đi kèm suy giảm tổng quát hoá có ý nghĩa thống kê (p<0,001)
   trên chữ viết tay ngoài domain (IAM), tuy nhẹ hơn cấu hình r=16 ban đầu.
3. **Lưu ý phương pháp cho Discussion**: một số model (đặc biệt VLM tổng quát) có phân phối CER đuôi dài —
   nên báo cáo median song song với mean.
