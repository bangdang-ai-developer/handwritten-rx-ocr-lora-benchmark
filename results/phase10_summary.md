# Phase 10 — Phân tích lỗi định tính (fine-tune sửa được loại lỗi nào, tạo ra loại lỗi nào mới)

> **⚠️ Cập nhật (17/09/2026):** Đã chạy lại hoàn toàn với **r=32** (cấu hình chính thức sau ablation, xem
> `phase7b_ablation_summary.md`) thay cho r=16. Kết luận tổng thể giữ nguyên, nhưng mức độ đánh đổi an toàn
> lâm sàng ở Mục 2 **nhẹ hơn** so với bản r=16 trước đây.

Script tái lập: `src/analyze_errors.py`. Phân loại lỗi dựa trên `results_master_combined.csv`, so sánh
**trocr-large-handwritten (zero-shot)** với **trocr-lora-finetuned (r=32)**, cả 2 dataset.

## 1. Đơn vị phân tích: 4 loại lỗi trên Kaggle-Rx (đóng, 78 tên thuốc)

| Loại | Định nghĩa |
|---|---|
| `correct` | Khớp chuỗi chính xác (`exact_match`) |
| `minor_ocr_noise_still_correct_drug` | Chuỗi không khớp tuyệt đối, nhưng ánh xạ về gần nhất trong 78 tên thuốc thì **đúng thuốc** |
| `confusable_wrong_drug` | Sai thuốc, và output **giống rõ một tên thuốc khác có thật** (khoảng cách edit chuẩn hoá ≤0,34) — **loại lỗi nguy hiểm nhất về mặt lâm sàng** |
| `hallucination_far_off` | Sai thuốc, và output không giống bất kỳ tên thuốc thật nào |

## 2. Kết quả — Kaggle-Rx (in-domain)

| Loại lỗi | Zero-shot | Fine-tuned (r=32) | Thay đổi | (so với r=16 trước đây) |
|---|---|---|---|---|
| `correct` | 8,08% | **68,08%** | +60,0 điểm % | (r=16: 60,00%) |
| `minor_ocr_noise_still_correct_drug` | 71,79% | 24,87% | −47,0 điểm % | (r=16: 30,77%) |
| `hallucination_far_off` | 18,59% | **4,49%** | −14,1 điểm % (giảm ~4,1 lần) | (r=16: 5,77%, giảm ~3,2 lần) |
| `confusable_wrong_drug` | 1,54% | **2,56%** | +1,0 điểm % (tăng ~1,7 lần) | (r=16: 3,46%, tăng ~2,2 lần) |

**Diễn giải chính:** cùng xu hướng như r=16 — fine-tuning dịch chuyển phân phối lỗi mạnh về phía "đúng hoàn
toàn" và giảm hẳn lỗi đoán bừa, nhưng **tăng nhẹ** tỷ lệ lỗi giống-thuốc-khác. Tuy nhiên r=32 có mức đánh đổi
**NHẸ HƠN r=16 trên cả 2 chiều**: giảm hallucination mạnh hơn (4,1 lần so với 3,2 lần) VÀ tăng confusable ít
hơn (tăng 1,7 lần so với 2,2 lần) — r=32 vừa cải thiện in-domain mạnh hơn, vừa có hồ sơ an toàn tốt hơn r=16,
không có đánh đổi giữa 2 mục tiêu này.

**Vẫn cần cảnh báo cho Thảo luận/An toàn lâm sàng**: dù nhẹ hơn r=16, tỷ lệ `confusable_wrong_drug` vẫn
**tăng so với zero-shot** (1,54%→2,56%) — kết luận về nguyên lý (lỗi còn sót lại có xu hướng "giống thuốc
thật" hơn là "rõ ràng vô nghĩa" khi model học tốt hơn) **vẫn đúng**, chỉ khác về độ lớn.

### Ví dụ cụ thể (fine-tuned r=32, kaggle_rx)

| reference | hypothesis | top1_pred (gần nhất) | CER | Loại |
|---|---|---|---|---|
| Amodis | Amodin | Axodin | 0,167 | confusable_wrong_drug |
| Dinafex | Fixal | Fixal | 0,714 | confusable_wrong_drug (output khớp đúng 1 thuốc khác) |
| Etizin | Eitrin | Zithrin | 0,500 | confusable_wrong_drug |
| Alatrol | KKKKK | Beklo | 1,000 | hallucination_far_off |
| Bacmax | Bac | Baclon | 0,500 | hallucination_far_off |
| Aceta | Acata | Aceta | 0,200 | minor_ocr_noise_still_correct_drug |

Nhóm tên thuốc "Bac-" (Bacaid, Backtone, Baclofen, Baclon, Bacmax) vẫn là điểm nghẽn khó nhất, nhất quán
với phát hiện ở bản r=16.

## 3. Kết quả — IAM (out-of-domain, chữ viết tay tổng quát)

| Mức lỗi (theo CER) | Zero-shot | Fine-tuned (r=32) | (so với r=16 trước đây) |
|---|---|---|---|
| `correct` (exact match) | 57,25% | 26,75% | (r=16: 25,75%) |
| `minor_error` (0<CER≤0,3) | 6,75% | 13,00% | (r=16: 13,50%) |
| `moderate_error` (0,3<CER≤0,7) | 12,75% | **32,75%** | (r=16: 32,00%) |
| `severe_error` (CER>0,7) | 23,25% | 27,50% | (r=16: 28,75%) |

**Diễn giải không đổi**: catastrophic forgetting biểu hiện chủ yếu qua dịch chuyển "đúng tuyệt đối" sang
"lỗi mức trung bình", không phải sụp đổ toàn phần — r=32 có phân bố gần như tương đương r=16, chỉ nhẹ hơn
một chút ở severe_error (27,50% so với 28,75%).

## 4. Kiểm tra an toàn: model fine-tuned có "rò rỉ" tên thuốc khi đọc chữ thường (IAM) không?

**Kết quả: 2/400 (0,5%)** (so với 1/400 = 0,25% ở r=16, và 0/400 ở zero-shot) — vẫn ở mức **rất thấp/không
đáng kể**, hai trường hợp:
- nhãn thật "at" (2 ký tự) bị đọc thành "Az" (trùng tên thuốc "Az" trong vocab) — nhiều khả năng trùng hợp
  ngẫu nhiên trên từ ngắn, giống pattern đã thấy ở r=16.
- nhãn thật "reason" bị đọc thành "Nexum" — CER=0,833, khác biệt lớn, có thể là 1 trường hợp thật của việc
  model "nhớ nhầm" ra tên thuốc khi gặp từ dài/khó đọc, nhưng chỉ 1/400 nên không đủ để kết luận đây là hiện
  tượng hệ thống.

**Kết luận không đổi**: không có bằng chứng đáng kể cho hiện tượng "rò rỉ tên thuốc" ở quy mô ảnh hưởng tới
kết luận chung, dù tỷ lệ tăng nhẹ (1 → 2 trường hợp) khi đổi từ r=16 sang r=32 — đáng ghi chú như 1 quan sát
nhỏ trong Limitations, không phải một finding chính.

## 5. Tóm tắt cho bài báo

1. Fine-tuning (r=32) dịch chuyển phân phối lỗi trên domain mục tiêu theo hướng tích cực rõ rệt hơn cả r=16:
   giảm hallucination_far_off mạnh hơn (4,1 lần) và tăng confusable_wrong_drug ít hơn (1,7 lần) — r=32 có hồ
   sơ an toàn tốt hơn r=16 trên đúng chiều đo này, không chỉ tốt hơn về CER trung bình.
2. Catastrophic forgetting trên IAM: bức tranh gần như không đổi so với r=16 (dịch từ "đúng tuyệt đối" sang
   "lỗi trung bình" là hiệu ứng chính, không phải sụp đổ hoàn toàn).
3. Không tìm thấy bằng chứng "rò rỉ" tên thuốc có ý nghĩa hệ thống ở cả r=16 và r=32.

## 6. Việc còn lại

- Phase 11 (tuỳ chọn): kiểm tra bổ sung RxHandBD.
- Phase 12-13: viết bài theo cấu trúc tạp chí (PeerJ Computer Science), nộp arXiv trước, sau đó nộp chính
  thức — tổng hợp toàn bộ Phase 1-10 (bao gồm ablation Phase 7b) làm phần Kết quả + Thảo luận.
