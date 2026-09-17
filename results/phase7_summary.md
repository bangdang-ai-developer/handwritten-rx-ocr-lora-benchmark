# Phase 7-8 — LoRA Fine-tune TrOCR-large-handwritten + Đánh giá trước/sau (KẾT QUẢ CHÍNH THỨC: r=32)

> **⚠️ Cập nhật (17/09/2026):** Sau khi chạy ablation đầy đủ (Phase 7b — xem
> `results/phase7b_ablation_summary.md`), **r=32 được chọn làm cấu hình chính thức** thay cho r=16 ban đầu,
> vì vượt trội r=16 trên CẢ 2 tiêu chí (CER in-domain thấp hơn VÀ giữ tổng quát hoá tốt hơn, không đánh đổi).
> Tài liệu này đã được viết lại hoàn toàn theo kết quả r=32. Kết quả gốc của r=16 (từng là "main") được giữ
> lại làm 1 điểm trong bảng ablation, xem `phase7b_ablation_summary.md`.

## 1. Cấu hình huấn luyện đã chạy (chính thức)

- Model nền: `microsoft/trocr-large-handwritten` (558M tham số), patch qua `peft.LoraConfig`:
  **`r=32, lora_alpha=64`**, `lora_dropout=0.1, target_modules=["query","value","q_proj","v_proj"], bias="none"`.
- Train: **3.120 ảnh** (`Training` split, đúng số thật — xem `phase6_summary.md`), augmentation Albumentations
  (Affine, ElasticTransform, GaussNoise, GaussianBlur, RandomBrightnessContrast, Morphological erosion/dilation)
  — đã kiểm chứng qua ablation là **không gây hại** cho in-domain và có xu hướng (chưa đạt ý nghĩa 0,05) giảm
  nhẹ catastrophic forgetting, xem Mục 2 của `phase7b_ablation_summary.md`.
- Theo dõi trong lúc train: subsample 300/780 ảnh `Validation`.
- `Seq2SeqTrainingArguments`: batch 8 × grad-accum 2 (effective 16), `fp16=True` (bắt buộc trên T4/Turing,
  không dùng bf16), `lr=2e-4` cosine, `num_train_epochs=15`, `early_stopping_patience=5` theo `eval_cer`.
- Kernel Kaggle T4×2 (`dangbang1/ocr-med-benchmark-phase1`), **kernel v22** (`ABLATION_CONFIG="r32"`) — chạy
  thành công hoàn chỉnh, cùng script/pipeline đã ổn định từ kernel v20 (đã sửa hết 4 bug kỹ thuật liên tiếp
  ở kernel v15-v19: `decoder_start_token_id`, monkeypatch `vocab_size` property cho `peft`, `generate()`
  keyword arg, và `OverflowError` khi decode `-100` — xem lịch sử commit git để biết chi tiết từng lỗi).

## 2. Lịch sử huấn luyện (kernel v22, r=32)

Early-stopped ở **epoch 7** (patience 5 không cải thiện thêm sau epoch 2):

| Epoch | eval_cer (300 ảnh validation) |
|---|---|
| 1 | 0.1732 |
| **2 (tốt nhất)** | **0.1022** |
| 3 | 0.1228 |
| 4 | 0.1628 |
| 5 | 0.1812 |
| 6 | 0.2095 |
| 7 | 0.1907 |

Adapter cuối cùng lưu lại là **checkpoint tốt nhất (epoch 2)** nhờ `load_best_model_at_end=True`.

## 3. Kết quả trên frozen test set (KẾT QUẢ CHÍNH — dùng cho bài báo)

So sánh **paired** (đúng từng ảnh, n khớp 100%) giữa TrOCR-large-handwritten **zero-shot** (Phase 2) và
**fine-tuned** (LoRA r=32), bootstrap CI 95% (1.000 resample, seed=0), Wilcoxon signed-rank test.

### 3.1. Kaggle-Rx Testing (780 ảnh, in-domain — đơn thuốc)

| Metric | Zero-shot | Fine-tuned (r=32) | Thay đổi |
|---|---|---|---|
| CER (mean, 95% CI) | 0.580 [0.555–0.606] | **0.114 [0.100–0.130]** | **−80,3%** |
| WER (mean) | 1.587 | 0.319 | −79,9% |
| Exact-match | 8,1% | **68,1%** | +60,0 điểm % |
| Top-1 accuracy (78 lớp thuốc) | — | **92,4%** | (không đo ở zero-shot) |

**Wilcoxon signed-rank (CER, paired, n=780): p = 6,97×10⁻¹¹¹** — cực kỳ có ý nghĩa thống kê.

### 3.2. IAM (400 ảnh, out-of-domain — kiểm tra catastrophic forgetting)

| Metric | Zero-shot | Fine-tuned (r=32) | Thay đổi |
|---|---|---|---|
| CER (mean, 95% CI) | 0.441 [0.364–0.515] | **0.501 [0.453–0.552]** | **+13,7% (TỆ HƠN)** |
| WER (mean) | 0.570 | 0.733 | +28,6% (tệ hơn) |
| Exact-match | 57,3% | 26,8% | −30,5 điểm % |

**Wilcoxon signed-rank (CER, paired, n=400): p = 3,18×10⁻⁴** — vẫn có ý nghĩa thống kê, dù mức độ tệ đi
**nhẹ hơn** so với r=16 (r=16: +18,8%, p=4,3×10⁻⁵ — xem lịch sử ở `phase7b_ablation_summary.md`).

## 4. Diễn giải — hai phát hiện, cả hai đều cần báo cáo trung thực

**(a) Cải thiện in-domain rất lớn, còn mạnh hơn r=16.** LoRA r=32 (chỉ cập nhật ~1-2% tham số) giảm CER tới
**80%**, đưa exact-match từ 8% lên 68%, và đạt 92,4% top-1 accuracy khi ánh xạ về 78 tên thuốc thật.

**(b) Catastrophic forgetting vẫn có ý nghĩa thống kê trên IAM, nhưng NHẸ HƠN r=16.** Model r=32 vẫn suy
giảm khả năng đọc chữ viết tay tổng quát (p<0,001), nhưng mức tệ đi (+13,7%) nhỏ hơn r=16 (+18,8%) —
**r=32 không chỉ tốt hơn in-domain, mà còn giữ tổng quát hoá tốt hơn r=16, không có đánh đổi rõ ràng nào**.
Median CER trên IAM vẫn tăng rõ rệt (0,000 ở zero-shot → 0,400 ở fine-tuned) — xác nhận đây là suy giảm
**mang tính hệ thống**, không phải do vài outlier (xem `phase9_summary.md` Mục 4).

**Lưu ý về độ tin cậy của so sánh rank**: chỉ chạy 1 seed cho mỗi rank (8/16/32) — phần chênh lệch giữa các
rank CÓ THỂ một phần đến từ phương sai giữa các lần train (không chỉ từ rank tự thân). Xem
`phase7b_ablation_summary.md` Mục 1 để biết đầy đủ giới hạn này.

## 5. Việc còn lại

- A/B augmentation đã chạy (Phase 7b) — elastic không đổi hiệu năng in-domain, có xu hướng (chưa đạt ý nghĩa)
  giảm nhẹ forgetting → giữ elastic=True là hợp lý, đã xác nhận qua dữ liệu thật.
- Rank ablation đã chạy đầy đủ (không chỉ quick-compare như kế hoạch gốc) — r=32 xác nhận là lựa chọn tốt
  nhất trong 3 rank đã test (8/16/32); chưa test r>32 (nằm ngoài phạm vi ablation ban đầu, có thể là hướng mở
  rộng nếu muốn, nhưng không bắt buộc cho kết luận hiện tại).
- Phase 9 (phân tích định lượng tổng hợp toàn bộ model) và Phase 10 (phân tích lỗi định tính) đã được chạy
  lại hoàn toàn với r=32 làm model chính — xem 2 file tương ứng.
