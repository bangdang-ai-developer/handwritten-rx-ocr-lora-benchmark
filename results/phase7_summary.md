# Phase 7-8 — LoRA Fine-tune TrOCR-large-handwritten + Đánh giá trước/sau (KẾT QUẢ THẬT)

## 1. Cấu hình huấn luyện đã chạy

- Model nền: `microsoft/trocr-large-handwritten` (558M tham số), patch qua `peft.LoraConfig`:
  `r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["query","value","q_proj","v_proj"], bias="none"`.
- Train: **3.120 ảnh** (`Training` split, đúng số thật — xem `phase6_summary.md`), augmentation Albumentations
  (Affine, ElasticTransform, GaussNoise, GaussianBlur, RandomBrightnessContrast, Morphological erosion/dilation).
- Theo dõi trong lúc train: subsample 300/780 ảnh `Validation`.
- `Seq2SeqTrainingArguments`: batch 8 × grad-accum 2 (effective 16), `fp16=True` (bắt buộc trên T4/Turing,
  không dùng bf16), `lr=2e-4` cosine, `num_train_epochs=15`, `early_stopping_patience=5` theo `eval_cer`.
- Kernel Kaggle T4×2 (`dangbang1/ocr-med-benchmark-phase1`), **kernel v20** — chạy thành công hoàn chỉnh sau
  4 lần sửa lỗi kỹ thuật liên tiếp ở các kernel v15-v19 (xem lịch sử commit git): thiếu
  `decoder_start_token_id`/`pad_token_id`/`eos_token_id`, xung đột `peft` với thuộc tính `vocab_size` của
  `VisionEncoderDecoderConfig` (phải monkeypatch thành class-level property), `PeftModelForSeq2SeqLM.generate()`
  chỉ nhận `pixel_values` dạng keyword argument, và cuối cùng là `OverflowError` khi decode giá trị đệm `-100`
  trong `pred_ids` lúc Trainer ghép các eval batch có độ dài sinh khác nhau.

## 2. Lịch sử huấn luyện (kernel v20)

Early-stopped ở **epoch 7** (patience 5 không cải thiện thêm sau epoch 2):

| Epoch | eval_cer (300 ảnh validation) | eval_loss |
|---|---|---|
| 1 | 0.1858 | 1.4946 |
| **2 (tốt nhất)** | **0.1233** | 0.4423 |
| 3 | 0.2043 | 0.2221 |
| 4 | 0.1736 | 0.1381 |
| 5 | 0.2015 | 0.1065 |
| 6 | 0.1657 | 0.0817 |
| 7 | 0.1876 | 0.0564 |

Adapter cuối cùng lưu lại (`model.save_pretrained()`) là **checkpoint tốt nhất (epoch 2)** nhờ
`load_best_model_at_end=True` — không phải checkpoint cuối cùng. `eval_loss` tiếp tục giảm sau epoch 2 (dấu hiệu
overfit nhẹ vào train set khi train lâu hơn), trong khi `eval_cer` không cải thiện thêm — đúng lý do cần theo dõi
CER (metric mục tiêu thật) thay vì chỉ loss khi early-stop.

*Ghi chú tái lập:* một lần chạy trước đó (kernel v18, cùng seed=42, cùng cấu hình, nhưng bị lỗi ở bước eval cuối
nên phải chạy lại từ đầu) hội tụ tốt nhất ở epoch 5 với CER 0.127 — cùng độ lớn nhưng khác epoch chính xác so với
lần này (epoch 2, CER 0.123). Chênh lệch này đến từ các nguồn không tất định còn sót lại (thứ tự dữ liệu qua
`DataLoader` trên 2 GPU song song, dropout) chứ không phải seed sai — hai lần chạy vẫn hội tụ về **cùng một mức
CER tối ưu (~0.12-0.13)**, nên kết luận về độ lớn cải thiện là ổn định qua các lần chạy.

## 3. Kết quả trên frozen test set (KẾT QUẢ CHÍNH — dùng cho bài báo)

So sánh **paired** (đúng từng ảnh, n khớp 100%) giữa TrOCR-large-handwritten **zero-shot** (Phase 2) và
**fine-tuned** (LoRA, adapter epoch 2), bootstrap CI 95% (1.000 resample, seed=0), Wilcoxon signed-rank test.

### 3.1. Kaggle-Rx Testing (780 ảnh, in-domain — đơn thuốc)

| Metric | Zero-shot | Fine-tuned | Thay đổi |
|---|---|---|---|
| CER (mean, 95% CI) | 0.580 [0.555–0.606] | **0.149 [0.132–0.166]** | **−74,4%** |
| WER (mean) | 1.587 | 0.402 | −74,7% |
| Exact-match | 8,1% | **60,0%** | +51,9 điểm % |
| Top-1 accuracy (78 lớp thuốc) | — | **89,9%** | (không đo ở zero-shot) |

**Wilcoxon signed-rank (CER, paired, n=780): p = 6,44×10⁻¹¹⁰** — khác biệt cực kỳ có ý nghĩa thống kê,
không thể là ngẫu nhiên.

### 3.2. IAM (400 ảnh, out-of-domain — kiểm tra catastrophic forgetting)

| Metric | Zero-shot | Fine-tuned | Thay đổi |
|---|---|---|---|
| CER (mean, 95% CI) | 0.441 [0.364–0.515] | **0.524 [0.472–0.579]** | **+18,8% (TỆ HƠN)** |
| WER (mean) | 0.570 | 0.745 | +30,7% (tệ hơn) |
| Exact-match | 57,3% | 25,8% | −31,5 điểm % |

**Wilcoxon signed-rank (CER, paired, n=400): p = 4,27×10⁻⁵** — khác biệt cũng có ý nghĩa thống kê, tức là
**không phải nhiễu ngẫu nhiên**: model fine-tune thật sự tệ hơn zero-shot trên chữ viết tay tổng quát.

## 4. Diễn giải — hai phát hiện, cả hai đều cần báo cáo trung thực

**(a) Cải thiện in-domain rất lớn và rất vững chắc.** LoRA fine-tune trên chỉ 3.120 ảnh giảm CER tới 74%,
đưa exact-match từ 8% lên 60%, và đạt 89,9% top-1 accuracy khi ánh xạ về 78 tên thuốc thật — đây là bằng chứng
phương pháp chính (trụ cột đóng góp Q2), khớp độ lớn cải thiện mà literature tham khảo (DLoRA-TrOCR và các bài
tương tự) đã báo cáo.

**(b) Catastrophic forgetting có ý nghĩa thống kê trên IAM — không thể bỏ qua.** Model fine-tune quá đặc
biệt hoá vào domain hẹp (từ đơn lẻ, thường ngắn, từ vựng dược phẩm) khiến khả năng đọc chữ viết tay tổng quát
(IAM) **suy giảm thật**, không phải sai số ngẫu nhiên (p<0,0001). Đây là phát hiện quan trọng cho phần Thảo luận
của bài báo: LoRA với rank thấp (r=16) trên tập nhỏ vẫn có thể đánh đổi tổng quát hoá lấy hiệu năng in-domain,
dù chỉ cập nhật <1% tham số — phản bác giả định thường gặp rằng PEFT "an toàn hơn" full fine-tune về mặt giữ
kiến thức gốc. Hướng giảm thiểu khả dĩ cho future work: trộn một phần nhỏ dữ liệu IAM vào quá trình fine-tune
(multi-task), hoặc rank thấp hơn nữa (r=8, đã có trong kế hoạch ablation ban đầu nhưng chưa chạy do giới hạn
thời gian).

## 5. Việc còn lại

- Ablation rank (r=8 vs r=16 vs r=32) và A/B augmentation (có/không elastic) đã được thiết kế trong kế hoạch
  ban đầu nhưng **chưa chạy** do ưu tiên có kết quả chính trước trong ngân sách thời gian — có thể bổ sung nếu
  còn thời gian, nếu không thì báo cáo r=16 làm cấu hình duy nhất và ghi rõ đây là giới hạn (không phải ablation
  đầy đủ) trong phần Limitations.
- Cập nhật `README.md` với dòng kết quả fine-tune (đã làm, xem bảng "Tiến độ thực tế").
- Phase 9 (phân tích định lượng tổng hợp toàn bộ model), Phase 10 (phân tích lỗi định tính — fine-tune có sửa
  được lỗi hallucination/degenerate không, hay chỉ sửa lỗi thông thường).
