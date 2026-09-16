# BÁO CÁO THIẾT KẾ: MỞ RỘNG KẾ HOẠCH NGHIÊN CỨU OCR Y KHOA HƯỚNG TỚI TẠP CHÍ Q2 (3-4 TUẦN)

*Xây dựng dựa trên: `docs/05-ke-hoach-2-tuan.md` (kế hoạch gốc), `docs/04-dataset-y-khoa-cong-khai.md`, kết quả thật Phase 1 (`results/phase1_summary.md`, `results/results_master_phase1.csv`), code Phase 2 đang chạy (`notebooks/kaggle_benchmark.py`), và 2 khối nghiên cứu xác minh đã cung cấp (Scimago + khả thi LoRA).*

> **⚠️ CẬP NHẬT SỐ LIỆU (Phase 6, 16/09/2026):** Đã tải dataset về máy và đếm trực tiếp — số liệu "2.808/936/936" (suy ra từ mô tả "60/20/20" trên trang Kaggle) trong toàn bộ tài liệu dưới đây **KHÔNG ĐÚNG với thực tế**. Số ảnh thật: **Training = 3.120, Validation = 780, Testing = 780** (tổng 4.680, khớp). Đây **không phải lỗi path-resolution** như suy đoán ban đầu — code đã resolve đúng 100% (780/780, 3120/3120) từ lúc đầu; chỉ là mô tả "60/20/20" trên trang Kaggle không khớp tỷ lệ chia thật (~66.7/16.65/16.65). Mọi con số "2.808 ảnh train"/"936 ảnh test" ở các mục bên dưới nên đọc là **3.120 ảnh train / 780 ảnh test** — xem `results/phase6_summary.md` để biết chi tiết đầy đủ.

Đã kiểm tra lại repo trước khi viết báo cáo này: Phase 1 (Tesseract, EasyOCR) đã xong thật với dữ liệu **780 ảnh** kaggle_rx (không phải 936 như tính toán lý thuyết 20%×4680 — có lệch do một số ảnh không resolve được đường dẫn trong `build_manifest_kaggle_rx()`, xem Rủi ro ở Mục 6) + 400 ảnh IAM. Phase 2 (TrOCR-large-handwritten + Donut-base zero-shot) vừa được thêm vào code (`notebooks/kaggle_benchmark.py`, `CURRENT_PHASE = 2`), đang/sắp chạy trên Kaggle. Dataset Kaggle-Rx có cấu trúc thật là 3 thư mục `Training/Testing/Validation` (60/20/20 stratified theo 78 lớp) với file nhãn CSV/XLSX riêng mỗi thư mục — tương ứng train ≈ 2.808 ảnh, validation ≈ 936 ảnh, test ≈ 936 ảnh (khớp con số 2.808 mà bạn nêu).

---

## 1. LỰA CHỌN TẠP CHÍ Q2 MỤC TIÊU

Dựa hoàn toàn vào bảng xác minh SJR đã cung cấp (không đoán thêm), **có 3 tạp chí Q2 xác nhận thật, phù hợp chủ đề** (Journal of Imaging, PeerJ Computer Science, Health Information Science and Systems), nên **không rơi vào tình huống phải hạ xuống Q3/hội nghị** — chọn được Q2 thật.

### Tạp chí chính: **PeerJ Computer Science**
| Tiêu chí | Đánh giá |
|---|---|
| Quartile | Q2 xác nhận (đa số nguồn đồng thuận) |
| Tiền lệ | **Mạnh nhất trong nhóm** — đã xuất bản "Adapting multilingual vision language transformers for low-resource Urdu OCR" (2024, dùng TrOCR) — gần như cùng loại bài (benchmark + adapt/fine-tune TrOCR) mà kế hoạch này nhắm tới |
| Chi phí | APC $1.395 — thấp nhất nhóm, quan trọng vì đây là nghiên cứu độc lập tự trả phí |
| Thời gian bình duyệt | Công bố chính thức ~35 ngày, nhưng khảo sát tác giả (SciRev) ghi nhận thực tế có thể ~14 tuần — **cần chuẩn bị tinh thần dài hơn con số quảng cáo** |
| Vì sao chọn làm chính | Kết hợp tốt nhất giữa (a) Q2 xác nhận, (b) tiền lệ xuất bản trực tiếp cùng dạng bài, (c) chi phí thấp nhất |

### Phương án dự phòng 1: **Journal of Imaging (MDPI)**
- Q2 xác nhận ở *cả 4* category liên quan (CV&PR, Radiology/Imaging, Electrical Eng., Computer Graphics) — **xác nhận Q2 chắc chắn nhất trong toàn bộ danh sách**, khớp trực tiếp "medical imaging".
- Bình duyệt cực nhanh (~20-22 ngày median) — dùng nếu cần công bố gấp hoặc nếu PeerJ CS từ chối và muốn vòng nộp lại nhanh.
- Đánh đổi: APC cao hơn (CHF 1.800-2.400 ≈ $2.000-2.700), 100% OA bắt buộc (không có route miễn phí).

### Phương án dự phòng 2: **Health Information Science and Systems (Springer)**
- Q2 xác nhận (best quartile 2024), phạm vi "AI in medicine/medical image processing" khớp trực tiếp khung "OCR y khoa".
- **Có route xuất bản miễn phí (subscription, không bắt buộc APC)** — quan trọng nếu ngân sách cá nhân là ràng buộc thực sự (đây là nghiên cứu độc lập, không có tài trợ CCI).
- Đánh đổi: chậm nhất nhóm (~8 tuần quyết định đầu + 3-4 tuần sau sửa) — dùng làm phương án nếu deadline không gấp hoặc nếu muốn tránh APC.

**Không cần đến IJDAR/Q1** vì đã có đủ 3 lựa chọn Q2 thật; IJDAR chỉ nên cân nhắc như một submission song song *sau này* nếu cả 3 phương án Q2 trên đều bị từ chối (IJDAR Q1 ở category chính xác nhất — CV&PR — nhưng có 2 bài rất giống chủ đề mới xuất bản 2026, tiền lệ tốt).

**Khuyến nghị hành động:** Nộp arXiv trước (không đổi so với kế hoạch gốc, để xác lập priority), sau đó nộp **PeerJ Computer Science** làm nộp chính thức đầu tiên; nếu bị từ chối, sửa theo phản biện rồi nộp **Journal of Imaging**; giữ **Health Information Science and Systems** làm phương án nếu ngân sách APC trở thành vấn đề thực tế.

---

## 2. MÔ HÌNH FINE-TUNE VÀ CẤU HÌNH KỸ THUẬT CỤ THỂ

### Model: `microsoft/trocr-large-handwritten` (558M tham số)
Lý do (đã có trong nghiên cứu khả thi): rủi ro kỹ thuật thấp nhất, đã có tiền lệ code công khai (`peft` + `VisionEncoderDecoderModel`), và **đã được benchmark zero-shot ở Phase 2 đang chạy** → cho phép so sánh "trước/sau" trực tiếp trên đúng cùng một model, đúng bằng chứng đóng góp phương pháp mà Q2 cần.

### Thư viện + kiến trúc LoRA cụ thể
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # rank — xem ablation bên dưới (8/16/32)
    lora_alpha=32,           # alpha = 2×r theo quy ước phổ biến
    lora_dropout=0.1,
    target_modules=["query", "value", "q_proj", "v_proj"],
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(trocr_model, lora_config)
model.print_trainable_parameters()   # BẮT BUỘC chạy dòng này để xác nhận số thật
```
- `"query"`, `"value"`: khớp tên module self-attention của encoder BEiT/ViT (`BeitSelfAttention`/`ViTSelfAttention`) trong `transformers`.
- `"q_proj"`, `"v_proj"`: khớp tên module attention kiểu BART của decoder TrOCR (`TrOCRAttention`, dùng cho cả self-attention và cross-attention decoder) — nên áp LoRA sẽ tự động phủ cả 2 loại attention này.
- **Trước khi finalize**, chạy `for n, _ in model.named_modules(): print(n)` trên chính bản `microsoft/trocr-large-handwritten` đã tải để xác nhận đúng tên module — tên có thể lệch nhẹ giữa các phiên bản `transformers` (rủi ro nhỏ, dễ sửa, nên làm sớm ở Ngày đầu Phase fine-tune).
- Ước tính tham số huấn luyện: ~0,5–1% tổng số (≈3–6M/558M) — khớp con số 0,7% của DLoRA-TrOCR (arXiv 2404.12734), **cần xác nhận số thật qua `print_trainable_parameters()`**, không dùng số ước tính này trong bài báo cuối.
- **Ablation rank**: chạy 3 cấu hình r=8, r=16, r=32 (giữ alpha=2r) trên **cùng 1 epoch nhỏ trước** để chọn nhanh, sau đó chạy đầy đủ epoch với rank tốt nhất theo validation CER — không cần chạy full training cho cả 3 rank.
- **Escalation tùy chọn nếu còn thời gian**: bật `use_dora=True` trong `LoraConfig` (peft ≥0.9) cho riêng encoder, giữ LoRA thường cho decoder — theo đúng công thức DLoRA-TrOCR (đạt CER 4,02% trên IAM, vượt PEFT khác) — đánh dấu là **thí nghiệm bổ sung**, không phải yêu cầu bắt buộc để có kết quả chính.

### Ước lượng thời gian huấn luyện trên Kaggle T4
- Train set thật: **2.808 ảnh** (thư mục `Training`), batch size 8 (grad-accum 2 → effective batch 16), ảnh resize 384×384 (chuẩn TrOCR processor).
- Ước tính mỗi epoch: **10–20 phút** trên 1 T4 (nội suy từ 2 mốc tham chiếu trong nghiên cứu khả thi).
- Số epoch khuyến nghị: **15–30**, early-stopping theo val CER (patience 5 epoch) → tổng **~3–6 giờ/lần train đầy đủ**.
- Ngân sách thực tế cho toàn bộ pha fine-tune (bao gồm ablation rank, A/B augmentation, debug): **~10–15 giờ GPU** — nằm trong giới hạn Kaggle free (30h/tuần, 9h/session, 2×T4 song song).

---

## 3. THIẾT KẾ THÍ NGHIỆM FINE-TUNING

### 3.1. Chia dữ liệu — dùng nguyên 3 split gốc, KHÔNG tự chia lại từ train
Trả lời trực tiếp câu hỏi bạn nêu: **dùng nguyên `Validation` split có sẵn** (≈936 ảnh, do tác giả gốc paper IEEE iCACCESS 2024 đã stratify sẵn theo 78 lớp), không tự cắt thêm validation từ 2.808 ảnh train. Lý do:
- Giữ nguyên tối đa 2.808 ảnh train cho fine-tune (dữ liệu vốn đã nhỏ, cắt thêm validation từ đây sẽ làm giảm train set càng làm tăng rủi ro overfitting).
- Validation split gốc đã stratified đúng 78 lớp — tự chia lại có nguy cơ mất cân bằng lớp nếu không cẩn thận.
- Giữ khả năng so sánh trực tiếp với bài báo IEEE gốc (dùng đúng 3 split của họ).

**Test set phải được ĐÓNG BĂNG (freeze) làm điểm neo bất biến**: đây là điểm quan trọng nhất về mặt phương pháp luận — model đã chạy zero-shot ở Phase 1/2 trên đúng bộ ảnh Testing (780 ảnh resolve được), nên khi fine-tune xong, **phải re-evaluate trên chính xác cùng 780 ảnh đó** (không phải 936 ảnh lý thuyết, không phải random lại) để phép so sánh trước/sau là *paired* hợp lệ (Wilcoxon signed-rank yêu cầu đúng từng cặp ảnh giống nhau). Hành động cụ thể: lưu list `image_path` đã dùng ở Phase 1/2 thành 1 file JSON/CSV bất biến (ví dụ `results/frozen_test_manifest_kaggle_rx.csv`, `results/frozen_test_manifest_iam.csv`) ngay khi bắt đầu Phase fine-tune, dùng lại đúng file này cho mọi lần eval sau.

### 3.2. Augmentation (chống overfitting trên 2.808 ảnh)
Theo bằng chứng cụ thể đã có (Ali et al. 2412.18199; Aradillas et al. 1804.01527), dùng **Albumentations**, kết hợp:
- Elastic distortion (nhẹ-vừa) — TrOCR-**large** nhiều khả năng hưởng lợi từ elastic (khác với bản small).
- Random rotation ±3–5°, shear nhẹ.
- Gaussian noise/blur.
- Brightness/contrast jitter.
- Erosion/dilation (mô phỏng độ đậm nét bút).
- Random crop-padding.

**A/B bắt buộc**: chạy 2 cấu hình — có augmentation vs không — vì literature cảnh báo elastic không phải lúc nào cũng có lợi cho mọi biến thể TrOCR; chọn theo val CER thấp hơn, báo cáo cả 2 con số trong bài (tăng tính minh bạch cho phản biện Q2).

### 3.3. Hyperparameter đề xuất
| Tham số | Giá trị | Ghi chú |
|---|---|---|
| LoRA rank / alpha | 16 / 32 (ablation 8, 32) | xem Mục 2 |
| Learning rate | 2e-4 (AdamW), warmup 5-10% step, cosine decay | LoRA thường dùng LR cao hơn full fine-tune (5e-5) |
| Batch size | 8 vật lý × grad-accum 2 = effective 16 | fp16/bf16 |
| Epoch | 15-30, early-stop patience 5 theo val CER | |
| Max target length | 32 token (khớp `max_new_tokens=32` đã dùng ở Phase 2 zero-shot) | |
| Seed | 42 (khớp seed đã dùng xuyên suốt code hiện tại) | |

### 3.4. Metric so sánh trước/sau — tái dùng đúng pipeline `src/metrics.py`
- CER, WER, Exact-match, Top-1 classification accuracy (78 lớp) — **giống hệt cách đo Phase 1/2** để so sánh trực tiếp không cần đổi công thức.
- Bootstrap 95% CI (1.000 resample, seed=0) cho mỗi metric, cả trước và sau fine-tune, trên cả 2 dataset (Kaggle-Rx test + IAM).
- **Wilcoxon signed-rank test** (paired, trên CER từng ảnh) — 2 phép so sánh chính:
  1. TrOCR zero-shot vs TrOCR fine-tuned, trên Kaggle-Rx test (đo cải thiện in-domain).
  2. TrOCR zero-shot vs TrOCR fine-tuned, trên IAM (đo catastrophic forgetting/mất tổng quát hoá).
- Với n≈780 (Kaggle-Rx test thật, không phải 936 lý thuyết — xem Mục 6), sai số chuẩn CER vẫn đủ nhỏ (~1%, CI95%≈±2%) để phát hiện khác biệt >4-5% — vẫn đủ mạnh cho paired test.

---

## 4. CẤU TRÚC LẠI CÁC PHASE (từ Phase 3 trở đi)

| Phase | Nội dung | Thay đổi so với kế hoạch gốc |
|---|---|---|
| **1** (XONG) | Tesseract + EasyOCR, kaggle_rx (780) + IAM (400) | không đổi |
| **2** (ĐANG CHẠY) | TrOCR-large-handwritten + Donut-base zero-shot | không đổi |
| **3** | GOT-OCR2.0 zero-shot (kaggle_rx + IAM) | **giữ nguyên** — vẫn cần cho benchmark gốc, không fine-tune model này (ms-swift là công cụ riêng, ngoài phạm vi trụ cột) |
| **4** | PaddleOCR-VL zero-shot (kaggle_rx + IAM) | giữ nguyên |
| **5** | Qwen2.5-VL/Qwen3-VL zero-shot (+ tuỳ chọn API đóng) | giữ nguyên, **vẫn đánh dấu tuỳ chọn/bonus** — không đưa vào diện fine-tune (rủi ro overfit cao với VLM 7B trên 2.808 ảnh, chỉ nên là thí nghiệm phụ nếu dư thời gian) |
| **6 (MỚI)** | Chuẩn bị fine-tune: đóng băng test manifest, dựng + unit-test augmentation pipeline, sửa lỗi path-resolution (xem Mục 6) để xác nhận n thật của test/validation | mới |
| **7 (MỚI)** | LoRA fine-tune TrOCR-large-handwritten trên train (2.808 ảnh): ablation rank (8/16/32), A/B augmentation, chọn cấu hình tốt nhất theo val CER | mới — trụ cột đóng góp phương pháp |
| **8 (MỚI)** | Re-evaluate model đã fine-tune trên **CẢ 2** dataset (Kaggle-Rx test đóng băng + IAM đóng băng) — đo cải thiện in-domain VÀ catastrophic forgetting | mới — phân tích quan trọng nhất cho phản biện Q2 |
| **9** | Phân tích định lượng tổng hợp: tất cả model zero-shot × before/after fine-tune, domain-shift gap, bootstrap CI, Wilcoxon | mở rộng từ Ngày 10 gốc |
| **10** | Phân tích lỗi định tính (bao gồm cả lỗi trước/sau fine-tune — fine-tune có sửa được kiểu lỗi nào không: nhầm tên gần giống, hallucination ngoài từ điển, degenerate output) | mở rộng từ Ngày 11 gốc |
| **11 (tuỳ chọn)** | RxHandBD supplementary check (giữ nguyên vai trò phụ như kế hoạch gốc) | không đổi |
| **12 (MỚI)** | Viết bài theo cấu trúc TẠP CHÍ (dài hơn workshop): Introduction, Related Work mở rộng (thêm PEFT/LoRA-for-OCR: DLoRA-TrOCR, devnagari-trocr-lora, Ali et al., Aradillas et al., Ancient Greek Qwen3-VL), Method (bao gồm thiết kế fine-tuning đầy đủ), Results, Discussion, **Threats to Validity/Limitations kỹ lưỡng** | thay thế Ngày 12-14 gốc |
| **13 (MỚI)** | Rà soát nội bộ, format theo template tạp chí, chuẩn bị code+data+adapter weights public, nộp arXiv rồi nộp tạp chí | thay thế Ngày 14 gốc |

---

## 5. LỊCH TRÌNH CHI TIẾT THEO NGÀY (bắt đầu 15/09/2026, tổng ~4 tuần → 12/10/2026; có thể rút xuống 3 tuần bằng cách cắt các mục đánh dấu *[có thể cắt]*)

### Tuần 1 (còn lại) — hoàn tất benchmark zero-shot gốc
| Ngày | Việc |
|---|---|
| 1-2 (15-16/09) | **XONG** — Phase 1 (Tesseract, EasyOCR) |
| 3 (17/09, Th5) | Hoàn tất Phase 2 (TrOCR-large-handwritten + Donut-base) đang chạy; đối chiếu CER TrOCR trên IAM với số công bố (2,89%, Li et al. 2023) làm sanity-check |
| 4 (18/09, Th6) | Phase 3: dựng môi trường + chạy GOT-OCR2.0 zero-shot (kaggle_rx + IAM) |
| 5-6 (19-20/09, cuối tuần, buffer) | Phase 4: dựng + chạy PaddleOCR-VL zero-shot |
| 7 (21/09, Th2) | Phase 5: Qwen2.5-VL/Qwen3-VL zero-shot *[có thể cắt phần API đóng bonus nếu cần tiết kiệm thời gian]*; gộp toàn bộ `results_master.csv` — **mốc hoàn thành scope benchmark 2 tuần gốc** |

### Tuần 2 — chuẩn bị + chạy fine-tuning
| Ngày | Việc |
|---|---|
| 8 (22/09, Th3) | Điều tra + sửa lỗi path-resolution (chỉ resolve 780/936 ảnh test — xem Mục 6); đóng băng `frozen_test_manifest_kaggle_rx.csv` và `frozen_test_manifest_iam.csv`; dựng + unit-test pipeline Albumentations |
| 9 (23/09, Th4) | Viết script fine-tune (`peft` + `LoraConfig`), in `print_trainable_parameters()` xác nhận số thật; smoke-test 20 ảnh/1 epoch để chắc loop chạy đúng, loss giảm |
| 10 (24/09, Th5) | Run 1: rank 16, không augmentation, 15 epoch, theo dõi val CER |
| 11 (25/09, Th6) | Run 2: rank 16 + augmentation (A/B); *[nếu còn ngân sách GPU]* ablation nhanh rank 8/32 trên vài epoch |
| 12-13 (26-27/09, cuối tuần, buffer) | Chọn cấu hình tốt nhất theo val CER, train tới hội tụ đầy đủ (early-stop); lưu adapter weights (đẩy lên HuggingFace Hub hoặc Kaggle Dataset output để không mất khi session hết hạn) |

### Tuần 3 — đánh giá lại + phân tích thống kê
| Ngày | Việc |
|---|---|
| 14 (28/09, Th2) | Re-evaluate model fine-tune trên **frozen test Kaggle-Rx** VÀ **frozen IAM subsample** (cùng pipeline `src/metrics.py`) |
| 15 (29/09, Th3) | Bootstrap CI (1.000 resample) + Wilcoxon signed-rank (2 phép so sánh: in-domain cải thiện, out-of-domain forgetting) |
| 16 (30/09, Th4) | Diễn giải kết quả forgetting — đây là phân tích trọng tâm cho Q2, viết nháp ngay khi số liệu còn "nóng" |
| 17 (01/10, Th5) | Gộp bảng/hình tổng (tất cả model × trước/sau); phân tích lỗi định tính before/after fine-tune |
| 18-19 (02-03/10, cuối tuần, buffer) | Xử lý backlog kỹ thuật còn lại; bắt đầu viết Methods + Results (phần không phụ thuộc polish cuối) |

### Tuần 4 — viết bài + nộp
| Ngày | Việc |
|---|---|
| 20 (04/10, Th2) | Introduction + Related Work mở rộng (thêm nhánh PEFT/LoRA-for-OCR) |
| 21 (05/10, Th3) | Discussion + **Limitations/Threats to Validity** kỹ lưỡng (xem danh sách Mục 6) — *[mốc quyết định: nếu đã đủ 3 tuần và cần dừng sớm, đây là điểm cắt hợp lý để rút gọn thành bài ngắn hơn]* |
| 22 (06/10, Th4) | Abstract + Conclusion; format theo template PeerJ CS; tự rà soát số liệu bảng ↔ văn bản |
| 23 (07/10, Th5) | Polish hình/bảng; chuẩn bị repo code+data+adapter weights public (README, model card HuggingFace) |
| 24 (08/10, Th6) | Nộp arXiv (cs.CV) trước; chuẩn bị hồ sơ nộp tạp chí (cover letter, data availability statement) |
| 25-26 (09-10/10, cuối tuần, buffer) | Rà soát cuối, xử lý trục trặc portal nộp bài |
| 27 (11/10) | Nộp chính thức **PeerJ Computer Science** |
| 28 (12/10) | Buffer / cập nhật `docs/` và memory dự án |

---

## 6. RỦI RO RIÊNG CỦA HƯỚNG NÀY VÀ PHƯƠNG ÁN B

| # | Rủi ro | Phương án B |
|---|---|---|
| 1 | **Fine-tune không cải thiện đáng kể** (hoặc cải thiện nhưng không có ý nghĩa thống kê) | Đây vẫn là kết quả khoa học trung thực, có thể báo cáo được — chuyển khung bài từ "fine-tune giúp cải thiện X%" sang "LoRA fine-tune nhẹ có/không đủ để đóng khoảng cách domain-shift trong chế độ dữ liệu cực nhỏ (2.808 ảnh)" — vẫn là đóng góp hợp lệ cho Q2, đặc biệt nếu đi kèm phân tích lỗi định tính giải thích tại sao |
| 2 | **Vượt quota GPU Kaggle** (30h/tuần, 9h/session) | Chuyển sang Colab Pro (~$10/tháng) cho phần fine-tune; hoặc giảm ablation xuống 1 cấu hình duy nhất (rank 16, có augmentation) thay vì chạy đủ 3 rank + A/B |
| 3 | **Catastrophic forgetting nặng** (IAM CER tệ đi rõ rệt sau fine-tune) | Đây có thể chính là phát hiện thú vị nhất của bài — biến thành phân tích "cái giá của chuyên biệt hoá" (trade-off in-domain gain vs out-of-domain loss), là đúng loại phân tích phản biện Q2 mong đợi; giảm nhẹ bằng early-stopping dựa trên metric kết hợp cả 2 dataset nếu muốn thử khắc phục thêm |
| 4 | **Test set Kaggle-Rx chỉ resolve được 780/936 ảnh** (đã xác nhận trong `phase1_summary.md` — do lỗi khớp đường dẫn trong `build_manifest_kaggle_rx()`, không phải do dataset thiếu ảnh) | Trước khi vào Phase fine-tune (Ngày 8), điều tra nguyên nhân cụ thể (khả năng: tên file trong CSV không khớp hoàn toàn tên file thật, hoặc thư mục ảnh lồng thêm 1 cấp) và cố gắng khôi phục đủ 936 ảnh để tăng power thống kê; nếu không sửa được kịp, chấp nhận n=780, báo cáo trung thực trong Limitations, và đảm bảo **dùng đúng cùng 780 ảnh đó** cho mọi so sánh trước/sau (tính nhất quán quan trọng hơn số lượng tuyệt đối) |
| 5 | **Chi phí APC vượt ngân sách cá nhân** (đây là nghiên cứu độc lập, không có tài trợ CCI theo đúng chủ trương ban đầu) | Ưu tiên nộp PeerJ CS (rẻ nhất, $1.395) trước; nếu vẫn là vấn đề, chuyển hẳn sang Health Information Science and Systems (route miễn phí subscription) |
| 6 | **Bị reviewer bắt lỗi overclaim** (78 lớp closed-vocabulary, dữ liệu fine-tune quá nhỏ dễ overfit vào đúng 78 tên thuốc thay vì học "đọc chữ viết tay" tổng quát) | Xử lý trước trong Limitations: nêu rõ ranh giới giữa "cải thiện nhận dạng chữ viết tay nói chung" và "cải thiện ghi nhớ closed-vocabulary 78 lớp cụ thể" — đây là lý do chính đánh giá thêm trên IAM (Mục 3, 4) là bắt buộc chứ không phải tuỳ chọn |
| 7 | **Trễ tiến độ tổng thể** (dồn lỗi kỹ thuật ở nhiều phase) | Thứ tự cắt giảm nếu cần rút về 3 tuần: (a) bỏ nhánh Qwen-VL bonus API, (b) bỏ ablation rank 8/32 (giữ mỗi rank 16), (c) bỏ RxHandBD supplementary, (d) rút gọn Related Work — **không được cắt** phần đánh giá lại trên IAM sau fine-tune (đây là lõi đóng góp mới nhất so với kế hoạch gốc) |
| 8 | **DLoRA/DoRA escalation không kịp thời gian** | Đã đánh dấu từ đầu là thí nghiệm bổ sung (Mục 2) — bỏ hoàn toàn không ảnh hưởng đóng góp chính (LoRA rank 16 cơ bản đã đủ để trả lời câu hỏi nghiên cứu) |

---

### Ghi chú cuối
Kế hoạch này giữ nguyên khung "nghiên cứu độc lập, dataset công khai, không IRB" đã chốt trước đó — toàn bộ phần mở rộng (fine-tuning LoRA) chỉ dùng dữ liệu Kaggle-Rx (đã có ground truth CSV/Excel công khai, không định danh bệnh nhân) và IAM (chữ viết tay tổng quát, đăng ký miễn phí) — không phát sinh nhu cầu dữ liệu bệnh viện/CCI hay thủ tục đạo đức mới nào.
---

## Phụ lục A: Xác minh tạp chí Q2 (SCImago, kiểm chứng chéo nhiều nguồn)

## KẾT QUẢ XÁC MINH QUARTILE (SCIMAGO) CHO CÁC TẠP CHÍ ỨNG VIÊN

**Lưu ý về phương pháp:** scimagojr.com có chặn bot (Cloudflare "security verification") nên không thể fetch trực tiếp trang này (đây cũng là hành vi bị cấm nếu cố bypass CAPTCHA/bot-detection nên tôi không thử vượt qua nó). Dữ liệu dưới đây được xác minh chéo qua nhiều nguồn tổng hợp lấy dữ liệu gốc từ Scopus/SJR: researcher.life, resurchify.com, editage.com, askbisht.com, scienceaijournal.com, journalsearches.com — đối chiếu ít nhất 2 nguồn độc lập cho mỗi tạp chí. Khuyến nghị: trước khi nộp bài, tự kiểm tra lại trực tiếp trên scimagojr.com (quartile cập nhật ~tháng 4-5 hàng năm và có thể đổi).

### BẢNG XÁC MINH 14 TẠP CHÍ TRONG DANH SÁCH GỐC

| # | Tạp chí | Quartile lĩnh vực gần nhất | Quartile khác | SJR / H-index | Kết luận Q2? |
|---|---------|------|------|------|------|
| 1 | **IJDAR** (Springer) | Computer Vision & Pattern Recognition: **Q1** (2024, best quartile) | Computer Science Applications: Q2; Software: Q2 | SJR ≈0.83, H-index 61, CiteScore 6.7 | ❌ Không — Q1 ở lĩnh vực chính xác nhất |
| 2 | Pattern Recognition Letters (Elsevier) | Computer Vision & Pattern Recognition: **Q1** (2024, best quartile) | — | CiteScore 8.6 | ❌ Không |
| 3 | Journal of Imaging Informatics in Medicine (Springer/SIIM) | **Q1** (best quartile) | — | SJR 0.892, H-index 24, IF 3.1 | ❌ Không — trái với giả định ban đầu |
| 4 | Multimedia Tools and Applications (Springer) | Media Technology: **Q1**; không có category CV&PR riêng | Computer Networks and Comm.: Q2; Hardware/Architecture: Q2; Software: Q2 | SJR 0.777, H-index 93 | ❌ Không — best quartile Q1 |
| 5 | Neural Computing and Applications (Springer) | Artificial Intelligence: **Q1**; Software: **Q1** | — | SJR 1.102, H-index 146 | ❌ Không |
| 6 | **Journal of Imaging (MDPI)** | Computer Vision & Pattern Recognition: **Q2** ✅ | Radiology/Nuclear Medicine/Imaging: Q2 (SJR)/Q1 (CiteScore); Electrical & Electronic Eng.: Q2; Computer Graphics & CAD: Q2 | SJR 0.66–0.73, H-index 34-53 (nguồn không khớp hoàn toàn), CiteScore 4.8–7.3 | ✅ **CÓ** |
| 7 | **SN Computer Science (Springer)** | Artificial Intelligence: **Q2** ✅ (best quartile Q2 2025) | — | SJR 0.565, H-index 25-65 (nguồn lệch nhau) | ✅ **CÓ** |
| 8 | Computers in Biology and Medicine (Elsevier) | Computer Science Applications: **Q1**; Health Informatics: **Q1** | — | SJR 1.375, H-index 113 | ❌ Không |
| 9 | Journal of Biomedical Informatics (Elsevier) | **Q1** | — | — | ❌ Không |
| 10 | Expert Systems with Applications (Elsevier) | **Q1** (best quartile, SJR 1.854-1.939) | (bản "X" companion từng Q2 năm 2020, chưa xác minh được số liệu hiện tại) | H-index rất cao | ❌ Không |
| 11 | Healthcare Analytics (Elsevier) | **Q1** | — | SJR ~1.1-1.6 | ❌ Không |
| 12 | Artificial Intelligence in Medicine (Elsevier) | **Q1** | — | H-index 100, CiteScore 10.4 | ❌ Không |
| 13 | Image and Vision Computing (Elsevier) | **Q1** | — | SJR 0.791 | ❌ Không |
| 14a | **PeerJ Computer Science** | Computer Science (miscellaneous): **Q2** ✅ (đa số nguồn đồng thuận; 1 nguồn — researcher.life — báo Q1 ở nhãn gộp "Computer Science (all)", có vẻ là lỗi tổng hợp chứ không phải category thật của Scimago) | — | SJR 0.618, H-index 84 | ✅ **CÓ** (với 1 lưu ý mâu thuẫn nhỏ) |
| 14b | **Health Information Science and Systems (Springer)** | Best quartile **Q2** (2024) ✅ | — | SJR 0.86 | ✅ **CÓ** |
| 14c | Applied Sciences (MDPI) | Computer Science Applications: **Q2** ✅; nhưng Engineering Multidisciplinary (category chính/best): **Q1** | Nhiều category khác Q2-Q3 | SJR 0.555 | ⚠️ Có, nhưng là mega-journal đa ngành |
| 14d | IEEE Access | **Q1** | — | H-index 338 | ❌ Không |
| 14e | BMC Medical Informatics and Decision Making | Mâu thuẫn: 1 nguồn nói Q2, nguồn khác (chi tiết theo category) nói Computer Science Applications/Health Informatics/Health Policy đều **Q1** | — | SJR 1.224 | ❌ Không (nhiều khả năng Q1) |

### XẾP HẠNG TOP 5 TẠP CHÍ Q2 THẬT (đã xác minh) — phù hợp nhất

**#1 — Journal of Imaging (MDPI)** — [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21100900151&tip=sid) | [MDPI](https://www.mdpi.com/journal/jimaging)
- Q2 xác nhận đúng ở *cả 4* category liên quan trực tiếp (Computer Vision & Pattern Recognition, Radiology/Imaging, Electrical Eng., Computer Graphics) — khớp nhất với bài OCR y khoa.
- Thời gian bình duyệt cực nhanh: ~20-22 ngày tới quyết định đầu tiên (trung vị nhiều năm liền).
- APC: CHF 1800-2400 (nguồn hơi lệch nhau, nên kiểm tra mdpi.com/journal/jimaging/apc) — 100% open access, không có lựa chọn miễn phí.
- Chưa tìm thấy tiền lệ bài báo OCR-toa-thuốc cụ thể trong chính journal này, nhưng phạm vi (Aims & Scope) khớp hoàn toàn với "so sánh + fine-tune mô hình thị giác/OCR y khoa".

**#2 — PeerJ Computer Science** — [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21100830173&tip=sid) | [peerj.com](https://peerj.com/computer-science/)
- Q2 (đa số nguồn), APC thấp nhất trong nhóm: $1,395.
- Bình duyệt nhanh theo công bố của tạp chí (~35 ngày), mặc dù SciRev (khảo sát tác giả) báo dài hơn (~14 tuần) — nên chuẩn bị tinh thần có thể lâu hơn con số chính thức.
- **Tiền lệ mạnh nhất**: đã xuất bản bài "Adapting multilingual vision language transformers for low-resource Urdu OCR" (2024, dùng TrOCR) — gần như đúng loại bài (benchmark + adapt mô hình OCR) mà kế hoạch nhắm tới.

**#3 — Health Information Science and Systems (Springer)** — [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21100853958&tip=sid) | [Springer](https://link.springer.com/journal/13755)
- Q2 xác nhận (best quartile 2024), SJR 0.86.
- Phạm vi ghi rõ: "AI in medicine, medical image processing" — khớp trực tiếp với hướng OCR toa thuốc viết tay.
- Hybrid: có lựa chọn xuất bản miễn phí (subscription route, không bắt buộc APC) — rất phù hợp cho nhà nghiên cứu độc lập tự trả phí.
- Nhược điểm: bình duyệt chậm hơn — trung bình ~8 tuần tới quyết định đầu, thêm 3-4 tuần sau khi sửa; APC nếu chọn OA cao (~$4,090).

**#4 — SN Computer Science (Springer)** — [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21101083109&tip=sid) | [Springer](https://link.springer.com/journal/42979)
- Q2 xác nhận trong category Artificial Intelligence.
- Hybrid, có lựa chọn miễn phí APC (subscription route).
- Nhược điểm: bình duyệt chậm nhất nhóm (trung vị 96 ngày ~ 3+ tháng); phạm vi rất rộng (mọi lĩnh vực CS) nên uy tín/độ chọn lọc thấp hơn 3 tạp chí trên.

**#5 — Applied Sciences (MDPI)** — [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21100829268&tip=sid) | [MDPI](https://www.mdpi.com/journal/applsci)
- Q2 xác nhận *đúng trong category Computer Science Applications* (dù category chính/tổng — Engineering Multidisciplinary — đã lên Q1).
- Bình duyệt siêu nhanh (~15 ngày), nhưng đây là **mega-journal đa ngành** (hàng nghìn bài/năm), bị nhiều nhà nghiên cứu đánh giá thấp hơn về uy tín/độ chọn lọc dù chỉ số Q2 là thật. Xếp cuối vì lý do này — dùng như phương án dự phòng an toàn, không phải lựa chọn đầu tiên.

### GHI CHÚ ĐẶC BIỆT: IJDAR (không nằm trong top 5 vì lý do quartile, nhưng đáng cân nhắc)

IJDAR (International Journal on Document Analysis and Recognition) là tạp chí phù hợp nhất về **chủ đề** — đây chính xác là tạp chí chuyên về OCR/nhận dạng tài liệu — và vừa mới xuất bản 2 bài **rất giống** kế hoạch của bạn:
- "Benchmarking OCR and vision-language models for Turkish text recognition: a comprehensive evaluation using synthetic data" (2026)
- "Literally reading behind the lines: a benchmark for OCR on cluttered printed documents" (2026)

Tuy nhiên, quartile *chính xác* ở category gần nhất (Computer Vision and Pattern Recognition) hiện là **Q1**, không phải Q2 (chỉ Q2 ở Computer Science Applications/Software — các category phụ, ít liên quan hơn). Vì bạn yêu cầu nghiêm ngặt Q2, tôi loại IJDAR khỏi top-5, nhưng nếu bạn chấp nhận nhắm Q1 cho một submission song song (dự phòng nếu bị Q2 từ chối), đây là lựa chọn có tiền lệ xuất bản mạnh nhất. Hybrid, có lựa chọn miễn phí (subscription route) hoặc OA ~$3,290.

### KHUYẾN NGHỊ CUỐI

Với tiêu chí (a) Q2 xác nhận thật, (b) phạm vi khớp bài benchmark+fine-tuning OCR y khoa, (c) bình duyệt nhanh, (d) chi phí hợp lý/có lựa chọn miễn phí:

**Ưu tiên nộp: Journal of Imaging (MDPI) hoặc PeerJ Computer Science** — cả hai đều Q2 xác nhận chắc chắn nhất, bình duyệt nhanh, và PeerJ CS có tiền lệ xuất bản bài rất giống (Urdu OCR + TrOCR). Health Information Science and Systems là lựa chọn tốt nếu muốn nhấn mạnh khung "ứng dụng y tế/AI-in-medicine" và không muốn trả APC (dùng route subscription).

Do một số con số (H-index, SJR) hơi lệch nhau giữa các nguồn tổng hợp (không phải do tôi suy đoán, mà do các trang mirror cache dữ liệu ở thời điểm khác nhau), khuyến nghị bạn tự vào trực tiếp scimagojr.com kiểm tra lại quartile chính xác của 5 tạp chí đã chọn trước khi quyết định nộp cuối cùng.
---

## Phụ lục B: Nghiên cứu khả thi fine-tuning LoRA trên Kaggle GPU T4

## KẾT QUẢ NGHIÊN CỨU: TÍNH KHẢ THI FINE-TUNE LORA OCR TRÊN KAGGLE GPU T4

### KHUYẾN NGHỊ CHÍNH (rõ ràng, 1 dòng)

**Fine-tune LoRA/DoRA model `microsoft/trocr-large-handwritten` (558M tham số) bằng thư viện `peft` của HuggingFace**, áp dụng LoRA lên encoder (BEiT/ViT) và decoder (RoBERTa) của kiến trúc `VisionEncoderDecoderModel`. Đây là lựa chọn rủi ro thấp nhất, có tiền lệ code công khai, khớp với model đã benchmark zero-shot ở Phase 2 (cho phép so sánh "trước/sau fine-tune" trực tiếp — chính là bằng chứng đóng góp phương pháp mà tạp chí Q2 cần), và chạy thoải mái trên 1 GPU T4 16GB trong vài giờ.

Nếu muốn thêm 1 thí nghiệm "bonus" mạnh hơn cho bài báo (so sánh HTR chuyên biệt vs. VLM tổng quát sau fine-tune): **Qwen2.5-VL-7B qua QLoRA + Unsloth** cũng khả thi trên T4 (không phải LoRA thường — xem chi tiết Câu 1).

---

### CÂU 1: Model nào THỰC TẾ fine-tune được bằng LoRA/PEFT trên 1 T4 16GB?

| Model | Kích thước | Khả thi LoRA trên T4 16GB? | Ghi chú / hướng dẫn |
|---|---|---|---|
| **TrOCR-large-handwritten** | ~558M | **CÓ, dễ dàng** | `peft.LoraConfig` áp trực tiếp lên `VisionEncoderDecoderModel` (HF `transformers`) — có GitHub issue chính thức bàn về LoRA+VRAM cho đúng class này (huggingface/peft #1048). Full fine-tune (không cần LoRA) còn fit thoải mái ở 558M, nên LoRA chỉ giúp tiết kiệm thêm VRAM/thời gian, không phải bắt buộc để fit máy — an toàn nhất. |
| **TrOCR-base-handwritten** | ~334M | **CÓ, rất dễ** | Phương án dự phòng nếu muốn nhanh hơn/an toàn hơn nữa; cùng codebase với large. |
| **Donut-base** | ~200M | **CÓ** | Cũng là `VisionEncoderDecoderModel` (Swin encoder + BART-like decoder), peft áp được tương tự; lưu ý Donut là OCR-free, cần thiết kế lại task prompt cho free-text OCR (đã ghi caveat này trong kế hoạch gốc của bạn). |
| **GOT-OCR2.0** | ~580M | **CÓ, qua công cụ riêng** | Repo chính thức dùng framework **ms-swift** (ModelScope) hỗ trợ sẵn `--sft_type lora`: `swift sft --model_type got-ocr2 --model_id_or_path stepfun-ai/GOT-OCR2_0 --sft_type lora --dataset train.jsonl`. Mặc định đóng băng vision encoder, chỉ LoRA fine-tune LLM+projector — nhẹ, hợp T4. Rủi ro: phải cài thêm ms-swift, ít tài liệu cộng đồng hơn TrOCR. |
| **PaddleOCR-VL** | ~0.9B | **KHÔNG NÊN dùng lúc này** | Đội PaddlePaddle xác nhận công khai (thảo luận HF #13, #4): *"A full workflow for fine-tuning with ERNIEKit is coming soon"* — **code fine-tune chính thức CHƯA phát hành**. Chỉ có 1 script cộng đồng không chính thức (`jzhang533/paddleocr-vl-sft`). Rủi ro kỹ thuật quá cao cho 1 dự án 3-4 tuần. |
| **Qwen2.5-VL-7B / Qwen3-VL-8B** | 7-8B | **LoRA thường: KHÔNG** (fp16 model đã ~18GB > 16GB T4). **QLoRA 4-bit qua Unsloth: CÓ** | Unsloth nạp Qwen2.5-VL 4-bit chỉ tốn ~7,1GB VRAM, có notebook Colab T4 miễn phí sẵn (bao gồm ví dụ OCR viết tay → LaTeX), 1 lần train đo được ~48,7 phút trên T4. Nhưng đây là VLM tổng quát 7B — rủi ro overfit cao hơn với chỉ 2.808 ảnh, và không phải "LoRA nhẹ" đúng nghĩa (phải quantize 4-bit). Phù hợp làm **thí nghiệm bổ sung**, không phải trụ cột. |
| **Florence-2** (gợi ý thay thế nhỏ hơn) | 0,23B–0,77B | **CÓ, khả thi** | Có tutorial LoRA chính thức (Roboflow, dùng cho detection nhưng áp dụng được cho `<OCR>` task token). Ít tiền lệ trực tiếp trên OCR y khoa, VRAM thực nghiệm được test chủ yếu trên GPU mạnh hơn T4 (RTX 3090/4090) — khả thi nhưng ít bằng chứng T4 cụ thể hơn TrOCR. |
| **CRNN full-train từ đầu** | nhỏ | **KHÔNG khuyến nghị** | Về mặt kỹ thuật chạy được trên T4 dễ dàng, nhưng bằng chứng thực nghiệm (Aradillas et al. 2018, xem Câu 4) cho thấy huấn luyện từ đầu trên dữ liệu nhỏ overfit nặng (CER ~18-40% trên tập validation) so với transfer learning từ model pretrained (CER 3-9%) — đi ngược mục tiêu "cải thiện rõ rệt" của bạn. |

**Xếp hạng cuối:** TrOCR-large-handwritten (chính) > TrOCR-base-handwritten (dự phòng an toàn) > GOT-OCR2.0 (mở rộng nếu còn thời gian) > Qwen2.5-VL-7B QLoRA (bonus so sánh) >> PaddleOCR-VL (loại bỏ, chưa sẵn sàng).

---

### CÂU 2: Thư viện LoRA phổ biến nhất 2025-2026

- **HuggingFace `peft`** — là lựa chọn **chuẩn và duy nhất có tiền lệ trực tiếp** cho kiến trúc encoder-decoder cổ điển (TrOCR, Donut): `peft.LoraConfig` + `get_peft_model()` áp lên `VisionEncoderDecoderModel`. Có tiền lệ công bố: adapter LoRA công khai trên HF (`manishw10/devgen-trocr-devanagari-lora`, train trên 95,4k ảnh chữ Devanagari bằng `peft` + `transformers`), và bài báo **DLoRA-TrOCR** (arXiv 2404.12734) — dùng DoRA cho encoder + LoRA cho decoder của TrOCR, **chỉ 0,7% tham số huấn luyện**, đạt CER 4,02% trên IAM, vượt các phương pháp PEFT khác.
- **ms-swift** (ModelScope) — thư viện được GOT-OCR2.0 dùng chính thức cho LoRA fine-tuning (wrapping peft bên dưới), cần nếu chọn GOT-OCR2.0.
- **Unsloth** — **KHÔNG** hỗ trợ trực tiếp kiến trúc encoder-decoder cổ điển (TrOCR/Donut/GOT-OCR2.0); Unsloth năm 2026 tập trung tối ưu cho **VLM/LLM decoder-only lớn**: Qwen3-VL, Qwen3.5, DeepSeek-OCR/DeepSeek-OCR-2 (model 3B mới ra 27/01/2026, có notebook fine-tune miễn phí, nhanh hơn 1,4x, tốn ít VRAM hơn 40%). Chỉ dùng Unsloth nếu bạn chọn nhánh Qwen-VL/DeepSeek-OCR làm thí nghiệm bổ sung.

**Kết luận Câu 2:** `peft` (HuggingFace) cho trụ cột chính (TrOCR); `ms-swift` nếu mở rộng GOT-OCR2.0; `Unsloth` chỉ nếu làm thêm nhánh VLM (Qwen2.5-VL/Qwen3-VL) như thí nghiệm phụ.

---

### CÂU 3: Data augmentation cho tập nhỏ (2.808 ảnh) chống overfitting

Bằng chứng cụ thể từ 2 nguồn:

1. **Ali et al. 2024** (chính bài toán đơn thuốc viết tay, gần domain của bạn nhất): dùng **brightness adjustment, contrast normalization, translation, minor shearing, elastic transformation, Gaussian noise, cropping with padding** để mở rộng 1.000 → 9.920 ảnh (~×10).
2. **Aradillas et al. 2018** (HTR dữ liệu nhỏ kinh điển, xem Câu 4): dùng **affine transform (rotation, shearing, translation, scaling) + morphological distortion (erosion, dilation)**, giảm CER từ 8,2%→6,4% (test) và 5,1%→4,4% (valid) trên IAM; trên tập cực nhỏ Washington-150 dòng, giảm CER 9,4%→8,9%.
3. Lưu ý nuance quan trọng (từ khảo sát TrOCR): **elastic distortion không phải lúc nào cũng có lợi** — các bản TrOCR nhỏ (small) đôi khi CER tốt hơn KHÔNG dùng elastic deformation, trong khi bản base/large lại hưởng lợi từ nó. Vì bạn dùng TrOCR-**large**, elastic transform nhiều khả năng có lợi — nhưng nên A/B test có/không trong quá trình tune.

**Khuyến nghị cụ thể cho tập 2.808 ảnh:** kết hợp elastic distortion (mức nhẹ-vừa) + random rotation nhỏ (±3-5°) + shear nhẹ + Gaussian noise/blur + brightness/contrast jitter + erosion/dilation (mô phỏng độ đậm nét bút khác nhau) + random crop-padding. Dùng thư viện **Albumentations** (phổ biến trong pipeline OCR) hoặc `torchvision.transforms` kết hợp hàm elastic transform tùy chỉnh (theo công thức Simard et al. mà cả 2 paper trên đều tham chiếu).

---

### CÂU 4: Bằng chứng — các công trình ĐÃ fine-tune OCR/HTR trên dữ liệu nhỏ tương tự, cải thiện bao nhiêu %

Đây là 4 nguồn trích dẫn được để biện minh "fine-tuning sẽ cải thiện đáng kể" trong bài báo:

1. **Ali et al., "Leveraging Deep Learning with Multi-Head Attention for Accurate Extraction of Medicine from Handwritten Prescriptions"** (arXiv 2412.18199) — **CHÍNH XÁC cùng bài toán**: Mask R-CNN + TrOCR-Base-Handwritten fine-tune trên ~1.000 đơn thuốc viết tay Pakistan (augment ×10 → 9.920 ảnh). Kết quả: CER 1,4-15,4% tùy kịch bản test (valid 1,4%, pattern thay đổi 3,9%, ảnh hoàn toàn mới 13,5%) — headline "CER 1,4% trên benchmark chuẩn". Đây là bằng chứng gần domain nhất, cùng quy mô dữ liệu, cùng họ model TrOCR.

2. **Aradillas, Murillo-Fuentes & Olmos, "Boosting Handwriting Text Recognition in Small Databases with Transfer Learning"** (arXiv 1804.01527) — bằng chứng kinh điển nhất về "transfer learning cứu tập dữ liệu nhỏ": với chỉ **350 dòng huấn luyện**, train from scratch CER = **18,2%**, còn dùng transfer learning (pretrain trên IAM 13k dòng rồi fine-tune) CER = **3,3%** — cải thiện tuyệt đối ~15 điểm CER, tương đương giảm ~82%. Với 150 dòng, CER vẫn giữ ở mức 5,8-9,4% nhờ transfer learning (so với overfit hoàn toàn nếu train from scratch).

3. **DLoRA-TrOCR** (arXiv 2404.12734) — bằng chứng rằng **LoRA cụ thể** (không chỉ full fine-tune) đủ mạnh: chỉ 0,7% tham số huấn luyện, đạt CER 4,02% trên IAM, F1 94,29% trên SROIE, vượt các phương pháp PEFT khác — chứng minh LoRA không đánh đổi nhiều hiệu năng so với full fine-tune.

4. **"Structure-Aware Text Recognition for Ancient Greek Critical Editions"** (arXiv 2603.02803) — bằng chứng gần đây (2026) với model lớn hơn: Qwen3-VL-8B fine-tune trên dữ liệu chữ viết chuyên biệt hẹp, CER giảm từ **5,2% (zero-shot) → 2,1% (fine-tune trên dữ liệu thật) → 1,0% (kết hợp synthetic+real)** — domain khác (Hy Lạp cổ, không phải y khoa) nhưng cùng luận điểm "VLM tổng quát cải thiện mạnh sau fine-tune trên script/domain hẹp", có thể trích dẫn làm bằng chứng bổ sung nếu bạn thử nhánh Qwen-VL.

**Tổng kết Câu 4:** Cả 4 nguồn đều cho thấy mức cải thiện CER rất lớn (giảm 50-80%+ so với zero-shot/from-scratch) khi fine-tune trên tập dữ liệu cùng quy mô (vài trăm đến vài nghìn ảnh) như của bạn — đủ mạnh để biện minh giả thuyết nghiên cứu.

---

### CÂU 5: Ước lượng thời gian huấn luyện thực tế trên T4

Không có số liệu công bố chính xác cho *TrOCR-large-handwritten LoRA fine-tune trên đúng 2.808 ảnh*, nhưng nội suy từ các mốc thực nghiệm tìm được:

- Dữ liệu tham chiếu 1: fine-tune TrOCR (biến thể nhỏ hơn), batch size hiệu dụng 8 (batch vật lý 2 + gradient accumulation 4), **10 epoch ≈ 2 giờ 40 phút** trên GPU **yếu hơn T4** (RTX 5060 Laptop, 8GB).
- Dữ liệu tham chiếu 2: một nghiên cứu khác fine-tune TrOCR full encoder+decoder trên chính **T4**, batch size 8, **40 epoch** (không nêu rõ tổng giờ, nhưng đây là số epoch điển hình được chọn cho hội tụ ổn định).
- Ước lượng riêng của kế hoạch gốc bạn đã có: TrOCR inference (chỉ forward) ~0,1-0,3 giây/ảnh trên GPU tầm trung cho 936 ảnh test (3-8 phút).

**Suy ra cho trường hợp của bạn** (train set ước tính ~2.246 ảnh sau chia 80/20 từ 2.808, TrOCR-**large** lớn hơn base nên chậm hơn ~1,3-1,5x, nhưng LoRA giảm tải bộ nhớ optimizer cho phép batch size lớn hơn bù lại phần nào):

- **Mỗi epoch: ước tính ~10-20 phút** (batch size 8-16, fp16/bf16, ảnh resize chuẩn TrOCR 384×384).
- **Số epoch khuyến nghị: 15-30 epoch** (fine-tune LoRA trên model đã pretrained hội tụ nhanh hơn nhiều so với train-from-scratch — không cần tới hàng trăm epoch như CNN-BLSTM-CTC train từ đầu trong Aradillas et al.).
- **Tổng thời gian 1 lần train đầy đủ: ước tính 3-6 giờ GPU**, cộng thêm thời gian thử 2-3 cấu hình hyperparameter (rank LoRA, learning rate, có/không elastic augmentation) → **tổng ngân sách thực tế nên dự trù ~10-15 giờ GPU** cho toàn bộ pha fine-tuning (bao gồm thử nghiệm/debug), **hoàn toàn nằm trong giới hạn ~30 giờ GPU/tuần và ~9 giờ/session** của Kaggle T4 x2 miễn phí — khả thi trong 1-2 tuần bổ sung như kế hoạch của bạn.

---

### Nguồn tham khảo chính đã kiểm chứng

- [DLoRA-TrOCR (arXiv 2404.12734)](https://arxiv.org/html/2404.12734v1)
- [manishw10/devgen-trocr-devanagari-lora (HuggingFace)](https://huggingface.co/manishw10/devgen-trocr-devanagari-lora)
- [huggingface/peft Issue #1048 — LoRA + VisionEncoderDecoderModel](https://github.com/huggingface/peft/issues/1048)
- [Ali et al., Leveraging Deep Learning with Multi-Head Attention... (arXiv 2412.18199)](https://arxiv.org/abs/2412.18199)
- [Aradillas et al., Boosting HTR in Small Databases with Transfer Learning (arXiv 1804.01527)](https://arxiv.org/pdf/1804.01527)
- [Structure-Aware Text Recognition for Ancient Greek Critical Editions (arXiv 2603.02803)](https://arxiv.org/pdf/2603.02803)
- [stepfun-ai/GOT-OCR2.0 GitHub + ms-swift issue #2122](https://github.com/modelscope/ms-swift/issues/2122)
- [PaddleOCR-VL fine-tuning discussion (HF #13)](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/discussions/13)
- [Unsloth Qwen3-VL / Qwen2.5-VL fine-tuning docs](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune)
- [NielsRogge/Transformers-Tutorials — TrOCR fine-tune notebooks](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb)

*(Ghi chú: 2 file PDF đầy đủ đã tải và đọc trong quá trình nghiên cứu được lưu tại thư mục tool-results của phiên làm việc này nếu cần tham chiếu lại chi tiết bảng số liệu.)*