# 5. Kế Hoạch Nghiên Cứu 2 Tuần (Nghiên Cứu Độc Lập, Dataset Công Khai)

*Bắt đầu 15/09/2026. Mục tiêu: hoàn thành thực nghiệm + bản thảo bài báo (dạng benchmark/technical report ngắn) và nộp arXiv trong 14 ngày — KHÔNG bao gồm thời gian bình duyệt (bình duyệt workshop/hội nghị luôn mất thêm hàng tuần đến hàng tháng, nằm ngoài phạm vi kiểm soát của kế hoạch này).*

---

## 5.1. Phân tích phương án và lựa chọn

## PHÂN TÍCH VÀ ĐỀ XUẤT KẾ HOẠCH NGHIÊN CỨU OCR Y KHOA — 2 TUẦN

*(Dựa trên kết quả tra cứu 3 nguồn đã cung cấp, đặc biệt phần "kiểm chứng sâu" — nguồn đáng tin nhất vì đã truy cập trực tiếp trang gốc)*

---

### TÓM TẮT NHANH KẾT LUẬN

Có **MỘT phương án sẵn sàng ngay hôm nay, không cần chờ credentialing, không cần IRB** (đúng yêu cầu context của bạn), rủi ro kỹ thuật thấp nhất, và tính mới được xác nhận rõ qua tra cứu tài liệu: **Phương án A — Benchmark các mô hình OCR/VLM pretrained (2024-2026) trên bộ dữ liệu Kaggle "Doctor's Handwritten Prescription BD dataset", đối chứng domain-shift bằng IAM.** Không có phương án nào trong 3 phương án dưới đây cần PhysioNet, nên **không có tình huống "phải chọn phương án dự phòng vì thiếu thời gian chờ"** — cả 3 đều dùng dữ liệu tải-ngay.

---

### PHƯƠNG ÁN A (đề xuất chính)

**Dataset:** Kaggle "Doctor's Handwritten Prescription BD dataset" (mamun1113) — https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset — dùng **IAM Handwriting Database** (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) làm đối chứng domain-shift.

**Lý do uy tín:**
- Kaggle dataset có **paper bình duyệt thật**: Mia et al., IEEE iCACCESS 2024 (document 10499631).
- Được cộng đồng dùng rộng, ổn định 2 năm: 34.700 views, 7.078 tải, 57 upvote, 29 notebook công khai, có badge "used in a publication".
- 4.680 ảnh, 78 lớp tên thuốc, ground truth CSV/Excel đầy đủ, license "Open Database" cho nghiên cứu.
- IAM là gold-standard HTR kinh điển (Marti & Bunke, IJDAR 2002, ~569+ trích dẫn), đăng ký miễn phí, không IRB.

**Câu hỏi nghiên cứu:** Các mô hình OCR-VLM pretrained thế hệ mới (GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, Qwen2.5-VL/Qwen3-VL) — đã được benchmark kỹ trên chữ viết tay tổng quát (IAM) hoặc hóa đơn (SROIE/CORD) nhưng **chưa từng được test trên chữ viết tay y khoa** — hoạt động ra sao khi đọc tên thuốc viết tay của bác sĩ? Mức độ suy giảm hiệu năng do domain-shift (tổng quát → y khoa) lớn tới đâu, và có khác biệt giữa các họ mô hình (OCR cổ điển vs. HTR transformer chuyên biệt vs. VLM tổng quát) không?

**Vì sao có tính mới:** Bảng "khoảng trống" trong tra cứu tài liệu xác nhận: GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, Qwen-VL **CHƯA** ai test trên bất kỳ dataset đơn thuốc/HTR y khoa công khai nào. Benchmark gần nhất (dev.to blog, không bình duyệt) chỉ test 4 engine cũ (Tesseract/EasyOCR/PP-OCRv5/GLM-OCR) trên RxHandBD — không đụng đến các model trên. Hai công trình cạnh tranh gần nhất (RxScribe Bench 09/2026 — dataset Ấn Độ khác, chỉ nói "frontier VLM" không nêu tên cụ thể; "From Handwriting to Structured Data" 04/2026 — chỉ test LLM đóng qua API trên FORM, không test OCR-VLM mã nguồn mở) đều **không** trùng dataset + danh mục model này → chỗ trống thật, khả năng bị "scoop" thấp trong 2 tuần tới.

**Danh sách mô hình/baseline (8):**
1. Tesseract (baseline cổ điển)
2. PaddleOCR hoặc EasyOCR (baseline deep-learning cổ điển)
3. TrOCR-large (HTR transformer pretrained, có số CER IAM công bố sẵn để đối chiếu)
4. Donut (VLM OCR-free, chưa test trên chữ viết tay/y khoa)
5. GOT-OCR2.0 (đã train trên IAM — cần ghi rõ caveat này — nhưng chưa test trên y khoa)
6. PaddleOCR-VL (chưa test trên IAM lẫn y khoa — mục tiêu mới quan trọng nhất)
7. Qwen2.5-VL/Qwen3-VL (VLM tổng quát, qua local hoặc API)
8. (Tùy chọn/bonus) 1 model đóng qua API (GPT-4V/Gemini/Claude) trên mẫu nhỏ để so sánh open-vs-closed

**Khối lượng công việc thực tế trong 2 tuần:** Khả thi cao. Dữ liệu nhỏ (~20MB Kaggle, IAM lấy subset), không cần huấn luyện — chỉ inference zero-shot, chạy được trên GPU tầm trung hoặc Colab/Kaggle-notebook free GPU. Rủi ro lớn nhất là dựng môi trường cho các VLM mới (PaddleOCR-VL, GOT-OCR2.0, DeepSeek-OCR — dependency/GPU/tải weight nhiều GB), nên cần buffer 2 ngày. IAM cần đăng ký trước (thủ tục nhanh, không rào cản) — nên làm ngay Ngày 1.

**Định dạng bài báo phù hợp:** **Benchmark/technical report ngắn (4-6 trang)**, đăng arXiv preprint và/hoặc nộp workshop (ICDAR workshop hoặc clinical-NLP workshop) — vì đóng góp là so sánh thực nghiệm hệ thống, không phải kiến trúc mới; khớp với format của các công trình liên quan gần nhất (RxScribe Bench, dev.to blog nâng cấp thành báo cáo học thuật đúng chuẩn).

---

### PHƯƠNG ÁN B

**Dataset:** RxHandBD (Zenodo DOI 10.5281/zenodo.18478741 / Mendeley DOI 10.17632/dsb5r6vskg.3) **+** Kaggle mamun1113 — dùng cả hai làm bài toán generalization chéo.

**Lý do uy tín:** Kaggle vẫn có paper bình duyệt như Phương án A. RxHandBD bổ sung vocab lớn hơn (1.559 từ vs 78 lớp) — nhưng **RxHandBD không có paper bình duyệt**, mới công bố 02-03/2026, license mâu thuẫn giữa Zenodo (MIT) và Mendeley (CC BY 4.0) — rủi ro uy tín học thuật nếu dùng làm trụ cột.

**Câu hỏi nghiên cứu:** Mô hình OCR-VLM pretrained zero-shot trên một dataset đơn thuốc viết tay có generalize sang dataset đơn thuốc khác (khác vocab, khác độ phân giải, khác nguồn thu thập) trong cùng miền hẹp (đơn thuốc Bangladesh) không? Có hiệu ứng của độ đa dạng từ vựng (78 lớp cố định vs. 1.559 từ mở) lên độ chính xác zero-shot?

**Vì sao có tính mới:** Chưa ai làm cross-dataset generalization trong nội miền "đơn thuốc Bangladesh" với các VLM-OCR mới — góc nhìn bổ sung cho Phương án A.

**Model list:** giống Phương án A.

**Khối lượng công việc:** Cao hơn A vì phải xử lý 2 format dữ liệu khác nhau, và **rủi ro cao hơn** vì reviewer có thể bắt lỗi "RxHandBD không đủ uy tín" (chưa bình duyệt, license mâu thuẫn) nếu dùng nó làm nguồn dữ liệu chính, không chỉ phụ.

**Định dạng:** Technical report/short paper — tương tự A nhưng cần đoạn "limitations" giải thích rõ RxHandBD chỉ là dữ liệu bổ sung/kiểm chứng, không phải trụ cột.

---

### PHƯƠNG ÁN C

**Dataset:** TCGA-Reports — ảnh gốc (`imgs_for_aws`) + kết quả OCR Textract (`aws_response`), Mendeley DOI 10.17632/hyg5xkznpx.1, cùng corpus text sạch trên GitHub jkefeli/tcga-path-reports.

**Lý do uy tín:** Paper Kefeli & Tatonetti, *Patterns* (Cell Press) 2024, license CC BY 4.0 rất thoải mái, dữ liệu đã có sẵn cặp ảnh+OCR để đối chiếu (không cần tự tạo ảnh giả).

**Câu hỏi nghiên cứu:** Các OCR/VLM mới (2024-2026) có vượt được AWS Textract (baseline dùng để tạo TCGA-Reports năm 2024) khi xử lý lại ảnh trang báo cáo giải phẫu bệnh gốc không? Cải thiện tập trung ở đâu (bảng dữ liệu, ảnh scan chất lượng thấp)?

**Vì sao có tính mới:** Chưa ai re-benchmark GOT-OCR2.0/Qwen-VL/PaddleOCR-VL/DeepSeek-OCR so với Textract trên chính bộ dữ liệu này.

**Model list:** giống trên, cộng Textract output có sẵn làm baseline (không cần chạy lại).

**Khối lượng công việc & rủi ro:** **Rủi ro TRUNG-CAO.** (1) Cấu trúc file trên trang Mendeley chưa xác minh được đầy đủ (tải bằng JS, chưa thấy hết trong lần kiểm tra) — cần dò lại ngay Ngày 1, có rủi ro cấu trúc phức tạp hơn dự kiến. (2) Đây là văn bản **in/đánh máy**, không phải chữ viết tay — bài toán "dễ hơn" về kỹ thuật, câu chuyện kém hấp dẫn hơn A. (3) Nếu muốn làm luôn phần downstream classification (như paper gốc, AU-ROC 0.992) để đo tác động OCR→NLP thì vượt quá 2 tuần — nên phải CẮT bỏ phần này, chỉ giữ so sánh chất lượng OCR thuần.

**Định dạng:** Technical report, phạm vi hẹp hơn A/B do phải cắt giảm.

---

### XẾP HẠNG

| Hạng | Phương án | Lý do |
|---|---|---|
| **1** | **A** | Dataset chính có paper bình duyệt + cộng đồng dùng rộng, tải ngay không rào cản, tính mới rõ nhất (đúng domain chữ viết tay y khoa — điểm nóng hiện tại theo RxScribe Bench/04-2026 paper), rủi ro kỹ thuật thấp nhất |
| 2 | B | Ý tưởng hay (cross-dataset) nhưng phụ thuộc RxHandBD chưa bình duyệt làm giảm độ tin cậy nếu dùng làm trụ cột — nên gộp làm phần *mở rộng* của A, không làm phương án độc lập |
| 3 | C | Câu chuyện kỹ thuật kém hấp dẫn hơn (văn bản in, không phải chữ viết tay), có lỗ hổng xác minh cấu trúc dữ liệu Mendeley chưa rõ, dễ vượt quá 2 tuần nếu tham vọng thêm downstream task |

**Chọn: PHƯƠNG ÁN A** làm trụ cột chính, có thể chèn RxHandBD (Phương án B) như một tiểu-mục "supplementary/robustness check" ở giai đoạn cuối nếu còn thời gian, và không cần tới TCGA-Reports (Phương án C) trong phạm vi 2 tuần.

**Xác nhận về mức độ sẵn sàng:** Không phương án nào trong 3 phương án trên cần chờ PhysioNet credentialing (đã bị loại từ vòng kiểm chứng vì không có dataset phù hợp) — Phương án A hoàn toàn **sẵn sàng thực hiện ngay hôm nay** (Kaggle: tài khoản miễn phí tải ngay; IAM: đăng ký online vài phút, không IRB/HREC). Vì vậy không cần phương án dự phòng do rào cản thời gian.

---

### KẾ HOẠCH CHI TIẾT 14 NGÀY — PHƯƠNG ÁN A

**Tên tạm:** *"Benchmarking Pretrained OCR and Vision-Language Models on Handwritten Medical Prescriptions: A Domain-Shift Study Against General Handwriting Recognition"*

**Metric:** CER, WER, Exact-Match accuracy (khớp với cách đo trong bài blog dev.to để so sánh được), cộng Top-1 classification accuracy trên 78 lớp cố định (khớp với cách đo của paper IEEE gốc để so sánh trực tiếp).

| Ngày | Việc chính |
|---|---|
| **1** | Đăng ký IAM (làm ngay để loại rủi ro chờ), tải Kaggle dataset qua Kaggle API, đọc kỹ paper IEEE iCACCESS 2024 để lấy đúng train/test split và số accuracy gốc, dựng repo + môi trường (Python, jiwer/editdistance để tính CER/WER) |
| **2** | Chuẩn hoá 2 dataset về format chung (path ảnh, label), viết + unit-test pipeline tính CER/WER/Exact-Match, chạy Tesseract + PaddleOCR/EasyOCR (baseline cổ điển) trên test split Kaggle |
| **3** | Chạy TrOCR-large trên test split Kaggle; chạy trên subsample IAM (để so sánh công bằng cùng code/prompt, đồng thời đối chiếu với số CER TrOCR đã công bố trên IAM làm sanity-check); bắt đầu dựng môi trường Donut |
| **4** | Chạy Donut zero-shot trên Kaggle (ghi rõ hạn chế nếu cần prompt OCR-free khác chuẩn); bắt đầu dựng môi trường GOT-OCR2.0 |
| **5** | Chạy GOT-OCR2.0 trên Kaggle + subsample IAM; buffer xử lý lỗi dependency/GPU (điểm rủi ro lớn nhất của cả kế hoạch) |
| **6** | *(buffer)* Hoàn tất các việc trễ của ngày 3-5; bắt đầu dựng môi trường PaddleOCR-VL (mới nhất, ít tài liệu — cần thêm thời gian); nếu ổn, viết sớm phần Methods (mô tả dataset, định nghĩa metric) |
| **7** | *(buffer)* Chạy PaddleOCR-VL trên Kaggle + subsample IAM; quyết định có đưa DeepSeek-OCR vào không (lit review cho thấy CER rất kém trên chữ viết tay không-Trung, có thể bỏ qua để ưu tiên Qwen-VL nếu thời gian gấp) |
| **8** | Dựng Qwen2.5-VL/Qwen3-VL (chọn API nếu không đủ GPU local cho model 7B+) |
| **9** | Chạy Qwen-VL trên Kaggle + subsample IAM; (tuỳ chọn) chạy 1 model đóng (GPT-4V/Gemini) trên mẫu nhỏ ~200 ảnh làm điểm so sánh bonus; gộp toàn bộ kết quả vào bảng tổng |
| **10** | Phân tích định lượng: CER/WER/Exact-Match/Top-1 accuracy theo model × dataset, khoảng tin cậy bootstrap, tính "domain-shift gap" (chênh lệch IAM vs Kaggle mỗi model), vẽ bảng/hình |
| **11** | Phân tích lỗi định tính: lấy mẫu case sai nặng nhất mỗi model, phân loại lỗi (nhầm tên thuốc gần giống, chữ không đọc được, VLM "hallucinate" từ ngoài từ vựng, lặp/loop) — nội dung phân biệt với bài blog dev.to hời hợt |
| **12** | *(tuỳ chọn/mở rộng)* Kiểm chứng bổ sung nhanh trên RxHandBD với 1-2 model tốt/kém nhất, ghi rõ là supplementary (không phải trụ cột) vì license/uy tín chưa rõ; viết Introduction + Related Work (định vị khác biệt với RxScribe Bench và "From Handwriting to Structured Data") |
| **13** | Hoàn thiện Results + Discussion + Limitations (phạm vi hẹp — closed-vocabulary 78 lớp, 1 quốc gia/ngôn ngữ, caveat GOT-OCR2.0 đã train trên IAM); rà soát nội bộ |
| **14** | Hoàn thiện Abstract, format theo template arXiv/workshop, chuẩn bị code + kết quả public trên GitHub (tăng độ tin cậy), sẵn sàng bản draft cuối |

**Rủi ro tổng thể và cách giảm:**
- **Lớn nhất:** dựng môi trường cho GOT-OCR2.0/PaddleOCR-VL/DeepSeek-OCR (dependency, tải weight nhiều GB) → đã có 2 ngày buffer (6-7); có thể chạy trên Kaggle Notebook (data đã ở đó, có GPU free-tier) hoặc Colab.
- **Đăng ký IAM chậm:** nếu trễ, dùng số CER TrOCR/GOT-OCR2.0 đã công bố trong literature làm điểm đối chiếu tổng quát thay cho tự chạy lại toàn bộ IAM — không chặn tiến độ.
- **Chi phí API model đóng:** giữ ở mức tuỳ chọn/mẫu nhỏ, không phải phần bắt buộc của đóng góp chính.
- **Rủi ro học thuật:** Kaggle dataset là closed-vocabulary 78 lớp (không phải free-text OCR hoàn toàn mở) — cần nêu rõ trong phần Limitations để không bị reviewer bắt lỗi overclaim, không phải rủi ro chặn tiến độ.
- **Đạo đức dữ liệu:** cả 2 dataset đều công khai, không có định danh bệnh nhân (IAM là chữ viết tay tổng quát; Kaggle là ảnh crop tên thuốc, không có thông tin bệnh nhân) — khớp với khung "nghiên cứu độc lập, không cần IRB/HREC" mà bạn đã xác định.
---

## 5.2. Kế hoạch hành động 14 ngày chi tiết (Phương án A)

## KẾ HOẠCH THỰC HIỆN 14 NGÀY — Benchmark OCR/VLM Pretrained trên Đơn thuốc Viết tay (Phương án A)

**Bắt đầu:** Thứ Ba 15/09/2026 → **Kết thúc:** Thứ Hai 28/09/2026 (bản thảo hoàn chỉnh + nộp arXiv)
**Lưu ý về kỳ vọng:** "Hoàn thành trong 2 tuần" = có bản thảo đầy đủ, code/kết quả public, đã nộp preprint — **KHÔNG** phải đã được bình duyệt/xuất bản. Bình duyệt workshop mất thêm 4–8 tuần, hội nghị chính mất 3–6 tháng nữa, nằm ngoài phạm vi 14 ngày này.

---

### 1. BẢNG KẾ HOẠCH THEO NGÀY

| Ngày (thứ, ngày) | Công việc cụ thể | Đầu ra kỳ vọng cuối ngày |
|---|---|---|
| **1** (Thứ Ba 15/09) | - Đăng ký tài khoản FKI để xin quyền tải IAM tại `https://fki.tic.heia-fr.ch/databases/iam-handwriting-database` (nộp form ngay sáng sớm vì thời gian duyệt không cố định, có thể vài giờ tới vài ngày)<br>- Tạo repo code (`git init`), venv, cài toàn bộ dependency nền (xem Mục 2)<br>- Tải Kaggle dataset qua `kaggle datasets download -d mamun1113/doctors-handwritten-prescription-bd-dataset` (4.680 ảnh, 78 lớp tên thuốc)<br>- Đọc kỹ paper Mia et al. (IEEE iCACCESS 2024, doc 10499631) để lấy đúng train/test split gốc và số accuracy baseline để đối chiếu<br>- Tạo file `manifest.csv` (path ảnh, label, split) | Repo khởi tạo, dữ liệu Kaggle sẵn sàng, đơn IAM đã nộp, môi trường cài xong ~70% |
| **2** (Thứ Tư 16/09) | - Chốt test split: dùng đúng split gốc của paper IEEE nếu có công bố, hoặc tự chia ngẫu nhiên theo tỉ lệ 80/20 có stratify theo 78 lớp (~936 ảnh test) — cố định seed, lưu file split để tái lập<br>- Viết + unit-test module tính metric: `compute_metrics.py` (CER, WER qua `jiwer`, Exact-Match, Top-1 accuracy trên 78 lớp)<br>- Chạy Tesseract (pytesseract) + 1 baseline deep-learning cổ điển (EasyOCR **hoặc** PaddleOCR PP-OCRv5) trên toàn bộ 936 ảnh test<br>- Kiểm tra IAM: nếu đã được cấp quyền → tải subset (xem Mục 3 về kích cỡ); nếu chưa → chuyển sang Rủi ro 3 (Mục 6) | Kết quả CER/WER/Exact-Match cho 2 baseline cổ điển; pipeline đo lường đã validate |
| **3** (Thứ Năm 17/09) | - Cài `transformers`, tải `microsoft/trocr-large-handwritten` từ HuggingFace<br>- Chạy TrOCR trên 936 ảnh test Kaggle (ước lượng: GPU tầm trung ~0,1–0,3 s/ảnh → 3–5 phút; CPU sẽ mất hàng giờ, **ưu tiên GPU/Colab/Kaggle-Notebook**)<br>- Chạy TrOCR trên subsample IAM (300–500 ảnh) để đối chiếu với số CER TrOCR đã công bố trong paper gốc (sanity-check pipeline)<br>- Bắt đầu dựng môi trường Donut (`naver-clova-ix/donut-base`) | Kết quả TrOCR trên cả 2 dataset; số CER IAM khớp/lệch bao nhiêu so với literature (ghi log) |
| **4** (Thứ Sáu 18/09) | - Chạy Donut zero-shot trên Kaggle test set (dùng task prompt `<s_synthdog>` hoặc tương đương cho "đọc chữ thô", KHÔNG dùng bản fine-tune CORD vì đó là task trích xuất trường hoá đơn, không phải free-text OCR — ghi rõ caveat này trong Method)<br>- Bắt đầu dựng môi trường GOT-OCR2.0 (`stepfun-ai/GOT-OCR-2.0-hf`, cần `transformers` mới hoặc cài từ git nếu lỗi tương thích) | Kết quả Donut (kèm ghi chú hạn chế); môi trường GOT-OCR2.0 cài xong hoặc log lỗi cụ thể |
| **5** (Thứ Bảy 19/09) | - Chạy GOT-OCR2.0 trên Kaggle test set + subsample IAM (ước lượng ~0,5–1 s/ảnh trên GPU → 936 ảnh ≈ 15–20 phút)<br>- **Buffer xử lý lỗi dependency/GPU** — đây là điểm rủi ro kỹ thuật lớn nhất, dành hẳn nửa ngày dự phòng<br>- Nếu máy cá nhân không đủ VRAM: chuyển sang chạy trên Kaggle Notebook (free GPU T4, dữ liệu đã có sẵn ở đó) hoặc Google Colab | Kết quả GOT-OCR2.0 trên cả 2 dataset (đã ghi rõ caveat: model này đã được train trên IAM nên số IAM không phải zero-shot thật) |
| **6–7** (CN 20/09 – Thứ Hai 21/09, **buffer**) | - Hoàn tất mọi việc trễ của Ngày 3–5<br>- Dựng môi trường PaddleOCR-VL: `pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/` rồi `pip install -U "paddleocr[doc-parser]>=3.6.0"` (dùng bản PaddleOCR-VL-1.6, phát hành 05/2026 — mới nhất, ổn định hơn bản gốc 10/2025); nếu không có GPU CUDA phù hợp, thử bản GGUF (`PaddlePaddle/PaddleOCR-VL-1.6-GGUF`) chạy qua CPU/llama.cpp<br>- Chạy PaddleOCR-VL trên Kaggle test set + subsample IAM<br>- Nếu còn dư thời gian trong buffer: bắt đầu viết sớm phần Methods (mô tả dataset, định nghĩa metric — phần này không phụ thuộc kết quả) | Kết quả PaddleOCR-VL; toàn bộ backlog kỹ thuật của tuần 1 đã xử lý xong; draft thô phần Methods |
| **8** (Thứ Ba 22/09) | - Quyết định danh mục model cuối: dựng `Qwen/Qwen3-VL-8B-Instruct` (ưu tiên, nếu đủ VRAM ≥16GB) hoặc `Qwen/Qwen2.5-VL-7B-Instruct` làm phương án nhẹ hơn<br>- Cân nhắc DeepSeek-OCR (`deepseek-ai/DeepSeek-OCR`, ~6,7GB, cần CUDA 11.8+/torch 2.6.0): theo literature CER trên chữ viết tay không-Trung khá kém — nếu thời gian gấp, **có thể bỏ qua để ưu tiên Qwen-VL** như phương án đã đề xuất | Môi trường Qwen-VL sẵn sàng; quyết định rõ có/không chạy DeepSeek-OCR |
| **9** (Thứ Tư 23/09) | - Chạy Qwen-VL trên Kaggle test set + subsample IAM (ước lượng trên GPU T4: 2–5 s/ảnh → 936 ảnh ≈ 30–75 phút; nếu chạy qua API, xem thời gian ở Mục 2)<br>- (Tuỳ chọn/bonus) Chạy 1 model đóng (GPT-5/GPT-5-mini vision hoặc Gemini 2.5/3.5 Flash) qua API trên mẫu nhỏ 200–300 ảnh<br>- Gộp toàn bộ kết quả các model vào bảng tổng `results_master.csv` | Toàn bộ 6–8 model đã có kết quả trên cả 2 dataset; bảng tổng hợp hoàn chỉnh |
| **10** (Thứ Năm 24/09) | - Phân tích định lượng: CER/WER/Exact-Match/Top-1 accuracy theo model × dataset<br>- Tính khoảng tin cậy 95% bằng bootstrap (1.000 lần resample) cho mỗi model<br>- Tính "domain-shift gap" = CER(IAM) − CER(Kaggle) cho mỗi model<br>- Kiểm định thống kê cặp (Wilcoxon signed-rank trên CER từng ảnh) để so sánh model tốt nhất vs model nhì<br>- Vẽ bảng kết quả chính + biểu đồ domain-shift gap | Bảng/hình kết quả chính hoàn chỉnh, có ý nghĩa thống kê rõ ràng |
| **11** (Thứ Sáu 25/09) | - Phân tích lỗi định tính: lấy 15–20 case sai nặng nhất mỗi model, phân loại lỗi (nhầm tên thuốc gần giống về hình dạng chữ, không đọc được, VLM "hallucinate" từ ngoài từ vựng 78 lớp, lặp/loop token)<br>- Đây là phần tạo khác biệt rõ với bài blog dev.to (chỉ báo cáo số, không phân tích lỗi)<br>- Ghi log tỉ lệ "output không hội tụ/degenerate" riêng cho từng model (không tính chung vào CER để tránh lệch số liệu) | Bảng phân loại lỗi + 3–5 ví dụ minh hoạ cho phần Discussion |
| **12** (Thứ Bảy 26/09, **tuỳ chọn/mở rộng**) | - Nếu đúng tiến độ: kiểm chứng bổ sung nhanh trên RxHandBD (Zenodo DOI 10.5281/zenodo.18478741) với 1–2 model tốt/kém nhất, ghi rõ là *supplementary check*, không phải trụ cột (vì license mâu thuẫn MIT/CC-BY-4.0 và chưa bình duyệt)<br>- Viết Introduction + Related Work, định vị rõ khác biệt với RxScribe Bench (09/2026) và "From Handwriting to Structured Data" (04/2026) | Draft Introduction + Related Work; (tuỳ chọn) 1 bảng phụ RxHandBD |
| **13** (Chủ Nhật 27/09) | - Hoàn thiện Results + Discussion + Limitations (phạm vi hẹp: closed-vocabulary 78 lớp, 1 quốc gia/ngôn ngữ, caveat GOT-OCR2.0 đã train trên IAM)<br>- Rà soát nội bộ toàn bài, kiểm tra số liệu khớp giữa bảng và văn bản<br>- Dọn code, viết README, đóng gói kết quả `results_master.csv` public trên GitHub | Bản thảo gần hoàn chỉnh (thiếu Abstract cuối + format) |
| **14** (Thứ Hai 28/09) | - Viết Abstract cuối cùng<br>- Format theo template arXiv (hoặc template workshop nếu đã chọn nơi nộp)<br>- Public code + data pointer trên GitHub (tăng độ tin cậy, đúng chuẩn reproducibility)<br>- Nộp arXiv (cs.CV hoặc cs.CL) | **Bản thảo hoàn chỉnh + đã nộp arXiv** |

---

### 2. DANH SÁCH CÀI ĐẶT NGÀY 1 + ƯỚC LƯỢNG CHI PHÍ/THỜI GIAN

#### 2.1. Môi trường nền
```bash
python -m venv venv && source venv/bin/activate     # (Windows: venv\Scripts\activate)
pip install torch torchvision --index-url <phù hợp CUDA của máy>
pip install jiwer editdistance python-Levenshtein pandas numpy matplotlib seaborn scipy statsmodels
pip install kaggle
kaggle datasets download -d mamun1113/doctors-handwritten-prescription-bd-dataset
```

#### 2.2. OCR cổ điển
```bash
pip install pytesseract            # + cài binary Tesseract của hệ điều hành
pip install easyocr
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -U "paddleocr[doc-parser]>=3.6.0"      # dùng chung cho PP-OCR cổ điển và PaddleOCR-VL
```

#### 2.3. HTR/VLM pretrained (HuggingFace)
```bash
pip install -U "huggingface_hub[cli]" transformers accelerate tiktoken verovio

hf download microsoft/trocr-large-handwritten
hf download naver-clova-ix/donut-base
hf download stepfun-ai/GOT-OCR-2.0-hf          # dùng AutoModelForImageTextToText
hf download PaddlePaddle/PaddleOCR-VL-1.6       # bản mới nhất (05/2026), migrate zero-cost từ 1.5
hf download deepseek-ai/DeepSeek-OCR            # ~6,7GB, cần CUDA 11.8+/torch 2.6.0 — tuỳ chọn, xem Ngày 8
hf download Qwen/Qwen3-VL-8B-Instruct           # hoặc Qwen/Qwen2.5-VL-7B-Instruct nếu VRAM hạn chế
```
*Lưu ý:* nếu `transformers` bản pip chưa hỗ trợ GOT-OCR2/PaddleOCR-VL, cài từ nguồn: `pip install git+https://github.com/huggingface/transformers.git`.

#### 2.4. API keys (chỉ cho phần bonus tuỳ chọn Ngày 9)
- **Google AI Studio** (Gemini API key, có free tier) — đăng ký tại aistudio.google.com
- **OpenAI Platform** (API key cho GPT-5/GPT-5-mini vision)

#### 2.5. Ước lượng chi phí & thời gian chạy thực tế

| Model | Cách chạy | Ước lượng thời gian (test set ~936 ảnh Kaggle) | Chi phí |
|---|---|---|---|
| Tesseract, EasyOCR/PaddleOCR | Local CPU/GPU | Vài phút | $0 |
| TrOCR-large, Donut-base | GPU tầm trung | ~3–8 phút | $0 |
| GOT-OCR2.0 (~580M) | GPU (T4 trở lên) | ~15–20 phút | $0 |
| PaddleOCR-VL (0,9B) | GPU hoặc CPU (bản GGUF) | ~15–30 phút (GPU) / lâu hơn nếu CPU | $0 |
| DeepSeek-OCR (~6,7GB weight) | GPU ≥16GB VRAM | ~30–60 phút | $0 (tự host) |
| Qwen3-VL-8B / Qwen2.5-VL-7B | GPU ≥16GB VRAM hoặc API | ~30–75 phút (GPU) | $0 (tự host) hoặc theo giá API |
| GPT-5(-mini) vision (bonus, ~250 ảnh) | API | Với concurrency 5, batch xử lý ~30–60 phút thực tế (kể cả rate-limit/retry) | Ước ~$0,001/ảnh × 250 ≈ **$0,25–$0,50** |
| Gemini 2.5/3.5 Flash (bonus, ~250 ảnh) | API | Tương tự, ~20–40 phút | Ước ~$0,0005–0,001/ảnh × 250 ≈ **$0,15–$0,25** |

**Tổng chi phí API dự kiến cho toàn bộ phần bonus: dưới $1**. Giá API tính theo token (ảnh + prompt + output), không phải giá cố định/ảnh — số trên là ước lượng dựa trên mức giá công khai tại thời điểm 09/2026 ($0,625/$5 triệu token GPT-5; $0,30/$2,50 triệu token Gemini 2.5 Flash), **cần kiểm tra lại trang pricing chính thức trước khi chạy thật** vì giá có thể thay đổi.

---

### 3. THIẾT KẾ THÍ NGHIỆM

**Metric:**
- **CER** (Character Error Rate, qua `jiwer`) — metric chính, so sánh được với số CER TrOCR/GOT-OCR2.0 đã công bố trên IAM (sanity-check)
- **WER** (Word Error Rate) — phụ, vì phần lớn nhãn Kaggle là 1 từ/cụm ngắn
- **Exact-Match accuracy** (chuẩn hoá: lowercase, trim khoảng trắng) — khớp cách đo của bài blog dev.to để so sánh chéo được
- **Top-1 classification accuracy trên 78 lớp cố định** (ánh xạ output OCR về nhãn gần nhất trong từ điển 78 tên thuốc) — khớp cách đo của paper IEEE gốc để so sánh trực tiếp với baseline 2024
- Tỉ lệ **"output không hội tụ/degenerate"** ghi riêng, không gộp vào CER trung bình

**Chia dữ liệu:**
- Kaggle: 4.680 ảnh, 78 lớp — dùng lại split gốc của paper IEEE nếu công bố, nếu không thì tự chia 80/20 stratify theo lớp → **test set ≈ 936 ảnh**. Với n≈936, sai số chuẩn của CER trung bình (giả định độ lệch chuẩn per-ảnh ~0,3) ≈ 1%, CI 95% ≈ ±2%, đủ để phát hiện khác biệt >4–5% CER giữa các model — phù hợp cho benchmark study trong 2 tuần, không cần toàn bộ 4.680 ảnh cho mỗi model (tiết kiệm thời gian, đặc biệt với VLM chậm).
- IAM: lấy **subsample 300–500 ảnh word/line** (không chạy toàn bộ IAM — không cần thiết cho câu hỏi domain-shift, và giữ thời gian/chi phí trong tầm kiểm soát với các model chạy qua API).
- Bonus API (GPT-5/Gemini): subsample nhỏ hơn nữa, **200–300 ảnh**, đủ để có xu hướng so sánh open-vs-closed mà không đội chi phí/thời gian.

**Kiểm tra thủ công:**
- Kiểm tra chéo tay ~50 ảnh ngẫu nhiên trong test set để xác nhận nhãn thư mục = ground truth chính xác (loại rủi ro lỗi gán nhãn của dataset crowd-sourced).
- Xem tay 15–20 case lỗi nặng nhất mỗi model (Ngày 11) để phân loại lỗi định tính — không phải để "sửa" ground truth, mà để phục vụ phần Discussion.

---

### 4. CẤU TRÚC BÀI BÁO NGẮN (ước tính ~5–6 trang, ~3.000 từ, khung ICDAR/DAS short paper)

| Phần | Nội dung | Ước tính |
|---|---|---|
| **Abstract** | Câu hỏi nghiên cứu, phương pháp, kết quả chính (số CER tốt nhất/kém nhất, domain-shift gap) | 150–200 từ |
| **1. Introduction** | Động lực (đơn thuốc viết tay là nguồn lỗi y khoa thực tế), khoảng trống cụ thể (chưa ai test GOT-OCR2.0/PaddleOCR-VL/DeepSeek-OCR/Qwen-VL trên HTR y khoa), liệt kê 3–4 đóng góp | 500–600 từ (~0,75–1 trang) |
| **2. Related Work** | IAM (Marti & Bunke 2002), dataset Kaggle (Mia et al. 2024), các công trình gần nhất (RxScribe Bench 09/2026, "From Handwriting to Structured Data" 04/2026, blog benchmark dev.to) — nêu rõ vì sao không trùng | 300–400 từ (~0,5 trang) |
| **3. Method/Experimental Setup** | Bảng mô tả 2 dataset, bảng danh mục model (kiến trúc, số tham số, nguồn), định nghĩa metric, chi tiết split | 600–800 từ + 2 bảng (~1–1,25 trang) |
| **4. Results** | Bảng kết quả chính (model × dataset × metric), hình domain-shift gap, khoảng tin cậy | 400–500 từ + 2–3 bảng/hình (~1,25–1,5 trang) |
| **5. Discussion** | Phân tích lỗi định tính, giải thích vì sao model train-trên-IAM (GOT-OCR2.0) vẫn kém trên y khoa, so sánh họ model | 400–500 từ (~0,5–0,75 trang) |
| **6. Limitations** | Closed-vocabulary 78 lớp, 1 ngôn ngữ/1 quốc gia, caveat GOT-OCR2.0, RxHandBD chỉ là supplementary | 150–200 từ (~0,25 trang) |
| **7. Conclusion** | Tóm 2–3 câu | 100–150 từ |
| **References** | ~25–35 trích dẫn | ~0,5–1 trang |

---

### 5. NƠI CÔNG BỐ NHANH (chỉ là NỘP, không đảm bảo CHẤP NHẬN trong 14 ngày)

1. **arXiv** (cs.CV hoặc cs.CL) — không cần bình duyệt, công khai gần như ngay (vài giờ–1–2 ngày moderation cho tài khoản mới). Đây là bước **duy nhất thực sự hoàn thành trong 14 ngày** và cũng là cách xác lập "priority" cho tính mới trước khi có nguy cơ bị người khác công bố trước.
2. **Workshop ngắn gắn với ICDAR/DAS** (Document Analysis Systems) — track "short paper"/"resource paper" nếu đang mở CFP. **Cần tự kiểm tra lịch CFP hiện hành** vì ICDAR/DAS họp theo chu kỳ 2 năm và lịch cụ thể tại thời điểm 09/2026 có thể không khớp với cửa sổ nộp 14 ngày này — nộp nếu còn hạn, nếu không thì giữ bản thảo cho vòng CFP kế tiếp.
3. **Workshop Clinical-NLP đi kèm hội nghị NLP lớn** (ví dụ track "Findings"/"short paper" của một workshop Clinical NLP tại ACL/EMNLP/NAACL, hoặc ML4Health-style workshop) — phù hợp vì bài nằm giữa OCR benchmark và ứng dụng y khoa; các track "findings" thường có deadline rolling/ngắn hơn track chính — **kiểm tra CFP cụ thể trước khi nộp**.
4. **Papers with Code / Hugging Face Papers** (liên kết tới bản arXiv) — không phải bình duyệt, nhưng tăng khả năng được cộng đồng biết đến nhanh và làm nơi neo code+data reproducibility.

---

### 6. RỦI RO CHÍNH VÀ PHƯƠNG ÁN B

| # | Rủi ro | Phương án B |
|---|---|---|
| 1 | Cài đặt GOT-OCR2.0/PaddleOCR-VL/DeepSeek-OCR lỗi dependency/CUDA (rủi ro lớn nhất theo đánh giá gốc) | Chuyển sang chạy trên Kaggle Notebook/Google Colab (GPU T4 free-tier, CUDA đã cấu hình sẵn); nếu vẫn lỗi, dùng bản GGUF/quantized (chạy CPU qua llama.cpp) hoặc loại bỏ model đó khỏi danh sách và ghi rõ "loại trừ do hạn chế môi trường" trong Limitations — không chặn tiến độ tổng thể vì đây chỉ là 1/7–8 model |
| 2 | API rate-limit hoặc chi phí vượt dự kiến (GPT-5/Gemini/Qwen-VL qua API) | Giảm mẫu bonus xuống 100–150 ảnh; thêm exponential backoff/retry; nếu vẫn kẹt, **bỏ hẳn phần bonus model đóng** — nó đã được đánh dấu tuỳ chọn từ đầu, không ảnh hưởng đóng góp chính (so sánh các model mã nguồn mở) |
| 3 | IAM chưa được cấp quyền đăng ký kịp trước Ngày 3 | Dùng tạm mirror công khai (Kaggle "iam_handwriting_word_database" hoặc HuggingFace "xReniar/IAM-Dataset") cho tiến độ, ghi rõ caveat license trong Limitations; hoặc với các model không kịp tự chạy lại trên IAM, dùng số CER IAM đã công bố sẵn trong paper gốc của model đó (TrOCR, GOT-OCR2.0) làm điểm đối chiếu domain-shift thay vì tự đo — không chặn tiến độ |
| 4 | Model cho output lặp/loop hoặc hallucinate hoàn toàn ngoài từ điển 78 lớp (đặc biệt Donut không đúng prompt, hoặc DeepSeek-OCR trên chữ không-Trung) | Giới hạn `max_new_tokens`, thêm repetition penalty; log các case này riêng thành "tỉ lệ không hội tụ" thay vì tính thẳng vào CER trung bình (tránh làm lệch thống kê) |
| 5 | Kết quả không có tính mới như kỳ vọng (ví dụ tất cả model đều fail giống nhau, hoặc domain-shift gap không đáng kể) | Chuyển trọng tâm bài báo từ "model nào tốt nhất" sang "đặc điểm hoá kiểu lỗi" (phần phân tích định tính Ngày 11 vẫn giữ giá trị đóng góp độc lập); hoặc mở rộng thêm nghiên cứu phụ RxHandBD (Ngày 12) thành một mục lớn hơn để tạo góc nhìn thứ hai trong cùng bài |
| 6 | Rủi ro bị reviewer bắt lỗi overclaim (dataset closed-vocabulary 78 lớp bị trình bày như OCR mở hoàn toàn) | Đã tách rõ 2 metric (CER free-text vs. Top-1 classification 78 lớp) trong Mục 3, và nêu rõ trong Limitations — xử lý trước, không phải phương án dự phòng |
| 7 | Trễ tiến độ tổng thể do dồn nhiều lỗi kỹ thuật | Buffer Ngày 6–7 đã dành sẵn; nếu vẫn chưa đủ, cắt theo thứ tự ưu tiên đã đánh dấu "tuỳ chọn" từ đầu: (a) bỏ RxHandBD Ngày 12, (b) bỏ bonus API model đóng Ngày 9, (c) bỏ DeepSeek-OCR — giữ nguyên lõi 5–6 model bắt buộc (Tesseract, 1 OCR cổ điển deep-learning, TrOCR, Donut, GOT-OCR2.0, PaddleOCR-VL hoặc Qwen-VL) để đảm bảo vẫn nộp được arXiv đúng hạn Ngày 14 |

---

#### Nguồn tra cứu dùng để cụ thể hoá kế hoạch
- [stepfun-ai/GOT-OCR-2.0-hf (Hugging Face)](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- [PaddlePaddle/PaddleOCR-VL-1.6 (Hugging Face)](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- [deepseek-ai/DeepSeek-OCR (Hugging Face)](https://huggingface.co/deepseek-ai/DeepSeek-OCR) / [GitHub deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- [Qwen/Qwen3-VL-8B-Instruct (Hugging Face)](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Doctor's Handwritten Prescription BD dataset (Kaggle)](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset)
- [IAM Handwriting Database (FKI)](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [Gemini API pricing 2026](https://ai.google.dev/gemini-api/docs/pricing)
- [OpenAI API pricing 2026 (tổng hợp)](https://www.finout.io/blog/openai-pricing-in-2026)