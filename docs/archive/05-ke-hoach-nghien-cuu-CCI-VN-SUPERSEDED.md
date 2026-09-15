> **⚠️ ĐÃ THAY THẾ (SUPERSEDED) — 15/09/2026.** Kế hoạch 12 tháng này giả định có dữ liệu bệnh viện/CCI thật và cần IRB/HREC — không tương thích với yêu cầu "nghiên cứu độc lập, gấp trong 2 tuần" của người dùng. Xem kế hoạch hiện tại tại [`05-ke-hoach-2-tuan.md`](../05-ke-hoach-2-tuan.md). Giữ lại file này chỉ để tham khảo/đối chiếu.

---

# KẾ HOẠCH NGHIÊN CỨU VÀ LỘ TRÌNH CÔNG BỐ
## Chủ đề: Pipeline OCR + LLM tự động trích xuất dữ liệu đăng ký ung thư nhi khoa

**Hướng chính (đã chọn): Hướng 1** — Pipeline OCR+LLM cho Pediatric Cancer Registry Automation
**Phương án dự phòng: Hướng 2 hoặc Hướng 3** — nếu không xin được dữ liệu bệnh nhi thực trong 3-4 tháng đầu, chuyển hướng nghiên cứu sinh sang Hướng 2 (chữ viết tay lâm sàng, dữ liệu công khai IAM/RxHandBD, không cần IRB nặng nề) hoặc Hướng 3 (benchmark OCR y tế tiếng Việt) — hai hướng này dùng chung phần lớn hạ tầng kỹ thuật (OCR/VLM, quy trình fine-tune, thiết kế đánh giá CER/WER/F1) nên việc chuyển hướng không làm mất công sức đã đầu tư ở giai đoạn build pipeline.

Ghi chú bối cảnh: đề xuất gốc đã gợi ý "CCIA/Children's Cancer Institute Úc" và Australian Childhood Cancer Registry (ACCR — do CCI/Cancer Council NSW/AIHW đồng vận hành) như một đối tác dữ liệu tự nhiên cho hướng này; nếu nghiên cứu sinh có quan hệ thể chế với CCIA thì đây là con đường tiếp cận dữ liệu khả thi nhất và nên được ưu tiên liên hệ ngay từ Tháng 1.

---

# PHẦN 1 — KẾ HOẠCH NGHIÊN CỨU CỤ THỂ

## 1.1. Câu hỏi nghiên cứu và giả thuyết

**Câu hỏi nghiên cứu chính (RQ):**
> Một pipeline kết hợp OCR/VLM + LLM có thể trích xuất tự động các biến đăng ký chuẩn hóa (theo phân loại ICCC-3 hoặc schema đăng ký ung thư nhi khoa quốc gia) từ báo cáo giải phẫu bệnh và tóm tắt bệnh án ung thư trẻ em dạng scan/PDF, đạt độ chính xác đủ để hỗ trợ (không thay thế hoàn toàn) nhân viên đăng ký ung thư hay không?

**Các câu hỏi phụ (sub-RQ):**
- RQ2: So với các mô hình đã công bố cho *người lớn* (F1 0,85–0,95), hiệu năng suy giảm bao nhiêu khi áp dụng trực tiếp (zero-shot/transfer) sang báo cáo ung thư *trẻ em* — nơi thuật ngữ mô bệnh học (embryonal tumors, đột biến di truyền, phân loại WHO cho u não trẻ em...) khác biệt đáng kể?
- RQ3: Với các nhóm bệnh "underrepresented" đã được Hands & Kavuluru (2025) chỉ ra (u hắc tố, lymphoma, ung thư nhi khoa nói chung), độ chính xác trích xuất có thấp hơn có ý nghĩa thống kê so với nhóm bệnh phổ biến (leukemia, u não) hay không?
- RQ4: Cơ chế "human-in-the-loop" (đánh dấu ca có độ tin cậy thấp để người kiểm tra) có thể giảm khối lượng công việc thủ công bao nhiêu % trong khi vẫn giữ độ chính xác tổng thể ≥ ngưỡng chấp nhận lâm sàng (ví dụ ≥95% cho trường "chẩn đoán chính")?

**Giả thuyết (H):**
- **H1:** Pipeline OCR+LLM đạt F1 ≥ 0,80 ở cấp độ trường (field-level) cho các biến cấu trúc chính (mã ICCC, tuổi chẩn đoán, ngày chẩn đoán), thấp hơn baseline người lớn nhưng vẫn vượt đáng kể so với rule-based NLP truyền thống.
- **H2:** Hiệu năng trên nhóm bệnh hiếm (melanoma, lymphoma nhi khoa) thấp hơn có ý nghĩa thống kê so với nhóm phổ biến (do out-of-distribution thuật ngữ), xác nhận định lượng khoảng trống mà Hands & Kavuluru nêu định tính.
- **H3:** Fine-tune nhẹ (LoRA, vài trăm–vài nghìn ca gán nhãn) trên domain nhi khoa cải thiện F1 đáng kể so với zero-shot của mô hình tổng quát (GPT-4o/Gemini), thu hẹp khoảng cách với baseline người lớn.

## 1.2. Phương pháp luận

### a) Dữ liệu

**Ưu tiên 1 — Dữ liệu thật (đường chính):**
- Liên hệ Australian Childhood Cancer Registry (ACCR) / CCIA, hoặc ANZCHOG (Australian and New Zealand Children's Haematology/Oncology Group), hoặc bệnh viện nhi tại Việt Nam (Bệnh viện Nhi Trung ương, Bệnh viện Ung Bướu TP.HCM) để xin bộ báo cáo giải phẫu bệnh/tóm tắt bệnh án ung thư trẻ em đã hồi cứu (retrospective), ẩn danh hóa theo quy trình chuẩn của tổ chức.
- Quy mô mục tiêu: 300–800 báo cáo (đủ cho fine-tune LoRA nhẹ + test set thống kê có ý nghĩa; các nghiên cứu tương tự — Tay et al. 2026 — dùng quy mô vài trăm đến vài nghìn ca).
- Schema gán nhãn: dựa theo ICCC-3 (International Classification of Childhood Cancer, 3rd edition) + các trường COG (Children's Oncology Group) chuẩn: chẩn đoán chính, mã hình thái học (morphology code), giai đoạn/stage, tuổi tại chẩn đoán, vị trí giải phẫu, ngày chẩn đoán, cơ sở điều trị.

**Ưu tiên 2 — Dữ liệu tổng hợp (synthetic) để làm proof-of-concept trước khi có dữ liệu thật (giảm rủi ro đạo đức/thời gian chờ IRB):**
- Sinh báo cáo giải phẫu bệnh tổng hợp bằng LLM (GPT-4/Claude) dựa trên template thật đã công khai hóa (case reports trên PubMed Central, sách giáo khoa mô bệnh học nhi khoa), sau đó render thành ảnh scan giả lập (thêm nhiễu, nghiêng, mờ — augmentation) để mô phỏng điều kiện scan bệnh viện thực tế.
- Mục đích: kiểm chứng kỹ thuật pipeline hoạt động đúng trước khi đưa vào dữ liệu thật nhạy cảm — đồng thời có thể công bố kèm bài báo như "synthetic benchmark" bổ sung (giải quyết một phần vấn đề thiếu benchmark công khai).

**Đạo đức và quyền riêng tư:**
- Nộp hồ sơ HREC/IRB ngay từ Tháng 1 (đây là bước có độ trễ dài nhất, thường 2-4 tháng) — xin phê duyệt dạng "retrospective, de-identified, waiver of consent" (thường khả thi hơn với dữ liệu hồi cứu ẩn danh).
- Toàn bộ xử lý PHI trẻ em (nhạy cảm hơn PHI người lớn) phải chạy on-premise (không gọi API cloud thương mại như GPT-4V trực tiếp trên dữ liệu chưa ẩn danh) — dùng VLM/LLM mã nguồn mở (Qwen2.5-VL, InternVL3.5, Llama) chạy nội bộ, hoặc chỉ gọi API cloud sau khi đã ẩn danh hóa (che tên, ID, ngày sinh chính xác) bằng bước tiền xử lý riêng.

### b) Kiến trúc mô hình / pipeline đề xuất

Pipeline 3 giai đoạn:

1. **Giai đoạn OCR/VLM (Document → Text có cấu trúc):**
   - Ứng viên chính: PaddleOCR-VL (nhẹ, đa ngôn ngữ, chạy on-premise tốt) hoặc GOT-OCR2.0 (mạnh về layout phức tạp).
   - So sánh với Tesseract (baseline OCR cổ điển) để định lượng lợi ích của VLM hiện đại.

2. **Giai đoạn trích xuất trường (Text → Structured fields theo schema ICCC/COG):**
   - LLM zero-shot/few-shot với prompt có cấu trúc (JSON schema-constrained generation) — thử nghiệm cả mô hình đóng (GPT-4o, Gemini — chỉ trên dữ liệu đã ẩn danh) và mô hình mở fine-tune (Llama-3, Qwen2.5 fine-tune LoRA trên vài trăm ca gán nhãn).
   - Kỹ thuật: chain-of-thought có kiểm soát (trích xuất từng trường kèm "trích dẫn nguồn" từ văn bản gốc để tăng khả năng kiểm tra/audit — quan trọng cho ứng dụng lâm sàng).

3. **Giai đoạn human-in-the-loop (Confidence scoring → Routing):**
   - Mô hình tự chấm điểm tin cậy cho từng trường (dựa trên log-probability hoặc self-consistency qua nhiều lần sinh) → trường có độ tin cậy thấp được gắn cờ để nhân viên đăng ký rà soát thủ công (theo mô hình Tay et al. 2026, chấp nhận ~23% cần rà soát).

### c) Baseline cần so sánh

| Baseline | Vai trò |
|---|---|
| Con người (double-entry bởi 2 nhân viên đăng ký, đối chiếu) | Gold standard / upper bound |
| Rule-based NLP truyền thống (regex + từ điển thuật ngữ, tương tự phương pháp Yoon et al. 2022 dùng cho 29.206 báo cáo ung thư trẻ em) | Baseline "cũ" — đo mức cải thiện của LLM hiện đại |
| OCR truyền thống (Tesseract) + LLM tổng quát zero-shot | Baseline "OCR rời rạc" — đo lợi ích của VLM end-to-end |
| Pipeline OCR+LLM đã công bố cho người lớn (tái lập theo mô tả Damani et al. 2026 / Tay et al. 2026), áp dụng zero-shot trên dữ liệu nhi khoa | Đo "domain gap" người lớn → trẻ em (trả lời trực tiếp RQ2) |
| Mô hình đề xuất (fine-tune LoRA trên domain nhi khoa) | Phương pháp chính |

### d) Thiết kế thí nghiệm

- **Thí nghiệm chính:** So sánh toàn bộ 5 pipeline trên cùng tập test giữ lại (held-out), báo cáo theo từng trường (field-level) và toàn bộ ca (document-level exact match).
- **Thí nghiệm phân tầng (stratified analysis):** Tách kết quả theo (i) nhóm bệnh phổ biến vs. hiếm (melanoma, lymphoma, embryonal tumors) để trả lời RQ3; (ii) chất lượng scan (rõ/mờ/nghiêng); (iii) có/không có trường viết tay xen kẽ.
- **Thí nghiệm robustness:** Thêm nhiễu nhân tạo (giảm độ phân giải, nghiêng góc, mất nét) để đo độ nhạy của pipeline — quan trọng vì dữ liệu hồi cứu nhiều năm thường có chất lượng scan không đồng đều.
- **External validation (nếu có thể — liên kết với Hướng 4):** thử pipeline đã tối ưu trên 1 cơ sở lên dữ liệu của cơ sở thứ 2 (ví dụ CCIA Úc ↔ một bệnh viện Việt Nam) để đo generalization gap — nếu làm được, đây là điểm cộng lớn cho tính mới của bài báo.

### e) Chỉ số đánh giá

- **Field-level:** Precision / Recall / F1 cho từng trường riêng biệt (đặc biệt nhấn "trường chẩn đoán chính" và "mã hình thái học" vì đây là trường quan trọng nhất cho dịch tễ học ung thư).
- **Document-level:** Exact match accuracy (tất cả trường đúng) và "clinically acceptable match" (cho phép sai lệch nhỏ ở trường phi cấu trúc).
- **CER/WER** cho riêng phần OCR thuần (tách biệt lỗi OCR khỏi lỗi trích xuất ngữ nghĩa — quan trọng để biết lỗi đến từ đâu).
- **Cohen's kappa** giữa pipeline và 2 người gán nhãn độc lập (đo độ tin cậy so với biến thiên giữa người với người — inter-annotator agreement là ngưỡng tham chiếu hợp lý).
- **Tỷ lệ ca cần rà soát thủ công** (% flagged for human review) và **thời gian tiết kiệm được** (so với quy trình nhập liệu thủ công hoàn toàn) — chỉ số tác động thực tiễn quan trọng cho bài báo hướng ứng dụng.

### f) Ablation study dự kiến

1. So sánh OCR engine (PaddleOCR-VL vs GOT-OCR2.0 vs Tesseract) — đo đóng góp của từng lựa chọn OCR vào F1 cuối cùng.
2. So sánh LLM extraction (GPT-4o vs Gemini vs Llama fine-tune vs Qwen2.5-VL fine-tune) — trade-off chi phí/bảo mật/độ chính xác.
3. Zero-shot vs few-shot vs fine-tune LoRA — đo lợi ích biên của từng lượng dữ liệu gán nhãn (data scaling curve: 0, 50, 100, 300, 500+ ca).
4. Có/không có cơ chế "trích dẫn nguồn" (grounded extraction) — đo ảnh hưởng đến khả năng phát hiện lỗi và độ tin cậy của human-in-the-loop.
5. Có/không bước ẩn danh hóa trước khi trích xuất — đo trade-off giữa bảo mật và độ chính xác (liên kết với Hướng 6 nếu muốn mở rộng).

## 1.3. Lộ trình theo thời gian (12 tháng, chia 6 giai đoạn)

| Giai đoạn | Thời gian | Công việc chính |
|---|---|---|
| **1. Tổng quan tài liệu + chuẩn bị hành chính** | Tháng 1–2 | Rà soát hệ thống (systematic review nhỏ) các nghiên cứu OCR/NLP cho cancer registry (mở rộng từ Hands & Kavuluru 2025); nộp hồ sơ HREC/IRB; liên hệ đối tác dữ liệu (ACCR/CCIA hoặc bệnh viện nhi VN); thiết kế schema gán nhãn ICCC/COG chi tiết |
| **2. Xây dựng dữ liệu proof-of-concept (synthetic)** | Tháng 2–3 | Sinh bộ báo cáo tổng hợp (song song chờ IRB); thiết lập pipeline kỹ thuật cơ bản (OCR + LLM extraction) trên dữ liệu synthetic; kiểm tra tính đúng đắn của code, schema JSON, quy trình đánh giá |
| **3. Nhận dữ liệu thật + gán nhãn gold standard** | Tháng 3–5 | Nhận dữ liệu (nếu IRB duyệt đúng tiến độ); 2 người độc lập gán nhãn tập gold standard (100–200 ca đầu để đo inter-annotator agreement); điều chỉnh schema nếu cần |
| **4. Xây dựng baseline + pipeline chính** | Tháng 4–7 (chạy song song giai đoạn 3 phần cuối) | Cài đặt đầy đủ 5 baseline; fine-tune LoRA mô hình chính; chạy thí nghiệm chính trên toàn bộ tập test |
| **5. Ablation, phân tích lỗi, external validation (nếu có)** | Tháng 7–9 | Chạy toàn bộ ablation study; phân tích lỗi định tính (error analysis) theo nhóm bệnh hiếm/phổ biến; nếu có đối tác thứ 2, chạy external validation |
| **6. Viết bài báo + nộp + phản biện** | Tháng 9–12 | Viết draft đầy đủ (song song từ tháng 8 với kết quả sơ bộ); nội bộ review; nộp hội nghị/tạp chí mục tiêu; chuẩn bị phản hồi reviewer |

*Lưu ý:* nếu IRB kéo dài quá Tháng 4 (rủi ro phổ biến nhất — xem mục 1.4), giai đoạn 2 (synthetic) có thể kéo dài thành nội dung chính của bài báo đầu tiên (proof-of-concept), còn dữ liệu thật dùng cho bài báo thứ hai/mở rộng.

## 1.4. Rủi ro chính và phương án giảm thiểu

| Rủi ro | Mức độ | Phương án giảm thiểu |
|---|---|---|
| **Không xin được dữ liệu bệnh nhi thực** (rào cản đạo đức cao nhất với trẻ em) | Cao | (1) Dùng dữ liệu tổng hợp (synthetic reports) làm nghiên cứu chính, công bố như "feasibility study" — vẫn có tính mới vì chưa ai làm cho nhi khoa; (2) chuyển hoàn toàn sang Hướng 3 (OCR y tế tiếng Việt tổng quát, không giới hạn ung thư nhi) hoặc Hướng 2 (chữ viết tay, dữ liệu công khai IAM/RxHandBD); (3) dùng dữ liệu ung thư *người lớn* công khai (TCGA-Reports, nếu có) để làm methodology paper, nêu rõ hạn chế và đề xuất mở rộng nhi khoa trong future work — giống chính cách Damani et al. 2026 đã làm |
| **IRB/HREC kéo dài hơn dự kiến** (rủi ro thời gian, không phải rủi ro chặn hoàn toàn) | Trung bình–Cao | Bắt đầu nộp hồ sơ ngay Tháng 1 (song song viết literature review); dùng thời gian chờ để hoàn thiện pipeline trên dữ liệu synthetic; có kế hoạch B là nộp bài "phần 1" (proof-of-concept + baseline) trước, "phần 2" (validation trên dữ liệu thật) sau như bài mở rộng |
| **Cỡ mẫu dữ liệu thật quá nhỏ** (ung thư nhi khoa hiếm, số ca/năm ít) | Trung bình | Dùng cross-validation thay vì train/test split cố định; báo cáo khoảng tin cậy (confidence interval) rõ ràng thay vì chỉ điểm số đơn; nhấn mạnh đây là "hard, low-resource setting" như một phần đóng góp khoa học (không che giấu hạn chế mà biến nó thành phát hiện) |
| **Mô hình cloud (GPT-4o/Gemini) không được phép dùng trên PHI trẻ em theo chính sách tổ chức** | Trung bình | Toàn bộ pipeline chính chạy on-premise (Qwen2.5-VL, Llama fine-tune); chỉ dùng API cloud trên dữ liệu synthetic hoặc đã ẩn danh hoàn toàn, dùng làm "upper-bound reference" chứ không phải giải pháp triển khai thực tế — điều này còn phù hợp với xu hướng thực tế mà tài liệu tổng hợp đã chỉ ra (Neveditsin 2025) |
| **Thiếu ground truth đáng tin cậy** (inter-annotator disagreement cao do bệnh án phức tạp) | Trung bình | Đo và báo cáo minh bạch Cohen's kappa giữa 2 người gán nhãn; với ca bất đồng, dùng ý kiến người thứ 3 (adjudication) làm gold standard cuối; dùng kappa này làm "ngưỡng trần" khi diễn giải kết quả mô hình (không kỳ vọng mô hình vượt độ đồng thuận giữa người với người) |
| **Kết quả kém hơn kỳ vọng so với baseline người lớn** (H1 sai) | Thấp–Trung bình | Đây vẫn là phát hiện có giá trị khoa học (định lượng domain gap = RQ2/H2) — không phải rủi ro thất bại mà là một đóng góp; cần thiết kế ablation đủ chi tiết để giải thích *tại sao* (thuật ngữ hiếm, layout khác biệt, chữ viết tay) thay vì chỉ báo cáo con số |

---

# PHẦN 2 — LỘ TRÌNH VIẾT BÀI BÁO

## 2.1. Cấu trúc bài báo đề xuất

| Phần | Gợi ý nội dung (1-2 câu) |
|---|---|
| **Abstract** | Nêu khoảng trống (ung thư nhi khoa "underrepresented" trong NLP/OCR cancer registry — chưa có nghiên cứu áp dụng LLM hiện đại); tóm tắt pipeline OCR+LLM và kết quả chính (F1, so với baseline người lớn/rule-based); kết luận về tính khả thi và hạn chế (domain gap ở nhóm bệnh hiếm). |
| **Introduction** | Mở đầu bằng tầm quan trọng của đăng ký ung thư nhi khoa cho dịch tễ học và nghiên cứu (mỗi ca đều quý vì bệnh hiếm); nêu hiện trạng NLP/OCR cho cancer registry đã tốt ở người lớn nhưng bỏ sót trẻ em (trích Hands & Kavuluru 2025); phát biểu rõ ràng research gap + đóng góp cụ thể của bài báo (liệt kê 3 đóng góp). |
| **Related Work** | Chia 3 nhóm: (1) OCR/Document AI hiện đại (VLM end-to-end, GOT-OCR2.0, PaddleOCR-VL) — nền tảng kỹ thuật; (2) NLP/LLM cho cancer registry ở người lớn (Yoon 2022, Damani 2026, Tay 2026, Wiest 2025) — baseline khái niệm; (3) Khoảng trống nhi khoa cụ thể (Hands & Kavuluru 2025) — định vị chỗ trống bài báo lấp vào. |
| **Method** | Mô tả chi tiết pipeline 3 giai đoạn (OCR/VLM → LLM extraction có schema → human-in-the-loop confidence routing); giải thích lựa chọn mô hình on-premise vì ràng buộc PHI trẻ em; mô tả schema ICCC/COG dùng làm nhãn. |
| **Experiments** | Mô tả dữ liệu (thật + synthetic), quy trình gán nhãn/đạo đức, các baseline, chỉ số đánh giá, thiết lập ablation. |
| **Results** | Bảng so sánh F1/CER/WER giữa các pipeline; phân tích phân tầng theo nhóm bệnh hiếm vs phổ biến (trả lời RQ3/H2 — đây là kết quả "mới" đáng chú ý nhất); kết quả ablation. |
| **Discussion** | Diễn giải domain gap người lớn→trẻ em; bàn về tính khả thi triển khai thực tế (thời gian tiết kiệm, % cần rà soát); nêu hạn chế (cỡ mẫu, một/hai cơ sở) trung thực. |
| **Conclusion** | Khẳng định đây là nghiên cứu đầu tiên áp dụng LLM hiện đại cho pediatric cancer registry; đề xuất hướng mở rộng đa trung tâm/đa ngôn ngữ (liên kết Hướng 3/4 làm future work — tạo mạch nối tự nhiên cho bài báo thứ hai). |

## 2.2. Danh sách công trình liên quan quan trọng cần trích dẫn

*(rút từ tài liệu tổng hợp đã cung cấp — cần tự bổ sung trích dẫn đầy đủ/DOI khi viết bản thảo chính thức, và kiểm tra lại qua PubMed/Google Scholar vì đây là tóm tắt gián tiếp)*

- **Hands & Kavuluru (2025)**, *AI Review* — rà soát 156 bài 2014–2024, nêu rõ ung thư nhi khoa/melanoma/lymphoma "underrepresented" — **trích dẫn nền tảng cho research gap**.
- **Yoon et al. (2022)**, *JAMIA Open* — trích xuất trên 29.206 báo cáo ung thư trẻ em, micro-F1 0,987 nhưng phương pháp cũ (pre-LLM) — **baseline lịch sử quan trọng nhất, phải trích dẫn và so sánh trực tiếp**.
- **Damani et al. (2026)**, Mayo Clinic — pipeline OCR-LLM cho multi-report PDF, minh thị đề xuất mở rộng đa chuyên khoa — **dùng làm baseline "người lớn" và làm điểm tựa cho khung "future work" của chính Damani** để chứng minh nghiên cứu này lấp đúng khoảng trống họ nêu.
- **Tay et al. (2026)**, Singapore — pipeline chấp nhận ~23% cần rà soát thủ công — **mô hình tham chiếu cho thiết kế human-in-the-loop**.
- **Wiest et al. (2025)** — F1 0,85–0,95+ cho cancer registry người lớn — **định lượng "upper bound" kỳ vọng**.
- **Nicora et al. (2026)** — case report form đột quỵ, VLM mở struggle với trường viết tay — **liên hệ trực tiếp nếu bài báo có trường viết tay trong bệnh án nhi khoa**.
- **Li et al. (2024)**, *J. Biomedical Informatics* — TSR end-to-end TEDS ~0,699 — **trích dẫn nếu báo cáo giải phẫu bệnh có bảng biểu phức tạp**.
- Các mô hình nền tảng OCR/VLM: **GOT-OCR2.0**, **PaddleOCR-VL**, **DeepSeek-OCR**, **Qwen-VL/Qwen3-VL**, **TrOCR**, **LayoutLMv3** — trích trong phần Method để biện minh lựa chọn kiến trúc.
- **OCRBench v2** — dẫn chứng cho luận điểm "ngay cả GPT-4o/Gemini cũng dưới 50/100 điểm ở tác vụ định vị/suy luận văn bản" — dùng trong Introduction để lập luận vì sao cần pipeline chuyên biệt thay vì dùng thẳng VLM tổng quát.

*Khuyến nghị:* trước khi nộp bản thảo, chạy lại tra cứu hệ thống trên PubMed/Google Scholar/Consensus với từ khóa "pediatric cancer registry NLP LLM OCR" và "childhood cancer information extraction" để xác nhận vẫn chưa có công trình cạnh tranh trực tiếp xuất bản trong lúc chuẩn bị bài (khoảng trống có thể bị "đóng" bởi nhóm khác — nên kiểm tra định kỳ mỗi 2-3 tháng trong suốt 12 tháng thực hiện).

## 2.3. Đề xuất hội nghị/tạp chí mục tiêu (xếp theo độ phù hợp)

1. **JAMIA Open** (Journal of the American Medical Informatics Association, open-access track) — *Phù hợp nhất*: đã có tiền lệ công bố các nghiên cứu OCR-LLM cancer registry tương tự (Yoon 2022, Tay 2026); tạp chí rolling submission, không có deadline cứng; thời gian phản biện trung bình 2-4 tháng; ưu tiên tính ứng dụng lâm sàng thực tế hơn độ mới thuần kỹ thuật — rất phù hợp với bài báo có dữ liệu thật + đánh giá lâm sàng.
2. **npj Digital Medicine** (Nature portfolio) — Uy tín cao hơn, chấp nhận scoping/tính hệ thống của external validation; rolling submission; đòi hỏi nội dung có tác động rộng (broad impact) — phù hợp nếu kết hợp thêm khía cạnh đa trung tâm (liên kết Hướng 4) để tăng sức nặng.
3. **Journal of Biomedical Informatics** (Elsevier) — Phù hợp về mặt kỹ thuật (đã công bố Li et al. 2024 về TSR y tế); chấp nhận cả phương pháp luận chi tiết và ablation sâu; rolling submission.
4. **AMIA Annual Symposium** (American Medical Informatics Association) — Hội nghị uy tín trong giới y tin học; **lưu ý**: abstract/paper deadline thường rơi vào khoảng tháng 3-4 hàng năm cho hội nghị tổ chức vào tháng 11 — cần kiểm tra lịch chính xác trên trang amia.org gần thời điểm nộp vì có thể thay đổi theo năm; phù hợp nếu muốn công bố nhanh dạng poster/short paper trước khi hoàn thiện bài journal đầy đủ.
5. **MICCAI Workshop** (ví dụ workshop về Clinical NLP/Document Analysis trong khuôn khổ MICCAI) — deadline chính thường vào khoảng tháng 3 cho hội nghị tháng 9-10; phù hợp nếu muốn nhấn mạnh khía cạnh kỹ thuật thị giác máy tính hơn là khía cạnh y tin học thuần túy; cần kiểm tra workshop cụ thể còn tổ chức năm nộp bài hay không (workshop thay đổi theo năm).

*Lưu ý quan trọng:* các deadline trên chỉ mang tính tham khảo dựa trên thông lệ các năm gần đây — **bắt buộc kiểm tra lại ngày chính xác trên trang chủ hội nghị/tạp chí ngay trước khi bắt tay viết bản thảo cuối** (khoảng Tháng 6-7 trong lộ trình 12 tháng), vì lịch hội nghị có thể xê dịch.

## 2.4. Lời khuyên thực tế để tăng cơ hội được chấp nhận

- **Khung tính mới (novelty framing) rõ ràng ngay từ câu đầu Abstract:** nêu thẳng "đây là nghiên cứu đầu tiên áp dụng LLM hiện đại cho pediatric cancer registry" và trích dẫn trực tiếp câu trong Hands & Kavuluru (2025) hoặc kết quả tra cứu phủ định (không tìm thấy công trình nào) để reviewer thấy ngay khoảng trống có bằng chứng, không phải suy diễn chủ quan.
- **Baseline phải đủ mạnh và đa dạng:** nhất định phải có baseline rule-based (kiểu Yoon 2022) *và* baseline LLM tổng quát zero-shot — reviewer y tin học rất hay bác bài vì thiếu so sánh với phương pháp "cũ nhưng đã hoạt động tốt" (Yoon đạt F1 0,987, dù dùng dữ liệu/kỹ thuật khác — cần giải thích rõ tại sao không so sánh được trực tiếp con số nếu schema khác).
- **Grounded extraction (trích dẫn nguồn) tăng độ tin cậy với reviewer lâm sàng:** cho mỗi trường trích xuất, hiển thị đoạn văn bản gốc làm bằng chứng — điều này giúp thuyết phục reviewer rằng hệ thống có thể kiểm toán được (auditable), một yêu cầu ngầm quan trọng trong các tạp chí y tin học.
- **Minh bạch về hạn chế thay vì né tránh:** vì cỡ mẫu ung thư nhi khoa chắc chắn nhỏ hơn nhiều so với nghiên cứu người lớn, hãy chủ động đóng khung phần Discussion là "định lượng domain gap và đề xuất hướng nghiên cứu tiếp theo" thay vì cố gắng chứng minh hệ thống đã "giải quyết xong" bài toán — cách này thực ra tăng độ tin cậy khoa học.
- **Reproducibility:** công khai (trên GitHub, sau khi loại bỏ hoàn toàn PHI) mã nguồn pipeline, prompt templates, và schema gán nhãn — kể cả khi không thể công khai dữ liệu thật, việc công khai bộ dữ liệu synthetic + code giúp bài báo dễ được chấp nhận hơn ở các tạp chí ngày càng yêu cầu reproducibility (JAMIA Open, npj Digital Medicine đều khuyến khích điều này).
- **Chuẩn bị sẵn "Data Availability Statement"** giải thích rõ vì sao dữ liệu thật không thể công khai (PHI trẻ em) nhưng mô tả quy trình xin truy cập qua đối tác registry — reviewer y tin học quen thuộc với ràng buộc này và sẽ không đánh giá thấp bài vì lý do đó nếu được giải thích minh bạch.
- **Nếu chuyển sang phương án dự phòng (Hướng 2/3):** giữ nguyên toàn bộ khung phương pháp luận (pipeline 3 giai đoạn, CER/WER/F1, ablation OCR/LLM) — chỉ thay đổi domain/dữ liệu, giúp không mất công đã đầu tư và có thể chuyển đổi hướng nhanh trong vòng 2-4 tuần nếu Tháng 3-4 xác nhận không xin được dữ liệu bệnh nhi thực.