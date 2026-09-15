> **⚠️ ĐÃ THAY THẾ (SUPERSEDED) — 15/09/2026.** Tài liệu này giả định hướng đăng ký ung thư nhi khoa gắn với CCI và/hoặc OCR tiếng Việt. Người dùng đã xác nhận đây là **nghiên cứu độc lập, không liên quan CCI**, **không làm OCR tiếng Việt**, và cần **kế hoạch gấp trong 2 tuần** — các điều kiện này khiến mọi hướng ở đây (đều cần dữ liệu bệnh viện/IRB hoặc tiếng Việt) không còn phù hợp. Xem hướng đi hiện tại tại [`04-dataset-y-khoa-cong-khai.md`](../04-dataset-y-khoa-cong-khai.md) và [`05-ke-hoach-2-tuan.md`](../05-ke-hoach-2-tuan.md). Giữ lại file này chỉ để tham khảo/đối chiếu.

---

# PHÂN TÍCH: HIỆN TRẠNG OCR VÀ CÁC HƯỚNG NGHIÊN CỨU MỞ (ƯU TIÊN Y KHOA)

---

## PHẦN 1 — TÓM TẮT HIỆN TRẠNG VÀ GIỚI HẠN HIỆN TẠI

- **Kiến trúc đang chuyển dịch nhanh sang OCR-free/VLM end-to-end** (GOT-OCR2.0, PaddleOCR-VL, dots.ocr, DeepSeek-OCR, Qwen3-VL...) thay cho pipeline module hóa (detect → recognize → parse layout), giảm lỗi tích lũy nhưng vẫn còn khoảng cách lớn giữa "đọc được chữ" và "hiểu sâu tài liệu": OCRBench v2 cho thấy **hầu hết LMM SOTA (kể cả GPT-4o, Gemini) đạt dưới 50/100 điểm** ở các tác vụ định vị văn bản và suy luận.
- **Chữ viết tay lâm sàng vẫn là điểm nghẽn lớn nhất, kể cả với VLM hiện đại nhất.** Nicora et al. (2026) thử 3 VLM mã nguồn mở trên case report form đột quỵ và kết luận "handwritten fields remained particularly challenging"; Helstad et al. (2025, kho lưu trữ y tế Na Uy) cũng nêu rõ hạn chế OCR với chữ viết tay lịch sử.
- **Layout phức tạp (bảng đa trang, form nhiều cột, scan mờ/nghiêng) làm giảm mạnh độ chính xác**: OCR mã nguồn mở truyền thống chỉ đạt 88–94% trên layout phức tạp; Table Structure Recognition (TSR) end-to-end vẫn khiêm tốn (TEDS ~0.699 dù từng module riêng lẻ cao hơn — Li et al. 2024) và thiếu chuẩn annotation/benchmark thống nhất (ACM Computing Surveys 2024).
- **Thuật ngữ/viết tắt y khoa chuyên biệt** (tên hoạt chất, mã ICD, "bid", "prn"...) không nằm trong phân bố huấn luyện của OCR/LLM tổng quát → tỷ lệ lỗi tăng đáng kể so với văn bản phổ thông (Datta et al. 2025 nêu rõ các phương pháp hiện có "struggle with multilingual text").
- **Ngôn ngữ ít tài nguyên bị bỏ lại phía sau đáng kể.** GlotOCRBench (2026) cho thấy trên 158 script Unicode, mô hình tốt nhất chỉ phiên âm đúng <7,7% câu (CER<5%) ở script ít tài nguyên. Tiếng Việt thuộc nhóm "trung-thấp tài nguyên" trong OCR y khoa toàn cầu, thiếu benchmark quy mô lớn tương đương OmniDocBench, dù đã có nỗ lực ban đầu (ViOCRVQA, ViConsFormer, khảo sát VDAR 2025).
- **Ràng buộc bảo mật/quyền riêng tư dữ liệu bệnh nhân (HIPAA, Privacy Act Úc, GDPR)** cản trở việc dùng trực tiếp API cloud (GPT-4V, Gemini) cho PHI, thúc đẩy xu hướng dùng multimodal LLM cỡ nhỏ chạy on-premise (Phi-4, Qwen-VL, InternVL3.5) — nhưng các mô hình này thường đánh đổi bằng độ chính xác thấp hơn trên dữ liệu nhiễu (Neveditsin et al. 2025).
- **Thiếu dữ liệu gán nhãn công khai quy mô lớn cho tài liệu y khoa**, đặc biệt đa chuyên khoa/đa trung tâm, do rào cản đạo đức. Phần lớn nghiên cứu hiện có (Hom 2022, Ma 2023, Hsu 2022, You 2025...) chỉ kiểm chứng tại **một cơ sở/một bệnh lý duy nhất**, thiếu external validation — chính các tác giả (Damani/Mayo Clinic 2026) minh thị đề xuất mở rộng đa chuyên khoa trong "future work".
- **Khoảng trống đặc biệt rõ và có thể định lượng**: ung thư nhi khoa, u hắc tố (melanoma), và lymphoma bị "underrepresented" trong nghiên cứu NLP/OCR cho cancer registry (Hands & Kavuluru, 2025, rà soát 156 bài 2014–2024); tra cứu Consensus chuyên sâu cũng **không tìm thấy nghiên cứu OCR/LLM nào áp dụng riêng cho đăng ký ung thư trẻ em** — đây là khoảng trống có bằng chứng phủ định trực tiếp, không chỉ suy luận.

---

## PHẦN 2 — CÁC HƯỚNG NGHIÊN CỨU MỞ ĐỀ XUẤT

### HƯỚNG 1 [Y KHOA — ƯU TIÊN CAO NHẤT] — Pipeline OCR+LLM tự động trích xuất dữ liệu đăng ký ung thư nhi khoa (Pediatric Cancer Registry Automation)

- **Câu hỏi nghiên cứu:** Một pipeline kết hợp OCR + LLM có thể trích xuất tự động, chính xác các biến đăng ký chuẩn ICCC (chẩn đoán, mô bệnh học, giai đoạn, tuổi tại chẩn đoán...) từ báo cáo giải phẫu bệnh/bệnh án ung thư trẻ em scan hay không, và độ tin cậy so với con người ra sao?
- **Tại sao là khoảng trống:** Hands & Kavuluru (2025, *AI Review*) chỉ rõ ung thư nhi khoa "underrepresented" trong 156 nghiên cứu NLP về cancer registry (2014–2024); tra cứu Consensus (3 truy vấn có hệ thống) **không tìm thấy bằng chứng trực tiếp** nào cho pediatric cancer registry, trong khi các mô hình tương tự trên người lớn đã đạt F1 ≥0,85–0,95 (Wiest 2025; Tay et al. 2026, Singapore) và Yoon et al. (2022, JAMIA Open) chứng minh tính khả thi kỹ thuật trên 29.206 báo cáo ung thư trẻ em (micro-F1 0,987) nhưng dùng phương pháp cũ (không phải LLM hiện đại) và chỉ ở Mỹ.
- **Hướng tiếp cận khả thi:** Pipeline 2 giai đoạn — (1) OCR/VLM (PaddleOCR-VL hoặc GOT-OCR2.0, chạy on-premise) chuyển báo cáo scan/PDF thành văn bản có cấu trúc; (2) LLM (GPT-4/Gemini qua API bảo mật, hoặc mô hình mở fine-tune) trích xuất trường theo schema ICCC/COG, có bước "human-in-the-loop" cho ca mơ hồ (theo mô hình Tay et al. 2026 chấp nhận ~23% cần rà soát).
- **Dữ liệu/tài nguyên:** Cần bộ báo cáo giải phẫu bệnh/bệnh án ung thư trẻ em thật (khả thi nếu hợp tác với một registry hoặc bệnh viện nhi — ví dụ CCIA/Children's Cancer Institute Úc, hoặc bệnh viện Việt Nam như Bệnh viện Nhi TW/Ung Bướu). **Độ khó:** cao về mặt đạo đức (IRB/HREC, PHI trẻ em nhạy cảm hơn người lớn) nhưng có thể giảm bằng dữ liệu ẩn danh hóa hồi cứu hoặc dữ liệu tổng hợp (synthetic reports) làm proof-of-concept trước khi xin dữ liệu thật.
- **Độ khả thi cho nhóm nhỏ:** Cao — không cần GPU lớn nếu dùng API LLM thương mại + OCR mã nguồn mở; điểm nghẽn chính là quyền truy cập dữ liệu, có thể giải quyết qua hợp tác với một trung tâm ung thư nhi.
- **Nguồn công bố phù hợp:** *JAMIA Open*, *npj Digital Medicine*, *Journal of Biomedical Informatics*, hội nghị AMIA; nếu nhấn khía cạnh kỹ thuật OCR: *MICCAI Workshop*, *DAS (Document Analysis Systems)*.
- **Đóng góp/tính mới dự kiến:** Là nghiên cứu đầu tiên (theo tra cứu hiện có) áp dụng LLM hiện đại cho cancer registry **trẻ em**, lấp khoảng trống đã được minh thị trong tài liệu; có thể mở rộng thành benchmark công khai (ẩn danh) cho cộng đồng.

### HƯỚNG 2 [Y KHOA] — Nhận dạng chữ viết tay lâm sàng bằng VLM nhỏ gọn chạy on-premise (Privacy-Preserving Clinical Handwriting Recognition)

- **Câu hỏi nghiên cứu:** Các VLM mã nguồn mở cỡ nhỏ (≤8B, chạy được on-premise) có thể đạt độ chính xác chấp nhận được trong nhận dạng trường viết tay trên case report form/đơn thuốc mà không cần gửi dữ liệu PHI ra cloud hay không?
- **Tại sao là khoảng trống:** Đây là khoảng trống được nêu **minh thị và lặp lại nhiều lần nhất** trong tài liệu đã tổng hợp — Nicora et al. (2026) kết luận trường viết tay "particularly challenging" ngay cả với Qwen2.5/Mistral/Granite Vision; Helstad et al. (2025) và khảo sát VDAR (2025) đều nêu hạn chế tương tự cho tiếng Na Uy/tiếng Việt.
- **Hướng tiếp cận khả thi:** Fine-tune/prompt-engineer một VLM mở (Qwen2.5-VL-7B, InternVL3.5, hoặc Granite-Docling) trên dữ liệu viết tay lâm sàng tổng hợp + thật (augmentation bằng font viết tay giả lập nếu thiếu dữ liệu thật), so sánh với pipeline CRNN+CTC cổ điển và với GPT-4V/Gemini làm upper-bound.
- **Dữ liệu/tài nguyên:** IAM Handwriting Database (công khai, không chuyên y khoa) để pretrain; RxHandBD (đơn thuốc, công khai) cho domain thuốc; cần thêm bộ case report form/ghi chú thật (khó, cần IRB) — có thể khởi đầu bằng dữ liệu công khai + mô phỏng trước khi mở rộng ra dữ liệu bệnh viện thật.
- **Độ khả thi cho nhóm nhỏ:** Cao — fine-tune LoRA trên VLM 7B khả thi với 1 GPU 24GB, không cần siêu máy tính; dữ liệu khởi đầu hoàn toàn công khai.
- **Nguồn công bố phù hợp:** *ICDAR*, *DAS*, *MICCAI (workshop Document/Clinical NLP)*, *JAMIA*.
- **Đóng góp/tính mới:** Benchmark có hệ thống đầu tiên so sánh VLM mở cỡ nhỏ on-premise vs. API cloud cho chữ viết tay lâm sàng, kèm khuyến nghị triển khai thực tế cho cơ sở y tế hạn chế tài nguyên/quyền riêng tư nghiêm ngặt.

### HƯỚNG 3 [Y KHOA] — Benchmark và mô hình OCR/HTR chuyên biệt cho tài liệu y tế tiếng Việt

- **Câu hỏi nghiên cứu:** Xây dựng benchmark quy mô vừa (ảnh scan/chụp hồ sơ bệnh án, đơn thuốc, kết quả xét nghiệm tiếng Việt) và đánh giá/fine-tune mô hình OCR-VLM cho đặc thù tiếng Việt (dấu thanh điệu, nguyên âm ghép) trong bối cảnh y khoa.
- **Tại sao là khoảng trống:** Khảo sát VDAR (2025, arXiv:2506.05061) khẳng định tiếng Việt thiếu dữ liệu gán nhãn quy mô lớn; GlotOCRBench (2026) định lượng khoảng cách hiệu năng giữa script giàu/ít tài nguyên; Dinh et al. (2023, ICIS) đã chứng minh khả thi cho hồ sơ uốn ván (CER 2%, WER 12%) nhưng phạm vi hẹp (1 bệnh viện, 1 loại hồ sơ) — chưa có benchmark tổng quát đa loại tài liệu y tế tiếng Việt tương đương OmniDocBench.
- **Hướng tiếp cận khả thi:** Thu thập/ẩn danh hóa đa dạng loại tài liệu y tế tiếng Việt (đơn thuốc in, phiếu xét nghiệm, ghi chú) từ 1–2 bệnh viện hợp tác; gán nhãn CER/WER; benchmark các mô hình hiện có (PaddleOCR-VL — vốn mạnh đa ngôn ngữ, VietOCR, Qwen-VL, GOT-OCR2.0) rồi fine-tune mô hình tốt nhất bằng LoRA trên domain y tế.
- **Dữ liệu/tài nguyên:** Cần hợp tác bệnh viện Việt Nam để có dữ liệu thật (mức độ khó: trung bình–cao do quy định bảo mật y tế Việt Nam, nhưng khả thi hơn dữ liệu quốc tế nhờ lợi thế ngôn ngữ/quan hệ địa phương của nhóm nghiên cứu); có thể bổ sung dữ liệu tổng hợp (rendering văn bản y khoa tiếng Việt lên nền ảnh scan giả lập) để giảm phụ thuộc dữ liệu thật ở giai đoạn đầu.
- **Độ khả thi cho nhóm nhỏ:** Cao — đây là dạng đề tài rất phù hợp cho một nghiên cứu sinh làm luận án, chi phí tính toán thấp, tài nguyên ngôn ngữ nằm trong lợi thế của nhóm nghiên cứu Việt Nam.
- **Nguồn công bố phù hợp:** *ICDAR*, *DAS*, *ACL/EMNLP Findings* (track đa ngôn ngữ/low-resource), hội nghị trong nước (NAFOSTED/VLSP), hoặc *Multimedia Systems* (nơi ViOCRVQA đã công bố).
- **Đóng góp/tính mới:** Bộ benchmark công khai đầu tiên chuyên biệt cho OCR y tế tiếng Việt (khác ViOCRVQA vốn là VQA tổng quát), cùng mô hình fine-tune làm baseline mới cho cộng đồng.

### HƯỚNG 4 [Y KHOA] — Đánh giá khái quát hóa đa trung tâm/đa chuyên khoa của pipeline OCR-LLM (External Validation Study)

- **Câu hỏi nghiên cứu:** Hiệu năng của một pipeline OCR+LLM đã công bố (ví dụ đạt F1 cao tại một bệnh viện/một bệnh lý) suy giảm bao nhiêu khi áp dụng sang cơ sở/chuyên khoa khác, và yếu tố nào (định dạng biểu mẫu, chất lượng scan, thuật ngữ địa phương) gây suy giảm?
- **Tại sao là khoảng trống:** Đây là hạn chế được **chính các tác giả gốc minh thị nêu ra** — Damani et al. (Mayo Clinic, 2026) viết rõ trong Future Work "evaluating the system across additional specialties and institutions"; Ma et al. (2023) và Hom et al. (2022) đều chỉ giới hạn ở một cơ sở; Nitayavardhana et al. (2025) tuy đa trung tâm (3 quốc gia) vẫn kêu gọi "wider uptake... to better understand strengths and limitations".
- **Hướng tiếp cận khả thi:** Chọn 1 pipeline mã nguồn mở đã công bố (ví dụ pipeline table-extraction của Li et al. 2024, hoặc pipeline OCR-LLM dạng Damani et al.), tái lập trên dữ liệu từ ≥2 cơ sở khác nhau (khác quốc gia/ngôn ngữ nếu có thể — ví dụ Việt Nam + Úc), đo suy giảm hiệu năng và phân tích lỗi theo nguyên nhân (domain shift).
- **Dữ liệu/tài nguyên:** Cần ít nhất 2 nguồn dữ liệu độc lập — khả thi nếu nhóm có quan hệ hợp tác đa cơ sở (ví dụ CCIA Úc + một bệnh viện Việt Nam); độ khó trung bình vì chỉ cần dữ liệu ở quy mô vừa (vài trăm tài liệu/cơ sở) để đo external validation, không cần dữ liệu huấn luyện lớn.
- **Độ khả thi cho nhóm nhỏ:** Trung bình-cao — công việc chủ yếu là thực nghiệm/đánh giá (không cần huấn luyện mô hình mới từ đầu), phù hợp với một nghiên cứu sinh có quan hệ hợp tác liên viện.
- **Nguồn công bố phù hợp:** *npj Digital Medicine* (đã có tiền lệ scoping review dạng này), *Journal of Biomedical Informatics*, *JAMIA Open*.
- **Đóng góp/tính mới:** Bằng chứng thực nghiệm hệ thống đầu tiên định lượng "generalization gap" của pipeline OCR-LLM y khoa qua nhiều cơ sở — một câu hỏi được nêu nhưng chưa ai trả lời trực tiếp trong tài liệu hiện có.

### HƯỚNG 5 [Y KHOA] — Table Structure Recognition end-to-end cho báo cáo xét nghiệm/bệnh lý phức tạp

- **Câu hỏi nghiên cứu:** Làm sao cải thiện khoảng cách giữa hiệu năng module (table detection, table recognition) riêng lẻ và hiệu năng end-to-end (hiện chỉ đạt TEDS ~0,699 dù từng phần đạt AP50=0,774 và TEDS=0,815 — Li et al. 2024) trên bảng biểu y tế scan/fax chất lượng thấp?
- **Tại sao là khoảng trống:** Li et al. (2024, *J. Biomedical Informatics*) minh thị chỉ ra khoảng cách này; khảo sát ACM Computing Surveys (2024) xác nhận thiếu chuẩn benchmark/annotation thống nhất cho TSR nói chung.
- **Hướng tiếp cận khả thi:** Thiết kế mô-đun "kết dính" (glue module) giữa detection và recognition — ví dụ dùng cơ chế attention chia sẻ đặc trưng giữa 2 giai đoạn thay vì pipeline rời rạc, hoặc thử nghiệm mô hình VLM end-to-end (Donut/Pix2Struct kiểu bảng) huấn luyện trực tiếp trên cặp (ảnh bảng, HTML) để tránh lỗi tích lũy.
- **Dữ liệu/tài nguyên:** Bộ dữ liệu của Li et al. (650 bảng, 632 báo cáo xét nghiệm) — cần liên hệ tác giả xin dùng lại/mở rộng, hoặc xây bộ tương tự từ dữ liệu công khai (PubTables-1M tổng quát + fine-tune domain y tế). Độ khó: trung bình, vì có thể tận dụng dữ liệu bảng tổng quát công khai làm pretrain.
- **Độ khả thi cho nhóm nhỏ:** Trung bình — cần một số kinh nghiệm về kiến trúc thị giác (TableFormer-like), nhưng không đòi hỏi hạ tầng tính toán lớn.
- **Nguồn công bố phù hợp:** *ICDAR*, *DAS*, *Journal of Biomedical Informatics*.
- **Đóng góp/tính mới:** Cải thiện trực tiếp một hạn chế đã được định lượng rõ ràng trong tài liệu, có baseline sẵn để so sánh.

### HƯỚNG 6 [Y KHOA] — Framework ẩn danh hóa tích hợp trong pipeline OCR (Privacy-by-Design OCR cho PHI)

- **Câu hỏi nghiên cứu:** Có thể thiết kế một pipeline OCR/VLM "ẩn danh hóa ngay tại nguồn" (phát hiện và che PHI ngay trong bước OCR, trước khi văn bản rời khỏi môi trường an toàn) mà không làm giảm đáng kể độ chính xác trích xuất thông tin lâm sàng cần giữ lại hay không?
- **Tại sao là khoảng trống:** Nhiều nghiên cứu (Helstad 2025 — Na Uy; You 2025 — audiogram) đã tích hợp de-identification nhưng đánh giá tách rời với hiệu năng OCR; chưa có nghiên cứu định lượng **đánh đổi (trade-off)** giữa mức độ ẩn danh hóa và độ chính xác trích xuất trong cùng một pipeline thống nhất.
- **Hướng tiếp cận khả thi:** Kết hợp mô hình phát hiện PHI cấp vùng ảnh (bounding-box level NER trên layout, tương tự CRAFT nhưng cho "vùng nhạy cảm" thay vì "vùng chữ") với OCR chọn lọc — chỉ trích xuất/lưu trữ trường không nhạy cảm, che các trường PHI trước khi log/lưu.
- **Dữ liệu/tài nguyên:** Có thể dùng dữ liệu tổng hợp (synthetic PHI chèn vào template hồ sơ thật đã ẩn danh) để tránh vấn đề đạo đức ở giai đoạn phát triển thuật toán, chỉ cần dữ liệu thật ở bước đánh giá cuối (dễ xin phép hơn vì mục đích là bảo vệ riêng tư).
- **Độ khả thi cho nhóm nhỏ:** Cao — đây là hướng có thể triển khai gần như hoàn toàn với dữ liệu tổng hợp + một tập nhỏ dữ liệu thật để validate cuối.
- **Nguồn công bố phù hợp:** *JAMIA*, *AMIA Annual Symposium*, hội nghị bảo mật y tế (nếu có track riêng), *npj Digital Medicine*.
- **Đóng góp/tính mới:** Framework "privacy-by-design" tích hợp thay vì ẩn danh hóa hậu kỳ — hướng đi khác biệt so với các phương pháp de-identification hiện có (vốn xử lý sau khi đã có văn bản đầy đủ).

### HƯỚNG 7 [Y KHOA — LMIC] — OCR/Document AI chi phí thấp cho hồ sơ y tế viết tay ở môi trường hạn chế tài nguyên

- **Câu hỏi nghiên cứu:** Một hệ thống OCR chạy trên thiết bị di động phổ thông (không cần GPU chuyên dụng) có thể số hóa đáng tin cậy hồ sơ y tế viết tay đa ngôn ngữ ở các cơ sở y tế tuyến dưới/vùng sâu vùng xa (Việt Nam, Đông Nam Á) hay không?
- **Tại sao là khoảng trống:** Kamanga et al. (2026, Malawi, "ScanForm") là một trong rất ít ví dụ ở LMIC; khung IDPA (2025) cho hồ sơ song ngữ Hindi-English chỉ đạt 74% chính xác, CER 13% — cho thấy dư địa cải thiện lớn và khoảng trống nghiên cứu rõ ràng ở khu vực có hạ tầng số hạn chế, đặc biệt chưa có nghiên cứu tương tự cho bối cảnh Việt Nam/Đông Nam Á.
- **Hướng tiếp cận khả thi:** Sử dụng mô hình OCR/VLM siêu gọn (PaddleOCR-VL 0.9B, HunyuanOCR 1B, Moondream2) lượng tử hóa INT8 để chạy trên smartphone/thiết bị biên (edge), thử nghiệm thực địa tại trạm y tế xã/huyện.
- **Dữ liệu/tài nguyên:** Cần hợp tác với cơ sở y tế tuyến cơ sở (trạm y tế, bệnh viện huyện) để thu thập ảnh chụp hồ sơ thật; độ khó cao về logistics (thực địa) nhưng thấp về rào cản kỹ thuật máy tính, và có ý nghĩa xã hội rõ ràng nên dễ nhận được sự đồng thuận/hỗ trợ từ y tế công.
- **Độ khả thi cho nhóm nhỏ:** Trung bình — cần hợp tác thực địa nhưng không cần hạ tầng tính toán lớn (chính điểm mạnh của hướng "on-device").
- **Nguồn công bố phù hợp:** *DAS*, *ICDAR*, *Digital Health*, hội nghị y tế toàn cầu (Global Health Informatics).
- **Đóng góp/tính mới:** Nghiên cứu thực địa đầu tiên (theo tài liệu hiện có) về OCR y tế edge-device cho bối cảnh Đông Nam Á, bổ sung khoảng trống LMIC ngoài châu Phi/Nam Á đã có.

### HƯỚNG 8 [TỔNG QUÁT — đối chiếu] — Nén ngữ cảnh quang học (Optical Context Compression) cho hồ sơ bệnh án dài nhiều trang

- **Câu hỏi nghiên cứu:** Kỹ thuật "nén ngữ cảnh quang học" của DeepSeek-OCR (dùng ảnh trang làm phương tiện nén token, giảm 7–20 lần so với văn bản thuần) có thể áp dụng để xử lý hồ sơ bệnh án nhiều trang (multi-report PDF) hiệu quả hơn, cho phép LLM "đọc" toàn bộ hồ sơ dài mà không vượt giới hạn ngữ cảnh?
- **Tại sao là khoảng trống:** DeepSeek-OCR (10/2025) và DeepSeek-OCR 2 (1/2026) là kỹ thuật rất mới, mới kiểm chứng trên OmniDocBench tổng quát, **chưa có nghiên cứu nào áp dụng cho domain y tế** — trong khi bài toán "multi-report PDF" y tế (Damani et al. 2026, Mayo Clinic) đúng là loại tài liệu dài, nhiều trang mà kỹ thuật này nhắm tới.
- **Hướng tiếp cận khả thi:** Áp dụng/so sánh DeepSeek-OCR (hoặc kỹ thuật tương tự) với pipeline OCR+LLM truyền thống trên hồ sơ y tế đa trang, đo cả độ chính xác và chi phí tính toán (số token, thời gian, chi phí API).
- **Dữ liệu/tài nguyên:** Có thể dùng dữ liệu công khai tổng quát trước (OmniDocBench) để tái lập kỹ thuật, sau đó thử trên tập nhỏ hồ sơ y tế thật/mô phỏng; độ khó thấp ở giai đoạn đầu vì mô hình đã mã nguồn mở.
- **Độ khả thi cho nhóm nhỏ:** Cao — đây chủ yếu là ứng dụng lại kỹ thuật đã công bố vào domain mới, phù hợp làm nghiên cứu đối chiếu/benchmark nhanh.
- **Nguồn công bố phù hợp:** *MICCAI Workshop*, *DAS*, *EMNLP Findings* (track hiệu quả tính toán/long-context).
- **Đóng góp/tính mới:** Nghiên cứu đầu tiên đánh giá "optical context compression" cho tài liệu y tế dài — đóng góp về hiệu quả chi phí, có giá trị thực tiễn cao cho triển khai quy mô lớn.

### HƯỚNG 9 [TỔNG QUÁT — đối chiếu] — So sánh có hệ thống VLM-OCR chuyên dụng siêu gọn vs. VLM tổng quát lớn trên dữ liệu y khoa

- **Câu hỏi nghiên cứu:** Trên cùng một bộ dữ liệu y khoa mới (ví dụ kết hợp với Hướng 3), các mô hình "OCR chuyên dụng siêu gọn" (PaddleOCR-VL 0.9B, HunyuanOCR 1B, GOT-OCR2.0 580M) có vượt trội các VLM tổng quát lớn (Qwen3-VL, Gemini, GPT) về tỷ lệ chi phí/hiệu năng hay không, đặc biệt trong điều kiện tài nguyên hạn chế của bệnh viện?
- **Tại sao là khoảng trống:** Nhận định #2 trong phần tổng quan 2024–2026 chỉ ra "cuộc đua song song" giữa hai nhóm mô hình này nhưng **chưa có so sánh trực tiếp trên domain y khoa** — hầu hết benchmark hiện có (OmniDocBench, OCRBench v2) đều tổng quát, không phản ánh đặc thù tài liệu lâm sàng (thuật ngữ, layout form y tế, chữ viết tay).
- **Hướng tiếp cận khả thi:** Benchmark thực nghiệm có kiểm soát: cùng input (đơn thuốc, phiếu xét nghiệm, báo cáo scan), đo CER/WER, F1 trích xuất trường, thời gian suy luận, chi phí (nếu dùng API) hoặc VRAM (nếu on-premise).
- **Dữ liệu/tài nguyên:** Có thể tái sử dụng dữ liệu từ Hướng 3 (tiếng Việt) hoặc dữ liệu công khai y tế tiếng Anh (RxHandBD, TCGA-Reports); độ khó thấp vì không cần huấn luyện mô hình mới, chỉ cần inference so sánh.
- **Độ khả thi cho nhóm nhỏ:** Rất cao — đây là dạng nghiên cứu "benchmark study" nhẹ về tính toán, phù hợp làm bài báo đầu tay cho nghiên cứu sinh mới.
- **Nguồn công bố phù hợp:** *DAS*, *ICDAR (short paper/benchmark track)*, *Journal of Imaging Informatics in Medicine*.
- **Đóng góp/tính mới:** Khuyến nghị thực tiễn có cơ sở thực nghiệm cho các cơ sở y tế khi chọn mô hình OCR triển khai — điền vào khoảng trống giữa nghiên cứu học thuật (benchmark tổng quát) và nhu cầu triển khai thực tế (domain y tế cụ thể).

---

## PHẦN 3 — XẾP HẠNG TOP 3 HƯỚNG TỐT NHẤT

### 🥇 #1 — Hướng 1: Pipeline OCR+LLM cho đăng ký ung thư nhi khoa
**Lý do chọn:** Đây là hướng có **bằng chứng khoảng trống mạnh nhất** trong toàn bộ tài liệu (được nêu minh thị bởi Hands & Kavuluru 2025 và xác nhận bằng tra cứu phủ định của Consensus), đồng thời có **tác động xã hội/lâm sàng rõ ràng nhất** (cải thiện tốc độ và độ chính xác đăng ký ung thư trẻ em — một nhóm bệnh hiếm, dữ liệu ít, mỗi ca đều quý giá cho nghiên cứu dịch tễ). Về khả thi, các kỹ thuật nền (OCR+LLM extraction) đã được chứng minh trên người lớn với hiệu năng cao (F1 0,85–0,95+), nên rủi ro kỹ thuật thấp — thách thức chính là tiếp cận dữ liệu, có thể giải quyết qua hợp tác thể chế.

### 🥈 #2 — Hướng 3: Benchmark và mô hình OCR/HTR y tế tiếng Việt
**Lý do chọn:** Kết hợp đồng thời **hai khoảng trống được tài liệu xác nhận độc lập** (ngôn ngữ ít tài nguyên theo GlotOCRBench/VDAR survey, và thiếu benchmark y khoa chuyên biệt) — tạo ra tính mới kép nhưng vẫn tập trung, không dàn trải. Có **lợi thế cạnh tranh tự nhiên** cho nhóm nghiên cứu (ngôn ngữ, quan hệ địa phương với bệnh viện Việt Nam), độ khả thi kỹ thuật rất cao (không cần dữ liệu khổng lồ, có thể bắt đầu bằng fine-tune mô hình mã nguồn mở sẵn có như PaddleOCR-VL), và là loại đề tài rất phù hợp cho luận án nghiên cứu sinh (rõ ràng, có thể chia giai đoạn: xây dữ liệu → benchmark → fine-tune → công bố).

### 🥉 #3 — Hướng 2: Nhận dạng chữ viết tay lâm sàng bằng VLM nhỏ on-premise
**Lý do chọn:** Đây là khoảng trống được **nhắc lại nhiều lần nhất và nhất quán nhất** trên khắp các nguồn tài liệu (Nicora 2026, Helstad 2025, VDAR 2025, khảo sát tổng hợp) — cho thấy đây là vấn đề cốt lõi, chưa giải quyết, được cộng đồng quốc tế công nhận rộng rãi (nên dễ định vị đóng góp và dễ được phản biện chấp nhận tính cấp thiết). Đồng thời giải quyết đồng thời bài toán kỹ thuật (chữ viết tay) và bài toán thực tiễn cấp bách (bảo mật dữ liệu y tế, xu hướng on-premise) — hai trục quan tâm lớn nhất hiện nay của ngành. Độ khả thi tính toán cao (fine-tune LoRA trên VLM 7B, 1 GPU), dữ liệu khởi đầu có sẵn công khai (IAM, RxHandBD).

**Ghi chú cho việc chọn đề tài cụ thể:** Hướng 1 và Hướng 3 có thể **kết hợp** một cách tự nhiên nếu nhóm nghiên cứu có quan hệ với cả hệ thống y tế Việt Nam lẫn một tổ chức ung thư nhi khoa quốc tế (ví dụ Children's Cancer Institute) — tạo thành một đề tài liên ngành mạnh: "OCR/LLM cho đăng ký ung thư nhi khoa trong bối cảnh đa ngôn ngữ/đa quốc gia", vừa lấp khoảng trống về đối tượng bệnh (nhi khoa) vừa lấp khoảng trống về ngôn ngữ (tiếng Việt) — hai khoảng trống độc lập nhưng bổ trợ nhau rất tốt cho một bài báo có tính mới cao.