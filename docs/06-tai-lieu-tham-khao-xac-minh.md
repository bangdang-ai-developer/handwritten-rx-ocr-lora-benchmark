# Tài liệu tham khảo đã xác minh (17/09/2026)

Tổng hợp từ 3 nghiên cứu độc lập (web search + Google Scholar/Semantic Scholar/arXiv + PubMed/Consensus),
dùng làm nguồn trích dẫn chính cho bản thảo bài báo (Phase 12). Mọi trích dẫn dưới đây đã được xác minh
qua ít nhất 1 nguồn (DOI/arXiv ID/venue), có ghi chú độ tin cậy và bất kỳ sai lệch nào so với giả định ban đầu.

---

## 1. Model/method cốt lõi (đã dùng trong benchmark + fine-tune)

| # | Trích dẫn | Venue | DOI/arXiv | Ghi chú |
|---|---|---|---|---|
| 1 | Li, Lv, Chen, Cui, Lu, Florencio, Zhang, Li, Wei. "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models." | **AAAI 2023**, Vol 37 No 11, pp 13094-13102 | DOI 10.1609/aaai.v37i11.26538 / arXiv:2109.10282 | ~480 trích dẫn. Model nền chính của toàn bộ dự án. |
| 2 | Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. "LoRA: Low-Rank Adaptation of Large Language Models." | **ICLR 2022** | arXiv:2106.09685 | ~15.183 trích dẫn. Nền tảng phương pháp fine-tune. |
| 3 | Kim, Hong, Yim, Nam, Park, Yim, Hwang, Yun, Han, Park. "OCR-free Document Understanding Transformer" (Donut). | **ECCV 2022**, Springer LNCS 13688 | DOI 10.1007/978-3-031-19815-1_29 / arXiv:2111.15664 | Xác nhận pretrain trên SynthDoG (tài liệu tổng hợp) — đúng bằng chứng giải thích domain-mismatch khi Donut fail trên ảnh từ đơn lẻ (Phase 2). |
| 4 | Wei, Liu, Chen, Wang, Kong, Xu, Ge, Zhao, Sun, Peng, Han, Zhang. "General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model" (GOT-OCR2.0). | **arXiv preprint** (chưa xác nhận bình duyệt) | arXiv:2409.01704 | ~104 trích dẫn. |
| 5 | Bai, Chen, Liu, et al. (Qwen Team). "Qwen2.5-VL Technical Report." | **arXiv preprint** (tech report, chuẩn cho model tự nhiên ngữ) | arXiv:2502.13923 | |
| 6 | Cui, Sun, Liang, Gao, Zhang, Liu, Wang, Zhou, Liu, Lin, Zhang, Zhang, Zheng, Zhang, Zhang, Liu, Yu, Ma (Baidu). "PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model." | **arXiv preprint** | arXiv:2510.14528 | Xác nhận đúng version đã test (`PaddlePaddle/PaddleOCR-VL`, không phải v1.5/v1.6 mới hơn). Model bị loại khỏi benchmark do bug thư viện (xem `phase4_summary.md`). |
| 7 | Chang, Li. "Mixed Text Recognition with Efficient Parameter Fine-Tuning and Transformer" (trước đây gọi "DLoRA-TrOCR"). | **ICONIP 2024**, Springer LNCS 15294 (**đã bình duyệt** — không còn chỉ là arXiv preprint) | DOI 10.1007/978-981-96-6599-0_2 / arXiv:2404.12734 (v4, đổi tên) | **⚠️ Đã sửa (17/09):** tựa đề + venue cũ ("DLoRA-TrOCR", arXiv-only) đã lỗi thời — dùng bản Springer/ICONIP khi trích dẫn. Bằng chứng LoRA (0,7% tham số) đạt CER 4,02% trên IAM. |

## 2. Bộ dữ liệu

| # | Trích dẫn | Venue | DOI | Ghi chú |
|---|---|---|---|---|
| 8 | Marti, Bunke. "The IAM-database: an English sentence database for offline handwriting recognition." | **IJDAR**, Vol 5(1), pp 39-46 (2002) | DOI 10.1007/s100320200071 | Xác nhận đúng — nguồn IAM chuẩn mực. |
| 9 | Mia, Chowdhury, Mamun, Ruddra, Tanny. "A Deep Neural Network Approach with Pioneering Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh." | **iCACCESS 2024** (IEEE) | DOI 10.1109/iCACCESS61735.2024.10499631 | Bài gốc của dataset Kaggle-Rx. **Cần tự kiểm tra lại** số liệu 78 lớp/4.680 ảnh trực tiếp trong PDF trước khi trích dẫn con số (agent không fetch được text đầy đủ do JS-rendering). |

## 3. Tiền lệ tạp chí mục tiêu

| # | Trích dẫn | Venue | DOI | Ghi chú |
|---|---|---|---|---|
| 10 | Cheema, Shaiq, Mirza, Kamal, Naeem. "Adapting multilingual vision language transformers for low-resource Urdu optical character recognition (OCR)." | **PeerJ Computer Science**, Vol 10, e1964 (2024) | DOI 10.7717/peerj-cs.1964 | **Xác nhận đúng** — tiền lệ PeerJ CS. Lưu ý khi viết: model chính của họ là ViLanOCR (Swin+mBART-50), **TrOCR chỉ là baseline so sánh trong bài** — không nên mô tả quá là "bài TrOCR gốc", mà là "PeerJ CS đã từng đăng bài dùng TrOCR làm baseline cho OCR chữ viết tay ít tài nguyên". |

## 4. Rủi ro trùng lặp novelty — BẮT BUỘC trích dẫn + phân biệt ở Related Work

| # | Trích dẫn | Venue | Mức overlap | Cách phân biệt trong bài |
|---|---|---|---|---|
| 11 | "RxScribe Bench: A Multi-Axis Benchmark for Evaluating Vision-Language Models on Indian Outpatient Prescriptions." | arXiv:2609.13280 (09/2026) | **Trung bình-cao** — cùng bài toán (benchmark VLM trên đơn thuốc viết tay) | Dataset của họ **riêng tư** (thu thập mới, không công khai); không có phần LoRA fine-tune/phân tích catastrophic forgetting. Bài của ta: **dataset công khai + fine-tune + forgetting analysis** — nêu rõ 2 khác biệt này ngay đầu Related Work. |
| 12 | Blog benchmark cộng đồng trên dataset "RxHandBD" (Zenodo/Mendeley, 02/2026, Md. Masudul Islam) | Không bình duyệt (dev.to, 04/2026) | Thấp-trung bình — dataset công khai Bangladesh khác, không cùng model set | Có thể nhắc ngắn ở Related Work như "một dataset đơn thuốc công khai khác mới xuất hiện", không phải đối thủ cạnh tranh trực tiếp. |
| 13 | "From Handwriting to Structured Data: Benchmarking AI Digitisation of Handwritten Forms" (dự án UBOMI BUHLE, Nam Phi, 2026) | Chưa rõ venue | Thấp — form y tế nói chung, không phải đơn thuốc; dataset riêng tư | Nhắc rất ngắn nếu cần, không phải nguy cơ chính. |
| 14 | "A Hybrid Deep Learning-Based OCR Model for Handwritten Medical Prescriptions." | **Frontiers in Medicine** (08/2026), DOI 10.3389/fmed.2026.1856485 | Thấp — bộ model khác (Tesseract+PaddleOCR cổ điển+GPT-4o), không có GOT-OCR2.0/Qwen2.5-VL/PaddleOCR-VL | Có thể trích dẫn như related work gần chủ đề (OCR y khoa) nhưng không trùng phương pháp. |

## 5. Nền tảng/khảo sát chung (Related Work/Introduction)

| # | Trích dẫn | Venue | Quartile | Ghi chú |
|---|---|---|---|---|
| 15 | Garrido-Munoz, Rios-Vila, Calvo-Zaragoza. "Handwritten Text Recognition: A Survey." | **IEEE TPAMI**, Vol 48(4), pp 4367-4387 (2026) | **Q1** | DOI 10.1109/TPAMI.2025.3646002. Khảo sát HTR uy tín nhất tìm được. |
| 16 | AlKendi, Gechter, Heyberger, Guyeux. "Advancements and Challenges in Handwritten Text Recognition: A Comprehensive Survey." | **Journal of Imaging** (MDPI), Vol 10(1), a18 (2024) | **Q2** | DOI 10.3390/jimaging10010018. Hữu ích vì Journal of Imaging là phương án dự phòng của ta. |
| 17 | Crosilla, Klic, Colavizza. "Benchmarking large language models for handwritten text recognition." | **Journal of Documentation** (Emerald), Vol 81(7), pp 334-354 (2025) | **Q1** | arXiv:2503.15195. Rất khớp khung bài của ta (so sánh VLM tổng quát vs model OCR chuyên biệt). |

## 6. Trích dẫn cho phần Thảo luận — catastrophic forgetting của PEFT/LoRA (Phase 7-9)

| # | Trích dẫn | Venue | Ghi chú |
|---|---|---|---|
| 18 | Biderman, Portes, Ortiz, Paul, Greengard, Jennings, King, Havens, Chiley, Frankle, Blakeney, Cunningham. "LoRA Learns Less and Forgets Less." | **TMLR 2024** (Featured Certification) | arXiv:2405.09673. **Lưu ý hướng ngược lại**: bài này cho rằng LoRA quên ÍT hơn full fine-tune — kết quả của ta (LoRA fine-tune vẫn gây forgetting có ý nghĩa thống kê trên IAM) là một **phát hiện bổ sung/phức tạp hoá** kết luận này, không mâu thuẫn hoàn toàn (ta không so sánh với full fine-tune) nhưng cần đóng khung cẩn thận: "LoRA có thể giảm forgetting so với full fine-tune, nhưng KHÔNG loại bỏ hoàn toàn forgetting — kết quả của chúng tôi cho thấy ngay cả <1% tham số cập nhật cũng đủ gây suy giảm tổng quát hoá có ý nghĩa thống kê". |
| 19 | Shuttleworth, Andreas, Torralba, Sharma. "LoRA vs Full Fine-tuning: An Illusion of Equivalence." | **NeurIPS 2025** | arXiv:2410.21228. Giải thích cơ chế ("intruder dimensions") — dùng để lý giải TẠI SAO LoRA vẫn forget được dù ít tham số. |
| 20 | Chen et al. "Bayesian Parameter-Efficient Fine-Tuning for Overcoming Catastrophic Forgetting." | **IEEE/ACM TASLP** | Xác nhận trực tiếp: "catastrophic forgetting remains an issue with PEFT" — trích dẫn mạnh nhất ủng hộ phát hiện của ta. |
| 21 | Kalajdzievski. "Scaling Laws for Forgetting When Fine-Tuning Large Language Models." | **arXiv preprint** (chưa xác nhận bình duyệt — ghi rõ khi trích) | arXiv:2401.05605 |

## 7. Trích dẫn cho phần Thảo luận — an toàn thuốc/tên thuốc dễ nhầm (Phase 10)

| # | Trích dẫn | Venue | Ghi chú |
|---|---|---|---|
| 22 | Bryan, Aronson, Williams, Jordan. "The problem of look-alike, sound-alike name errors: Drivers and solutions." | **British Journal of Clinical Pharmacology**, 87(2), 386-394 (2020) | PMID 32198938. Bằng chứng nền cho khái niệm LASA (look-alike sound-alike). |
| 23 | Ostini, Roughead, Kirkpatrick, Monteith, Tett. "Quality Use of Medicines - medication safety issues in naming; look-alike, sound-alike medicine names." | **International Journal of Pharmacy Practice**, 20(6), 349-357 (2012) | PMID 23134093. |
| 24 | Karet. "Linguistic Analysis of Generic-Generic Drug Name Pairs Prone to Wrong-Drug Errors for which Tall-Man Lettering is Recommended." | **Therapeutic Innovation & Regulatory Science**, 57(4), 751-758 (2023) | PMID 37171707. **Hữu ích nhất về phương pháp**: định lượng độ giống chữ/âm giữa các cặp thuốc dễ nhầm theo FDA/ISMP — có thể tham chiếu khi giải thích cách định nghĩa `confusable_wrong_drug` (khoảng cách edit chuẩn hoá ≤0,34) trong `src/analyze_errors.py`. |
| 25 | Lizano-Díez et al. "Prevention strategies to identify LASA errors: building and sustaining a culture of patient safety." | **BMC Health Services Research**, 20, 63 (2020) | PMID 31996197. |
| 26 | Zhou, Blackley, Kowalski, Doan, Acker, Landman, Kontrient, Mack, Meteer, Bates, Goss. "Analysis of Errors in Dictated Clinical Documents Assisted by Speech Recognition Software and Professional Transcriptionists." | **JAMA Network Open**, 1(3), e180530 (2018) | PMID 30370424. Tiền lệ gần nhất cho luận điểm "phân loại lỗi theo mức độ nguy hiểm lâm sàng quan trọng hơn chỉ nhìn tỷ lệ lỗi thô" — nhưng ở domain ASR/ghi chú lâm sàng, KHÔNG phải OCR/đơn thuốc. |

**Kết luận quan trọng cho Mục 4 (Phase 10) của bài báo:** research agent xác nhận **KHÔNG tìm thấy công trình nào** kết nối trực tiếp "loại lỗi OCR khi đọc đơn thuốc" với "rủi ro an toàn dùng thuốc" — đây là **góc tiếp cận thực sự mới** của bài báo (không phải chỉ là claim marketing), nên nhấn mạnh rõ trong Discussion: dùng văn tự tương tự "Đến nay, theo hiểu biết của chúng tôi, chưa có nghiên cứu nào phân tích sự dịch chuyển LOẠI lỗi OCR (không chỉ tỷ lệ lỗi) dưới góc độ an toàn dùng thuốc; nguyên lý tổng quát rằng phân loại lỗi theo mức độ nguy hiểm lâm sàng quan trọng hơn tỷ lệ lỗi thô đã có tiền lệ trong lĩnh vực ghi chú lâm sàng bằng ASR [Zhou et al. 2018] và đánh giá an toàn LLM lâm sàng, và cơ sở dữ liệu LASA [Bryan 2020; Ostini 2012; Karet 2023] xác lập rằng lỗi nhầm-tên-thuốc-thật là một nhóm lỗi nguy hiểm được công nhận riêng."

---

## Việc đã sửa trong session này (17/09/2026)

1. **Đã sửa** `src/augmentation.py` docstring — không còn dùng Ali et al. (arXiv:2412.18199) để biện minh riêng cho lựa chọn augmentation (bài đó không phải về augmentation methodology).
2. **Đã sửa** `docs/05-ke-hoach-Q2.md` Mục 3.2 — thêm ghi chú cảnh báo + xoá claim sai.
3. **Đã cập nhật** trích dẫn DLoRA-TrOCR (Mục 6, dòng "CÂU 4") sang bản đã bình duyệt (ICONIP 2024/Springer).
4. **Đã thêm** rủi ro #9 (RxScribe Bench) vào bảng rủi ro Mục 6 của kế hoạch.

## Việc CẦN làm trước khi nộp bài (chưa làm trong session này)

1. Tự kiểm tra lại số liệu 78 lớp/4.680 ảnh trực tiếp trong PDF gốc của Mia et al. (iCACCESS 2024) — agent không fetch được toàn văn.
2. Kiểm tra lại Qwen2.5-VL Technical Report (arXiv:2502.13923) và GOT-OCR2.0 (arXiv:2409.01704) xem đã được chấp nhận ở venue bình duyệt nào chưa tính đến thời điểm nộp bài chính thức (cả 2 vẫn là arXiv-only tại thời điểm kiểm tra 17/09/2026).
3. Đọc kỹ RxScribe Bench (arXiv:2609.13280) đầy đủ trước khi viết đoạn phân biệt novelty ở Related Work — agent chỉ tóm tắt qua abstract/mô tả, chưa đọc toàn văn.
4. Xác minh câu trích "Structure-Aware Text Recognition for Ancient Greek Critical Editions" (arXiv:2603.02803) ở `docs/05-ke-hoach-Q2.md` dòng ~341 — trích dẫn này **CHƯA được kiểm chứng** trong 3 đợt nghiên cứu vừa rồi (nằm ngoài phạm vi câu hỏi), chỉ nên dùng nếu tự kiểm tra lại trước.
