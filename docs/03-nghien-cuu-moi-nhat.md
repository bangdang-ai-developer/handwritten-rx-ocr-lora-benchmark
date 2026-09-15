# 3. Các Nghiên Cứu Mới Nhất Về OCR (2023–2026)

Phần này tổng hợp bốn nguồn độc lập: (3.1) khảo sát tổng quát qua web search, (3.2) khảo sát chuyên biệt về ứng dụng y khoa qua web search, (3.3) tra cứu tài liệu đã bình duyệt trên PubMed, và (3.4) bằng chứng khoa học tổng hợp qua Consensus. Mỗi nguồn có danh sách tham khảo/trích dẫn riêng ở cuối mục — nên đối chiếu lại bản gốc (DOI/arXiv) trước khi trích dẫn chính thức trong bài báo.

---

## 3.1. Xu hướng nghiên cứu OCR tổng quát (không giới hạn y khoa)

## Nghiên cứu tổng quan (Literature Review): OCR / Document AI — Cập nhật 2024–2026

### 1. Bối cảnh và xu hướng lớn

Giai đoạn 2024–2026 đánh dấu sự chuyển dịch rõ rệt từ kiến trúc OCR truyền thống (pipeline tách rời: phát hiện vùng chữ → nhận dạng ký tự → phân tích layout hậu xử lý) sang các mô hình **thị giác-ngôn ngữ (Vision-Language Model, VLM) end-to-end**, thường được gọi là hướng **"OCR-free"** hay **"OCR 2.0"**. Thay vì huấn luyện riêng từng module, các mô hình mới đọc trực tiếp ảnh trang tài liệu và sinh ra văn bản có cấu trúc (Markdown/HTML/JSON), xử lý đồng thời nhận dạng ký tự, bảng biểu, công thức toán, layout và ngữ nghĩa trong một lượt suy luận. Xu hướng này được các bài phân tích ngành gọi là "Vision Models Replace OCR" (Extend.ai, 8/2026) và được thúc đẩy thêm bởi yêu cầu truy vết/tuân thủ (EU AI Act) trong các ngành tài chính, bảo hiểm.

### 2. Scene Text Detection & Recognition (STDR)

Với văn bản trong ảnh tự nhiên (scene text), nghiên cứu 2024–2026 tiếp tục xoay quanh nhận dạng chữ có hình dạng bất kỳ (arbitrary-shape), dùng kiến trúc transformer/contour-based thay cho box hình chữ nhật: ví dụ **PBFormer** (Polynomial Band Transformer, 2023) và các mô hình contour-transformer (CT-Net) tiếp tục được mở rộng. Xu hướng đáng chú ý là dùng mô hình nền tảng thị giác-ngôn ngữ (CLIP) làm bộ phát hiện chữ (2023–2024) và các mô hình end-to-end "spotting" tích hợp ước lượng thứ tự đọc (reading-order) với lấy mẫu động (2024). Một khảo sát toàn diện mới (dự kiến đăng ScienceDirect 2026) tập trung vào chữ viết trong ảnh tự nhiên với ngôn ngữ Ấn Độ (Indic scripts), cho thấy trọng tâm nghiên cứu đang mở rộng ra ngoài tiếng Anh/Latin sang các script phức tạp hơn.

### 3. OCR-free / VLM cho Document Understanding — nhóm mô hình trung tâm

Đây là mảng phát triển sôi động nhất trong 2024–2026:

- **GOT-OCR2.0** (StepFun/UCAS, 9/2024, arXiv:2409.01704) — đề xuất "General OCR Theory", một mô hình thống nhất 580M tham số (encoder nén cao + decoder ngữ cảnh dài) xử lý mọi loại "ký tự quang học": văn bản thường, công thức toán/hoá học, bảng, biểu đồ, thậm chí bản nhạc; đầu ra có thể là Markdown/LaTeX/SMILES.
- **Nougat** (Meta, 2023) và **Kosmos-2.5** (Microsoft, arXiv:2309.11419) — hai mô hình "literate" chuyển PDF khoa học/tài liệu thành Markdown, giữ cấu trúc và công thức; Kosmos-2.5 (1.3B tham số) vượt trội Vary-Base dù nhỏ hơn nhiều, đạt kết quả tốt nhất trên MarkdownEval; khác biệt chính là Nougat dùng LaTeX cho bảng còn Kosmos-2.5 dùng Markdown.
- **Qwen2.5-VL** (Alibaba, 2/2025, arXiv:2502.13923) và **Qwen3-VL** (arXiv:2511.21631) — vượt GPT-4o/Claude 3.5 Sonnet ở nhiều tác vụ document/diagram understanding; bản 7B đạt 883 điểm OCRBench, 94.91 DocVQA.
- **InternVL3** (4/2025, arXiv:2504.10479) và **InternVL3.5** (8/2025, arXiv:2508.18265) — InternVL3-2B đạt 88.3 DocVQA; InternVL3.5-2B đạt 89.4 DocVQA, 70.8 InfoVQA.
- **DeepSeek-VL2** (MoE, các bản Tiny/Small/Base 1.0B–4.5B tham số kích hoạt) là nền tảng cho **DeepSeek-OCR** (10/2025, arXiv:2510.18234) — điểm nhấn là "Contexts Optical Compression": dùng ảnh làm lớp nén ngữ cảnh, giảm 7–20 lần số token cần thiết so với văn bản thuần, đạt Edit Distance <0.25 trên OmniDocBench (gần mức người). **DeepSeek-OCR 2** (1/2026, arXiv:2601.20552) — mô hình 3B, đạt 91.09 điểm trên OmniDocBench v1.5, SOTA mới về hiểu tài liệu có cấu trúc.
- **MinerU / MinerU2.5** (OpenDataLab, arXiv:2604.04771) — hệ thống mã nguồn mở tách 2 giai đoạn "coarse-to-fine": phân tích layout toàn cục trên ảnh thumbnail, sau đó nhận dạng độ phân giải cao cục bộ; mạnh về học thuật, sách giáo khoa, tài liệu scan nhiễu.
- **olmOCR / olmOCR 2** (Allen Institute for AI, olmOCR ra mắt 2/2025; olmOCR 2 phát hành 10/2025, arXiv:2510.19817) — mô hình mã nguồn mở hoàn toàn (dữ liệu, code, trọng số), fine-tune từ Qwen2.5-VL-7B, dùng RL (GRPO) với "phần thưởng kiểm thử đơn vị" (unit-test rewards) để cải thiện xử lý công thức/bảng khó; benchmark riêng **olmOCR-Bench** (7.000+ test case, 1.400 tài liệu).
- **PaddleOCR-VL** (Baidu, 10/2025, arXiv:2510.14528) — mô hình siêu gọn 0.9B (NaViT visual encoder + ERNIE-4.5-0.3B), đạt 96.33% trên OmniDocBench v1.6, mạnh về đa ngôn ngữ; bản **PaddleOCR-VL-1.5/1.6** cải tiến thêm cho robust in-the-wild parsing.
- **dots.ocr** (rednote-hilab/Xiaohongshu, 8/2025) — VLM 1.7B tích hợp phát hiện layout + nhận dạng nội dung trong một mô hình duy nhất, mạnh về ngôn ngữ thiểu số, sau đổi tên thành **dots.mocr** (3/2026).
- **HunyuanOCR** (Tencent, 11/2025, arXiv:2511.19575) — VLM nhẹ 1B, đạt hạng nhất ICDAR 2025 DIMT Challenge (Small Model Track), SOTA OCRBench trong nhóm dưới 3B; bản **HunyuanOCR-1.5** tiếp tục cải thiện tốc độ.
- **GLM-OCR** (arXiv:2603.10910), **MonkeyOCR** (mô hình "structure-recognition-relation triplet"), **Dolphin** (parsing qua "heterogeneous anchor prompting"), và **Granite-Docling** (IBM, kế thừa **SmolDocling**, dùng backbone Granite 3 + SigLIP2, tích hợp pipeline **Docling** mã nguồn mở, arXiv:2408.09869) là các hướng bổ sung đáng chú ý cuối 2025–2026.

### 4. Benchmark / Leaderboard mới

- **OmniDocBench** (CVPR 2025, arXiv:2412.07626) — benchmark toàn diện nhất hiện nay cho document parsing: 1.651 trang PDF, 10 loại tài liệu (báo cáo tài chính, báo in, sách giáo khoa, ghi chú viết tay...), 5 loại layout, 5 ngôn ngữ, chú thích tới 28 loại khối và 4 loại span, hỗ trợ đánh giá cả LaTeX/HTML cho công thức và bảng. Đây gần như là "chuẩn vàng" để so sánh MinerU, PaddleOCR-VL, dots.ocr, DeepSeek-OCR, GOT-OCR2...
- **OCRBench** và **OCRBench v2** (1/2025, arXiv:2501.00321) — OCRBench v2 mở rộng gấp 4 lần số tác vụ so với bản gốc, phủ 31 kịch bản, 10.000 cặp hỏi-đáp được người kiểm chứng, tập trung vào các điểm yếu: định vị văn bản (text localization), chữ viết tay, suy luận logic trên văn bản. Kết quả đáng chú ý: hầu hết LMM SOTA (kể cả GPT-4o, Gemini, Qwen) đều đạt **dưới 50/100 điểm**, cho thấy khoảng cách lớn giữa "đọc được chữ" và "hiểu sâu tài liệu".
- Các benchmark phái sinh/bổ sung khác xuất hiện 2025–2026: **CC-OCR** và **CC-OCR v2** (ICCV 2025, arXiv:2605.03903, đánh giá tính "literacy" trong xử lý tài liệu thực tế), **MosaicDoc** (song ngữ, arXiv:2511.09919), **GlotOCRBench** (đa script, xem mục 6), **Dr. DocBench** (tài liệu khó cấp chuyên gia, arXiv:2606.01393).

### 5. Xử lý layout phức tạp: bảng biểu, form nhiều cột, scan chất lượng thấp

Nhận dạng cấu trúc bảng (Table Structure Recognition) vẫn là bài toán khó: các khảo sát (ACM Computing Surveys 2024, arXiv:2211.08469; arXiv:2312.04808) chỉ ra thiếu chuẩn annotation/benchmark thống nhất. **TableFormer** (IBM, nền tảng cho Docling) dùng CNN + transformer encoder-decoder để dự đoán đồng thời cấu trúc HTML và bounding box ô, xử lý tốt bảng không viền, ô trống, ô gộp. **PP-StructureV2** (Baidu) là hệ thống phân tích tài liệu mạnh cho layout + bảng + trích xuất thông tin. Xu hướng 2025 là dùng chính các VLM lớn (Vision-LLM) để tăng cường nhận dạng bảng, kèm benchmark và "toolchain reasoner" mới (IJCAI 2025).

Về form và trích xuất thông tin có cấu trúc (Key Information Extraction – KIE), **LayoutLMv3** vẫn là baseline mã nguồn mở phổ biến năm 2025 (kết hợp masking văn bản + ảnh thống nhất), nhưng hạn chế rõ với chữ viết tay. Với layout nhiều cột và tài liệu scan chất lượng thấp, các nghiên cứu (LlamaIndex blog, Extend.ai 2025) cho thấy: OCR mã nguồn mở truyền thống chỉ đạt 88–94% độ chính xác khi gặp bảng đa trang, layout nhiều cột hoặc scan mờ; các mô hình VLM lớn (vd. GPT-5.5 trong một số so sánh 2026) thể hiện khả năng giữ đúng thứ tự đọc tốt hơn khi số cột tăng, trong khi một số mô hình khác (Doubao) suy giảm mạnh — cho thấy khả năng khái quát hoá với layout phức tạp vẫn rất khác biệt giữa các mô hình.

### 6. OCR/HTR đa ngôn ngữ và ngôn ngữ ít tài nguyên (bao gồm tiếng Việt)

**GlotOCRBench** (arXiv:2604.12978) là benchmark quan trọng nhất mới về đa script: đánh giá trên 158 script Unicode, cho thấy mô hình hiện nay mạnh với Latin nhưng suy giảm nghiêm trọng ở script trung-tài nguyên và gần như thất bại toàn diện ở script ít tài nguyên (mô hình tốt nhất chỉ phiên âm đúng dưới 7,7% câu với CER<5%). **Nayana OCR** (ACL 2025 workshop) là framework mở rộng cho các ngôn ngữ Ấn Độ ít tài nguyên (Odia, Punjabi, Tamil, Telugu). Một khảo sát chuyên biệt về OCR cho ngôn ngữ ít tài nguyên (George Mason NLP, 2024) và các nghiên cứu benchmark LLM-OCR cho script bị "bỏ quên" (arXiv:2412.16119) khẳng định: dữ liệu tổng hợp (synthetic rendering) trên hàng chục triệu dòng văn bản, ở hàng chục ngôn ngữ, là chìa khoá cải thiện độ bền của mô hình.

Với **tiếng Việt**, phát hiện đáng chú ý nhất là bài khảo sát **"A Survey on Vietnamese Document Analysis and Recognition: Challenges and Future Directions"** (2025, arXiv:2506.05061) — tổng hợp toàn diện các thách thức đặc thù: dấu thanh điệu phức tạp, thiếu dữ liệu gán nhãn quy mô lớn, và đề xuất hướng tận dụng LLM/VLM đa phương thức. Các công trình cụ thể khác: mô hình **transformer-based OCR cho chữ viết tay tiếng Việt** (Springer 2025, WER/SER ~9%/24%), nghiên cứu dùng **vision-based LLM cho nhận dạng chữ viết tay tiếng Việt** (Springer 2025), pipeline số hoá **hồ sơ bệnh án viết tay tiếng Việt** (phát hiện vùng chữ → tăng cường ảnh → phiên âm → tự sửa lỗi), và bộ dữ liệu **ViOCRVQA** (4/2024, arXiv:2404.18397, Multimedia Systems 2025) — 28.000+ ảnh, 120.000+ cặp hỏi-đáp về văn bản trong ảnh tiếng Việt, cùng mô hình đề xuất VisionReader (EM 0.4116, F1 0.6990); mở rộng gần đây là **ViInfographicVQA** (arXiv:2512.12424) và **ViConsFormer** cho cụm từ scene-text tiếng Việt (arXiv:2410.14132). Nhìn chung, tiếng Việt vẫn thuộc nhóm ngôn ngữ "trung-thấp tài nguyên" trong bối cảnh OCR toàn cầu, thiếu benchmark quy mô lớn tương đương OmniDocBench.

### 7. Nhận dạng chữ viết tay lịch sử (Historical HTR)

**Transkribus** tiếp tục là nền tảng thương mại/học thuật dẫn đầu: hơn 300 mô hình công khai, hỗ trợ 100+ ngôn ngữ, mô hình flagship "Text Titan I" huấn luyện trên hơn 30 triệu từ tài liệu lịch sử đa thế kỷ. Hạn chế được ghi nhận: độ chính xác phụ thuộc mạnh vào chất lượng nguồn (mực phai, vết ố, giấy da hỏng), vẫn cần hiệu đính thủ công đáng kể. Song song đó, hệ sinh thái mã nguồn mở **eScriptorium** (xây trên engine **Kraken**, dùng RNN + CTC) tiếp tục được cộng đồng nhân văn số dùng cho các dự án như thử nghiệm HTR cho bản thảo Latin thời Trung cổ (KBLab, 6/2025) và các dự án 2025–2026 áp dụng cho bản thảo tiếng Ý tiền hiện đại và chữ Devanagari.

### 8. Hiệu quả / OCR trên thiết bị (on-device, edge)

Xu hướng "nhỏ mà mạnh" nổi bật rõ trong 2025–2026: **GOT-OCR2.0** (580M), **PaddleOCR-VL** (0.9B), **dots.ocr** (1.7B), **HunyuanOCR** (1B) đều là các VLM chuyên OCR có kích thước dưới 2B nhưng đạt hiệu năng cạnh tranh với mô hình lớn hơn nhiều lần. **Moondream2** (<2B, ~1GB) được thiết kế riêng cho thiết bị di động/edge, đạt 61,2 điểm OCRBench, tốt với form/bảng in nhưng yếu với chữ viết tay. **RolmOCR** được định vị cho triển khai OCR nhẹ khi không đủ tài nguyên GPU cho mô hình 30B+. Về mặt kỹ thuật nén, các phương pháp lượng tử hoá INT8, cắt tỉa trọng số (weight pruning) và backbone nhẹ (MobileNetV3, CRNN) tiếp tục được dùng để đạt suy luận độ trễ thấp trên CPU/ARM. Một hướng kỹ thuật mới, riêng biệt và có ảnh hưởng lớn là "nén ngữ cảnh quang học" của DeepSeek-OCR — dùng chính ảnh làm phương tiện nén token, giảm 7–20 lần chi phí token so với biểu diễn văn bản thuần, mở ra hướng tiết kiệm tài nguyên cho ứng dụng xử lý tài liệu quy mô lớn (không chỉ riêng edge).

### 9. Nhận định chung

Ba xu hướng lớn nhất giai đoạn 2024–2026: (1) hội tụ về kiến trúc VLM end-to-end thay thế pipeline OCR module hoá truyền thống; (2) cuộc đua song song giữa mô hình "khổng lồ đa năng" (Qwen3-VL, InternVL3.5, Gemini, GPT) và mô hình "OCR chuyên dụng siêu gọn" (PaddleOCR-VL, HunyuanOCR, dots.ocr, GOT-OCR2) — nhóm sau thường đạt hiệu năng/chi phí tốt hơn cho tác vụ document parsing thuần tuý; (3) khoảng cách chất lượng ngày càng rõ giữa ngôn ngữ giàu tài nguyên (Anh, Trung, các script Latin) và ngôn ngữ ít tài nguyên (bao gồm tiếng Việt) — được OCRBench v2 và GlotOCRBench định lượng cụ thể — cho thấy đây vẫn là khoảng trống nghiên cứu lớn, đặc biệt về benchmark quy mô lớn và dữ liệu huấn luyện tổng hợp cho tiếng Việt.

---

### Danh sách tham khảo

1. General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model (GOT-OCR2.0), 2024 — https://arxiv.org/abs/2409.01704
2. Kosmos-2.5: A Multimodal Literate Model, 2023 — https://arxiv.org/abs/2309.11419
3. Qwen2.5-VL Technical Report, 2025 — https://arxiv.org/abs/2502.13923
4. Qwen3-VL Technical Report, 2025 — https://arxiv.org/pdf/2511.21631
5. InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models, 2025 — https://arxiv.org/pdf/2504.10479
6. InternVL3.5: Advancing Open-Source Multimodal Models, 2025 — https://arxiv.org/pdf/2508.18265
7. DeepSeek-OCR: Contexts Optical Compression, 2025 — https://arxiv.org/html/2510.18234v1 (GitHub: https://github.com/deepseek-ai/DeepSeek-OCR)
8. DeepSeek-OCR 2: Visual Causal Flow, 2026 — https://arxiv.org/pdf/2601.20552
9. MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale, 2026 — https://arxiv.org/pdf/2604.04771
10. olmOCR 2: Unit Test Rewards for Document OCR, 2025 — https://arxiv.org/pdf/2510.19817 (blog: https://allenai.org/blog/olmocr-2)
11. PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact VLM, 2025 — https://arxiv.org/pdf/2510.14528
12. dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model, 2025 — https://github.com/rednote-hilab/dots.ocr
13. HunyuanOCR Technical Report, 2025 — https://arxiv.org/abs/2511.19575
14. OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations, CVPR 2025 — https://arxiv.org/abs/2412.07626
15. OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning, 2025 — https://arxiv.org/pdf/2501.00321
16. Docling Technical Report (IBM), 2024 — https://arxiv.org/pdf/2408.09869
17. Deep Learning for Table Detection and Structure Recognition: A Survey, ACM Computing Surveys, 2024 — https://arxiv.org/pdf/2211.08469
18. GlotOCR Bench: OCR Models Still Struggle Beyond a Handful of Unicode Scripts, 2026 — https://arxiv.org/pdf/2604.12978
19. A Survey on Vietnamese Document Analysis and Recognition: Challenges and Future Directions, 2025 — https://arxiv.org/html/2506.05061
20. ViOCRVQA: Novel Benchmark Dataset and Vision Reader for VQA by Understanding Vietnamese Text in Images, 2024 — https://arxiv.org/abs/2404.18397
21. ViConsFormer: Constituting Meaningful Phrases of Scene Texts in Vietnamese Text-based VQA, 2024 — https://arxiv.org/html/2410.14132
22. Transkribus — AI platform for historical documents — https://www.transkribus.org/
23. eScriptorium — Open Source Platform for Historical Document Analysis — https://escriptorium.eu/about/
24. OCR & AI: Vision Models Replace OCR, Extend.ai, 2026 — https://www.extend.ai/resources/ocr-and-ai-vision-language-models-replacing-text-recognition

**Lưu ý phương pháp luận:** Tài liệu này được tổng hợp qua WebSearch/WebFetch (không truy cập trực tiếp toàn văn từng bài báo trừ bài khảo sát tiếng Việt), nên các số liệu benchmark nên được đối chiếu lại với bản PDF gốc trước khi trích dẫn trong công trình chính thức. Một số nguồn về leaderboard OCRBench thời điểm 9/2026 (tên phiên bản mô hình rất mới) có độ tin cậy chưa xác minh cao và đã được loại khỏi phần số liệu cụ thể trong báo cáo này.
---

## 3.2. Ứng dụng OCR / Document AI trong Y khoa

## Tổng quan tài liệu: Ứng dụng OCR / Document AI trong Y khoa (ưu tiên 2023–2026)

### 1. Số hóa hồ sơ bệnh án điện tử (EHR) từ tài liệu scan/viết tay

Phần lớn cơ sở y tế trên thế giới vẫn còn một tỷ lệ đáng kể hồ sơ giấy dù đã triển khai EHR. Xu hướng 2024–2025 cho thấy việc số hóa ngày càng dựa vào ảnh chụp bằng điện thoại (thay vì máy scan chuyên dụng), khiến văn bản OCR "nhiễu" hơn — lỗi nhận nhầm ký tự, ngắt dòng sai lệch nghiêm trọng đến chất lượng trích xuất dữ liệu ở bước sau [Wang et al., 2025]. Các giải pháp AI-OCR hiện đại (kết hợp OCR + NLP + deep learning) đang chuyển đổi ghi chú viết tay/in thành các định dạng chuẩn hóa như HL7 FHIR, JSON để tích hợp trực tiếp vào EHR. Một hướng nghiên cứu mới nổi là dùng các mô hình ngôn ngữ đa phương thức (multimodal LLM) cỡ nhỏ — Phi-4, Qwen-VL, InternVL-3.5 — thay thế OCR cổ điển (Tesseract, PaddleOCR) cho báo cáo lâm sàng nhiễu, với ưu điểm bảo mật (chạy on-premise) phù hợp môi trường y tế hạn chế tài nguyên [Neveditsin et al., 2025]. Về mặt thị trường, quy mô OCR toàn cầu đạt 13,95 tỷ USD năm 2024 và được dự báo lên 46 tỷ USD vào 2033, với y tế là một trong các ngành tăng trưởng nhanh nhất.

### 2. Nhận dạng chữ viết tay đơn thuốc (Handwritten Prescription Recognition)

Đây được xem là bài toán OCR y khoa khó nhất do sự đa dạng cực lớn trong chữ viết bác sĩ, khoảng cách chữ không đều, viết tắt phi chuẩn và thuật ngữ dược phẩm liên tục thay đổi. Một thống kê được trích dẫn nhiều: 35,7% đơn thuốc viết tay chứa lỗi, so với chỉ 2,5% ở đơn thuốc điện tử — minh chứng cho nhu cầu cấp thiết của tự động hóa. Về mặt bộ dữ liệu, **RxHandBD** (Zenodo/Mendeley, 2024–2025) là dataset chuyên biệt gồm 5.578 từ viết tay được cắt từ đơn thuốc thật, chia theo tỷ lệ 80/20 train/test, chuẩn hóa ảnh 128×128px — dùng để benchmark 4 engine OCR mã nguồn mở. Về kiến trúc mô hình, các nghiên cứu 2024–2025 sử dụng CRNN kết hợp CTC loss (không cần phân đoạn ký tự) và đáng chú ý là việc fine-tune mô hình **Donut** (NAVER Clova AI) chuyên biệt cho đơn thuốc viết tay, xử lý ảnh-đến-văn-bản end-to-end (Hugging Face: chinmays18/medical-prescription-ocr). Với ngữ cảnh song ngữ/đa ngôn ngữ, hệ thống **ViLanOCR** thích ứng mô hình vision-language đa ngôn ngữ để đọc đơn thuốc viết tay chứa cả tiếng Anh và tiếng Urdu (tên bệnh, tên bệnh nhân, tuổi, thuốc, liều lượng) [PMC11065407].

### 3. Số hóa báo cáo giải phẫu bệnh, chẩn đoán hình ảnh, kết quả xét nghiệm

**Giải phẫu bệnh (Pathology):** Nghiên cứu gần đây tập trung vào việc dùng LLM để trích xuất thông tin có cấu trúc từ báo cáo pathology. Năm 2023, GPT-3.5 được đánh giá trích xuất 17 đặc trưng từ 340 báo cáo ung thư vú; năm 2024, GPT-4 được thử nghiệm trên 5 đặc trưng từ báo cáo ung thư đại trực tràng (TCGA) và 11 đặc trưng từ báo cáo u thần kinh đệm (glioma) của University College London Hospitals [arXiv:2502.12183]. Bộ dữ liệu **TCGA-Reports** (công bố 2024, tạp chí *Patterns*) cung cấp kho báo cáo pathology dạng máy đọc được, dùng làm chuẩn benchmark cho các mô hình AI dựa trên văn bản y khoa.

**Chẩn đoán hình ảnh (Radiology):** Một tổng quan hệ thống kết hợp phân tích gộp (2025, *Journal of Imaging Informatics in Medicine*, DOI: 10.1007/s10278-025-01728-8) khảo sát 28 nghiên cứu, 421.692 thực thể được trích xuất từ 51.187 báo cáo radiology tự do dạng văn bản, cho kết quả pooled sensitivity 91% (KTC 95%: 87–93), specificity 96% (93–97), AUC 0,98 — cho thấy NLP đã đạt hiệu năng rất cao trên loại tài liệu này, dù còn khác biệt đáng kể giữa vị trí giải phẫu đơn lẻ so với đa vị trí [Yang et al., 2025]. Một scoping review khác trên *npj Digital Medicine* (2024, DOI: 10.1038/s41746-024-01219-0) rà soát 34 nghiên cứu dùng LLM để trích xuất thông tin từ báo cáo radiology, ghi nhận báo cáo CT/MRI và lồng ngực chiếm ưu thế, đồng thời chỉ ra hạn chế phổ biến: thiếu kiểm định trên dữ liệu ngoài (external validation) khiến hiệu năng giảm khi áp dụng liên viện. Khung **RadEx** (2024, arXiv:2406.15465) là một framework dựa trên LLM chuyên biệt để trích xuất thông tin có cấu trúc từ radiology report.

**Xét nghiệm (Lab results):** Nghiên cứu tiêu biểu nhất trong nhóm này là của Ma và cộng sự (2023, *BMC Medical Informatics and Decision Making*, DOI: 10.1186/s12911-023-02346-6), xây dựng pipeline OCR + NLP hai giai đoạn trên 153 báo cáo xét nghiệm giấy từ Bệnh viện số 1 Đại học Bắc Kinh (PKU1). Module OCR đạt độ chính xác trung bình 0,93 ở ba mức đánh giá; module trích xuất thông tin (dùng CRF-based NER) đạt F1 = 0,86 khi trích 4 loại thực thể (tên xét nghiệm, kết quả, đơn vị, khoảng tham chiếu), thời gian suy luận chỉ 0,78 giây/báo cáo trên một CPU — chứng minh tính khả thi triển khai thực tế với tài nguyên tính toán tối thiểu.

### 4. Số hóa hồ sơ bệnh án lịch sử/lưu trữ phục vụ nghiên cứu, gồm đăng ký ung thư (cancer registry)

Đây là mảng liên quan trực tiếp đến ung thư trẻ em/nhi khoa. Nghiên cứu nổi bật của Yoon và cộng sự (2022, *JAMIA Open*, DOI: 10.1093/jamiaopen/ooac049) phát triển mô hình deep learning trích xuất thông tin tự động từ báo cáo giải phẫu bệnh ung thư trẻ em, huấn luyện/kiểm định trên 29.206 báo cáo (bệnh nhân 0–19 tuổi) từ 6 cơ quan đăng ký ung thư bang tại Mỹ, phân loại theo chuẩn ICCC (International Classification of Childhood Cancer). Mô hình phân loại trực tiếp ICCC đạt micro-F1 = 0,987, chỉ "từ chối" (không đủ tin cậy để gán mã) 14,8% báo cáo mơ hồ — cho thấy AI có thể hỗ trợ đáng kể nhân sự đăng ký ung thư trong việc đọc và trích xuất phần lớn báo cáo một cách chính xác, đáng tin cậy.

Một khảo sát tổng quan gần đây (Hands & Kavuluru, 2025, *Artificial Intelligence Review*, DOI: 10.1007/s10462-025-11316-5) rà soát 156 bài báo (2014–2024) về NLP trong vận hành cancer registry, phân loại theo phương pháp (rule-based n=70, machine learning n=66, deep learning truyền thống n=70, transformer n=29 — tăng mạnh từ 2019). Điểm đáng chú ý: **ung thư nhi khoa, u hắc tố và lymphoma bị đại diện thiếu** (underrepresented) trong nghiên cứu NLP hiện có — một khoảng trống nghiên cứu rõ ràng cho các tổ chức tập trung vào ung thư trẻ em. Nghiên cứu mới hơn (McPhaul et al., 2026, medRxiv, DOI: 10.64898/2026.03.20.26348915) so sánh hai nền tảng trích xuất tự động: **Brim Analytics** (dựa trên LLM) và **DeepPhe** (dựa trên ontology), thử nghiệm trên 330 báo cáo ung thư tụy và 34 báo cáo ung thư vú từ Johns Hopkins Hospital. Brim Analytics đạt độ chính xác trung bình 96,7% (ung thư tụy) trên 7 biến số đăng ký (T stage 96,4%, độ mô học 97,0%), trong khi DeepPhe có điểm yếu rõ ở T stage (chỉ 83,6% với tụy, 70,6% với vú) — cho thấy cách tiếp cận LLM-based hiện vượt trội hơn ontology-driven cho tác vụ này. Ngoài ra, dự án đăng ký ung thư trẻ em tại Thụy Sĩ (2025, medRxiv) minh họa việc duy trì registry quốc gia liên tục cập nhật phương pháp thu thập dữ liệu.

Ở mảng lưu trữ lịch sử nói chung, các nghiên cứu 2024–2025 (arXiv:2510.06743, arXiv:2509.13236) đánh giá LLM cho OCR tài liệu lịch sử, với bằng chứng sơ bộ rằng LLM vượt trội hệ thống truyền thống trên văn bản viết tay/lịch sử, dù kết quả về hiệu chỉnh hậu-OCR (post-OCR correction) bằng LLM còn mâu thuẫn giữa các nghiên cứu.

### 5. Xử lý tài liệu bảo hiểm y tế, phiếu đồng ý tham gia nghiên cứu, form y tế

**Bảo hiểm y tế:** Các hệ thống OCR AI hiện trích xuất tự động từ mẫu đơn CMS-1500, UB-04 — thông tin bệnh nhân, mã bảo hiểm, chi tiết nhà cung cấp — với độ chính xác được nhà cung cấp thương mại tuyên bố trên 99%, kết hợp NLP để kiểm tra điều kiện chi trả, phát hiện trùng lặp/gian lận. Đáng lưu ý, tỷ lệ từ chối yêu cầu bồi thường ban đầu năm 2024 đạt 11,81%, làm nổi bật tầm quan trọng tài chính của độ chính xác ngay từ khâu nhập liệu.

**Phiếu đồng ý tham gia nghiên cứu (Informed Consent):** Một scoping review (2025, *BMC Health Services Research*) rà soát 27 nghiên cứu (2012–2024) về số hóa quy trình đồng ý, cho thấy số hóa giúp cải thiện hiểu biết của người tham gia về thủ tục, rủi ro/lợi ích. Về mặt kỹ thuật OCR, tài liệu thử nghiệm lâm sàng (bao gồm ICF và case report forms) thường chứa cả văn bản in và chữ ký/phản hồi viết tay — việc kết hợp OCR + layout parsing giúp tăng tốc nhập liệu thử nghiệm. Hệ thống **InformGen** (2025, arXiv:2504.00934) là một AI copilot hỗ trợ soạn thảo tài liệu đồng ý tuân thủ quy định, dùng kỹ thuật RAG với AWS Textract để chuyển đổi protocol PDF (hàng chục đến hàng trăm trang) sang Markdown máy đọc được trước khi trích xuất nội dung liên quan.

### 6. Kết hợp OCR + NLP để trích xuất thông tin lâm sàng có cấu trúc

Đây là lớp "hạ tầng" chung cho tất cả ứng dụng trên. Các nền tảng thương mại lớn đã tích hợp sẵn: **Amazon Textract** (OCR nâng cao, nhận diện field/table trong form) kết hợp **Amazon Comprehend Medical** (NLP phát hiện bệnh lý, thuốc, PHI từ văn bản lâm sàng tự do — ghi chú bác sĩ, tóm tắt xuất viện); **Google Cloud Healthcare Natural Language API** dùng Document AI (OCR) làm bước tiền xử lý trước khi mô hình NLP y tế phân tích. Về nghiên cứu học thuật, các pipeline NER lâm sàng hiện đại (2024–2025) đã có 151 mô hình NER lâm sàng huấn luyện sẵn (John Snow Labs), với hướng mới là kết hợp BERT fine-tuned với retrieval-augmented generation (RAG) có từ điển y khoa để chuẩn hóa thuật ngữ, cũng như framework zero-shot NER dựa trên LLM (2025, ACL) không cần dữ liệu huấn luyện gán nhãn cho từng loại thực thể mới.

### 7. Các thách thức riêng của OCR y khoa

**Bảo mật/quyền riêng tư:** Không giống OCR thông thường, tài liệu y tế luôn chứa PHI (protected health information), đòi hỏi pipeline tuân thủ HIPAA (hoặc luật tương đương như Privacy Act ở Úc) — bao gồm chiến lược token hóa/ẩn danh hóa (pseudonymization) trước khi lưu trữ/xử lý, mã hóa toàn phần thiết bị chụp ảnh di động, và tiêu hủy tài liệu giấy đạt chuẩn sau khi số hóa. Các mô hình NER de-identification (2023, arXiv:2312.08495) đóng vai trò cốt lõi trong việc tự động ẩn danh dữ liệu lâm sàng thực tế quy mô lớn.

**Đa dạng chữ viết tay bác sĩ:** Đã nêu ở mục 2 — đây là nguyên nhân hàng đầu gây lỗi thuốc và là rào cản kỹ thuật lớn nhất.

**Thuật ngữ/viết tắt y khoa:** OCR tổng quát không được huấn luyện trên từ vựng y khoa chuyên biệt (tên hoạt chất, mã ICD, viết tắt như "bid", "prn") nên tỷ lệ lỗi tăng đáng kể so với văn bản phổ thông.

**Hồ sơ đa ngôn ngữ:** Một khung xử lý chi phí thấp, khả năng mở rộng (IDPA, 2025, IET Conference Proceedings, DOI: 10.1049/icp.2025.3682) kết hợp OCR + Vision-Language Model để số hóa hồ sơ song ngữ Hindi-English, viết tay, chứa số liệu — thử nghiệm trên 150 hồ sơ bệnh nhân Ấn Độ, mô hình PaliGemma đạt 74% độ chính xác, CER 13%. Với tiếng Việt, nghiên cứu của Dinh và cộng sự (ICIS 2023) xây dựng pipeline nhận dạng chữ viết tay tiếng Việt chuyên biệt cho hồ sơ cấp cứu uốn ván (Bệnh viện Bệnh Nhiệt Đới), giải quyết thách thức riêng của tiếng Việt (6 thanh điệu, dấu nguyên âm) bằng BLSTM xử lý vấn đề "delayed stroke" (nét bút trễ) — đạt Character Error Rate 2% và Word Error Rate 12% trên hồ sơ của 30 bác sĩ/điều dưỡng, dưới áp lực thời gian cấp cứu.

**Chất lượng ảnh tài liệu cũ:** Tài liệu lưu trữ lâu năm (mờ, ố, cong vênh) khiến engine OCR cổ điển (Tesseract) rất nhạy với nhiễu — đây là động lực chính thúc đẩy nghiên cứu chuyển sang multimodal LLM (mục 1) vốn chịu nhiễu tốt hơn.

### 8. Bộ dữ liệu benchmark trong lĩnh vực này

- **RxHandBD** (2024–2025): 5.578 từ viết tay từ đơn thuốc thật, dùng cho HTR y khoa.
- **TCGA-Reports** (2024, *Patterns*): kho báo cáo giải phẫu bệnh dạng máy đọc, benchmark cho mô hình AI văn bản y khoa.
- **Named Clinical Entity Recognition Benchmark** (2024, arXiv:2410.05046): benchmark NER lâm sàng đa thực thể.
- **IAM Handwriting Database**: dataset chữ viết tay tổng quát (13.353 dòng văn bản, 657 người viết) — vẫn là benchmark HTR được trích dẫn nhiều nhất 2024–2025, dù không chuyên biệt y khoa.
- **METATR** (2026, arXiv:2605.26712): benchmark đa ngôn ngữ, đang phát triển cho nhận dạng văn bản tự động, có thể áp dụng vào tài liệu y khoa đa ngôn ngữ.
- **KITAB-Bench** (2025, arXiv:2502.14949): benchmark OCR/Document Understanding đa lĩnh vực cho tiếng Ả Rập, tham chiếu hữu ích cho bài toán ngôn ngữ ít tài nguyên tương tự tiếng Việt.

### Kết luận ngắn

Giai đoạn 2023–2026 chứng kiến sự dịch chuyển rõ rệt từ OCR cổ điển (Tesseract, CRNN+CTC) sang các mô hình vision-language/LLM đa phương thức cho hầu hết ứng dụng — từ đơn thuốc viết tay, báo cáo pathology/radiology, đến đăng ký ung thư — mang lại độ chính xác cao hơn trên dữ liệu nhiễu nhưng đặt ra thách thức mới về kiểm định ngoài (external validation), chi phí tính toán, và tuân thủ quyền riêng tư. Mảng ung thư nhi khoa/đăng ký ung thư trẻ em vẫn được xác định là "underrepresented" trong nghiên cứu NLP hiện có [Hands & Kavuluru, 2025] — đây vừa là thách thức vừa là cơ hội nghiên cứu rõ ràng cho các tổ chức chuyên về ung thư trẻ em.

---

### Tài liệu tham khảo

1. RxHandBD: A Handwritten Prescription Word Image Dataset. Zenodo, 2025. https://zenodo.org/records/18478741 ; Mendeley Data, 2024. https://data.mendeley.com/datasets/dsb5r6vskg/3
2. Pather, N., Fouché, J., Mundia, S. và cộng sự. *From Handwriting to Structured Data: Benchmarking AI Digitisation of Handwritten Forms*. arXiv:2604.16504, 2026.
3. Dinh, M.N., Le, M.T., Bui, T. và cộng sự. *A Vietnamese Handwritten Text Recognition Pipeline for Tetanus Medical Records*. ICIS 2023 Proceedings. https://aisel.aisnet.org/icis2023/ishealthcare/ishealthcare/2/
4. Fine-tuned Donut model cho đơn thuốc viết tay. Hugging Face: chinmays18/medical-prescription-ocr.
5. Neveditsin, N., Lingras, P., Patil, S., Patil, S., Mago, V. *Compact Multimodal Language Models as Robust OCR Alternatives for Noisy Textual Clinical Reports*. arXiv:2511.13523, 2025.
6. Wang, Y., Li, Y., Qin, Y., Qian, H. *Key Coverage Matters: Semi-Structured Extraction of OCR Clinical Reports*. arXiv:2605.09440, 2025.
7. Ma, M.-W., Gao, X.-S., Zhang, Z.-Y. và cộng sự. *Extracting laboratory test information from paper-based reports*. BMC Medical Informatics and Decision Making, 2023, 23:251. DOI: 10.1186/s12911-023-02346-6 (theo PubMed, PMID 37932733).
8. *TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models*. Patterns, 2024. DOI: 10.1016/j.patter.2024.100933.
9. *Leveraging large language models for structured information extraction from pathology reports*. arXiv:2502.12183, 2025.
10. Reichenpfader, D., Müller, H., Denecke, K. *A scoping review of large language model based approaches for information extraction from radiology reports*. npj Digital Medicine, 2024, 7:222. DOI: 10.1038/s41746-024-01219-0 (theo PubMed, PMID 39182008).
11. Yang, Q., Jiang, J., Dong, X. và cộng sự. *Performance of Natural Language Processing Model in Extracting Information from Free-Text Radiology Reports: A Systematic Review and Meta-Analysis*. Journal of Imaging Informatics in Medicine, 2025, 39(4):3639–3653. DOI: 10.1007/s10278-025-01728-8 (theo PubMed, PMID 41152658).
12. *RadEx: A Framework for Structured Information Extraction from Radiology Reports based on Large Language Models*. arXiv:2406.15465, 2024.
13. Yoon, H.-J., Peluso, A., Durbin, E.B. và cộng sự. *Automatic information extraction from childhood cancer pathology reports*. JAMIA Open, 2022, 5(2):ooac049. DOI: 10.1093/jamiaopen/ooac049 (theo PubMed, PMID 35721398).
14. Hands, I., Kavuluru, R. *A survey of NLP methods for oncology in the past decade with a focus on cancer registry applications*. Artificial Intelligence Review, 2025, 58(10):314. DOI: 10.1007/s10462-025-11316-5 (theo PubMed, PMID 40688631).
15. McPhaul, T., Kreimeyer, K., Baras, A., Botsis, T. *Automated Extraction of Cancer Registry Data from Pathology Reports: Comparing LLM-Based and Ontology-Driven NLP Platforms*. medRxiv, 2026. DOI: 10.64898/2026.03.20.26348915 (theo PubMed, PMID 41929331).
16. *The childhood cancer registry in Switzerland: methods and results in 2025*. medRxiv, 2025.10.26.25338836.
17. *Digitalizing informed consent in healthcare: a scoping review*. BMC Health Services Research, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12225439/
18. *InformGen: An AI Copilot for Accurate and Compliant Clinical Research Consent Document Generation*. arXiv:2504.00934, 2025.
19. Amazon Comprehend Medical & Amazon Textract — AWS Documentation, 2024–2025. https://docs.aws.amazon.com/comprehend-medical/
20. Google Cloud Blog. *Medical Text Processing with the Healthcare Natural Language API*. https://cloud.google.com/blog/topics/healthcare-life-sciences/medical-text-processing-on-google-cloud
21. *A scalable, low-cost framework for multilingual intelligent document processing for continuity of care*. IET Conference Proceedings, 2025. DOI: 10.1049/icp.2025.3682.
22. *Adapting multilingual vision language transformers for low-resource Urdu optical character recognition (OCR)* (ViLanOCR). PMC11065407, 2024 (theo PubMed, PMID 38699211).
23. *Building a HIPAA-Compliant OCR Pipeline: A Technical Guide*. IntuitionLabs, 2024–2025. https://intuitionlabs.ai/articles/hipaa-compliant-ocr-pipeline
24. *Beyond Accuracy: Automated De-Identification of Large Real-World Clinical Text Datasets*. arXiv:2312.08495, 2023.
25. *Named Clinical Entity Recognition Benchmark*. arXiv:2410.05046, 2024.

*Ghi chú phương pháp:* Tổng quan này được tổng hợp bằng WebSearch/WebFetch và tra cứu PubMed (theo PubMed) trong tháng 9/2026, ưu tiên các công bố 2023–2026. Một số nguồn là preprint (arXiv, medRxiv) chưa qua bình duyệt đầy đủ — cần đối chiếu bản xuất bản chính thức khi trích dẫn học thuật.
---

## 3.3. Tài liệu khoa học đã bình duyệt (PubMed) về OCR trong Y tế

According to PubMed (NCBI), dưới đây là kết quả tổng quan tài liệu khoa học đã bình duyệt về ứng dụng OCR (Optical Character Recognition – nhận dạng ký tự quang học) trong y tế/lâm sàng, tập trung giai đoạn 2022–2026.

**Ghi chú phương pháp tìm kiếm:** Đã chạy 6 truy vấn trên PubMed với các tổ hợp từ khóa được yêu cầu. Đáng lưu ý: truy vấn "OCR pathology report" trả về 112 kết quả nhưng khi kiểm tra toàn bộ đều KHÔNG liên quan — vì trong sinh học ty thể, "OCR" là viết tắt của "oxygen consumption rate" (tốc độ tiêu thụ oxy), không phải nhận dạng ký tự quang học; đã loại bỏ toàn bộ nhánh này. Sau khi lọc thủ công ~60 bài báo ứng viên (đọc tiêu đề + tóm tắt đầy đủ), chọn ra 15 bài liên quan trực tiếp và có chất lượng cao nhất, trải đều các lĩnh vực ứng dụng (nghiên cứu lâm sàng, xét nghiệm, X-quang/DXA, nhãn khoa, thính học, hồ sơ bệnh án giấy, y tế toàn cầu).

---

### DANH SÁCH 15 BÀI BÁO LIÊN QUAN NHẤT

#### 1. Facilitating clinical research through automation: Combining optical character recognition with natural language processing
- **Tác giả chính:** Hom J, et al. | **Năm:** 2022 | **Tạp chí:** Clinical Trials (Clin Trials) | **PMID:** 35608136 | DOI: 10.1177/17407745221093621
- **Tóm tắt:** Nhóm nghiên cứu tại City of Hope kết hợp OCR (phần mềm ABBYY FineReader) với NLP (Linguamatics i2e) để trích xuất chỉ số performance status từ hồ sơ bệnh án scan của 189 bệnh nhân u lympho tế bào B lớn lan tỏa. Pipeline đạt độ chính xác cao (<1% lỗi), giảm 83% thời gian trích xuất dữ liệu (từ 108 phút xuống 18 phút/bệnh nhân đối với tài liệu không chứa thông tin cần tìm).
- **Khoảng trống nghiên cứu:** Không nêu rõ future work, nhưng nghiên cứu chỉ thực hiện trên một bệnh lý (u lympho) tại một trung tâm — hàm ý cần kiểm chứng khả năng khái quát hóa sang bệnh lý/cơ sở khác.

#### 2. Development of novel optical character recognition system to reduce recording time for vital signs and prescriptions: A simulation-based study
- **Tác giả chính:** Soeno S, et al. | **Năm:** 2024 | **Tạp chí:** PLoS ONE | **PMID:** 38241403 | DOI: 10.1371/journal.pone.0296319
- **Tóm tắt:** Thử nghiệm mô phỏng tại 3 bệnh viện và 2 trạm cứu hỏa ở Nhật Bản với 38 nhân viên y tế (paramedic, điều dưỡng, bác sĩ), so sánh OCR với gõ tay khi ghi sinh hiệu và đơn thuốc từ ảnh chụp. OCR nhanh hơn đáng kể khi ghi đơn thuốc (18s so với 144s cho 6 loại thuốc) và có tỷ lệ lỗi thấp hơn ở cả hai loại dữ liệu.
- **Khoảng trống nghiên cứu:** Tác giả tự nhận định **OCR giúp giảm thời gian ghi đơn thuốc nhưng KHÔNG giúp giảm thời gian ghi sinh hiệu** — cho thấy hiệu quả OCR phụ thuộc mạnh vào loại dữ liệu/độ phức tạp bố cục, đây là khoảng trống cần cải thiện.

#### 3. Extracting laboratory test information from paper-based reports
- **Tác giả chính:** Ma MW, et al. | **Năm:** 2023 | **Tạp chí:** BMC Medical Informatics and Decision Making | **PMID:** 37932733 | DOI: 10.1186/s12911-023-02346-6
- **Tóm tắt:** Xây dựng pipeline NLP hai module (OCR + trích xuất thông tin dùng CRF) để số hóa 153 báo cáo xét nghiệm giấy tại Bệnh viện Đại học Bắc Kinh số 1. Độ chính xác OCR trung bình 0.93; F1-score trích xuất thực thể (tên xét nghiệm, kết quả, đơn vị, khoảng tham chiếu) đạt 0.86, thời gian suy luận chỉ 0.78s/báo cáo trên CPU đơn.
- **Khoảng trống nghiên cứu:** Chỉ đánh giá trên dữ liệu một bệnh viện (PKU1) — cần thử nghiệm đa trung tâm để xác nhận khả năng khái quát hóa với các định dạng báo cáo khác.

#### 4. Deep learning-based NLP data pipeline for EHR-scanned document information extraction
- **Tác giả chính:** Hsu E, et al. | **Năm:** 2022 | **Tạp chí:** JAMIA Open | **PMID:** 35702624 | DOI: 10.1093/jamiaopen/ooac045
- **Tóm tắt:** Đánh giá tổ hợp tiền xử lý ảnh + OCR (Tesseract) + NLP (bao gồm ClinicalBERT) trên 955 báo cáo giấc ngủ (sleep study) scan để trích xuất AHI và SaO2. Mô hình ClinicalBERT đạt AUROC 0.97 (AHI) và 0.95 (SaO2), độ chính xác tài liệu >91%.
- **Khoảng trống nghiên cứu:** Tác giả nêu rõ: **"tài liệu scan sẽ vẫn là một phần của y tế trong nhiều năm tới, việc phát triển hệ thống NLP để trích xuất thông tin then chốt là rất cấp thiết"** — đồng thời thừa nhận không thể thử hết mọi tổ hợp tiền xử lý/mô hình do giới hạn nguồn lực.

#### 5. Improving tabular data extraction in scanned laboratory reports using deep learning models
- **Tác giả chính:** Li Y, et al. | **Năm:** 2024 | **Tạp chí:** Journal of Biomedical Informatics | **PMID:** 39393477 | DOI: 10.1016/j.jbi.2024.104735
- **Tóm tắt:** Phát triển pipeline OCR chuyên biệt cho bảng biểu (table detection bằng DETR R18/YOLOv8s, table recognition bằng PaddleOCR/EDD) trên 650 bảng từ 632 báo cáo xét nghiệm scan/fax. DETR R18 đạt AP50=0.774; EDD đạt điểm TEDS 0.815; pipeline tổng thể đạt TEDS 0.699.
- **Khoảng trống nghiên cứu:** Điểm TEDS tổng thể (0.699) còn khiêm tốn so với từng module riêng lẻ — cho thấy khoảng trống ở bước kết hợp end-to-end giữa phát hiện bảng và nhận dạng cấu trúc bảng phức tạp.

#### 6. Automating the segmentation, date extraction, and classification of multi-report PDFs in outside medical records using optical character recognition and generative artificial intelligence
- **Tác giả chính:** Damani S, et al. (Mayo Clinic) | **Năm:** 2026 | **Tạp chí:** JAMIA Open | **PMID:** 41924015 | DOI: 10.1093/jamiaopen/ooag027
- **Tóm tắt:** Kết hợp OCR với Gemini 1.5 (LLM) để phân đoạn, phân loại và trích xuất ngày tháng từ 1303 hồ sơ y tế ngoài viện (multi-report PDF) từ 116 cơ sở khác nhau, tập trung vào ung thư vú. Đạt F1 0.95 (phân đoạn), 0.96 (phân loại), 0.90 (trích xuất ngày); giảm 40% thời gian xem xét hồ sơ theo đánh giá của bác sĩ lâm sàng.
- **Khoảng trống nghiên cứu:** Tác giả nêu **RÕ RÀNG trong phần Future Work**: "Future work will focus on evaluating the system across additional specialties and institutions" — nghiên cứu hiện chỉ giới hạn ở một chuyên khoa (ung thư vú) và một hệ thống bệnh viện.

#### 7. Evaluating Open and Accessible Visual Language Models for Optical Character Recognition in Clinical Case Report Forms
- **Tác giả chính:** Nicora G, et al. | **Năm:** 2026 | **Tạp chí:** Studies in Health Technology and Informatics | **PMID:** 42174982 | DOI: 10.3233/SHTI260308
- **Tóm tắt:** So sánh 3 mô hình Visual Language Model (VLM) mã nguồn mở (Qwen 2.5, Mistral Small 3.1, Granite 3.2 Vision) trong nhận dạng biểu mẫu báo cáo ca bệnh (Case Report Form) giấy từ một thử nghiệm đột quỵ tại Ý, trên 80 ảnh chụp bằng smartphone. Qwen tốt nhất ở nhận dạng tiêu đề (91%) và ngày viết tay (75%); Mistral tốt hơn ở Record ID viết tay (53%) và checkbox (80%).
- **Khoảng trống nghiên cứu:** Tác giả nhấn mạnh **"handwritten fields remained particularly challenging"** (các trường viết tay vẫn là thách thức lớn) và kết luận nghiên cứu "highlight both the promise and current limitations of open VLMs" — đây là khoảng trống nghiên cứu then chốt cho nhận dạng chữ viết tay lâm sàng.

#### 8. Streamlining data recording through optical character recognition: a prospective multi-center study in intensive care units
- **Tác giả chính:** Nitayavardhana P, et al. | **Năm:** 2025 | **Tạp chí:** Critical Care | **PMID:** 40102894 | DOI: 10.1186/s13054-025-05347-1
- **Tóm tắt:** Nghiên cứu tiến cứu đa trung tâm (3 quốc gia) tại các đơn vị ICU, dùng OCR để nhập dữ liệu từ ảnh chụp thiết bị y tế (thở máy, ECMO). Độ đầy đủ dữ liệu 98.5%, độ chính xác 96.9%, giảm 43.9% thời gian nhập liệu (3.4 phút so với 6.0 phút/bệnh nhân), mức hài lòng người dùng cao (4.25/5).
- **Khoảng trống nghiên cứu:** Tác giả kêu gọi rõ: **"Wider uptake of these systems should be encouraged to better understand their strengths and limitations in both clinical and research settings"** — cần thêm nghiên cứu về triển khai rộng rãi để hiểu giới hạn thực tế.

#### 9. Vendor-Agnostic Multisite Automated Dual-Energy X-Ray Absorptiometry Reporting Using Artificial Intelligence-Based Optical Character Recognition: Impact on Workflow Efficiency and Accuracy
- **Tác giả chính:** Lakhani P, et al. | **Năm:** 2026 | **Tạp chí:** Journal of the American College of Radiology (JACR) | **PMID:** 42595274 | DOI: 10.1016/j.jacr.2026.08.007
- **Tóm tắt:** Triển khai hệ thống OCR-AI để tự động soạn báo cáo đo mật độ xương DXA tại 4 cơ sở (2 học thuật, 2 cộng đồng). Thời gian tạo báo cáo giảm từ 3.48 xuống 0.87 phút (học thuật); thời gian trả kết quả (TAT) giảm ~4 ngày tại cơ sở cộng đồng; độ chính xác số liệu ≥99.5%, tăng độ đầy đủ báo cáo tại cơ sở cộng đồng (100% so với 45% trước đây).
- **Khoảng trống nghiên cứu:** Không nêu rõ hạn chế trong tóm tắt, nhưng thiết kế nghiên cứu chỉ áp dụng cho một loại báo cáo (DXA) — cần mở rộng sang các loại hình ảnh chẩn đoán khác.

#### 10. Ocular Biometry OCR: a machine learning algorithm leveraging optical character recognition to extract intra ocular lens biometry measurements
- **Tác giả chính:** Salvi A, et al. (Stanford) | **Năm:** 2025 | **Tạp chí:** Frontiers in Artificial Intelligence | **PMID:** 39834877 | DOI: 10.3389/frai.2024.1428716
- **Tóm tắt:** So sánh PaddleOCR và Gemini để trích xuất chỉ số đo sinh trắc học mắt (trục nhãn cầu, độ dày thủy tinh thể...) từ gần 6700 báo cáo (Lenstar, IOL Master 500/700). Điểm đồng thuận (agreement score) đạt 0.985–0.999; PaddleOCR-to-Annotator đạt độ chính xác gần tuyệt đối (0.999).
- **Khoảng trống nghiên cứu:** Tác giả tự thừa nhận **"in the absence of ground truth"** đối với 2/3 thiết bị (chỉ IOLM 500 có ground truth từ annotator) — hạn chế về khả năng kiểm chứng độc lập, cần bộ dữ liệu chuẩn (benchmark) có nhãn đầy đủ hơn trong tương lai.

#### 11. Optical Character Recognition-based Biometry Scanner for Easy and Efficient Biometry Calculations
- **Tác giả chính:** Levy I, et al. | **Năm:** 2026 | **Tạp chí:** Journal of Refractive Surgery | **PMID:** 42715024 | DOI: 10.3928/1081597X-20260701-02
- **Tóm tắt:** Phát triển và kiểm định ứng dụng di động dùng OCR (Google Gemini 2.5) để tự động nhập dữ liệu sinh trắc học vào công cụ tính công suất thủy tinh thể nhân tạo (ESCRS IOL calculator). Khảo sát 12-21 bác sĩ nhãn khoa cho điểm hài lòng cao (4.5-4.67/5); giảm 48.9% thời gian nhập liệu so với thủ công.
- **Khoảng trống nghiên cứu:** Cỡ mẫu người dùng đánh giá còn nhỏ (12 bác sĩ hoàn thành khảo sát) — cần nghiên cứu đa trung tâm quy mô lớn hơn để khẳng định độ tin cậy lâm sàng.

#### 12. Digitising health history: The creation, function and implementation of the Norwegian Health Archives Registry
- **Tác giả chính:** Helstad G, et al. | **Năm:** 2025 | **Tạp chí:** Health Information Management (Health Inf Manag) | **PMID:** 41204650 | DOI: 10.1177/18333583251389095
- **Tóm tắt:** Mô tả sáng kiến quốc gia của Na Uy số hóa 1.7 triệu hồ sơ bệnh án giấy (từ năm 1875) bằng công cụ OCR tùy chỉnh cho thuật ngữ y khoa tiếng Na Uy, kết hợp hệ thống AI ẩn danh hóa thông tin cá nhân tự động.
- **Khoảng trống nghiên cứu:** Tác giả nêu rõ: **"challenges persist in processing handwritten and historical PHRs due to OCR limitations and language-specific complexities. Key challenges include improving data quality, enhancing OCR accuracy"** — đây là bằng chứng mạnh cho khoảng trống về OCR chữ viết tay lịch sử và đặc thù ngôn ngữ ít phổ biến (non-English).

#### 13. Developing a surveillance system for HIV pre-exposure prophylaxis (PrEP) use in pregnancy in Malawi
- **Tác giả chính:** Kamanga F, et al. | **Năm:** 2026 | **Tạp chí:** BMC Pregnancy and Childbirth | **PMID:** 42098649 | DOI: 10.1186/s12884-026-09190-2
- **Tóm tắt:** Xây dựng hệ thống đăng ký giám sát an toàn thuốc PrEP ở phụ nữ mang thai tại Malawi, dùng công cụ "ScanForm" — ứng dụng OCR hỗ trợ AI để số hóa và phân tích dữ liệu viết tay từ 30+ cơ sở y tế, tích hợp với các chỉ số dược cảnh giác (pharmacovigilance) của WHO.
- **Khoảng trống nghiên cứu:** Không đề cập giới hạn kỹ thuật OCR cụ thể, nhưng đây là ví dụ điển hình cho **khoảng trống ứng dụng OCR ở các nước thu nhập thấp/trung bình (LMIC)** — nơi hạ tầng số hóa còn hạn chế và phụ thuộc nhiều vào biểu mẫu giấy viết tay.

#### 14. Preserving medical information from doctor's prescription ensuring relation among the terminology
- **Tác giả chính:** Datta A, et al. | **Năm:** 2025 | **Tạp chí:** Computers in Biology and Medicine | **PMID:** 39983357 | DOI: 10.1016/j.compbiomed.2025.109812
- **Tóm tắt:** Đề xuất hệ thống EHR tích hợp YOLO (phát hiện vùng quan tâm - ROI, độ chính xác 99.6%) và OCR để số hóa đơn thuốc viết tay/in, kết hợp thuật toán sửa lỗi chính tả tên thuốc (độ chính xác 96%), liên kết tên thuốc với liều lượng và thông tin nhà sản xuất.
- **Khoảng trống nghiên cứu:** Tác giả chỉ ra hạn chế của các phương pháp hiện có: **"struggle with multilingual text"** — cho thấy khoảng trống về xử lý đa ngôn ngữ trong đơn thuốc, dù giải pháp đề xuất cũng chưa được kiểm chứng đa ngôn ngữ đầy đủ.

#### 15. Digitizing audiograms with deep learning: structured data extraction and pseudonymization for hearing big data
- **Tác giả chính:** You S, et al. | **Năm:** 2025 | **Tạp chí:** Hearing Research | **PMID:** 40532492 | DOI: 10.1016/j.heares.2025.109337
- **Tóm tắt:** Xây dựng hệ thống CNN kết hợp OCR để số hóa biểu đồ thính lực đồ (audiogram) thành dữ liệu có cấu trúc, đồng thời ẩn danh hóa thông tin bệnh nhân. Độ chính xác 95-98%, tốc độ xử lý nhanh hơn 17.72 lần so với số hóa thủ công (3.57s so với 63.27s/biểu đồ).
- **Khoảng trống nghiên cứu:** Không nêu future work minh thị, nhưng mô hình chỉ huấn luyện/kiểm tra trên bộ ký hiệu thính lực đồ từ một hệ thống — hàm ý cần đánh giá khả năng khái quát hóa với định dạng audiogram khác nhau giữa các thiết bị/quốc gia.

---

### PHÂN TÍCH TỔNG HỢP: CÁC KHOẢNG TRỐNG NGHIÊN CỨU (RESEARCH GAPS) NỔI BẬT

Từ 15 bài báo trên, có thể rút ra một số xu hướng và khoảng trống nghiên cứu quan trọng cho việc định hướng đề tài tiếp theo:

1. **Nhận dạng chữ viết tay (handwriting recognition) vẫn là điểm yếu lớn nhất.** Cả bài #7 (VLM cho case report form) và #12 (Kho lưu trữ Na Uy) đều nêu rõ ràng rằng các trường viết tay là thách thức dai dẳng, ngay cả với các mô hình VLM/AI hiện đại nhất. Đây có thể là khoảng trống nghiên cứu ưu tiên nhất nếu nhóm muốn có đóng góp mới.

2. **Thiếu tính khái quát hóa đa trung tâm/đa chuyên khoa.** Nhiều nghiên cứu (bài #1, #3, #6, #15) chỉ được kiểm chứng tại một cơ sở/một chuyên khoa duy nhất, và một số tác giả (đặc biệt bài #6 - Mayo Clinic) minh thị đề xuất mở rộng thử nghiệm sang nhiều chuyên khoa/tổ chức khác trong "future work".

3. **Thiếu bộ dữ liệu chuẩn có ground truth đầy đủ.** Bài #10 (Ocular Biometry OCR) minh họa rõ vấn đề khi phải đánh giá độ tin cậy liên mô hình (inter-model agreement) thay vì so với nhãn chuẩn thực sự, do thiếu ground truth annotate đầy đủ.

4. **Khoảng trống về đa ngôn ngữ và ngôn ngữ ít phổ biến (low-resource languages).** Bài #12 (tiếng Na Uy) và #14 (đa ngôn ngữ trong đơn thuốc) đều chỉ ra OCR y tế hiện tại chủ yếu được tối ưu cho tiếng Anh; các ngôn ngữ khác đòi hỏi tùy biến riêng và có độ chính xác thấp hơn.

5. **Hiệu quả OCR không đồng đều theo loại dữ liệu.** Bài #2 cho thấy một nghịch lý thú vị: OCR giúp ích rõ rệt cho việc ghi đơn thuốc (text dài) nhưng KHÔNG cải thiện đáng kể việc ghi sinh hiệu (số liệu ngắn, có thể do sai số làm tròn hoặc bố cục màn hình theo dõi phức tạp) — gợi ý cần nghiên cứu sâu hơn về loại dữ liệu nào OCR mang lại giá trị thực sự.

6. **Xu hướng chuyển dịch từ OCR truyền thống sang kết hợp với Generative AI/LLM/VLM** (bài #6, #7, và một phần bài #10) — đây là hướng nghiên cứu mới nổi (2025-2026) thay thế cho pipeline OCR + NLP cổ điển, nhưng các mô hình VLM mã nguồn mở vẫn còn hạn chế về tuân thủ định dạng đầu ra (structured output formatting) như bài #7 đã chỉ ra với mô hình Granite.

7. **Ứng dụng tại các nước thu nhập thấp/trung bình (LMIC) còn ít được nghiên cứu** — bài #13 (Malawi) là một trong số ít ví dụ, cho thấy khoảng trống lớn về việc đánh giá độ tin cậy và khả năng mở rộng của OCR y tế trong bối cảnh hạ tầng số hạn chế.

---

**Nguồn dữ liệu:** PubMed/PubMed Central (NCBI), truy xuất qua công cụ tìm kiếm PubMed ngày 15/09/2026. Tất cả DOI đã được liệt kê kèm theo từng bài để tiện tra cứu và trích dẫn.
---

## 3.4. Bằng chứng tổng hợp từ Consensus

## Bằng chứng khoa học: OCR/AI trong số hóa hồ sơ y tế lâm sàng

### 1. Độ chính xác của hệ thống OCR/AI khi số hóa hồ sơ y tế

Một nghiên cứu đa trung tâm tiến cứu tại các đơn vị hồi sức tích cực (ICU) ở 3 quốc gia cho thấy hệ thống OCR (được huấn luyện trên 868 ảnh, kiểm định trên 469 ảnh) đạt độ hoàn chỉnh dữ liệu 98,5% và độ chính xác dữ liệu 96,9%, đồng thời giảm 43,9% thời gian nhập liệu so với nhập tay thủ công [1]. Một phân tích tổng hợp (meta-analysis) về ứng dụng AI-OCR trong khâu văn thư tiền phân tích của phòng xét nghiệm lâm sàng ghi nhận: độ chính xác OCR khi chưa huấn luyện (kể cả chữ viết tay) là 91,08%, tăng lên 98,4% sau khi huấn luyện với cơ sở dữ liệu quan hệ/phân cấp, và có thể đạt 99,9–100% khi huấn luyện thêm về trường biểu mẫu; khi kết hợp AI-OCR với bước xác minh của con người, độ chính xác tổng thể đạt 99,99% [2]. Với đơn thuốc viết tay, một hệ thống EHR tích hợp YOLO (phát hiện vùng quan tâm, độ chính xác 99,6%) kết hợp OCR và thuật toán sửa lỗi chính tả (độ chính xác 96%) cho phép số hóa và cấu trúc hóa thông tin đơn thuốc một cách toàn diện [3].

### 2. Tỷ lệ lỗi của OCR trên hồ sơ bệnh án, đơn thuốc, báo cáo bệnh lý

Một nghiên cứu mô phỏng trong bệnh viện tại Nhật Bản so sánh trực tiếp OCR với gõ tay thủ công trên dữ liệu sinh hiệu và đơn thuốc: tỷ lệ lỗi ký tự của OCR là 0% (0/1056) với dữ liệu sinh hiệu so với 1,32% (14/1056) khi gõ tay, và 0,62% (30/4814) với đơn thuốc so với 1,10% (53/4814) khi gõ tay (p<0,001 cho sinh hiệu) — OCR không chỉ nhanh hơn đáng kể khi ghi đơn thuốc (18 giây so với 144 giây cho đơn 6 loại thuốc) mà còn có tỷ lệ lỗi thấp hơn nhập tay [4]. Nghiên cứu về AI-OCR trong phòng xét nghiệm cũng chỉ ra rằng lỗi văn thư giai đoạn tiền phân tích (bao gồm nhập đơn giấy) chiếm tới 65,09% tổng số lỗi xét nghiệm — cho thấy dư địa cải thiện lớn nếu ứng dụng AI-OCR thay thế nhập tay [2]. Nghiên cứu tại ICU nói trên cũng ghi nhận khoảng dao động độ chính xác dữ liệu OCR từ 95,3% đến 100% tùy trung tâm, phản ánh sự biến thiên theo chất lượng ảnh chụp/thiết bị nguồn [1].

### 3. Ứng dụng OCR/document understanding (AI/LLM) trong ung thư học và đăng ký ung thư

Trong lĩnh vực ung thư học, một khung phần mềm mã nguồn mở (LLM-AIx) sử dụng LLM để trích xuất thực thể lâm sàng có cấu trúc (ví dụ giai đoạn TNM) từ 100 báo cáo giải phẫu bệnh (pathology reports) thuộc bộ dữ liệu TCGA, chạy trên hạ tầng bệnh viện nội bộ để bảo vệ quyền riêng tư dữ liệu bệnh nhân, không cần chuyển dữ liệu ra ngoài [5]. Một nghiên cứu khác xây dựng pipeline kết hợp OCR (chuyển báo cáo giải phẫu bệnh dạng ảnh/PDF quét sang văn bản) với LLM để trích xuất chẩn đoán ung thư, mô bệnh học, độ mô học và giai đoạn từ 829 báo cáo giải phẫu bệnh và 569 ghi chú tiến triển bệnh trên 40 loại ung thư (26 khối u đặc, 14 bệnh máu ác tính); kết quả đạt điểm F1 ≥0,85 cho hầu hết các biến số (ví dụ loại ung thư từ báo cáo giải phẫu bệnh: precision 0,86, recall 0,89, F1 0,87) [6]. Gần với chủ đề đăng ký ung thư nhất, một nghiên cứu tại Trung tâm Ung thư Quốc gia Singapore đã triển khai pipeline LLM (GPT-5 nội bộ) tự động trích xuất mã ICD-10-AM, mô bệnh học, vị trí (laterality) và ngày chẩn đoán từ tài liệu lâm sàng không cấu trúc phục vụ báo cáo đăng ký ung thư (cancer registry); trên hai đoàn hệ (760 bệnh nhân, 859 chẩn đoán), độ chính xác tổng thể đạt 84,0–91,5%, với F1 cho các biến riêng lẻ thường trên 0,95, giúp rút ngắn đáng kể thời gian xử lý (136–144 phút/đoàn hệ) so với quy trình thủ công vốn có độ trễ tới 6 tháng — tuy khoảng 23% ca vẫn cần con người rà soát lại (human-in-the-loop) [7].

**Lưu ý về đăng ký ung thư trẻ em (pediatric cancer registry):** Trong phạm vi 3 truy vấn đã thực hiện, tôi **không tìm thấy** nghiên cứu nào áp dụng riêng OCR/LLM cho đăng ký ung thư **trẻ em**. Ba nghiên cứu liên quan đến ung thư học ở trên [5][6][7] đều thực hiện trên dữ liệu người lớn (TCGA, EHR đa loại ung thư, đăng ký ung thư quốc gia Singapore).

---

## TÓM TẮT NGẮN (Tiếng Việt)

- **Độ chính xác OCR/AI trong hồ sơ y tế nhìn chung khá cao**, dao động 95–99%+ tùy bối cảnh: khoảng 96,9% với dữ liệu ICU thu qua ảnh chụp thiết bị [1], có thể đạt 98,4–99,99% khi OCR/AI được huấn luyện chuyên biệt và có bước con người xác minh trong phòng xét nghiệm [2], và 96–99,6% cho các bước phụ trợ (sửa lỗi chính tả thuốc, phát hiện vùng chữ) trong hệ thống số hóa đơn thuốc [3].
- **Tỷ lệ lỗi OCR thường thấp hơn nhập liệu thủ công**: trong một thử nghiệm trực tiếp, OCR có tỷ lệ lỗi ký tự 0–0,62% so với 1,1–1,32% khi gõ tay [4], đồng thời tiết kiệm đáng kể thời gian nhập liệu (giảm 44–92% tùy tác vụ) [1][4].
- **Trong ung thư học**, các hệ thống kết hợp OCR + LLM đã chứng minh khả năng trích xuất tự động chẩn đoán, mô bệnh học, giai đoạn TNM từ báo cáo giải phẫu bệnh và ghi chú lâm sàng với độ chính xác F1 thường ≥0,85, có nghiên cứu đạt tới 91–100% cho một số biến số cụ thể [5][6][7], cho thấy tiềm năng lớn để hỗ trợ đăng ký ung thư (cancer registry) tự động, giảm khối lượng công việc thủ công và độ trễ dữ liệu.
- **Riêng về đăng ký ung thư trẻ em (pediatric cancer registry)**, chưa tìm thấy bằng chứng trực tiếp trong các truy vấn này — đây có thể là khoảng trống nghiên cứu (research gap) đáng lưu ý nếu tổ chức đang cân nhắc ứng dụng công nghệ này cho đăng ký ung thư nhi khoa tại Việt Nam/khu vực.
- Nhìn chung, tất cả các nghiên cứu đều nhấn mạnh: hiệu suất OCR/AI cao nhất khi được **huấn luyện chuyên biệt theo lĩnh vực** và **có bước rà soát/xác minh của con người** (human-in-the-loop) đối với các ca phức tạp hoặc mơ hồ, thay vì triển khai hoàn toàn tự động.

---

### Tài liệu tham khảo

[1] [Streamlining data recording through optical character recognition: a prospective multi-center study in intensive care units](https://consensus.app/papers/details/a65c5bc292395b4a9d2965e9729538e0/?utm_source=claude_code) (P. Nitayavardhana et al., 2025, 10 citations, *Critical Care*)

[2] [A-182 Applications of artificial intelligence optical character recognition in laboratory clerical functions offsetting staffing shortages and error reduction](https://consensus.app/papers/details/a03f77bcc1755c17856038bfdb88e9a4/?utm_source=claude_code) (L. Springer et al., 2024, 0 citations, *Clinical Chemistry*)

[3] [Preserving medical information from doctor's prescription ensuring relation among the terminology](https://consensus.app/papers/details/2d782d05ef945c7e917b7bba51361369/?utm_source=claude_code) (A. Datta et al., 2025, 9 citations, *Computers in Biology and Medicine*)

[4] [Development of novel optical character recognition system to reduce recording time for vital signs and prescriptions: A simulation-based study](https://consensus.app/papers/details/9096b96a0ffc5ce18b3329f0908166c9/?utm_source=claude_code) (S. Soeno et al., 2024, 15 citations, *PLOS ONE*)

[5] [A software pipeline for medical information extraction with large language models, open source and suitable for oncology](https://consensus.app/papers/details/ba10b7953a4155c48f3a42de5043a59b/?utm_source=claude_code) (I. Wiest et al., 2025, 23 citations, *NPJ Precision Oncology*)

[6] [Use of large language models to extract cancer diagnosis, histology, grade, and staging from unstructured electronic health records](https://consensus.app/papers/details/5034d4c85acd593ba198b3f974cb5f8c/?utm_source=claude_code) (Gayathri Namasivayam et al., 2025, 0 citations, *Journal of Clinical Oncology*)

[7] [Automated cancer data extraction using large language models: A scalable workflow for clinical documentation processing](https://consensus.app/papers/details/a3ffc59ecb8e56fcb38b9e1e2b5541e0/?utm_source=claude_code) (See Boon Tay et al., 2026, 0 citations, *Journal of Clinical Oncology*)

Create or connect a free Consensus account to return more than 3 results per search in Claude Code.: https://consensus.app/sign-up/?utm_source=claude_code&auth=claude_code