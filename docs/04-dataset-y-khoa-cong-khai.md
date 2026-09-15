# 4. Bộ Dữ Liệu OCR Y Khoa Công Khai, Uy Tín (Không Tiếng Việt, Không Cần Hợp Tác Bệnh Viện)

*Cập nhật 15/09/2026. Đây là kết quả tra cứu cho hướng nghiên cứu ĐỘC LẬP, dùng dữ liệu công khai sẵn có — không tiếp tục hướng OCR tiếng Việt hay đăng ký ung thư nhi khoa gắn CCI (xem lý do tại [archive/](archive/)).*

Phần này gồm 3 mục: (4.1) khảo sát web rộng liệt kê ~22 bộ dữ liệu ứng viên, (4.2) tra cứu tài liệu khoa học để xác định dataset nào là "chuẩn" và mô hình nào CHƯA được test trên đó (chỗ tạo tính mới), và (4.3) kiểm chứng sâu trực tiếp từng ứng viên mạnh nhất (link tải, license, ground truth, độ uy tín thực tế).

---

## 4.1. Khảo sát rộng các bộ dữ liệu ứng viên

## BỘ DỮ LIỆU OCR/DOCUMENT AI CÔNG KHAI TRONG LĨNH VỰC Y KHOA/LÂM SÀNG
*(Tổng hợp phục vụ kế hoạch nghiên cứu benchmark 2 tuần — không giới hạn bệnh lý cụ thể, không dùng tiếng Việt)*

---

### NHÓM 1 — CHỮ VIẾT TAY Y KHOA / ĐƠN THUỐC (Handwritten Medical/Prescription)

#### 1. RxHandBD — Handwritten Prescription Word Image Dataset
- **Link tải:** [Zenodo](https://zenodo.org/records/18478741) | [Mendeley Data](https://data.mendeley.com/datasets/dsb5r6vskg/3)
- **Quy mô:** 5.578 ảnh từ đơn thuốc viết tay đã cắt (crop) sẵn theo từng từ, chuẩn hoá 128×128px; từ vựng 1.559 từ duy nhất (tên thuốc generic, brand, dạng bào chế, chỉ dẫn). Đã chia sẵn train/test (4.463/1.115) kèm file CSV nhãn.
- **Ground truth:** Có transcription đầy đủ ở cấp độ từ (word-level label CSV) — dùng ngay cho HTR/word recognition, không cần tự gán nhãn.
- **License:** Host trên Zenodo/Mendeley (mặc định CC-BY dạng open dataset, nên kiểm tra lại trang cụ thể trước khi trích dẫn, nhưng về nguyên tắc tự do dùng cho nghiên cứu).
- **Độ uy tín:** Dataset rất mới (2025-2026), có bài báo IEEE liên quan ("A Deep Neural Network Approach with Pioneering Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh", iCACCESS 2024) — còn ít trích dẫn nhưng là nguồn dữ liệu chuyên biệt hiếm cho domain này, tải được ngay không cần xin phép.

#### 2. Doctor's Handwritten Prescription BD dataset (Kaggle)
- **Link:** https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset
- **Quy mô:** 4.680–4.688 ảnh (~20 MB), 78 từ/thuật ngữ thuốc khác nhau, thu thập từ nhiều bệnh viện Bangladesh.
- **Ground truth:** Nhãn từ (word label) sẵn theo tên file/CSV — transcription đầy đủ ở mức từ.
- **License:** Theo license Kaggle mặc định của dataset (thường CC0/Open, cần kiểm tra tab "License" trên trang).
- **Độ uy tín:** Dataset Kaggle phổ biến nhất trong nhóm "handwritten prescription", được dùng trong nhiều notebook/bài báo con (Swin Transformer, CNN-SSD-ResNet...); tải tức thời, chỉ cần tài khoản Kaggle miễn phí.

#### 3. Handwritten Medical Prescriptions Collection (Illegible Medical Prescription Images Dataset)
- **Link:** https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset
- **Quy mô:** Chưa công bố số lượng ảnh chính xác trong metadata tìm được — cần vào trang Kaggle kiểm tra trực tiếp (cập nhật 2024).
- **Ground truth:** Tập trung vào ảnh chữ viết tay khó đọc ("illegible"), dùng cho OCR + NLP để nhận diện tên bệnh/thuốc — có bài báo Springer liên quan ("Disease Identification from Illegible Medical Prescriptions Using OCR and NLP Techniques").
- **License:** Theo điều khoản Kaggle của dataset.
- **Độ uy tín:** Mới, ít trích dẫn nhưng phù hợp làm case-study "worst-case OCR" (chữ khó đọc) — tải ngay.

#### 4. A Curated Bangladesh-Based Dataset of Handwritten and Printed Prescription Images
- **Link:** https://data.mendeley.com/datasets/k62rfd23kz/1
- **Đặc điểm:** Có cả đơn thuốc viết tay VÀ in máy — phù hợp bài toán phân loại handwritten-vs-printed hoặc OCR hỗn hợp.
- **License:** Mendeley Data (thường CC BY 4.0).
- **Độ uy tín:** Dataset ngách, mới, chưa có nhiều trích dẫn nhưng bổ sung tốt cho RxHandBD.

#### 5. MIRAGE — Multimodal Identification and Recognition of Annotations in Indian General Prescriptions
- **Link:** https://arxiv.org/abs/2410.09729 (paper); phần dữ liệu công khai là **subset nhỏ ~100 bản ghi có nhãn** (dataset gốc 743.118 bản ghi thuộc sở hữu công ty Medyug Technology — KHÔNG công khai đầy đủ).
- **Ground truth:** Có annotation multimodal (bounding box + transcription) cho subset công khai.
- **License:** Theo điều khoản đi kèm bài báo arXiv/HuggingFace Papers.
- **Độ uy tín:** Bài báo 2024 mới, dùng fine-tune LLaVA 1.6 và Idefics2 — hữu ích để tham khảo phương pháp nhưng dữ liệu đầy đủ không tải được tự do (chỉ dùng subset demo).
- **Lưu ý:** Vì phần lớn dữ liệu không công khai, khuyến nghị chỉ dùng làm tài liệu tham khảo phương pháp luận, KHÔNG phải nguồn dữ liệu chính cho benchmark 2 tuần.

---

### NHÓM 2 — CHỮ VIẾT TAY TỔNG QUÁT (đối chiếu/pretraining bổ trợ)

#### 6. IAM Handwriting Database
- **Link:** https://fki.tic.heia-fr.ch/databases/iam-handwriting-database (bản mirror tiện lợi: [Teklia/IAM-line trên HuggingFace](https://huggingface.co/datasets/Teklia/IAM-line))
- **Quy mô:** 1.539 trang viết tay từ 657 người viết, 13.353 dòng, 115.320 từ. Kích thước ~vài trăm MB đến ~22GB tùy phiên bản (word/line/form).
- **Ground truth:** Transcription đầy đủ cấp từ + bounding box (XML), chuẩn "vàng" trong HTR benchmark.
- **License:** Miễn phí cho nghiên cứu phi thương mại, cần đăng ký (registration) đơn giản, không cần IRB.
- **Độ uy tín:** RẤT cao — bài báo gốc Marti & Bunke (2002, IJDAR) là benchmark HTR kinh điển nhất, được trích dẫn trong hầu như mọi paper HTR/OCR chữ viết tay tiếng Anh 20+ năm qua (hàng nghìn trích dẫn).

#### 7. CVL Database
- **Link:** https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/
- **Quy mô:** 7 văn bản mẫu (1 tiếng Đức, 6 tiếng Anh), 310 người viết, ảnh RGB 300dpi + bounding box từ (XML).
- **Ground truth:** Transcription đầy đủ ở cấp từ.
- **License:** CC BY-NC 3.0 (chỉ nghiên cứu phi thương mại).
- **Độ uy tín:** Benchmark phụ trợ phổ biến, thường dùng cùng IAM trong các paper writer-identification/HTR.

#### 8. RIMES-2011
- **Link:** [HuggingFace Teklia/RIMES-2011-line](https://huggingface.co/datasets/Teklia/RIMES-2011-line) | trang gốc a2ialab.com (cần email xin user agreement)
- **Quy mô:** 12.723 trang, ~5.605 thư viết tay tiếng Pháp, hơn 1.300 người viết.
- **Ground truth:** Transcription đầy đủ cấp dòng/trang.
- **License:** Cần ký user agreement (miễn phí, gửi email) hoặc dùng bản mirror HuggingFace/Zenodo không cần xin phép.
- **Độ uy tín:** Benchmark chuẩn cho HTR ngôn ngữ Latin ngoài tiếng Anh — dùng phổ biến trong ICDAR 2011 competition, được trích dẫn rộng.

---

### NHÓM 3 — VĂN BẢN BÁO CÁO Y KHOA MÁY ĐỌC ĐƯỢC (nguồn để render giả-scan hoặc OCR)

#### 9. TCGA-Reports
- **Link:** [Mendeley Data (Kefeli et al.)](https://data.mendeley.com/datasets/hyg5xkznpx/1)
- **Quy mô:** 9.523 báo cáo giải phẫu bệnh (pathology reports) từ 32 loại mô/ung thư, đã OCR sẵn qua AWS Textract từ PDF gốc TCGA.
- **Ground truth:** Đây CHÍNH là sản phẩm sau-OCR (text đã làm sạch) — cực kỳ hữu ích: có thể lấy PDF/ảnh scan gốc để tự OCR lại rồi so sánh với bản text "gold" này để tính CER/WER.
- **License:** Mendeley Data, thường CC BY 4.0 (kiểm tra trang cụ thể).
- **Độ uy tín:** Bài báo 2024 trên *Patterns* (Cell Press) — mới nhưng chất lượng cao, chuyên thiết kế để "benchmark AI dựa trên văn bản y khoa"; đúng mục tiêu nghiên cứu OCR của bạn.

#### 10. MIMIC-IV-Note / MIMIC-III discharge summaries (PhysioNet)
- **Link:** https://physionet.org/content/mimic-iv-note/2.2/
- **Quy mô:** 331.794 discharge summary phi định danh từ 145.915 bệnh nhân (MIMIC-IV-Note); MIMIC-III có 52.746 discharge summaries.
- **Ground truth:** Text sạch (không phải ảnh scan) — dùng làm nguồn "ground truth text" để tự tạo ảnh giả-scan (render ra PDF/ảnh rồi làm nhiễu) phục vụ benchmark OCR có kiểm soát.
- **License:** PhysioNet Credentialed Health Data License — **CẦN credentialing**: hoàn thành khóa CITI training (vài giờ) + PhysioNet review nhân sự thường **24–48 giờ** (có thể lâu hơn vài ngày nếu hồ sơ thiếu), sau đó ký DUA riêng cho từng dataset (thường tức thời sau khi đã credentialed). Tổng thời gian thực tế: **thường 3–7 ngày**, đủ nằm trong kế hoạch 2 tuần nếu làm ngay từ Ngày 1.
- **Độ uy tín:** Cực cao trong y tế-AI, hàng nghìn bài báo dùng MIMIC.

#### 11. n2c2 (i2b2) Clinical NLP Shared Task Corpora — đặc biệt Track 1 "2022 n2c2 Contextualized Medication Event" (CMED)
- **Link:** https://n2c2.dbmi.hms.harvard.edu/data-sets → đăng ký qua https://portal.dbmi.hms.harvard.edu/
- **Quy mô (ví dụ Track 1/2022):** 500 EHR notes được annotate, 9.013 mention thuốc, chia train 350/val 50/test 100 notes.
- **Ground truth:** Annotation NER + context (change/no-change, action, negation, temporality...) rất chi tiết.
- **License:** Data Use Agreement (DUA) miễn phí nhưng bắt buộc ký; **CẢNH BÁO QUAN TRỌNG:** một số báo cáo gần đây (giữa 2026) cho biết cổng n2c2 trên DBMI Portal đang hiển thị "Temporarily Unavailable" và đăng ký tạm đóng — cần kiểm tra lại tình trạng truy cập trước khi đưa vào kế hoạch, có rủi ro không kịp trong 2 tuần.
- **Độ uy tín:** Chuẩn vàng NLP lâm sàng (từ 2006), rất nhiều trích dẫn — nhưng đây là text thuần, không phải ảnh scan, chỉ hữu ích nếu bạn tự render thành ảnh để test OCR.

---

### NHÓM 4 — BỘ DỮ LIỆU BIỂU MẪU/HÓA ĐƠN (benchmark kỹ thuật, không phải y khoa nhưng hữu ích để pretraining/baseline)

#### 12. SROIE (ICDAR 2019 Robust Reading Challenge — Scanned Receipts OCR)
- **Link:** https://rrc.cvc.uab.es/?ch=13 | mirror GitHub: https://github.com/zzzDavid/ICDAR-2019-SROIE
- **Quy mô:** 1.000 ảnh hóa đơn scan (600 train + 400 test), ~542MB.
- **Ground truth:** Text localization + OCR transcription + key-info extraction (4 trường: tên hàng, đơn giá, tổng tiền, ngày).
- **License:** CC-BY-4.0 cho annotation redistributed.
- **Độ uy tín:** ~262 trích dẫn (IEEE Xplore) — benchmark OCR+KIE kinh điển, dùng để pretraining pipeline text-detection/OCR trước khi áp dụng sang domain y tế (layout tài liệu bán-cấu trúc tương tự đơn thuốc/hóa đơn viện phí).

#### 13. FUNSD (Form Understanding in Noisy Scanned Documents)
- **Link:** https://guillaumejaume.github.io/FUNSD/
- **Quy mô:** 199 form scan thực (149 train/50 test), 9.707 semantic entity, 31.485 từ.
- **Ground truth:** OCR text + bounding box + entity label + linking (đầy đủ, hình thức annotation dùng làm chuẩn cho LayoutLM và các mô hình document-AI).
- **License:** Không có license thương mại rõ ràng — dùng cho nghiên cứu (theo trang gốc).
- **Độ uy tín:** ~402 trích dẫn (Semantic Scholar) — rất phổ biến, là benchmark chuẩn để test document-layout understanding (kỹ thuật tương tự cần cho form y tế/đơn thuốc có cấu trúc bán-form).

#### 14. CORD (Consolidated Receipt Dataset)
- **Link:** https://github.com/clovaai/cord | HuggingFace: https://huggingface.co/datasets/naver-clova-ix/cord-v2
- **Quy mô:** >11.000 hóa đơn Indonesia, 30 nhãn ngữ nghĩa chi tiết.
- **Ground truth:** OCR box-level + parsing class (multi-level) — đầy đủ nhất trong nhóm receipt dataset.
- **License:** CC BY 4.0.
- **Độ uy tín:** ~294 trích dẫn; là dataset huấn luyện/test chính của mô hình **Donut** (OCR-free Document Understanding) — rất liên quan nếu bạn benchmark Donut/GOT-OCR2.0/Qwen-VL.

#### 15. DocVQA
- **Link:** https://www.docvqa.org/datasets/docvqa (RRC portal, cần đăng nhập miễn phí)
- **Quy mô:** 50.000 câu hỏi trên 12.767 ảnh tài liệu; ~8,3GB.
- **Ground truth:** QA pairs + OCR text (được cung cấp sẵn OCR tokens).
- **License:** Theo điều khoản RRC (dùng nghiên cứu, cần đăng ký tài khoản miễn phí trên trang, không cần IRB).
- **Độ uy tín:** ~1.086 trích dẫn (Semantic Scholar) — RẤT phổ biến, chuẩn benchmark VQA-trên-tài-liệu cho các LMM như GPT-4V/Gemini/Qwen-VL mà bạn định test.

#### 16. PubTables-1M
- **Link:** https://github.com/microsoft/table-transformer (Microsoft Research Open Data)
- **Quy mô:** ~947.642 bảng (table) được annotate đầy đủ từ bài báo khoa học.
- **Ground truth:** Cấu trúc bảng (header, cell, location) chi tiết, giải quyết vấn đề "oversegmentation" của dataset bảng trước đó.
- **License:** Microsoft Research Open Data license (dùng nghiên cứu).
- **Độ uy tín:** ~145 trích dẫn — hữu ích nếu paper của bạn có phần trích xuất bảng dữ liệu xét nghiệm/kết quả lâm sàng dạng bảng trong tài liệu y tế.

---

### NHÓM 5 — CUỘC THI ICDAR/ICFHR LIÊN QUAN

#### 17. ICDAR 2024 Competition on Recognition and VQA on Handwritten Documents (HWD)
- **Link:** https://ilocr.iiit.ac.in/icdar_2024_hwd/
- **Đặc điểm:** 3 task — isolated word recognition, page-level recognition, VQA trên tài liệu viết tay. KHÔNG chuyên về y khoa nhưng có thể dùng testset/leaderboard để đối chiếu SOTA HTR hiện tại.

#### 18. ICDAR 2023 Competition on Indic Handwriting Text Recognition (IHTR)
- **Link:** paper DOI 10.1007/978-3-031-41679-8_25
- **Đặc điểm:** 18 team đăng ký, 8 team nộp kết quả — benchmark chữ viết tay đa ngôn ngữ Ấn Độ, có thể liên hệ gián tiếp tới domain đơn thuốc Ấn Độ (MIRAGE ở trên).
- **Kết luận cho nhóm 5:** Qua tìm kiếm, **KHÔNG tồn tại một track ICDAR/ICFHR chính thức chuyên biệt cho "handwritten medical prescription/clinical document"** tính đến 2026 — đây là điểm trống thị trường (gap) đáng để paper của bạn nêu như motivation ("chưa có benchmark ICDAR chuẩn cho domain y tế, khác với biên lai/hóa đơn đã có SROIE/CORD").

---

### NHÓM 6 — ẢNH Y KHOA KÈM VĂN BẢN/BÁO CÁO CẦN OCR

#### 19. MIMIC-CXR (PhysioNet)
- **Link:** https://physionet.org/content/mimic-cxr/2.1.0/
- **Quy mô:** 377.110 ảnh X-quang ngực, 227.835 lượt khám của 65.379 bệnh nhân, kèm báo cáo free-text bán cấu trúc.
- **Ground truth:** Báo cáo là text gốc (không phải ảnh scan) — nhưng bạn có thể tự render báo cáo thành ảnh/PDF để tạo synthetic OCR benchmark, hoặc dùng ảnh X-quang có nhãn kèm để test pipeline multimodal.
- **License:** PhysioNet Credentialed Health Data License (cùng quy trình credentialing như MIMIC-IV, ~1 tuần).
- **Độ uy tín:** ~1.594 trích dẫn — cực kỳ uy tín, gần như "chuẩn vàng" cho nghiên cứu ảnh+báo cáo X-quang.

#### 20. IU X-Ray / OpenI (Indiana University Chest X-rays)
- **Link:** [Kaggle mirror](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) | gốc: https://openi.nlm.nih.gov/
- **Quy mô:** 7.470 ảnh X-quang ngực + 3.955 báo cáo (Comparison/Indication/Findings/Impression).
- **License:** CC BY-NC-ND 4.0 — **tải được ngay, KHÔNG cần credentialing** (khác biệt lớn so với MIMIC), phù hợp nếu deadline gấp.
- **Độ uy tín:** Dataset kinh điển trong report-generation (dùng trong hàng trăm paper "chest X-ray report generation").

#### 21. PadChest
- **Link:** http://bimcv.cipf.es/bimcv-projects/padchest/ (cần điền form xin quyền truy cập — không phải IRB, chỉ là form đăng ký nghiên cứu đơn giản)
- **Quy mô:** >160.000 ảnh X-quang, 67.000 bệnh nhân, báo cáo bằng **tiếng Tây Ban Nha** (lưu ý: có thể không phù hợp nếu bạn chỉ muốn OCR tiếng Anh, nhưng vẫn hữu ích để test đa ngôn ngữ ngoài tiếng Việt).
- **License:** Theo form xin quyền của nhóm nghiên cứu gốc (miễn phí, phi thương mại).
- **Độ uy tín:** Một trong các dataset X-quang lớn nhất kèm báo cáo gốc, được trích dẫn rất nhiều.

#### 22. ROCOv2 (Radiology Objects in COntext v2)
- **Link:** https://zenodo.org/records/8333645 | HuggingFace: https://huggingface.co/datasets/eltorio/ROCOv2-radiology
- **Quy mô:** 79.789 ảnh X-quang/CT/MRI kèm caption + concept y khoa, trích từ PubMed Open Access.
- **Ground truth:** Caption/mô tả (không phải scan văn bản, nhưng là text-image pair chuẩn để test image-to-text/report generation).
- **License:** Cần ký user agreement đơn giản trên Zenodo (không cần IRB).
- **Độ uy tín:** Dùng trong ImageCLEFmedical Caption 2023 — benchmark chuẩn quốc tế.

---

### GHI CHÚ CHIẾN LƯỢC CHO KẾ HOẠCH 2 TUẦN

1. **Nhóm ưu tiên tải ngay (không cần credentialing, <1 ngày):** RxHandBD, Doctor's Handwritten Prescription BD (Kaggle), IAM, CVL, RIMES (mirror HF), TCGA-Reports, SROIE, FUNSD, CORD, DocVQA, PubTables-1M, IU X-Ray/OpenI, ROCOv2 — đủ để chạy benchmark ngay từ Ngày 1-2.
2. **Nhóm cần credentialing (bắt đầu xin ngay Ngày 1, chờ song song 3-7 ngày):** MIMIC-CXR, MIMIC-IV-Note/MIMIC-III (PhysioNet) — vẫn nằm trong 2 tuần nếu nộp hồ sơ đầu tiên.
3. **Nhóm rủi ro/không chắc kịp:** n2c2/i2b2 (cổng DBMI có báo cáo tạm đóng giữa 2026 — nên có kế hoạch B), MIRAGE (dữ liệu đầy đủ không công khai — chỉ dùng subset demo hoặc bỏ qua).
4. **Khoảng trống nghiên cứu đáng nêu trong paper:** không có track ICDAR/ICFHR chính thức cho "handwritten clinical/prescription document" — luận điểm tốt để định vị đóng góp benchmark của bạn.

---

**Nguồn tham khảo chính** (đã trích trong quá trình tìm kiếm): Zenodo, Mendeley Data, Kaggle, PhysioNet/MIMIC, n2c2 DBMI Portal, arXiv (SROIE 2103.10213, FUNSD 1905.13538, CORD, DocVQA 2007.00398, PubTables-1M 2110.00061, ROCOv2 2405.10004, MIRAGE 2410.09729, PadChest 1901.07441, MIMIC-CXR Scientific Data 2019), GitHub (clovaai/cord, microsoft/table-transformer, zzzDavid/ICDAR-2019-SROIE), Semantic Scholar (số liệu trích dẫn).
---

## 4.2. Dataset "chuẩn", SOTA hiện tại, và khoảng trống để tạo tính mới

## KẾT QUẢ NGHIÊN CỨU: Bộ dữ liệu OCR/HTR y khoa công khai — chuẩn vàng, SOTA hiện tại, và khoảng trống cho benchmark 2024-2026

### 1. Xác định dataset "chuẩn" (gold standard)

#### 1.1. IAM Handwriting Database — chuẩn vàng của HTR nói chung (KHÔNG phải dataset y khoa)
- **Nguồn gốc**: Marti & Bunke, "The IAM-database: an English sentence database for offline handwriting recognition", *International Journal on Document Analysis and Recognition (IJDAR)*, vol. 5, tr. 39–46, 2002 (DOI: 10.1007/s100320200071). Phân phối qua Đại học Bern (FKI).
- **Độ uy tín**: ~569 lượt trích dẫn trên Google Scholar (đã tồn tại >20 năm) — đây là benchmark tổng quát lâu đời và được trích dẫn nhiều nhất trong lĩnh vực HTR (Marti & Bunke, 2002; kiểm chứng qua nhiều bài báo 2024-2026 vẫn dẫn nó làm baseline).
- **Lưu ý quan trọng**: IAM **không phải** dataset y khoa — nó là 1.066 trang chữ viết tay tiếng Anh tổng quát (dựa trên corpus LOB) của ~400 người viết, 82.227 từ. Nó được dùng làm "sân chơi chung" để đo năng lực HTR nền tảng, và các bài báo OCR y khoa thường dẫn số liệu IAM để so sánh mức độ tổng quát hoá của mô hình, KHÔNG phải vì nó chứa nội dung y khoa. → Nên đưa IAM vào nghiên cứu như **benchmark đối chứng tổng quát** (không phải trọng tâm y khoa) để so sánh với dataset y khoa thật.

#### 1.2. RxHandBD / "Doctor's Handwritten Prescription BD dataset" — chuẩn "y khoa cầm tay" gần nhất hiện có
Đây thực chất là một **họ dataset** đơn thuốc viết tay tiếng Anh/Bangla của bác sĩ Bangladesh, có 2 phiên bản liên quan:
- **Bản gốc**: "Doctor's Handwritten Prescription BD dataset" (Kaggle, Mia, Mamun, Sajid & Ruddra, 2024) — 4.680 ảnh, 78 lớp tên thuốc. Giới thiệu lần đầu trong: *"A Deep Neural Network Approach with Pioneering Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh"*, IEEE, 2024 (ieeexplore.ieee.org/document/10499631) — dùng VGG16, đây là bài báo gốc "khai sinh" dataset này ở Bangladesh (97,1% bác sĩ Bangladesh viết tay đơn thuốc).
- **Bản mở rộng/tái đóng gói**: **RxHandBD** (Md. Masudul Islam, Zenodo, DOI 10.5281/zenodo.18478741, công bố 04/02/2026; cũng có trên Mendeley Data) — 5.578 ảnh từ, 1.559 nhãn văn bản duy nhất, chia sẵn train/test 80/20, license MIT. Trang Zenodo không nêu rõ liên kết trực tiếp với bản Kaggle 2024, nhưng về bản chất mô tả (đơn thuốc viết tay Bangladesh) là cùng dòng nghiên cứu.
- **Độ uy tín**: chưa có "bài báo gốc" bình duyệt riêng cho RxHandBD (mới đăng 02/2026, gần như chưa có trích dẫn), nhưng dòng dữ liệu này đã được dùng trong ≥8 công trình 2019-2026 (xem mục 2.2), khiến nó là dataset **chuyên biệt y khoa** được dùng lại nhiều nhất hiện có công khai, dù quy mô nhỏ và chưa "kinh điển" như IAM.

#### 1.3. TCGA-Reports — chuẩn cho NLP lâm sàng, KHÔNG phải chuẩn để đo độ chính xác OCR
- **Nguồn**: Kefeli et al., *"TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models"*, *Patterns* (Cell Press), 21/02/2024, DOI: 10.1016/j.patter.2024.100933; PMID 38487800; PMC10935496.
- **Nội dung**: 9.523 báo cáo giải phẫu bệnh (pathology reports) từ TCGA, 32 loại mô/ung thư, được OCR qua AWS Textract + xử lý hậu kỳ, dùng để benchmark phân loại loại ung thư bằng NLP/LLM (đạt AU-ROC 0,992).
- **Điểm mấu chốt cần lưu ý cho nghiên cứu của bạn**: TCGA-Reports **không cung cấp cặp ảnh+nhãn ground-truth ở mức từ/ký tự** để tính CER/WER — OCR ở đây chỉ là bước tiền xử lý để tạo văn bản cho bài toán NLP hạ nguồn (phân loại). Nếu mục tiêu là "benchmark độ chính xác OCR" (CER/WER) thì TCGA-Reports **không phù hợp** làm dataset chính; nó phù hợp hơn cho một nghiên cứu về "OCR làm tiền xử lý cho NLP lâm sàng", khác hướng với RxHandBD/IAM.

---

### 2. Mô hình/phương pháp đã được thử nghiệm và kết quả (SOTA hiện tại)

#### 2.1. Trên IAM (SOTA tổng quát, không riêng y khoa)
| Mô hình | Loại | CER (IAM) | Nguồn |
|---|---|---|---|
| GPT-5 | VLM API | ~1,22% | CodeSOTA benchmark registry, 2026 (codesota.com/benchmark/iam) |
| Claude Opus 4.7 | VLM API | ~1,31% | như trên |
| Gemini 3 | VLM API | ~1,44% | như trên |
| GPT-4o | VLM API | 1,69% (03/2025) — "thời điểm VLM vượt mô hình HTR chuyên biệt" | như trên |
| Azure Document Intelligence v4.0 | Dịch vụ thương mại | ~1,8% | như trên |
| DTrOCR | Mô hình HTR chuyên biệt (không-VLM) | 2,38% (WACV 2024) | như trên |
| **TrOCR-Large** | Transformer pretrained | **2,89%** | Li et al., *"TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models"*, AAAI 2023 / arXiv:2109.10282 |
| CNN-BiLSTM+CTC | CNN-RNN | 3,59% CER / 9,44% WER | arXiv:2307.00664, "CNN-BiLSTM model for English Handwriting Recognition" |
| **GOT-OCR2.0** | VLM 580M | *(số liệu cụ thể không công bố rõ trong phần tóm tắt được truy xuất)* — **quan trọng: GOT-OCR2.0 dùng chính IAM làm dữ liệu HUẤN LUYỆN** (cùng với CASIA-HWDB2 tiếng Trung, NorHand-v3 tiếng Na Uy) | Wei et al., *"General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model"*, arXiv:2409.01704 |
| Donut | OCR-free VDU | Không tìm thấy số liệu IAM — Donut được đánh giá chủ yếu trên CORD, SROIE (không phải HTR viết tay dòng) | Kim et al., ECCV 2022, arXiv:2111.15664 |

**Lưu ý quan trọng**: GOT-OCR2.0 đã **huấn luyện trực tiếp trên IAM** — nên nếu bạn dự định "test GOT-OCR2.0 trên IAM" như một điểm mới, cần làm rõ đây là đánh giá trên tập test giữ lại (held-out) chứ không phải mô hình hoàn toàn chưa từng thấy IAM — ảnh hưởng đến cách trình bày tính mới trong bài báo.

#### 2.2. Trên RxHandBD / dòng dữ liệu đơn thuốc Bangladesh (SOTA y khoa cụ thể)
| Nghiên cứu | Mô hình | Kết quả | Nguồn |
|---|---|---|---|
| Bài gốc dataset (2024) | VGG16/CNN | Accuracy trên "Doctor's Handwritten Prescription BD" (4.680 ảnh, 78 lớp) | IEEE 10499631, 2024 |
| CRNN cơ bản | CRNN (13 conv + 3 BiLSTM) | 72% accuracy | ResearchGate 336419684, "Medical Handwritten Prescription Recognition Using CRNN" |
| CRNN (dataset khác) | CRNN | 95% accuracy | tìm qua WebSearch, chưa xác minh DOI cụ thể |
| BiLSTM + tăng cường dữ liệu SRP | BiLSTM | 93,0% accuracy trung bình (max 94,5%, min 92,1%) | Scientific Reports, *"An online cursive handwritten medical words recognition system..."*, Nature, 2022, PMC8897401 |
| Mô hình attention (F1) | CNN+Attention | 89,0% test accuracy, F1 macro/weighted = 0,88 | arXiv:2412.18199, *"Leveraging Deep Learning with Multi-Head Attention for Accurate Extraction of Medicine from Handwritten Prescriptions"*, 12/2024 |
| Multi-backbone feature-concat | Ensemble CNN | 88,44% ± 0,99% accuracy, F1 = 0,88 ± 0,01 | *Advances in Technology Innovation*, ojs.imeti.org/AITI/16426 |
| CNN Bi-LSTM + Lexicon Search | CNN-BiLSTM + hậu xử lý từ điển | (kết quả cụ thể chưa trích xuất được) | ResearchGate 375414751, 2024 |
| RxVLM | VLM contrastive learning | (đề xuất mô hình mới, không phải benchmark off-the-shelf) | *The Visual Computer* (Springer), 2026, DOI 10.1007/s00371-026-04420-2 |
| **Benchmark trực tiếp trên RxHandBD (5.578 ảnh)** — **gần nhất với ý tưởng của bạn** | Tesseract, EasyOCR, PP-OCRv5, **GLM-OCR** (VLM 0,9B) | Tesseract: CER 0,785/Exact Match 2,5%; EasyOCR: CER 0,695/2,6%; PP-OCRv5: CER 0,477/WER 0,789/**21,4%** exact match; **GLM-OCR: CER 0,328 (thấp nhất)/WER 0,801/32,6% exact match (cao nhất)** | Bài blog kỹ thuật (KHÔNG bình duyệt): *"How Well Can OCR Read Doctor Handwriting in 2026?"*, DEV Community, 2026, dev.to/kaniel_outis/how-well-can-ocr-read-doctor-handwriting-in-2026-54hn |

**Kết luận chính từ benchmark trên**: "ngay cả mô hình tốt nhất (GLM-OCR) cũng chỉ đúng chính xác 1/3 số từ" → chưa đủ tin cậy để tự động hoá hoàn toàn, cần con người rà soát (theo bài blog trên).

#### 2.3. Công trình liên quan gần nhất (rất mới, cần biết để tránh trùng lặp)
- **RxScribe Bench** (arXiv:2609.13280, nộp 08/09/2026) — benchmark đa trục (correctness/hallucination/engagement/robustness) cho VLM trên **đơn thuốc ngoại trú Ấn Độ** (KHÁC dataset RxHandBD). Không nêu rõ tên các VLM cụ thể trong abstract, chỉ nói "frontier vision-language models". Đây là công trình **rất mới, cùng chủ đề** (đơn thuốc viết tay + VLM) — bạn cần định vị khác biệt (dataset khác, khung đánh giá theo rủi ro lâm sàng khác PP-OCRv5/GLM-OCR/CER-WER truyền thống).
- **"From Handwriting to Structured Data: Benchmarking AI Digitisation of Handwritten Forms"** (arXiv:2604.16504, 14/04/2026) — benchmark 17 mô hình MLLM frontier (GPT-5.4, Claude Sonnet 4.6, Gemini 3.1...) trên **một mẫu FORM y tế thực tế** (không phải đơn thuốc). Gemini 3.1: WER=0,50/CER=0,31 (tốt nhất phần văn bản tự do); GPT-5.4 tốt nhất về trích ngày/ít "hallucination" (6%). **Không** test các mô hình OCR-VLM mã nguồn mở chuyên biệt (GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, Qwen-VL) — chỉ test các LLM đóng (closed) qua API.
- **OmniHandwritingOCR** (arXiv:2608.18586, CIKM 2026) — benchmark chẩn đoán MLLM cho HTR tổng quát (77,57K ảnh, gồm cả công thức toán viết tay), KHÔNG chuyên về y khoa.
- **ICDAR 2024-HWD** (English/Hindi/Bangla/Telugu) và **ICDAR 2025 Indic HDR** — cuộc thi nhận dạng chữ viết tay Ấn Độ, không có nội dung y khoa cụ thể (theo ilocr.iiit.ac.in/icdar_2024_hwd).

---

### 3. KHOẢNG TRỐNG (gap) để tạo tính mới cho bài báo 2 tuần

Đây là phát hiện quan trọng nhất (mục tiêu 3):

**Chưa có công trình nào (bình duyệt hoặc preprint) kiểm thử các mô hình VLM-OCR mã nguồn mở thế hệ mới (2024-2026) trên bất kỳ dataset đơn thuốc/HTR y khoa công khai nào** — cụ thể:

| Mô hình | Đã test trên RxHandBD/BD dataset? | Đã test trên IAM? | Nguồn xác nhận |
|---|---|---|---|
| GOT-OCR2.0 (arXiv:2409.01704) | **CHƯA** | Đã dùng làm dữ liệu HUẤN LUYỆN (không phải test độc lập) | Wei et al. 2024 |
| PaddleOCR-VL / -1.5 / -1.6 (arXiv:2510.14528, 2601.21957, 2606.03264) | **CHƯA** | **CHƯA** (chỉ test trên CASIA-HWDB, GNHK, BRUSH — không phải IAM) | PaddleOCR-VL papers |
| DeepSeek-OCR / -OCR2 (arXiv:2510.18234, 2601.20552) | **CHƯA** | **CHƯA** (chỉ test Fox benchmark, OmniDocBench, chữ viết tay Trung Quốc — CER 154,81 rất kém) | DeepSeek-OCR paper |
| Qwen2.5-VL (arXiv:2502.13923) / Qwen3-VL (arXiv:2511.21631) | **CHƯA** | Không xác nhận rõ | Qwen technical reports |
| TrOCR | Đã test trên IAM (2,89% CER) nhưng **CHƯA test trên RxHandBD/dữ liệu y khoa** | Có | Li et al. 2023 |
| Donut | **CHƯA test cả hai** (chỉ CORD/SROIE — biên lai/hoá đơn) | Không rõ | Kim et al. 2022 |
| GPT-4V/Gemini/Claude (qua API) | Có test trên FORM y tế (2604.16504) nhưng **CHƯA test trên đơn thuốc RxHandBD** | Có (leaderboard IAM) | Nhiều nguồn |

**→ Khoảng trống rõ ràng nhất và khả thi nhất trong 2 tuần**: Benchmark một tập hợp mô hình pretrained (Tesseract làm baseline, TrOCR, Donut, GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, Qwen2.5-VL/Qwen3-VL, và tuỳ chọn GPT-4V/Gemini qua API) trên **RxHandBD** (đã có sẵn train/test 80/20, 5.578 ảnh, license MIT — sẵn sàng dùng ngay, không cần IRB vì là dữ liệu công khai không định danh) và đối chứng bằng **IAM** để đo mức độ tổng quát hoá domain-shift (in-domain tổng quát vs out-of-domain y khoa). Công trình gần nhất (dev.to blog, không bình duyệt, chỉ test 4 engine cũ) chưa đụng đến bất kỳ mô hình nào trong danh sách trên — đây là "chỗ trống" thật, chưa bị "chiếm" bởi RxScribe Bench (dataset khác — Ấn Độ) hay "From Handwriting to Structured Data" (dataset khác — form, không phải đơn thuốc, chỉ test LLM đóng qua API).

**Cảnh báo về tốc độ cạnh tranh**: cả RxScribe Bench (08/09/2026) và From Handwriting to Structured Data (04/2026) cho thấy chủ đề "VLM benchmark trên chữ viết tay y khoa" đang rất "nóng" trong nửa cuối 2026 — nên cần công bố nhanh (khớp với timeline 2 tuần bạn đề ra) và định vị rõ sự khác biệt: **dataset RxHandBD (Bangladesh, đơn thuốc thật) + danh mục mô hình OCR-VLM mã nguồn mở chuyên biệt (chưa ai test) + đối chứng domain-shift với IAM**, thay vì chỉ lặp lại phương pháp multi-axis của RxScribe Bench hay tập mô hình đóng của 2604.16504.

---

### Nguồn đã trích dẫn (tổng hợp)
1. Marti, U.-V. & Bunke, H. (2002). *The IAM-database*. IJDAR 5, 39–46. DOI: 10.1007/s100320200071
2. Kefeli, J. et al. (2024). *TCGA-Reports*. Patterns, Cell Press. DOI: 10.1016/j.patter.2024.100933; PMID 38487800
3. RxHandBD, Zenodo (2026). DOI: 10.5281/zenodo.18478741
4. "Doctor's Handwritten Prescription BD dataset", Kaggle (2024) — IEEE 10499631 (2024)
5. arXiv:2412.18199 — Multi-Head Attention for Handwritten Prescriptions (2024)
6. PMC8897401 — Scientific Reports/Nature (2022), BiLSTM+SRP
7. dev.to/kaniel_outis — "How Well Can OCR Read Doctor Handwriting in 2026?" (blog, không bình duyệt)
8. arXiv:2609.13280 — RxScribe Bench (2026)
9. arXiv:2604.16504 — From Handwriting to Structured Data (2026)
10. arXiv:2608.18586 — OmniHandwritingOCR (2026)
11. arXiv:2409.01704 — GOT-OCR2.0 (2024)
12. arXiv:2510.14528, 2601.21957, 2606.03264 — PaddleOCR-VL series
13. arXiv:2510.18234, 2601.20552 — DeepSeek-OCR series
14. arXiv:2109.10282 — TrOCR (AAAI 2023)
15. arXiv:2111.15664 — Donut (ECCV 2022)
16. ilocr.iiit.ac.in — ICDAR 2024-HWD, ICDAR 2025 Indic HDR

*Ghi chú phương pháp*: một số số liệu (đặc biệt số trích dẫn chính xác của RxHandBD/IEEE 10499631, và số CER cụ thể của GOT-OCR2.0 trên IAM) không truy xuất được đầy đủ qua WebSearch/WebFetch trong phạm vi thời gian nghiên cứu này — nên xác minh lại trực tiếp trên Google Scholar/Semantic Scholar và bản PDF gốc của GOT-OCR2.0 trước khi đưa vào bài báo chính thức.
---

## 4.3. Kiểm chứng sâu từng ứng viên (link, license, ground truth — đã xác nhận trực tiếp)

## BÁO CÁO KIỂM CHỨNG SÂU: CÁC BỘ DỮ LIỆU OCR Y KHOA CÔNG KHAI (cập nhật 15/09/2026)

Đã truy cập trực tiếp trang Zenodo, Mendeley Data, Kaggle (qua browser render vì WebFetch tĩnh không đọc được SPA của Kaggle), GitHub, trang chính thức FKI/Bern (qua Wayback Machine vì domain gốc bị chặn TLS/kết nối), PubMed Central, và PhysioNet. Dưới đây là kết quả cho từng bộ.

---

### 1. RxHandBD — "A Handwritten Prescription Word Image Dataset"

- **Link tải trực tiếp:** Zenodo DOI `10.5281/zenodo.18478741` (https://zenodo.org/records/18478741) VÀ Mendeley Data DOI `10.17632/dsb5r6vskg.3` (https://data.mendeley.com/datasets/dsb5r6vskg/3)
- **Số lượng mẫu:** 5.578 ảnh từ cắt (cropped word images), .jpg, vocab 1.559 từ duy nhất (tên thuốc generic, tên biệt dược, dạng dùng, chỉ dẫn lâm sàng). Chia sẵn train/test 4.463/1.115 (80/20).
- **Ground truth:** Có — `train_labels.csv` và `test_labels.csv` kèm mỗi ảnh (không phải JSON/XML).
- **License — MÂU THUẪN giữa 2 nơi host:**
  - Zenodo v1 (04/02/2026): ghi **MIT License**, ảnh chuẩn hóa 128×128px.
  - Mendeley v3 (20/03/2026, phiên bản mới nhất): ghi **CC BY 4.0**, ảnh chuẩn hóa 512×512px, dung lượng 244 MB.
  - Đây là hai phiên bản khác nhau (độ phân giải khác, license khác) của "cùng" bộ dữ liệu — cần trích dẫn rõ phiên bản nào bạn dùng.
- **Bài báo gốc:** **KHÔNG có** paper học thuật/peer-reviewed nào giới thiệu bộ này. Chỉ có metadata dataset (tác giả duy nhất: Md Masudul Islam, Bangladesh University of Business and Technology). Đây chỉ là self-publish dataset, chưa qua bình duyệt.
- **Độ mới/rủi ro uy tín:** Tạo v1 ngày 02/03/2026, v2 13/03, v3 20/03/2026 — cực kỳ mới (chỉ ~6 tháng trước "hôm nay"), lặp phiên bản rất nhanh, một tác giả duy nhất, chưa được cộng đồng khoa học kiểm chứng qua peer review. Đã có 1 bài blog benchmark độc lập (không peer-reviewed, dev.to, tác giả tự ghi "no affiliation") test Tesseract/EasyOCR/PP-OCRv5/GLM-OCR trên bộ này — cho thấy bộ dữ liệu *dùng được về mặt kỹ thuật*, nhưng **không có "uy tín quốc tế" theo đúng nghĩa** (không có DOI citation dataset paper, không ICDAR/PhysioNet/HuggingFace chính thống công nhận).
- **Kết luận:** Tải được ngay, ground truth đầy đủ, nhưng **rủi ro cao về uy tín học thuật** nếu dùng làm bộ dữ liệu chính cho paper — nên dùng làm bộ phụ/bổ sung, không nên là trụ cột duy nhất.

---

### 2. "Doctor's Handwritten Prescription BD dataset" trên Kaggle (mamun1113)

**Đây là một bộ dữ liệu KHÁC HẲN RxHandBD**, không phải cùng một bộ (số lượng ảnh, số lớp, cấu trúc đều khác nhau):

- **Link:** https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset
- **Số lượng mẫu:** 4.680 ảnh từ đã cắt, **78 lớp tên thuốc** (khác với 1.559 vocab của RxHandBD). Chia 60% train / 20% validation / 20% test, stratified theo lớp.
- **Ground truth:** Có — file Excel (.xlsx) và CSV index tên từ, riêng cho mỗi thư mục train/val/test.
- **License:** Kaggle ghi "**Database: Open Database, Contents: © Original Authors**" + ghi chú của tác giả "miễn phí dùng cho mục đích giáo dục và nghiên cứu" (có giới hạn phi thương mại ngầm).
- **Bài báo gốc — CÓ, đã bình duyệt:** A. R. Mia, M. A.-A.-S. Chowdhury, A. A. Mamun, A. M. Ruddra, N. T. Tanny, **"A Deep Neural Network Approach with Pioneering Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh,"** IEEE iCACCESS 2024, Dhaka. (IEEE Xplore document 10499631)
- **Độ tin cậy/uy tín:** Rất tốt trong nhóm Kaggle — **34.700 views, 7.078 lượt tải, 57 upvotes**, cập nhật lần cuối ~2 năm trước (2024), có 29 notebook code công khai dùng bộ này, có cả badge "used in a publication". Đây là bộ **ổn định, đã được kiểm chứng bởi số đông người dùng và có paper hội nghị IEEE thật.**
- **Kết luận: Đây là ứng viên MẠNH NHẤT trong toàn bộ danh sách** — sẵn sàng dùng ngay, ground truth đầy đủ, có paper trích dẫn được, được cộng đồng dùng rộng rãi.

---

### 3. IAM Handwriting Database (Đại học Bern)

- **Link chính thức:** https://fki.tic.heia-fr.ch/databases/iam-handwriting-database (trang gốc hiện bị lỗi TLS/không kết nối được trực tiếp — đã xác minh qua Wayback Machine, nội dung xác thực)
- **Số lượng mẫu (v3.0):** 657 người viết, 1.539 trang scan, 5.685 câu, 13.353 dòng, **115.320 từ** — tất cả đã gán nhãn.
- **Ground truth:** Ảnh PNG (scan 300dpi, grayscale 256 mức) + file XML metadata (segmentation + tham số) đi kèm mỗi form/line/word.
- **Điều kiện dùng (trích nguyên văn từ trang chính thức):** *"This database may be used for non-commercial research purpose only. If you publish material based on this database, we request you to include a reference to paper [4]"* (Marti & Bunke, IJDAR 2002). Phải **đăng ký (registration)** trước khi tải — miễn phí, không cần IRB/HREC, chỉ cần điền form biết ai đang dùng.
- **Bài báo gốc:** U. Marti, H. Bunke, ICDAR 1999 (giới thiệu đầu), IJDAR 2002 (mô tả đầy đủ v3.0) — đây là bộ dữ liệu **gold-standard kinh điển nhất trong ngành HTR**, được trích dẫn hàng ngàn lần.
- **QUAN TRỌNG — không phải dữ liệu y khoa:** IAM là câu tiếng Anh tổng quát (rút từ LOB Corpus – văn bản báo chí/tiểu thuyết), **không liên quan đến y khoa/đơn thuốc**. Chỉ dùng được như **benchmark tổng quát/baseline uy tín quốc tế** để đối chiếu (ví dụ: "model X đạt CER=5% trên IAM nhưng chỉ 40% trên đơn thuốc y khoa" — làm nổi bật độ khó domain y khoa), không thể dùng làm trụ cột chính cho một paper OCR y khoa.
- **Kết luận:** Sẵn sàng dùng trong 2 tuần (đăng ký nhanh, không rào cản), nhưng chỉ nên đóng vai trò **benchmark phụ/baseline chuẩn quốc tế**, không phải dữ liệu y khoa.

---

### 4. TCGA-Reports (Patterns 2024, Kefeli & Tatonetti)

- **Bài báo:** Kefeli J, Tatonetti N. "TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models." *Patterns*, Cell Press, tháng 2/2024. (PMC10935496; DOI báo: 10.1016/j.patter.2024.100965)
- **Bản chất — ĐÃ XÁC MINH RÕ theo đúng câu hỏi:** Bộ dữ liệu công bố chính (`TCGA_Reports.csv.zip`, trên GitHub `jkefeli/tcga-path-reports`) là **VĂN BẢN MÁY ĐỌC ĐƯỢC (text thuần), KHÔNG PHẢI ảnh scan** — 9.523 báo cáo giải phẫu bệnh, 23.909 trang, 842.134 dòng text, 32 loại mô/ung thư.
- **Nguồn gốc OCR:** PDF gốc từ Genomic Data Commons (GDC) → cắt thành ảnh trang → OCR bằng **AWS Textract** → hậu xử lý loại bỏ artifact, **loại bỏ luôn cả chú thích viết tay (handwritten annotation)** coi đó là "nhiễu". → Bản final KHÔNG hướng tới chữ viết tay.
- **PHÁT HIỆN QUAN TRỌNG (trả lời đúng câu hỏi "có kèm ảnh scan gốc không"):** README của GitHub xác nhận **ảnh trang gốc dùng làm input cho Textract (`imgs_for_aws`) VÀ file OCR-response gốc từ Textract (`aws_response`) ĐƯỢC LƯU RIÊNG trên Mendeley Data** (do dung lượng lớn), cùng DOI với bản text: `10.17632/hyg5xkznpx.1` (https://data.mendeley.com/datasets/hyg5xkznpx/1). Tức là **có tồn tại cặp ảnh scan + kết quả OCR thật** — nhưng đây là văn bản **in/máy đánh chữ (typed)**, không phải chữ viết tay bác sĩ.
- **License:** CC BY 4.0 (ghi rõ trên cả Mendeley và GitHub LICENSE file).
- **Định dạng file:** CSV (báo cáo text) trên GitHub; ảnh + response JSON/text trên Mendeley (chưa xác minh được cấu trúc file chi tiết trang Mendeley do phần "Files" load bằng JS không hiện hết trong lần truy cập).
- **Hướng sử dụng thay thế nếu muốn bài toán OCR:** 
  - (a) Dùng cặp `imgs_for_aws` + `aws_response` có sẵn trên Mendeley làm bài toán OCR **văn bản y khoa in/scan chất lượng thấp** (không phải chữ viết tay) — đỡ công tự tạo ảnh giả.
  - (b) Nếu vẫn muốn bài toán synthetic, có thể tự render lại text sạch thành ảnh giả lập scan (thêm nhiễu, xoay, mờ) như đề xuất ban đầu của bạn.
- **Kết luận:** Sẵn sàng dùng trong 2 tuần cho OCR văn bản y khoa **in ấn/scan** (không phải handwriting), license rất thoải mái (CC BY 4.0), có paper Patterns 2024 uy tín cao.

---

### 5. PhysioNet — dataset scanned clinical documents / chữ viết tay

- Đã kiểm tra trực tiếp `physionet.org/about/database/` (danh sách đầy đủ) và trang chỉ mục PhysioNet (physionet.org).
- **KẾT LUẬN: KHÔNG tồn tại** bộ dữ liệu nào trên PhysioNet tập trung vào scanned documents, OCR, hoặc handwriting/prescription. PhysioNet gần như toàn bộ là tín hiệu sinh lý (ECG/EEG/PPG), ảnh y tế (X-quang, CT, đáy mắt), và dữ liệu EHR có cấu trúc (MIMIC, eICU). Vài bộ có "báo cáo X-quang dạng text" (MIMIC-CXR, REFLACX) nhưng đó là text gõ máy trong EHR, không phải OCR/scan/handwriting.
- **Khuyến nghị:** Bỏ hướng PhysioNet — không có ứng viên phù hợp.

---

### 6. CMATERdb (và các bộ liên quan Ấn Độ/Bangladesh)

- **Link:** https://code.google.com/archive/p/cmaterdb (đã archive; mirror trên TensorFlow Datasets/GitHub)
- Đã kiểm tra toàn bộ nội dung CMATERdb (tạo tại CMATER Lab, Jadavpur University, Kolkata): 
  - CMATERdb 3.1.1/3.2.1/3.3.1/3.4.1: **chữ số viết tay** (Bangla/Devanagari/Telugu/Arabic numerals), 32×32px, ~3.000-6.000 ảnh mỗi bộ, license CC BY 4.0.
  - CMATERdb1: 150 trang văn bản viết tay tổng quát (100 trang Bangla, 50 trang Bangla-English), không liên quan y khoa.
- **KẾT LUẬN: KHÔNG có nội dung y khoa/đơn thuốc nào trong toàn bộ họ CMATERdb.** Không phù hợp cho nghiên cứu OCR y khoa.
- **Ghi chú thêm:** Tìm thấy một bộ liên quan thật của nhóm Bangladesh/Kyushu University: **"Handwritten Medical Term Corpus"** (17.431 mẫu, 480 từ y khoa — 360 Anh + 120 Bangla, từ 39 nhân viên y tế), gắn với bài báo *Scientific Reports* (Nature, 2022, PMC8897401) và IEEE 2021 (document 9488622). Tuy nhiên **không tìm thấy link tải công khai trực tiếp** (không có trên Kaggle/Zenodo/GitHub) — dữ liệu có vẻ chỉ dùng nội bộ nhóm nghiên cứu Kyushu University, cần liên hệ tác giả để xin — **không đáp ứng yêu cầu "sẵn sàng dùng ngay trong 2 tuần."**

---

### BẢNG TỔNG KẾT — SẴN SÀNG DÙNG TRONG 2 TUẦN?

| Bộ dữ liệu | Tải được ngay? | Ground truth đầy đủ? | License rõ ràng cho nghiên cứu? | Uy tín/paper? | Domain y khoa chữ viết tay? |
|---|---|---|---|---|---|
| **Kaggle "Doctor's Handwritten Prescription BD dataset" (mamun1113)** | ✅ Có | ✅ CSV/Excel | ✅ Open Database + ghi chú nghiên cứu | ✅ **Có paper IEEE iCACCESS 2024**, rất phổ biến (7k+ downloads) | ✅ Đúng — đơn thuốc viết tay |
| RxHandBD (Zenodo/Mendeley) | ✅ Có | ✅ CSV | ⚠️ Mâu thuẫn MIT/CC-BY-4.0 giữa 2 bản | ❌ Không có paper bình duyệt, quá mới (2026) | ✅ Đúng — đơn thuốc viết tay |
| IAM Handwriting Database | ✅ Có (cần đăng ký) | ✅ PNG+XML | ✅ Phi thương mại, phải cite | ✅✅ Gold-standard (ICDAR/IJDAR) | ❌ Không — tiếng Anh tổng quát, KHÔNG y khoa |
| TCGA-Reports (+ imgs_for_aws/aws_response) | ✅ Có (GitHub+Mendeley) | ✅ CSV text; ảnh+OCR-response riêng | ✅ CC BY 4.0 | ✅✅ Paper Patterns 2024 (Cell Press) | ⚠️ Y khoa nhưng là **văn bản in/scan**, KHÔNG phải chữ viết tay (đã lọc bỏ) |
| PhysioNet (bất kỳ) | ❌ Không tồn tại | — | — | — | — |
| CMATERdb | ✅ Có nhưng không liên quan | — | ✅ CC BY 4.0 | ✅ Có paper IJDAR | ❌ Không — chữ số/văn bản tổng quát |

#### Khuyến nghị cho kế hoạch 2 tuần
1. **Trụ cột chính:** Kaggle "Doctor's Handwritten Prescription BD dataset" (mamun1113) — bộ duy nhất vừa có ground truth đầy đủ, vừa có paper bình duyệt thật, vừa được cộng đồng dùng rộng rãi và ổn định nhiều năm.
2. **Bổ sung/đối chiếu domain y khoa:** RxHandBD — dùng thêm để tăng độ đa dạng vocab (1.559 từ vs 78 từ của bộ Kaggle), nhưng ghi rõ trong paper đây là dataset mới/chưa bình duyệt để tránh bị reviewer bắt lỗi "nguồn không uy tín".
3. **Benchmark tổng quát/baseline quốc tế (không y khoa):** IAM — dùng để so sánh "model tốt trên text tổng quát nhưng kém trên đơn thuốc y khoa", tăng tính thuyết phục về challenge domain-specific.
4. **Không dùng:** PhysioNet (không tồn tại) và CMATERdb (không liên quan y khoa) — nên loại khỏi kế hoạch.
5. TCGA-Reports phù hợp nếu muốn mở rộng sang OCR văn bản y khoa **in/scan chất lượng thấp** (khác khía cạnh với "chữ viết tay"), tận dụng luôn cặp ảnh+OCR-response có sẵn trên Mendeley thay vì tự tạo ảnh giả.