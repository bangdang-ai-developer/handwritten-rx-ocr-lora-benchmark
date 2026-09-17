# Dự án nghiên cứu OCR — Tổng quan

Bộ tài liệu này được tổng hợp để chuẩn bị cho một dự án nghiên cứu khoa học **độc lập** về OCR (Optical Character Recognition), ứng dụng y khoa, mục tiêu công bố tại **tạp chí Q2**, chỉ dùng **dữ liệu công khai, uy tín quốc tế** (không tiếng Việt, không cần hợp tác bệnh viện/CCI). Toàn bộ nội dung chi tiết nằm trong thư mục [`docs/`](docs/):

1. [Kiến thức nền tảng của OCR](docs/01-kien-thuc-co-so.md) — định nghĩa, pipeline, các phương pháp cổ điển và deep learning, metric đánh giá, benchmark.
2. [Lịch sử phát triển OCR](docs/02-lich-su.md) — từ optophone 1913 đến các mô hình đa phương thức 2026.
3. [Nghiên cứu mới nhất (2023–2026)](docs/03-nghien-cuu-moi-nhat.md) — tổng hợp từ web search (chung + y khoa), PubMed (15 bài đã bình duyệt), và Consensus.
4. [Bộ dữ liệu OCR y khoa công khai, uy tín](docs/04-dataset-y-khoa-cong-khai.md) — ~22 ứng viên đã khảo sát, kiểm chứng sâu link tải/license/ground-truth, xác định dataset "chuẩn" và khoảng trống chưa ai khai thác.
5. **[→ Kế hoạch hướng Q2, ~3-4 tuần (đọc phần này trước)](docs/05-ke-hoach-Q2.md)** — tạp chí mục tiêu (đã xác minh SJR thật), thiết kế LoRA fine-tuning, cấu trúc lại toàn bộ phase, lịch trình theo tuần/ngày.

> `docs/archive/` chứa các bản kế hoạch/hướng nghiên cứu **cũ đã bị thay thế** (gắn CCI+ung thư nhi khoa+cần IRB; và bản benchmark-2-tuần trước khi nâng mục tiêu lên Q2) — giữ lại chỉ để đối chiếu, Phase 1-2 mô tả trong đó vẫn là nền tảng không đổi.

## Tiến độ thực tế (cập nhật liên tục)

**Benchmark zero-shot (Phase 1-5) ĐÃ XONG** — 5 model chạy thật thành công + 1 loại trừ có ghi chú:

| Model | CER kaggle_rx | CER iam | Ghi chú |
|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | **0.434 (tốt nhất)** | 1.10* | *mean bị outlier kéo lệch — median=0.0, xem [phase5](results/phase5_summary.md) |
| **GOT-OCR2.0** | 0.479 | **0.386 (tốt nhất)** | phải ghim `transformers==4.57.0` mới chạy đúng, xem [phase3](results/phase3_summary.md) |
| EasyOCR | 0.552 | 0.736 | |
| TrOCR-large-handwritten | 0.580 | 0.441 | top-1 acc cao nhất nhóm OCR chuyên biệt (79.5%) |
| Tesseract | 0.625 | 0.836 | baseline cổ điển |
| Donut-base | 1.00 (thất bại) | 1.00 (thất bại) | phát hiện hợp lệ — xem [phase2](results/phase2_summary.md) |
| ~~PaddleOCR-VL~~ | loại trừ | loại trừ | lỗi tương thích thượng nguồn chưa có fix, xem [phase4](results/phase4_summary.md) |

**Phát hiện nổi bật nhất:** trên đúng domain mục tiêu (đơn thuốc), VLM tổng quát nhỏ (Qwen2.5-VL-3B) vượt qua mọi model OCR chuyên biệt — kể cả GOT-OCR2.0 vốn được thiết kế riêng cho OCR.

Toàn bộ dữ liệu thô: [results/results_master_combined.csv](results/results_master_combined.csv) (8.260 dòng).

**LoRA fine-tune TrOCR-large-handwritten (Phase 7-8) ĐÃ XONG** — kết quả thật, 3.120 ảnh train, so sánh paired với zero-shot (n khớp 100%, Wilcoxon signed-rank):

| Dataset | CER zero-shot → fine-tuned | Exact-match | Wilcoxon p-value | Ý nghĩa |
|---|---|---|---|---|
| **Kaggle-Rx test (in-domain, 780 ảnh)** | 0,580 → **0,149** (−74,4%) | 8,1% → **60,0%** | p=6,4×10⁻¹¹⁰ | Cải thiện rất lớn, cực kỳ có ý nghĩa |
| **IAM (out-of-domain, 400 ảnh)** | 0,441 → **0,524** (+18,8%, TỆ HƠN) | 57,3% → 25,8% | p=4,3×10⁻⁵ | Catastrophic forgetting có ý nghĩa thống kê |

Top-1 accuracy (ánh xạ về 78 tên thuốc thật) sau fine-tune: **89,9%**. Chi tiết đầy đủ (cấu hình LoRA, lịch sử train, diễn giải song song 2 phát hiện) ở [results/phase7_summary.md](results/phase7_summary.md).

**Phân tích tổng hợp (Phase 9) ĐÃ XONG** — phát hiện quan trọng nhất: sau fine-tune, TrOCR-large-handwritten
(558M) **vượt qua mọi model zero-shot, kể cả VLM lớn hơn nhiều (Qwen2.5-VL-3B, GOT-OCR2.0)** trên domain mục
tiêu (CER 0,149 vs 0,434 vs 0,479, Wilcoxon p<10⁻⁴⁷ cho cả hai so sánh). Chi tiết đầy đủ (bảng 8 model × 2
dataset, domain-shift gap, lưu ý về phân phối đuôi dài/outlier-robustness) ở [results/phase9_summary.md](results/phase9_summary.md), script tái lập ở [src/analyze_aggregate.py](src/analyze_aggregate.py).

**Phân tích lỗi định tính (Phase 10) ĐÃ XONG** — fine-tuning dịch chuyển hẳn phân phối lỗi trên domain mục
tiêu: lỗi "đoán bừa" (hallucination) giảm ~3,2 lần (18,6%→5,8%), nhưng lỗi "giống một tên thuốc thật khác"
(nguy hiểm hơn về an toàn lâm sàng vì khó bị phát hiện thủ công) tăng gần gấp đôi (1,5%→3,5%). Trên IAM,
catastrophic forgetting chủ yếu là dịch chuyển từ "đúng tuyệt đối" sang "lỗi mức trung bình", không phải sụp
đổ hoàn toàn; kiểm tra riêng cho thấy **không có bằng chứng "rò rỉ" tên thuốc** khi model đọc chữ viết tay
tổng quát (0,25% trên IAM, gần như trùng hợp ngẫu nhiên). Chi tiết + ví dụ cụ thể ở [results/phase10_summary.md](results/phase10_summary.md), script tái lập ở [src/analyze_errors.py](src/analyze_errors.py).

**Ablation (Phase 7b) ĐÃ XONG** — chạy full training cho r=8/16/32 (elastic=True) + r=16 không-elastic.
**Phát hiện quan trọng: r=16 (cấu hình "main" ban đầu) KHÔNG phải rank tốt nhất** — cả r=8 và r=32 đều vượt
r=16 có ý nghĩa thống kê mạnh trên kaggle_rx (CER 0,113/0,114 vs 0,149, p<10⁻⁶), và **r=32 tốt nhất trên cả 2
tiêu chí** (kaggle_rx VÀ giữ tổng quát hoá IAM tốt nhất, không đánh đổi). Augmentation elastic: không ảnh
hưởng in-domain, có xu hướng (chưa đạt ý nghĩa 0,05) giảm nhẹ forgetting. Chi tiết + khuyến nghị (đổi model
chính sang r=32?) ở [results/phase7b_ablation_summary.md](results/phase7b_ablation_summary.md) — **đang chờ
quyết định của bạn** trước khi viết lại Phase 9/10 với r=32.

**Tiếp theo:** Phase 11 (tuỳ chọn, RxHandBD) → Phase 12-13 (viết bài + nộp). Chi tiết đầy đủ ở [docs/05-ke-hoach-Q2.md](docs/05-ke-hoach-Q2.md). Toàn bộ pipeline chạy qua điều khiển trực tiếp Kaggle API (`kaggle kernels push/status/output`, không cần mở trình duyệt) — xem [notebooks/kaggle_benchmark.py](notebooks/kaggle_benchmark.py).

## Điểm mấu chốt (tóm tắt nhanh)

- **Dataset trụ cột:** Kaggle "Doctor's Handwritten Prescription BD dataset" (`mamun1113/doctors-handwritten-prescription-bd-dataset`, đơn thuốc viết tay, có paper bình duyệt IEEE iCACCESS 2024, license mở) — đối chứng domain-shift bằng **IAM Handwriting Database** (`nibinv23/iam-handwriting-word-database`).
- **Tính mới benchmark:** GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, Qwen-VL chưa từng được test trên dataset đơn thuốc/HTR y khoa công khai nào — nhưng chủ đề đang "nóng" (2 công trình rất gần xuất hiện 04/2026, 09/2026), nên cần công bố nhanh.
- **Tính mới cho Q2 (mới thêm 16/09/2026):** không chỉ benchmark model có sẵn — sẽ **LoRA fine-tune TrOCR-large-handwritten** trên 2.808 ảnh train, đo cải thiện in-domain (Kaggle-Rx) VÀ nguy cơ "quên" tổng quát hóa (catastrophic forgetting trên IAM) — đây là đóng góp phương pháp thật sự, có bằng chứng từ nhiều bài báo tương tự cho thấy fine-tuning có thể giảm CER 50-80%.
- **Tạp chí mục tiêu:** **PeerJ Computer Science** (Q2 xác nhận qua SCImago, APC thấp nhất $1.395, có tiền lệ xuất bản bài rất giống — TrOCR cho OCR tiếng Urdu). Dự phòng: *Journal of Imaging* (MDPI, Q2 chắc chắn nhất, bình duyệt ~20 ngày) và *Health Information Science and Systems* (Springer, Q2, có route miễn phí không cần trả APC).
- **Kỳ vọng đúng với timeline:** ~3-4 tuần cho thực nghiệm + bản thảo + nộp — **không phải** được chấp nhận/xuất bản (bình duyệt tạp chí luôn mất thêm nhiều tuần đến vài tháng, ngoài tầm kiểm soát).

## Lưu ý về độ tin cậy nguồn

Phần 1–2 (kiến thức nền tảng, lịch sử) là kiến thức đã được xác lập rộng rãi. Phần 3–5 dựa nhiều vào tra cứu web/PubMed/Consensus/SCImago — một số nguồn là preprint (arXiv) chưa qua bình duyệt đầy đủ, quartile SJR có thể đổi hàng năm (cập nhật ~tháng 4-5), và scimagojr.com có chặn bot nên số liệu quartile trong doc 5 được kiểm chứng chéo qua nhiều nguồn thứ cấp — **tự kiểm tra lại trực tiếp trên scimagojr.com trước khi quyết định nộp bài chính thức**.

Vì chủ đề "benchmark + fine-tune VLM trên chữ viết tay y khoa" đang được cộng đồng quan tâm mạnh cuối 2026, nên ưu tiên nộp arXiv sớm (ngay sau khi xong benchmark, không cần chờ fine-tuning hoàn tất) để giữ tính ưu tiên (priority) của đóng góp, rồi cập nhật bản mở rộng khi có kết quả fine-tuning.
