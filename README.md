# Dự án nghiên cứu OCR — Tổng quan

Bộ tài liệu này được tổng hợp để chuẩn bị cho một dự án nghiên cứu khoa học **độc lập** về OCR (Optical Character Recognition), ứng dụng y khoa, mục tiêu công bố tại **tạp chí Q2**, chỉ dùng **dữ liệu công khai, uy tín quốc tế** (không tiếng Việt, không cần hợp tác bệnh viện/CCI). Toàn bộ nội dung chi tiết nằm trong thư mục [`docs/`](docs/):

1. [Kiến thức nền tảng của OCR](docs/01-kien-thuc-co-so.md) — định nghĩa, pipeline, các phương pháp cổ điển và deep learning, metric đánh giá, benchmark.
2. [Lịch sử phát triển OCR](docs/02-lich-su.md) — từ optophone 1913 đến các mô hình đa phương thức 2026.
3. [Nghiên cứu mới nhất (2023–2026)](docs/03-nghien-cuu-moi-nhat.md) — tổng hợp từ web search (chung + y khoa), PubMed (15 bài đã bình duyệt), và Consensus.
4. [Bộ dữ liệu OCR y khoa công khai, uy tín](docs/04-dataset-y-khoa-cong-khai.md) — ~22 ứng viên đã khảo sát, kiểm chứng sâu link tải/license/ground-truth, xác định dataset "chuẩn" và khoảng trống chưa ai khai thác.
5. **[→ Kế hoạch hướng Q2, ~3-4 tuần (đọc phần này trước)](docs/05-ke-hoach-Q2.md)** — tạp chí mục tiêu (đã xác minh SJR thật), thiết kế LoRA fine-tuning, cấu trúc lại toàn bộ phase, lịch trình theo tuần/ngày.

> `docs/archive/` chứa các bản kế hoạch/hướng nghiên cứu **cũ đã bị thay thế** (gắn CCI+ung thư nhi khoa+cần IRB; và bản benchmark-2-tuần trước khi nâng mục tiêu lên Q2) — giữ lại chỉ để đối chiếu, Phase 1-2 mô tả trong đó vẫn là nền tảng không đổi.

## Tiến độ thực tế (cập nhật liên tục)

- ✅ **Phase 1** (Tesseract, EasyOCR) — xong, kết quả thật tại [results/phase1_summary.md](results/phase1_summary.md). CER 0.55–0.84, EasyOCR thắng rõ nhưng cả 2 đều sai >50% ký tự trên chữ viết tay.
- 🔄 **Phase 2** (TrOCR-large-handwritten, Donut-base zero-shot) — đang chạy trên Kaggle GPU (kernel `dangbang1/ocr-med-benchmark-phase1`, điều khiển trực tiếp qua `kaggle kernels push/status/output` — xem [notebooks/kaggle_benchmark.py](notebooks/kaggle_benchmark.py)).
- ⏭️ Phase 3-5 (GOT-OCR2.0, PaddleOCR-VL, Qwen-VL) → Phase 6-8 (**LoRA fine-tune TrOCR-large-handwritten**, đánh giá lại + catastrophic-forgetting check) → Phase 9-13 (phân tích + viết bài + nộp). Chi tiết đầy đủ ở [docs/05-ke-hoach-Q2.md](docs/05-ke-hoach-Q2.md).

## Điểm mấu chốt (tóm tắt nhanh)

- **Dataset trụ cột:** Kaggle "Doctor's Handwritten Prescription BD dataset" (`mamun1113/doctors-handwritten-prescription-bd-dataset`, đơn thuốc viết tay, có paper bình duyệt IEEE iCACCESS 2024, license mở) — đối chứng domain-shift bằng **IAM Handwriting Database** (`nibinv23/iam-handwriting-word-database`).
- **Tính mới benchmark:** GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, Qwen-VL chưa từng được test trên dataset đơn thuốc/HTR y khoa công khai nào — nhưng chủ đề đang "nóng" (2 công trình rất gần xuất hiện 04/2026, 09/2026), nên cần công bố nhanh.
- **Tính mới cho Q2 (mới thêm 16/09/2026):** không chỉ benchmark model có sẵn — sẽ **LoRA fine-tune TrOCR-large-handwritten** trên 2.808 ảnh train, đo cải thiện in-domain (Kaggle-Rx) VÀ nguy cơ "quên" tổng quát hóa (catastrophic forgetting trên IAM) — đây là đóng góp phương pháp thật sự, có bằng chứng từ nhiều bài báo tương tự cho thấy fine-tuning có thể giảm CER 50-80%.
- **Tạp chí mục tiêu:** **PeerJ Computer Science** (Q2 xác nhận qua SCImago, APC thấp nhất $1.395, có tiền lệ xuất bản bài rất giống — TrOCR cho OCR tiếng Urdu). Dự phòng: *Journal of Imaging* (MDPI, Q2 chắc chắn nhất, bình duyệt ~20 ngày) và *Health Information Science and Systems* (Springer, Q2, có route miễn phí không cần trả APC).
- **Kỳ vọng đúng với timeline:** ~3-4 tuần cho thực nghiệm + bản thảo + nộp — **không phải** được chấp nhận/xuất bản (bình duyệt tạp chí luôn mất thêm nhiều tuần đến vài tháng, ngoài tầm kiểm soát).

## Lưu ý về độ tin cậy nguồn

Phần 1–2 (kiến thức nền tảng, lịch sử) là kiến thức đã được xác lập rộng rãi. Phần 3–5 dựa nhiều vào tra cứu web/PubMed/Consensus/SCImago — một số nguồn là preprint (arXiv) chưa qua bình duyệt đầy đủ, quartile SJR có thể đổi hàng năm (cập nhật ~tháng 4-5), và scimagojr.com có chặn bot nên số liệu quartile trong doc 5 được kiểm chứng chéo qua nhiều nguồn thứ cấp — **tự kiểm tra lại trực tiếp trên scimagojr.com trước khi quyết định nộp bài chính thức**.

Vì chủ đề "benchmark + fine-tune VLM trên chữ viết tay y khoa" đang được cộng đồng quan tâm mạnh cuối 2026, nên ưu tiên nộp arXiv sớm (ngay sau khi xong benchmark, không cần chờ fine-tuning hoàn tất) để giữ tính ưu tiên (priority) của đóng góp, rồi cập nhật bản mở rộng khi có kết quả fine-tuning.
