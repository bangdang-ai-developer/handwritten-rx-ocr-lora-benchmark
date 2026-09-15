# Dự án nghiên cứu OCR — Tổng quan

Bộ tài liệu này được tổng hợp để chuẩn bị cho một dự án nghiên cứu khoa học **độc lập** về OCR (Optical Character Recognition), ứng dụng y khoa, cần hoàn thành **gấp trong 2 tuần**, chỉ dùng **dữ liệu công khai, uy tín quốc tế** (không tiếng Việt, không cần hợp tác bệnh viện/CCI). Toàn bộ nội dung chi tiết nằm trong thư mục [`docs/`](docs/):

1. [Kiến thức nền tảng của OCR](docs/01-kien-thuc-co-so.md) — định nghĩa, pipeline, các phương pháp cổ điển và deep learning, metric đánh giá, benchmark.
2. [Lịch sử phát triển OCR](docs/02-lich-su.md) — từ optophone 1913 đến các mô hình đa phương thức 2026.
3. [Nghiên cứu mới nhất (2023–2026)](docs/03-nghien-cuu-moi-nhat.md) — tổng hợp từ web search (chung + y khoa), PubMed (15 bài đã bình duyệt), và Consensus.
4. [Bộ dữ liệu OCR y khoa công khai, uy tín](docs/04-dataset-y-khoa-cong-khai.md) — ~22 ứng viên đã khảo sát, kiểm chứng sâu link tải/license/ground-truth, xác định dataset "chuẩn" và khoảng trống chưa ai khai thác.
5. **[→ Kế hoạch 2 tuần (đọc phần này trước)](docs/05-ke-hoach-2-tuan.md)** — phương án đã chọn + lịch trình hành động 14 ngày, cài đặt cụ thể, chi phí, cấu trúc bài báo, nơi nộp.

> `docs/archive/` chứa 2 file hướng nghiên cứu/kế hoạch **cũ đã bị thay thế** (gắn với CCI + đăng ký ung thư nhi khoa + cần dữ liệu bệnh viện/IRB, không hợp với yêu cầu 2 tuần độc lập) — giữ lại chỉ để đối chiếu.

## Điểm mấu chốt (tóm tắt nhanh)

- **Phương án đã chọn:** Benchmark các mô hình OCR/VLM pretrained (2024–2026: Tesseract, EasyOCR/PaddleOCR, TrOCR, Donut, GOT-OCR2.0, PaddleOCR-VL, Qwen2.5/3-VL, tùy chọn GPT-5/Gemini qua API) trên bộ **Kaggle "Doctor's Handwritten Prescription BD dataset"** (đơn thuốc viết tay, có paper bình duyệt IEEE iCACCESS 2024, ~7.000 lượt tải, license mở) — đối chứng domain-shift bằng **IAM Handwriting Database** (chuẩn HTR kinh điển, ~thiên trích dẫn).
- **Vì sao có tính mới:** tra cứu tài liệu xác nhận GOT-OCR2.0, PaddleOCR-VL, DeepSeek-OCR, Qwen-VL **chưa từng được test trên bất kỳ dataset đơn thuốc/HTR y khoa công khai nào**. Hai công trình cạnh tranh gần nhất (RxScribe Bench 09/2026, "From Handwriting to Structured Data" 04/2026) dùng dataset/model khác — chỗ trống còn mở nhưng **đang "nóng"**, nên cần công bố nhanh.
- **Không dùng:** RxHandBD (Zenodo/Mendeley) làm trụ cột — dataset đúng domain nhưng chưa có paper bình duyệt, license mâu thuẫn giữa 2 bản host (chỉ dùng làm supplementary); PhysioNet — không có dataset OCR/scan/handwriting phù hợp; TCGA-Reports — là văn bản in/scan (không phải chữ viết tay), phù hợp hướng khác nếu muốn mở rộng sau.
- **Kỳ vọng đúng với "2 tuần":** hoàn thành thực nghiệm + bản thảo + nộp arXiv preprint — **không phải** được bình duyệt/xuất bản (luôn mất thêm nhiều tuần đến vài tháng, ngoài tầm kiểm soát).

## Lưu ý về độ tin cậy nguồn

Phần 1–2 (kiến thức nền tảng, lịch sử) là kiến thức đã được xác lập rộng rãi. Phần 3–5 dựa nhiều vào tra cứu web/PubMed/Consensus thực hiện ngày 15/09/2026 — một số nguồn là preprint (arXiv) chưa qua bình duyệt đầy đủ, và một số số liệu benchmark rất mới (2026, kể cả tên phiên bản model) có độ tin cậy chưa xác minh cao. **Trước khi trích dẫn chính thức trong bài báo, hãy đối chiếu lại DOI/link gốc** — mỗi phần đều có danh sách tham khảo kèm link.

Vì chủ đề "benchmark VLM trên chữ viết tay y khoa" đang được cộng đồng quan tâm mạnh cuối 2026 (xem mục 4.2), nên bắt tay thực hiện ngay và ưu tiên nộp arXiv sớm để giữ tính ưu tiên (priority) của đóng góp.
