# Phase 4 — PaddleOCR-VL: LOẠI KHỎI BENCHMARK (lỗi tương thích thượng nguồn)

Chạy trên Kaggle (kernel v12, 16/09/2026). Kết quả: `TypeError: create_causal_mask() got an unexpected keyword argument 'inputs_embeds'`, lỗi xảy ra bên trong chính code `modeling_paddleocr_vl.py` (tải qua `trust_remote_code=True` từ HF repo của PaddlePaddle, không phải code của dự án này).

## Đã tra cứu — đây là lỗi cộng đồng đã biết, chưa có fix

Thảo luận chính thức trên HuggingFace: ["Newest commit breaks compatibility with transformers==4.57.6, while 5.3.0 is broken as well"](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/discussions/22) (đóng ngày 30/04, không có giải pháp được ghi nhận). Trích nguyên văn: *"Anything above `transformers==5` with `AutoModelForImageTextToText` has seemingly always been broken."* Một người dùng khác báo đã "lock ở `4.57.6`" như giải pháp tạm trước khi một commit mới của chính model repo làm hỏng luôn cả cách đó.

**Kết luận:** đây không phải lỗi do T4/dtype/prompt của dự án (đã áp dụng đầy đủ bài học từ Phase 3: ghim `transformers==4.57.0`, dùng `float16`, debug 5 ảnh trước khi chạy full) — mà là xung đột API nội bộ giữa code tùy chỉnh của PaddleOCR-VL và các phiên bản `transformers` hiện có, chưa có tổ hợp version nào được xác nhận hoạt động ổn định tại thời điểm 16/09/2026.

## Quyết định

**Loại PaddleOCR-VL khỏi benchmark zero-shot**, không đầu tư thêm thời gian debug một lỗi thượng nguồn chưa có lời giải. Code gọi model vẫn giữ lại trong `notebooks/kaggle_benchmark.py` (đã vô hiệu hóa bằng `return` sớm, kèm comment giải thích) để dễ thử lại nếu PaddlePaddle phát hành bản fix trước khi nộp bài. Chuyển sang **Phase 5: Qwen2.5-VL-3B-Instruct** — model nằm trong `transformers` chính thức (không cần `trust_remote_code`), hệ sinh thái ổn định hơn nhiều.

## Ghi chú cho bài báo

Đây vẫn là một điểm dữ liệu đáng đưa vào Limitations/Discussion: không phải mọi mô hình VLM-OCR "mới nhất" đều sẵn sàng dùng ngay trong thực tế nghiên cứu — sự phụ thuộc vào `trust_remote_code` và tốc độ phát triển nhanh của các repo này có thể gây ra rào cản tái lập (reproducibility) thực sự, một quan sát có giá trị cho cộng đồng OCR/benchmark.
