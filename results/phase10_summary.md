# Phase 10 — Phân tích lỗi định tính (fine-tune sửa được loại lỗi nào, tạo ra loại lỗi nào mới)

Script tái lập: `src/analyze_errors.py`. Phân loại lỗi dựa trên `results_master_combined.csv`
(cột `exact_match`, `degenerate`, `top1_pred`/`top1_correct` có sẵn từ `src/metrics.py`), so sánh
**trocr-large-handwritten (zero-shot)** với **trocr-lora-finetuned**, cả 2 dataset.

## 1. Đơn vị phân tích: 4 loại lỗi trên Kaggle-Rx (đóng, 78 tên thuốc)

| Loại | Định nghĩa |
|---|---|
| `correct` | Khớp chuỗi chính xác (`exact_match`) |
| `minor_ocr_noise_still_correct_drug` | Chuỗi không khớp tuyệt đối, nhưng khi ánh xạ về gần nhất trong 78 tên thuốc (`top1_pred`) thì **đúng thuốc** — lỗi chính tả không ảnh hưởng ý nghĩa lâm sàng |
| `confusable_wrong_drug` | Sai thuốc, và bản thân chuỗi output **giống rõ một tên thuốc khác có thật** (khoảng cách edit chuẩn hoá ≤0,34 tới `top1_pred` sai) — **loại lỗi nguy hiểm nhất về mặt lâm sàng** (dễ bị đọc nhầm thành đơn thuốc hợp lệ nhưng sai) |
| `hallucination_far_off` | Sai thuốc, và output không giống bất kỳ tên thuốc thật nào (không degenerate, nhưng cũng không phải near-miss) — dễ bị người đọc/dược sĩ phát hiện là "rõ ràng sai" |

## 2. Kết quả — Kaggle-Rx (in-domain)

| Loại lỗi | Zero-shot | Fine-tuned | Thay đổi |
|---|---|---|---|
| `correct` | 8,08% | **60,00%** | +51,9 điểm % |
| `minor_ocr_noise_still_correct_drug` | 71,79% | 30,77% | −41,0 điểm % |
| `hallucination_far_off` | 18,59% | **5,77%** | −12,8 điểm % (giảm ~3,2 lần) |
| `confusable_wrong_drug` | 1,54% | **3,46%** | **+1,9 điểm % (tăng ~2,2 lần)** |

**Diễn giải chính:**
- Fine-tuning không chỉ giảm CER trung bình — nó **dịch chuyển hẳn phân phối lỗi**: phần lớn trường hợp
  trước đây "đúng thuốc nhưng chính tả nhiễu" (71,8%) nay trở thành **khớp chính xác hoàn toàn** (60%),
  và quan trọng hơn, **lỗi sai-hoàn-toàn-không-giống-gì (hallucination) giảm mạnh** (18,6%→5,8%) — mô hình
  ít "đoán bừa" hơn hẳn sau khi thấy đủ ví dụ thật.
- **Phát hiện cần cảnh báo cho phần Thảo luận/An toàn lâm sàng**: tỷ lệ lỗi `confusable_wrong_drug`
  (output giống hệt MỘT tên thuốc khác có thật) tăng gần gấp đôi (1,54%→3,46%). Điều này hợp logic: khi
  model học được "hình dạng" chung của 78 tên thuốc thật, phần lỗi còn sót lại có xu hướng rơi vào không
  gian "trông giống thuốc thật" thay vì "rõ ràng vô nghĩa". Về mặt an toàn: lỗi loại `hallucination_far_off`
  dễ bị dược sĩ/nhân viên y tế phát hiện ngay ("chữ này không phải tên thuốc nào cả"), trong khi
  `confusable_wrong_drug` (ví dụ đơn thuốc ghi "Amodis" bị đọc thành "Axodin" — cả hai đều là thuốc thật)
  **nguy hiểm hơn vì trông hợp lý, khó bị nghi ngờ khi kiểm tra thủ công**. Đây là lý do một hệ thống OCR y
  khoa thực tế **không nên chỉ tối ưu CER trung bình**, mà cần đo riêng loại lỗi "sound-alike/look-alike
  giữa các thuốc thật" — gợi ý hướng cải tiến tương lai: thêm loss/metric phạt nặng riêng cho nhầm lẫn giữa
  các cặp thuốc dễ nhầm (đã biết trong dược học là "confusable drug names", ví dụ FDA có danh sách riêng
  cho vấn đề này).

### Ví dụ cụ thể (fine-tuned model, kaggle_rx)

| reference | hypothesis | top1_pred (gần nhất) | CER | Loại |
|---|---|---|---|---|
| Amodis | Amodin | Axodin | 0,167 | confusable_wrong_drug |
| Backtone | Baczin | Bicozin | 0,500 | confusable_wrong_drug |
| Bacmax | Bucmon | Baclon | 0,500 | confusable_wrong_drug |
| Beklo | Bexo | Fexo | 0,400 | confusable_wrong_drug |
| Ace | Aen | Tamen | 0,667 | hallucination_far_off |
| Bacaid | Berc | Beklo | 0,833 | hallucination_far_off |
| Aceta | Acata | Aceta | 0,200 | minor_ocr_noise_still_correct_drug (top1 vẫn đúng) |

Quan sát thêm: nhóm tên thuốc bắt đầu bằng "Bac-" (Bacaid, Backtone, Baclofen, Baclon, Bacmax) là điểm
nghẽn khó nhất — các thuốc này rất giống nhau về hình dạng chữ, cả zero-shot lẫn fine-tuned đều nhầm lẫn
nội bộ trong nhóm này, đúng bản chất bài toán "confusable drug names" nêu trên.

## 3. Kết quả — IAM (out-of-domain, chữ viết tay tổng quát)

| Mức lỗi (theo CER) | Zero-shot | Fine-tuned | Thay đổi |
|---|---|---|---|
| `correct` (exact match) | 57,25% | 25,75% | −31,5 điểm % |
| `minor_error` (0<CER≤0,3) | 6,75% | 13,50% | +6,8 điểm % |
| `moderate_error` (0,3<CER≤0,7) | 12,75% | **32,00%** | +19,3 điểm % |
| `severe_error` (CER>0,7) | 23,25% | 28,75% | +5,5 điểm % |

**Diễn giải:** khác với giả thuyết ban đầu ("model có thể vỡ hoàn toàn trên domain khác"), dữ liệu cho thấy
bức tranh **tinh vi hơn**: catastrophic forgetting ở đây **không phải sụp đổ toàn phần** (severe_error chỉ
tăng nhẹ +5,5 điểm %), mà chủ yếu là **một khối lớn từ trước đây đọc đúng tuyệt đối nay bị dịch sang lỗi
mức trung bình** (`moderate_error` tăng gần gấp 2,5 lần, từ 12,75% lên 32%). Nói cách khác: fine-tune trên
domain hẹp làm model "mất đi sự chính xác tuyệt đối" trên chữ viết tay tổng quát nhiều hơn là làm nó "hỏng
hoàn toàn" — khớp với quan sát median CER tăng từ 0 lên 0,429 đã nêu ở Phase 9.

## 4. Kiểm tra an toàn quan trọng: model fine-tuned có "rò rỉ" tên thuốc khi đọc chữ thường (IAM) không?

Đây là câu hỏi an toàn cụ thể: liệu việc fine-tune trên 78 tên thuốc có khiến model **hay "nhớ nhầm" ra tên
thuốc** ngay cả khi đang đọc một từ tiếng Anh thông thường không liên quan (một dạng lỗi rất đáng lo nếu
model này được dùng ngoài phạm vi đơn thuốc)? Kiểm tra: đếm số dự đoán trên IAM khớp *chính xác* với 1 trong
78 tên thuốc trong khi nhãn thật không phải vậy (CER>0,3 so với nhãn thật).

**Kết quả: chỉ 1/400 (0,25%)** — và trường hợp duy nhất đó là nhãn thật "a" (1 ký tự) bị đọc thành "az" (2
ký tự, trùng với tên thuốc "Az" trong vocab), nhiều khả năng là trùng hợp ngẫu nhiên với 1 từ ngắn mơ hồ hơn
là bằng chứng "rò rỉ" thật sự. Zero-shot có tỷ lệ nền là 0/400 (0%). **Kết luận: không có bằng chứng đáng kể
cho hiện tượng "rò rỉ tên thuốc"** — đây là một kết quả âm tính đáng báo cáo (loại trừ trước một mối lo ngại
an toàn cụ thể mà phản biện có thể nêu ra).

## 5. Tóm tắt cho bài báo

1. Fine-tuning dịch chuyển phân phối lỗi trên domain mục tiêu theo hướng tích cực rõ rệt: giảm mạnh lỗi
   "đoán bừa" (hallucination_far_off, −12,8 điểm %), nhưng tăng nhẹ tỷ lệ lỗi "giống thuốc khác thật"
   (confusable_wrong_drug, +1,9 điểm %) — cần thảo luận như một đánh đổi an toàn lâm sàng tinh vi, không chỉ
   nhìn CER trung bình.
2. Catastrophic forgetting trên IAM biểu hiện chủ yếu qua việc dịch chuyển khối lớn "đúng tuyệt đối" sang
   "lỗi mức trung bình", không phải sụp đổ hoàn toàn — một phát hiện có sắc thái hơn so với chỉ nhìn CER
   trung bình tăng ở Phase 7-9.
3. Không tìm thấy bằng chứng "rò rỉ" tên thuốc khi model đọc chữ viết tay tổng quát ngoài domain — kết quả
   âm tính hữu ích, giúp giới hạn phạm vi lo ngại về an toàn của phát hiện catastrophic forgetting.

## 6. Việc còn lại

- Phase 11 (tuỳ chọn): kiểm tra bổ sung RxHandBD.
- Phase 12-13: viết bài theo cấu trúc tạp chí (PeerJ Computer Science), nộp arXiv trước, sau đó nộp chính
  thức — tổng hợp toàn bộ Phase 1-10 làm phần Kết quả + Thảo luận.
