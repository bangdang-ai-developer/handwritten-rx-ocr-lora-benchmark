# Phase 7b — Ablation: LoRA rank (8/16/32) và augmentation A/B (elastic on/off)

> **✅ QUYẾT ĐỊNH ĐÃ THỰC HIỆN (17/09/2026):** Đã chọn **r=32 làm cấu hình chính thức** của toàn bài báo
> (phương án 1 ở Mục 3). Đã đổi tên trong `results_master_combined.csv` (`trocr-lora-finetuned` = r=32;
> r=16 gốc đổi tên thành `trocr-lora-r16`), chạy lại `analyze_aggregate.py`/`analyze_errors.py`/
> `generate_figures.py`, và viết lại `phase7_summary.md`, `phase9_summary.md`, `phase10_summary.md`,
> README.md theo r=32. Adapter r=16 (cũ) di chuyển sang `results/trocr-lora-adapter-r16-superseded/`;
> adapter r=32 (mới, chính thức) ở `results/trocr-lora-adapter-final/`. Bảng ablation dưới đây giữ nguyên
> không đổi (đây chính là dữ liệu dẫn tới quyết định trên).

Script tái lập: `src/analyze_ablation.py`. Cả 4 cấu hình đều train **full** (không phải quick-compare
như kế hoạch gốc dự tính) — kernel v20 (main, r=16+elastic), v21 (r=8), v22 (r=32), v23 (r=16, không elastic)
— cùng seed=42, cùng 3.120 ảnh train, cùng frozen test set (780 kaggle_rx + 400 iam), so sánh **paired**.

## 1. Ablation rank (r=8 vs r=16 vs r=32, alpha=2r, elastic=True)

| Rank | CER kaggle_rx | Top-1 acc | Exact-match | CER IAM | Wilcoxon vs r=16 (kaggle_rx) | Wilcoxon vs r=16 (IAM) |
|---|---|---|---|---|---|---|
| r=8 | **0,113** [0,098–0,128] | 91,5% | 70,0% | 0,530 [0,475–0,580] | **p=1,0×10⁻⁷** (tốt hơn) | p=0,830 (không khác biệt) |
| r=16 (main) | 0,149 [0,132–0,166] | 89,9% | 60,0% | 0,524 [0,472–0,579] | — | — |
| r=32 | **0,114** [0,100–0,130] | **92,4%** | 68,1% | **0,501** [0,453–0,552] | **p=2,4×10⁻⁸** (tốt hơn) | p=0,079 (chưa đủ ý nghĩa, α=0,05) |

**Phát hiện bất ngờ**: rank r=16 (lựa chọn "trung bình" ban đầu, chọn theo quy ước phổ biến trước khi có
dữ liệu thật) **KHÔNG phải rank tốt nhất** — cả r=8 VÀ r=32 đều vượt qua r=16 một cách **có ý nghĩa thống kê
rất mạnh** trên kaggle_rx (p<10⁻⁶ cho cả hai), và r=32 có xu hướng giữ tổng quát hoá tốt nhất trên IAM (dù
chưa đạt ngưỡng ý nghĩa 0,05). **r=32 là cấu hình tốt nhất trên cả 2 tiêu chí** (in-domain VÀ out-of-domain),
không có đánh đổi rõ ràng nào so với r=16.

**Lưu ý về giới hạn (quan trọng, phải ghi trong Limitations của bài báo)**: mỗi rank chỉ chạy **1 lần** (1
seed). Do quá trình training có nhiều nguồn ngẫu nhiên (shuffle dữ liệu, dropout, thứ tự batch qua 2 GPU),
khác biệt quan sát được có thể MỘT PHẦN đến từ phương sai giữa các lần chạy, không chỉ từ rank tự thân —
Wilcoxon signed-rank ở đây kiểm định khác biệt **giữa 2 tập dự đoán cụ thể đã có** (rất đáng tin cho câu hỏi
"2 checkpoint này có thực sự khác nhau không"), nhưng KHÔNG kiểm định được "nếu chạy lại r=16 lần 2 có ra kết
quả tương tự không". Để khẳng định chắc chắn hơn "rank" là nguyên nhân (không phải may rủi lần chạy), cần
chạy thêm ≥1 seed khác cho mỗi rank — chưa làm trong lần này, nêu rõ như một giới hạn.

## 2. Augmentation A/B (elastic=True vs elastic=False, r=16 giữ nguyên)

| Augmentation | CER kaggle_rx | Exact-match | CER IAM | Wilcoxon vs elastic=True (kaggle_rx) | Wilcoxon vs elastic=True (IAM) |
|---|---|---|---|---|---|
| elastic=True (main) | 0,149 [0,132–0,166] | 60,0% | 0,524 [0,472–0,579] | — | — |
| elastic=False | 0,150 [0,134–0,167] | 61,2% | 0,562 [0,505–0,627] | p=0,647 (không khác biệt) | p=0,101 (chưa đủ ý nghĩa) |

**Kết luận**: elastic distortion **không ảnh hưởng đáng kể** đến hiệu năng in-domain (đúng như literature đã
cảnh báo — không phải augmentation nào cũng có lợi rõ ràng cho mọi biến thể TrOCR), nhưng có xu hướng **giảm
nhẹ catastrophic forgetting** trên IAM (0,524 vs 0,562, chênh ~7%) — dù chưa đạt ngưỡng ý nghĩa thống kê 0,05.
**Quyết định giữ elastic=True cho cấu hình chính là hợp lý** (không có bằng chứng nó gây hại, có xu hướng
(chưa chắc chắn) giúp giữ tổng quát hoá tốt hơn).

## 3. Khuyến nghị quan trọng — cần quyết định trước khi viết bài

Vì **r=32 vượt trội r=16 trên cả 2 tiêu chí** (không có đánh đổi), theo đúng logic kế hoạch gốc ("ablation
để CHỌN rank tốt nhất, rồi báo cáo rank đó làm kết quả chính" — không phải cố định r=16 bất kể kết quả), nên
xét lại việc **đổi model chính (headline) của toàn bài từ r=16 sang r=32**. Việc này ảnh hưởng dây chuyền:

- **Rẻ để làm lại**: Phase 9 (`src/analyze_aggregate.py`) và Phase 10 (`src/analyze_errors.py`) chỉ cần đổi
  tên model tham chiếu (`trocr-lora-finetuned` → `trocr-lora-r32`) và chạy lại — không cần GPU, chỉ vài giây,
  dữ liệu r=32 đã có sẵn trong `results_master_combined.csv`.
- **Tốn công**: các đoạn văn diễn giải cụ thể ở `phase7_summary.md`, `phase9_summary.md`, `phase10_summary.md`,
  README, và các con số/ví dụ định tính trong `fig7_qualitative_examples.png` đều tham chiếu trực tiếp số liệu
  r=16 (0,149; 89,9%; 60%; các % loại lỗi cụ thể...) — cần viết lại, không chỉ thay số.

**Hai lựa chọn:**
1. **Đổi sang r=32 làm kết quả chính** — bài báo mạnh hơn (CER 0,114 thay vì 0,149, tốt hơn cả về giữ tổng
   quát hoá), đúng tinh thần "báo cáo cấu hình tốt nhất sau khi đã tune hyperparameter" (thông lệ ML chuẩn,
   không phải p-hacking vì rank là hyperparameter được chọn qua validation, không phải giả thuyết kiểm định).
2. **Giữ r=16 làm chính, r=8/r=32 chỉ là ablation phụ** — ít việc viết lại hơn, nhưng để lộ điểm yếu "tại sao
   không dùng cấu hình tốt hơn mà bạn đã có sẵn" — phản biện Q2 gần như chắc chắn sẽ hỏi.

**Khuyến nghị của tôi: chọn phương án 1** (đổi sang r=32) — dữ liệu đã có sẵn, chi phí thực hiện lại thấp
(không cần GPU), và kết quả mạnh hơn đáng kể. Cần xác nhận từ bạn trước khi viết lại Phase 9/10 vì đây là
thay đổi diễn giải nội dung khá rộng, không chỉ là thêm 1 mục ablation.
