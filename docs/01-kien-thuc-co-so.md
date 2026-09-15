# 1. Kiến Thức Nền Tảng Về OCR (Optical Character Recognition)

# Kiến Thức Nền Tảng Về OCR (Optical Character Recognition) — Tài Liệu Ôn Tập Cho Người Mới Bắt Đầu Nghiên Cứu

## Mở đầu

OCR (Optical Character Recognition — nhận dạng ký tự quang học) là một trong những lĩnh vực lâu đời nhất và vẫn còn rất "nóng" của thị giác máy tính (computer vision) và xử lý ngôn ngữ tự nhiên (NLP). Trước khi bắt tay vào một đề tài nghiên cứu khoa học về OCR, người mới cần nắm được bức tranh tổng thể: OCR là gì, nó gồm những bài toán con nào, pipeline xử lý ra sao, các thế hệ phương pháp đã phát triển như thế nào, và làm sao để đánh giá một hệ thống OCR một cách khoa học. Bài viết này tổng hợp các kiến thức nền tảng đó theo trình tự logic từ định nghĩa đến các mô hình hiện đại nhất.

## 1. Định nghĩa OCR và các bài toán liên quan

**OCR (Optical Character Recognition)** là quá trình chuyển đổi hình ảnh chứa văn bản (ảnh chụp, ảnh scan, ảnh cảnh tự nhiên...) thành văn bản dạng mã hóa (machine-encoded text) mà máy tính có thể đọc, chỉnh sửa và tìm kiếm được. Về bản chất, OCR là một bài toán ánh xạ từ không gian pixel sang không gian ký tự/từ.

Tuy nhiên, "OCR" trong nghiên cứu hiện đại thường được phân rã thành nhiều bài toán con, mỗi bài toán có đặc thù và phương pháp riêng:

- **Text detection**: xác định vị trí (bounding box, polygon, hoặc mask) của các vùng chứa chữ trong ảnh, nhưng chưa đọc nội dung. Đầu ra thường là tọa độ khung bao quanh dòng chữ hoặc từ.
- **Text recognition**: với một vùng ảnh đã được cắt ra (đã biết là có chữ), mô hình đọc và giải mã thành chuỗi ký tự văn bản.
- **End-to-end OCR**: kết hợp detection và recognition trong một pipeline (hoặc một mô hình duy nhất) để đi từ ảnh gốc ra thẳng văn bản, không cần bước trung gian thủ công.
- **Scene text recognition (STR)**: nhánh nhận dạng văn bản xuất hiện trong ảnh chụp cảnh tự nhiên (biển hiệu, nhãn sản phẩm, ảnh đường phố...) — khó hơn nhiều so với văn bản in trên nền trắng vì chữ có thể bị nghiêng, cong, mờ, bị che khuất, phông chữ trang trí, ánh sáng không đều.
- **Document layout analysis (DLA)**: phân tích cấu trúc bố cục của một trang tài liệu — xác định đâu là tiêu đề, đoạn văn, bảng biểu, hình ảnh, cột báo, footer/header — trước hoặc song song với việc đọc chữ.
- **Handwriting recognition (HTR/ICR)**: Handwritten Text Recognition (HTR) và Intelligent Character Recognition (ICR) là các bài toán nhận dạng chữ viết tay — khó hơn chữ in vì nét chữ biến thiên rất lớn giữa người viết, tốc độ viết, và kiểu chữ (chữ rời rạc hay chữ liền nét — cursive).
- **Key information extraction (KIE)**: sau khi đã có văn bản (và vị trí của nó), KIE trích xuất các trường thông tin có ý nghĩa (ví dụ: tên, ngày tháng, số tiền, mã số thuế trên hóa đơn) — đây là bước biến "đọc được chữ" thành "hiểu được nội dung".
- **Document understanding / Document AI**: khái niệm rộng hơn, bao trùm toàn bộ chuỗi xử lý tài liệu có cấu trúc — từ OCR, layout analysis, KIE, đến trả lời câu hỏi trên tài liệu (Document VQA), phân loại tài liệu, và tóm tắt.

Việc phân tách rõ các bài toán này rất quan trọng khi viết một đề cương nghiên cứu, vì mỗi bài toán có bộ dữ liệu, chỉ số đánh giá (metric) và các baseline khác nhau.

## 2. Pipeline OCR cổ điển

Một hệ thống OCR truyền thống (trước hoặc song song với deep learning) thường được chia thành các giai đoạn tuần tự sau:

**(1) Thu nhận ảnh (image acquisition):** ảnh đầu vào có thể đến từ máy scan, camera điện thoại, hoặc ảnh chụp cảnh tự nhiên. Chất lượng nguồn ảnh ảnh hưởng trực tiếp đến toàn bộ pipeline phía sau.

**(2) Tiền xử lý (pre-processing):** mục tiêu là chuẩn hóa ảnh để các bước sau hoạt động ổn định hơn, gồm:
- *Binarization*: chuyển ảnh xám/màu về ảnh nhị phân (đen-trắng) để tách chữ khỏi nền, thường dùng các thuật toán như Otsu hoặc adaptive thresholding.
- *Deskew*: xoay ảnh để chỉnh lại góc nghiêng của văn bản (do quét lệch hoặc chụp không thẳng).
- *Denoise*: khử nhiễu (nhiễu muối tiêu, vết bẩn, nhòe mực) để làm sạch tín hiệu chữ.
- *Contrast enhancement*: tăng cường độ tương phản giữa chữ và nền, đặc biệt quan trọng với ảnh chụp thiếu sáng hoặc ám màu.

**(3) Phân đoạn bố cục (layout segmentation):** chia ảnh trang thành các khối văn bản, sau đó phân tách tiếp thành dòng (line segmentation), từ (word segmentation), và cuối cùng là ký tự (character segmentation) — đây là bước rất dễ lỗi trong pipeline cổ điển, đặc biệt với chữ viết tay hoặc chữ dính liền nhau.

**(4) Trích xuất đặc trưng (feature extraction):** với mỗi ký tự/từ đã tách được, hệ thống trích ra các đặc trưng mô tả hình dạng — ví dụ: histogram of oriented gradients (HOG), profile chiếu (projection profile), số điểm giao cắt nét, tỷ lệ khung, các đặc trưng hình học/topo (số lỗ, số điểm cuối nét).

**(5) Phân loại/nhận dạng (classification):** dùng một bộ phân loại để ánh xạ vector đặc trưng sang nhãn ký tự cụ thể (ví dụ 'A', 'b', '7'...).

**(6) Hậu xử lý (post-processing):** dùng mô hình ngôn ngữ (language model) hoặc từ điển để sửa lỗi chính tả, loại bỏ các kết quả nhận dạng vô nghĩa, và tăng độ chính xác tổng thể của chuỗi văn bản đầu ra — ví dụ sửa "rnodel" thành "model" dựa trên xác suất n-gram hoặc từ điển hợp lệ.

Pipeline cổ điển này có ưu điểm là minh bạch, dễ debug từng bước, nhưng nhược điểm lớn là lỗi tích lũy qua từng giai đoạn (error propagation) — nếu phân đoạn ký tự sai thì bước nhận dạng gần như chắc chắn sai theo.

## 3. Các phương pháp OCR truyền thống (pre-deep-learning)

Trước khi deep learning phổ biến, nhận dạng ký tự chủ yếu dựa vào các kỹ thuật machine learning cổ điển:

- **Template matching**: so khớp trực tiếp ảnh ký tự cần nhận dạng với một tập "mẫu" (template) đã biết trước, tính độ tương đồng (thường bằng correlation hoặc khoảng cách pixel) và chọn mẫu gần nhất. Phương pháp này đơn giản, hiệu quả với phông chữ cố định (ví dụ chữ in một font) nhưng rất nhạy với biến dạng, xoay, tỷ lệ, và không mở rộng tốt sang nhiều font/chữ viết tay.
- **Feature-based classification**: thay vì so khớp pixel thô, trích xuất các đặc trưng bất biến (invariant features) rồi dùng bộ phân loại như k-Nearest Neighbors (k-NN), Decision Tree, hoặc Neural Network nông (shallow) để phân loại.
- **Hidden Markov Model (HMM)**: mô hình hóa chuỗi ký tự như một quá trình Markov ẩn, rất phù hợp với các bài toán có tính tuần tự như nhận dạng chữ viết tay liền nét (cursive handwriting) hoặc nhận dạng từ mà không cần phân đoạn ký tự tường minh trước — đây là một bước tiến quan trọng vì nó giảm phụ thuộc vào segmentation hoàn hảo.
- **Support Vector Machine (SVM)**: được dùng rộng rãi làm bộ phân loại ký tự nhờ khả năng phân tách tốt trong không gian đặc trưng chiều cao, đặc biệt hiệu quả khi kết hợp với các đặc trưng HOG hoặc Gabor filter.

Nhìn chung, các phương pháp truyền thống đòi hỏi thiết kế đặc trưng thủ công (hand-crafted features) rất công phu, và hiệu năng phụ thuộc mạnh vào chất lượng của bước tiền xử lý/phân đoạn.

## 4. Cách mạng deep learning trong OCR

Deep learning đã thay đổi hoàn toàn cách tiếp cận OCR bằng cách cho phép học đặc trưng trực tiếp từ dữ liệu (representation learning) thay vì thiết kế thủ công, đồng thời giải quyết được vấn đề cốt lõi: *segmentation-free recognition*.

- **CNN (Convolutional Neural Network) làm feature extractor**: CNN trích xuất bản đồ đặc trưng (feature map) từ ảnh, nắm bắt các mẫu hình cục bộ (nét, cạnh, góc) một cách phân cấp — thay thế hoàn toàn cho các đặc trưng thủ công như HOG.
- **RNN/LSTM/GRU cho sequence modeling**: vì văn bản có bản chất là chuỗi ký tự tuần tự, các mạng hồi quy (Recurrent Neural Network) như LSTM (Long Short-Term Memory) hoặc GRU (Gated Recurrent Unit) được dùng để mô hình hóa phụ thuộc ngữ cảnh giữa các ký tự liền kề trong một dòng ảnh.
- **CTC loss (Connectionist Temporal Classification)** — đây là bước ngoặt quan trọng nhất: CTC cho phép huấn luyện mô hình sequence-to-sequence mà **không cần biết trước alignment** (tức không cần biết chính xác ký tự nào tương ứng với vị trí pixel nào trong ảnh). CTC tự động tính tổng xác suất trên mọi cách gióng hàng (alignment) hợp lệ giữa chuỗi đặc trưng đầu vào và chuỗi nhãn đầu ra, dùng một ký tự "blank" đặc biệt để xử lý các trường hợp lặp/không phát âm. Nhờ đó, mô hình có thể học trực tiếp từ cặp (ảnh dòng chữ, chuỗi văn bản) mà không cần nhãn ở mức ký tự — giảm chi phí gán nhãn dữ liệu rất lớn.
- **Kiến trúc CRNN (Convolutional Recurrent Neural Network — Shi et al., 2016)**: là kiến trúc kinh điển kết hợp CNN (trích đặc trưng) + RNN hai chiều (BiLSTM, mô hình hóa ngữ cảnh) + CTC loss (giải mã chuỗi). CRNN trở thành baseline chuẩn cho text recognition trong nhiều năm vì đơn giản, hiệu quả, và không cần phân đoạn ký tự.
- **Attention-based encoder-decoder**: thay vì CTC, một hướng khác dùng cơ chế attention để mô hình decoder "nhìn" vào các phần khác nhau của ảnh khi sinh từng ký tự đầu ra tuần tự — giống mô hình dịch máy neural (neural machine translation). Cách này thường xử lý tốt hơn các trường hợp văn bản bị cong, xoay, hoặc có thứ tự đọc phức tạp.
- **Kiến trúc dựa trên Transformer và Vision Transformer (ViT)**: các mô hình gần đây thay thế RNN bằng self-attention (Transformer) để mô hình hóa quan hệ toàn cục giữa các phần tử trong chuỗi song song hóa tốt hơn, huấn luyện nhanh hơn, và ViT dùng cơ chế chia ảnh thành patch rồi áp Transformer trực tiếp lên chuỗi patch, giúp trích đặc trưng ảnh mà không cần convolution.

## 5. Text detection hiện đại

Các phương pháp deep learning cho text detection tập trung giải quyết bài toán phát hiện vùng chữ có hình dạng đa dạng (ngang, nghiêng, cong):

- **EAST (Efficient and Accurate Scene Text detector)**: một mô hình dự đoán trực tiếp (single-shot), dự đoán điểm ảnh nào thuộc vùng chữ và hồi quy trực tiếp ra hình học của bounding box (góc quay hoặc tứ giác) cho mỗi điểm, bỏ qua các bước trung gian như region proposal — nổi tiếng vì tốc độ nhanh và độ chính xác tốt.
- **CRAFT (Character Region Awareness For Text detection)**: thay vì dự đoán trực tiếp box của từ/dòng, CRAFT dự đoán bản đồ điểm số ở mức ký tự (character region score) và điểm liên kết giữa các ký tự (affinity score), sau đó nhóm chúng lại — cách tiếp cận này xử lý rất tốt văn bản có hình dạng bất kỳ (cong, biến dạng) vì không phụ thuộc vào hình học box cố định.
- **DBNet (Differentiable Binarization Network)**: cải tiến bước binarization (vốn không khả vi và khó tối ưu end-to-end) bằng một hàm xấp xỉ khả vi (differentiable approximation), cho phép học ngưỡng phân ngưỡng thích ứng ngay trong quá trình huấn luyện mạng — giúp tăng tốc độ suy luận đáng kể trong khi vẫn giữ độ chính xác cao, và trở thành một trong các baseline phổ biến nhất hiện nay. Các biến thể sau này (DBNet++) tiếp tục cải thiện khả năng biểu diễn đặc trưng đa tỷ lệ (multi-scale).

## 6. Text recognition hiện đại

Sau khi có vùng ảnh chứa chữ, các mô hình recognition hiện đại đã vượt xa CRNN cơ bản:

- **ASTER**: kết hợp một module rectification (dựa trên Spatial Transformer Network) để "làm phẳng" các vùng chữ bị cong/nghiêng trước khi đưa vào bộ mã hóa-giải mã có attention, giúp xử lý tốt scene text có hình dạng bất thường.
- **SATRN (Self-Attention Text Recognition Network)**: thay thế hoàn toàn RNN bằng self-attention hai chiều, mô hình hóa được cả phụ thuộc theo chiều ngang lẫn chiều dọc trong ảnh — hữu ích với chữ có bố cục phức tạp, xoay hoặc cong mạnh.
- **SVTR (Scene Text Recognition with a Single Visual Model)**: một kiến trúc "thuần thị giác" (visual-only), không cần module ngôn ngữ riêng, khai thác các khối trộn đặc trưng cục bộ và toàn cục (local/global mixing) lấy cảm hứng từ Vision Transformer, đạt tốc độ suy luận rất nhanh trong khi vẫn cạnh tranh về độ chính xác — phù hợp triển khai thực tế.
- **PARSeq (Permuted Autoregressive Sequence model)**: huấn luyện mô hình theo nhiều thứ tự sinh chuỗi khác nhau (permutation) trong cùng một kiến trúc, cho phép mô hình vừa giải mã theo kiểu autoregressive (tuần tự, tận dụng ngữ cảnh đã sinh) vừa có khả năng giải mã song song (non-autoregressive) khi cần tốc độ — là một trong các mô hình đạt kết quả tốt nhất (state-of-the-art) trên nhiều benchmark scene text gần đây.

Xu hướng chung của các mô hình recognition hiện đại là ngày càng dựa nhiều vào attention/transformer, giảm phụ thuộc vào RNN tuần tự (vốn khó song song hóa), và tích hợp ngữ cảnh ngôn ngữ ngay trong kiến trúc thị giác thay vì tách rời thành bước hậu xử lý riêng.

## 7. Document AI / hiểu tài liệu có cấu trúc

Khi bài toán mở rộng từ "đọc chữ" sang "hiểu tài liệu" (document understanding), một hướng nghiên cứu riêng đã hình thành:

- **LayoutLM family (LayoutLM, LayoutLMv2, LayoutLMv3)**: mở rộng mô hình ngôn ngữ kiểu BERT bằng cách đưa thêm thông tin vị trí không gian (2D position embedding) và đặc trưng hình ảnh của từng token văn bản, cho phép mô hình học đồng thời nội dung chữ, vị trí trên trang, và hình ảnh xung quanh — rất mạnh cho các bài toán như KIE và phân loại tài liệu.
- **Donut (Document understanding transformer)** và **Pix2Struct**: đại diện cho hướng **"OCR-free"** — mô hình đọc trực tiếp ảnh tài liệu bằng kiến trúc encoder-decoder hình ảnh-văn bản (image-to-text), sinh ra chuỗi kết quả (có thể là JSON có cấu trúc) mà **không cần một OCR engine riêng biệt** ở bước tiền xử lý.
- **TrOCR (Transformer-based OCR)**: dùng một encoder ảnh (image Transformer, ví dụ ViT/BEiT đã pretrain) kết hợp với một decoder văn bản (text Transformer, ví dụ RoBERTa/GPT đã pretrain), tận dụng sức mạnh của các mô hình pretrained hai phía để nhận dạng chữ, đặc biệt hiệu quả với cả chữ in lẫn chữ viết tay.
- **Vì sao hướng "OCR-free" đang trở thành xu hướng**: pipeline OCR truyền thống (detect → crop → recognize → parse layout → extract field) có nhiều bước rời rạc, mỗi bước là một nguồn lỗi tiềm ẩn (error propagation), khó tối ưu end-to-end, và tốn chi phí gán nhãn ở nhiều mức (box, ký tự, trường thông tin). Các mô hình OCR-free coi toàn bộ bài toán là một phép ánh xạ ảnh → chuỗi đầu ra mong muốn, học end-to-end bằng một mục tiêu duy nhất, đơn giản hóa pipeline triển khai và thường tổng quát hóa tốt hơn sang các định dạng tài liệu mới lạ.

## 8. Các mô hình ngôn ngữ đa phương thức (multimodal LLM) và OCR chuyên biệt mới

Sự bùng nổ của các mô hình ngôn ngữ lớn đa phương thức (multimodal Large Language Model) đã mở ra một hướng tiếp cận OCR hoàn toàn mới: dùng một mô hình nền tảng tổng quát để "đọc" ảnh như một khả năng phụ trong tập kỹ năng rộng hơn, thay vì huấn luyện một mô hình OCR chuyên biệt.

- **GPT-4V, Gemini, Qwen-VL**: các mô hình vision-language tổng quát này có thể đọc văn bản trong ảnh khá tốt nhờ được huấn luyện trên khối lượng dữ liệu ảnh-văn bản khổng lồ, đồng thời có khả năng suy luận (reasoning) trên nội dung đọc được — ví dụ trả lời câu hỏi về một biểu đồ hoặc tóm tắt nội dung một trang tài liệu — điều mà OCR truyền thống không làm được.
- **GOT-OCR2.0**: một mô hình OCR chuyên biệt thế hệ mới ("OCR-2.0"), được thiết kế để xử lý đa dạng loại đầu vào (văn bản thường, công thức toán, bảng biểu, nốt nhạc, biểu đồ...) trong một kiến trúc thống nhất, nhắm tới việc trở thành một "OCR engine tổng quát" thế hệ tiếp theo.
- **Kosmos-2.5**: mô hình đa phương thức của Microsoft được thiết kế chuyên cho các tác vụ liên quan đến ảnh văn bản, có thể sinh ra văn bản kèm thông tin không gian (markdown có cấu trúc layout) trực tiếp từ ảnh tài liệu.
- **Nougat (Neural Optical Understanding for Academic Documents)**: mô hình OCR-free chuyên biệt cho tài liệu khoa học, được huấn luyện để chuyển ảnh trang PDF của bài báo khoa học thành văn bản định dạng markup (bao gồm công thức toán LaTeX, bảng biểu) — giải quyết một điểm yếu lớn của OCR truyền thống là không xử lý tốt công thức toán học phức tạp.

Điểm cần lưu ý khi nghiên cứu: các mô hình multimodal LLM tổng quát thường mạnh về suy luận ngữ nghĩa nhưng có thể kém chính xác hơn OCR chuyên biệt ở mức ký tự (character-level fidelity), đặc biệt với văn bản dày đặc, số liệu, hoặc ngôn ngữ ít phổ biến — đây là một hướng đánh giá thực nghiệm rất đáng làm cho một đề tài nghiên cứu.

## 9. Các chỉ số đánh giá (evaluation metrics)

Đánh giá một hệ thống OCR cần tách riêng theo từng bài toán con:

**Cho text recognition:**
- **CER (Character Error Rate)**: tỷ lệ lỗi ở mức ký tự, tính bằng công thức `CER = (S + D + I) / N`, trong đó S (substitution — số ký tự bị thay thế sai), D (deletion — số ký tự bị thiếu), I (insertion — số ký tự thừa) được tính từ **edit distance** (khoảng cách Levenshtein) giữa chuỗi dự đoán và chuỗi ground truth, còn N là tổng số ký tự trong ground truth. CER càng thấp càng tốt.
- **WER (Word Error Rate)**: tương tự CER nhưng tính ở mức từ thay vì ký tự — thường nghiêm khắc hơn CER vì chỉ cần sai một ký tự trong từ là cả từ đó bị tính sai.
- **Edit distance**: là nền tảng toán học cho cả CER và WER, đo số phép biến đổi tối thiểu (thêm/xóa/thay) để biến chuỗi A thành chuỗi B.

**Cho text detection:**
- **Precision**: tỷ lệ các vùng dự đoán đúng là chữ trên tổng số vùng được dự đoán.
- **Recall**: tỷ lệ các vùng chữ thật sự được phát hiện đúng trên tổng số vùng chữ có trong ground truth.
- **F1-score**: trung bình điều hòa (harmonic mean) của precision và recall, thường dùng làm chỉ số tổng hợp duy nhất để so sánh các phương pháp detection — một vùng dự đoán thường được tính là "đúng" (true positive) khi độ chồng lấp (IoU — Intersection over Union) với ground truth vượt một ngưỡng cho trước (ví dụ 0.5).

**Cách diễn giải chung**: không có chỉ số nào là "đủ" một mình — một hệ thống có Recall cao nhưng Precision thấp nghĩa là phát hiện được nhiều chữ thật nhưng cũng báo nhầm nhiều vùng không phải chữ; CER thấp nhưng WER cao nghĩa là lỗi tuy nhỏ ở mức ký tự nhưng rải đều khắp các từ. Khi viết báo cáo nghiên cứu, nên báo cáo đồng thời nhiều chỉ số và làm rõ điều kiện đo (ví dụ ngưỡng IoU, có chuẩn hóa chữ hoa/thường hay không, có loại bỏ dấu câu khi tính CER hay không) vì các quy ước đo này khác nhau khá nhiều giữa các bài báo.

## 10. Các bộ dữ liệu/benchmark phổ biến

Việc chọn đúng benchmark là yếu tố quan trọng để định vị đề tài nghiên cứu so với state-of-the-art:

- **ICDAR series** (ICDAR 2013, 2015, 2017, 2019...): chuỗi cuộc thi và bộ dữ liệu benchmark lâu đời nhất và có ảnh hưởng lớn nhất trong cộng đồng OCR, bao phủ cả text detection, recognition, và nhiều bài toán con khác qua các năm (ICDAR2015 nổi tiếng với scene text nghiêng/incidental).
- **IIIT5K**: bộ dữ liệu scene text recognition với 5.000 ảnh từ, thu thập từ Google Image Search, là benchmark chuẩn cho bài toán cropped word recognition.
- **SVT (Street View Text)**: ảnh chữ thu thập từ Google Street View, đặc trưng bởi chất lượng ảnh thấp và nhiễu nền phức tạp.
- **COCO-Text**: bộ dữ liệu văn bản cảnh tự nhiên quy mô lớn, xây dựng trên nền ảnh của MS COCO, dùng cho cả detection và recognition.
- **SROIE**: benchmark trích xuất thông tin từ hóa đơn bán lẻ (scanned receipts), tiêu chuẩn phổ biến cho bài toán KIE.
- **FUNSD**: bộ dữ liệu cho bài toán hiểu form (form understanding), gán nhãn quan hệ giữa các trường (ví dụ cặp "key-value") trên tài liệu dạng biểu mẫu.
- **CORD**: bộ dữ liệu hóa đơn tiếng Indonesia có gán nhãn chi tiết theo cấu trúc phân cấp, dùng phổ biến cho KIE trên tài liệu bán cấu trúc.
- **DocVQA**: benchmark hỏi-đáp trực tiếp trên ảnh tài liệu (Document Visual Question Answering), đòi hỏi mô hình vừa đọc được chữ vừa suy luận vị trí/ngữ cảnh để trả lời — rất phù hợp đánh giá các mô hình document understanding hiện đại.
- **Các benchmark mới hơn**: có thể kể thêm **Union14M** (tập hợp dữ liệu scene text recognition quy mô lớn, đa dạng nguồn, nhằm khắc phục tình trạng benchmark cũ đã gần bão hòa về độ chính xác), **HierText** (benchmark có gán nhãn phân cấp dòng/đoạn/khối cho layout), và các benchmark tài liệu đa trang, đa ngôn ngữ hướng tới đánh giá khả năng của mô hình OCR-free/multimodal LLM trên toàn bộ trang tài liệu phức tạp thay vì chỉ từng từ đơn lẻ.

## Kết luận

Tổng quan trên cho thấy OCR đã đi qua một hành trình phát triển rõ rệt: từ các phương pháp thống kê/hình học cổ điển dựa trên đặc trưng thủ công, qua thời kỳ CRNN và CTC loss giải quyết bài toán alignment, đến các kiến trúc attention/transformer hiện đại, và gần đây nhất là xu hướng hợp nhất toàn bộ pipeline vào các mô hình OCR-free hoặc tận dụng năng lực đọc-hiểu sẵn có của các mô hình đa phương thức tổng quát. Đối với một dự án nghiên cứu khoa học mới bắt đầu, việc nắm vững bản đồ khái niệm này — biết phân biệt bài toán con nào cần giải quyết, phương pháp nào phù hợp, và benchmark/metric nào dùng để đo lường — là bước chuẩn bị quan trọng trước khi đi vào thiết kế thực nghiệm hay đề xuất phương pháp mới.
