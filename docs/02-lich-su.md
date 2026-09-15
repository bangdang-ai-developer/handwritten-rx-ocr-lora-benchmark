# Lịch Sử Phát Triển Của Công Nghệ Nhận Dạng Ký Tự Quang Học (OCR)

## Mở đầu

OCR (Optical Character Recognition – Nhận dạng ký tự quang học) là công nghệ cho phép máy móc "đọc" được chữ viết từ hình ảnh hay tài liệu giấy, chuyển nó thành văn bản số mà máy tính có thể xử lý, tìm kiếm và chỉnh sửa. Từ những ý tưởng sơ khai cách đây hơn một thế kỷ, OCR đã trải qua nhiều giai đoạn phát triển: từ các thiết bị cơ-điện thô sơ, qua thời kỳ nhận dạng mẫu (pattern matching) và thống kê, cho đến cuộc cách mạng học sâu (deep learning) và ngày nay là các mô hình ngôn ngữ đa phương thức khổng lồ. Bài viết này điểm lại hành trình đó theo dòng thời gian.

## Cuối thế kỷ 19 – đầu thế kỷ 20: Những ý tưởng khai sinh

Ý tưởng "máy đọc chữ giúp người khiếm thị" xuất hiện rất sớm, gắn liền với mong muốn nhân đạo hơn là mục tiêu xử lý dữ liệu. Năm 1913-1914, nhà vật lý người Ireland **Edmund Fournier d'Albe** phát minh ra **optophone** – một thiết bị dùng tế bào quang điện (photosensor) quét qua từng ký tự in và chuyển hình dạng chữ thành các âm thanh có cao độ khác nhau để người khiếm thị "nghe" ra chữ. Đây được xem là một trong những nỗ lực đầu tiên biến tín hiệu quang học của chữ viết thành thông tin mà con người (hoặc sau này là máy) có thể diễn giải được – tiền thân về mặt ý tưởng của OCR hiện đại, dù bản thân nó chưa "nhận dạng" ký tự theo nghĩa tự động hoàn toàn.

Cùng thời kỳ này, các nhà phát minh khác như **Emanuel Goldberg** ở Đức cũng nghiên cứu các hệ thống quét và tìm kiếm ký tự trên vi phim (microfilm) vào thập niên 1920, đặt nền móng cho ý tưởng lưu trữ và truy xuất văn bản bằng máy.

## Thập niên 1920-1930: Gustav Tauschek và "Lesemaschine"

Bước tiến quan trọng tiếp theo đến từ kỹ sư người Áo **Gustav Tauschek**. Năm 1929, ông nộp bằng sáng chế cho một thiết bị gọi là **"Lesemaschine"** (tiếng Đức nghĩa là "máy đọc"). Thiết bị này sử dụng các mẫu (template) quang học đặt chồng lên hình ảnh ký tự cần nhận dạng: nếu ánh sáng xuyên qua khớp với một mẫu ký tự cụ thể, máy sẽ "nhận ra" đó là ký tự nào. Đây chính là kỹ thuật **template matching** – một trong hai nguyên lý nền tảng của OCR (bên cạnh trích xuất đặc trưng – feature extraction – ra đời sau này). Tauschek sau đó tiếp tục đăng ký nhiều bằng sáng chế liên quan tại Mỹ vào đầu thập niên 1930, mở rộng ý tưởng sang cả nhận dạng số và ký tự in.

## Thập niên 1950: OCR thương mại đầu tiên

Giai đoạn hậu Thế chiến II chứng kiến OCR bước từ phòng thí nghiệm ra thị trường. Năm 1951, kỹ sư người Mỹ **David Shepard** chế tạo ra **"Gismo"**, một trong những máy đọc ký tự đầu tiên có khả năng chuyển văn bản in thành mã điện báo (telegraph code) mà máy đánh chữ có thể tái tạo lại. Cùng năm đó, Shepard thành lập công ty **Intelligent Machines Research Corporation (IMRC)** – được coi là công ty OCR thương mại đầu tiên trên thế giới. IMRC bán các hệ thống đọc tài liệu cho khách hàng lớn như Reader's Digest (đọc số liệu bán hàng) và hãng dầu Standard Oil. Năm 1954, IMRC bị Farrington Manufacturing (sau này gắn liền với tên tuổi trong ngành thẻ tín dụng và bán lẻ) mua lại, đưa công nghệ này vào ứng dụng thương mại rộng hơn.

## Thập niên 1960-1970: Chuẩn hóa font, ứng dụng ngân hàng – bưu chính, và Ray Kurzweil

Đây là giai đoạn OCR "trưởng thành" về mặt công nghiệp.

- **Chuẩn hóa phông chữ máy đọc được**: Vì các máy OCR thời kỳ đầu chỉ đọc chính xác một kiểu chữ cố định, ngành công nghiệp đã phát triển các bộ font chuyên dụng: **OCR-A** (ra đời khoảng 1968, chuẩn hóa bởi ANSI năm 1970) với các ký tự có hình dạng đơn giản, dễ phân biệt cho máy nhưng khá "cứng" với mắt người; và **OCR-B** (do Adrian Frutiger thiết kế, chuẩn hóa bởi ECMA/ISO năm 1973) mềm mại, dễ đọc hơn với con người. Hai font này được dùng phổ biến trên hộ chiếu, vé máy bay, hóa đơn.
- **Ngành ngân hàng**: song song đó, ngành ngân hàng phát triển công nghệ **MICR (Magnetic Ink Character Recognition)** – nhận dạng ký tự bằng mực từ tính, in ở đáy tấm séc (chuẩn E-13B ra đời cuối thập niên 1950, phổ biến rộng suốt thập niên 1960-1970). MICR không hẳn là OCR quang học thuần túy nhưng có vai trò tương tự trong tự động hóa xử lý chứng từ tài chính.
- **Ray Kurzweil và Omni-font OCR**: Bước đột phá lớn nhất giai đoạn này là công của nhà phát minh **Ray Kurzweil**. Thay vì chỉ đọc được vài font cố định, ông phát triển công nghệ **"omni-font OCR"** – có khả năng nhận dạng ký tự in ở hầu như bất kỳ kiểu chữ nào. Năm 1976, kết hợp công nghệ này với bộ tổng hợp giọng nói (text-to-speech) do công ty của ông phát triển, Kurzweil cho ra mắt **Kurzweil Reading Machine** – cỗ máy đọc sách thành tiếng đầu tiên dành cho người khiếm thị, được ca ngợi là phát minh quan trọng nhất kể từ máy in Gutenberg đối với người mù. Danh ca mù Stevie Wonder là một trong những khách hàng đầu tiên và cũng là người ủng hộ nhiệt thành cho sản phẩm này.

## Thập niên 1980-1990: Máy tính cá nhân, phần mềm thương mại và mầm mống Tesseract

Sự phổ biến của **máy quét (scanner) cho máy tính cá nhân** trong thập niên 1980 đã đưa OCR đến gần hơn với người dùng phổ thông, không còn giới hạn trong các hệ thống công nghiệp đắt tiền.

- Nhiều phần mềm OCR thương mại ra đời để chạy trên PC, nổi bật nhất là **OmniPage** của công ty **Caere Corporation** (ra mắt cuối thập niên 1980), từng là tiêu chuẩn vàng trong ngành trong nhiều năm.
- Tại châu Âu, công ty **ABBYY** được thành lập năm 1989 tại Nga (khi đó còn là Liên Xô, dưới tên gọi ban đầu khác trước khi đổi thành ABBYY), và sau này phát triển sản phẩm **FineReader** – một trong những phần mềm OCR chính xác nhất, đặc biệt mạnh với các ngôn ngữ đa dạng và tài liệu phức tạp.
- Đáng chú ý, dự án **Tesseract** – sau này trở thành công cụ OCR mã nguồn mở phổ biến nhất thế giới – khởi nguồn từ năm **1985 tại phòng thí nghiệm Hewlett-Packard (HP)**, phát triển liên tục đến khoảng năm 1994-1995 rồi bị dừng lại. Nó "ngủ đông" hơn một thập kỷ trước khi HP và Đại học Nevada, Las Vegas hợp tác đưa mã nguồn ra công khai, và đến năm **2005 Google tiếp nhận, phát triển tiếp và mở mã nguồn (open-source)** hoàn toàn, biến Tesseract thành nền tảng cho vô số ứng dụng OCR miễn phí sau này.
- Giai đoạn này cũng chứng kiến sự phát triển của **ICR (Intelligent Character Recognition)** – công nghệ nhận dạng chữ viết tay, phức tạp hơn nhiều so với đọc chữ in vì nét chữ mỗi người mỗi khác. ICR được ứng dụng mạnh trong xử lý biểu mẫu (form processing), đặc biệt là ngành bưu chính (nhận dạng địa chỉ viết tay để phân loại thư) và ngân hàng (đọc số tiền viết tay trên séc).

## Thập niên 2000: Phương pháp thống kê và OCR di động sơ khai

Trong thập niên 2000, các kỹ thuật nhận dạng dựa trên **mô hình thống kê**, đặc biệt là **Hidden Markov Model (HMM)** – vốn rất thành công trong nhận dạng giọng nói – được áp dụng sang OCR, đặc biệt hữu ích cho nhận dạng chữ viết tay liền mạch (cursive handwriting) và chữ viết của nhiều ngôn ngữ không có ranh giới ký tự rõ ràng. Đây cũng là thời kỳ manh nha của **OCR trên thiết bị di động**: khi điện thoại thông minh bắt đầu có camera chất lượng khá hơn, các ứng dụng quét danh thiếp, quét văn bản đơn giản bắt đầu xuất hiện, dù độ chính xác và tốc độ còn hạn chế so với các máy quét chuyên dụng.

## Từ 2012 trở đi: Cách mạng học sâu

Sự kiện **AlexNet** chiến thắng cuộc thi ImageNet năm 2012 đã tạo ra làn sóng học sâu (deep learning) lan tỏa sang mọi lĩnh vực thị giác máy tính, và OCR không ngoại lệ. Các **mạng nơ-ron tích chập (CNN)** thay thế các đặc trưng thủ công (hand-crafted features) trong việc trích xuất hình ảnh ký tự. Với văn bản chuỗi dài, mô hình kết hợp **RNN/LSTM với hàm mất mát CTC (Connectionist Temporal Classification)** giải quyết vấn đề căn chỉnh (alignment) giữa ảnh và chuỗi ký tự mà không cần phân đoạn từng ký tự thủ công. Tiêu biểu là kiến trúc **CRNN (Convolutional Recurrent Neural Network)** do Shi và cộng sự công bố năm 2015, trở thành nền tảng cho vô số hệ thống nhận dạng văn bản trong ảnh (scene text recognition) sau này.

Song song với nhận dạng, bài toán **phát hiện văn bản trong ảnh tự nhiên (text detection)** – tức xác định vùng nào trong ảnh có chữ trước khi đưa vào nhận dạng – cũng phát triển mạnh với các mô hình như **EAST** (2017), **CRAFT** (Character Region Awareness for Text detection, 2019), và **DBNet** (Differentiable Binarization, 2019-2020), giúp OCR xử lý tốt các trường hợp khó như biển hiệu, ảnh chụp nghiêng, chữ cong.

## Từ 2017 trở đi: Attention, Transformer và Document AI

Cơ chế **attention** và kiến trúc **Transformer** (giới thiệu năm 2017) nhanh chóng được đưa vào OCR, giúp mô hình "tập trung" vào từng phần của ảnh khi giải mã từng ký tự, cải thiện đáng kể độ chính xác với văn bản dài và bố cục phức tạp.

Lĩnh vực **Document AI** (hiểu tài liệu có cấu trúc) cũng bùng nổ:
- **LayoutLM** (Microsoft, 2020) là mô hình tiên phong kết hợp thông tin văn bản, vị trí (layout) và hình ảnh để hiểu tài liệu như hóa đơn, biểu mẫu.
- **TrOCR** (Microsoft, 2021) áp dụng kiến trúc Transformer end-to-end (encoder ảnh – decoder văn bản) cho bài toán OCR, đơn giản hóa pipeline truyền thống vốn cần nhiều bước riêng lẻ (phát hiện, nhận dạng, hậu xử lý).
- **Donut** (2022, do Naver phát triển) là bước ngoặt quan trọng theo hướng **"OCR-free document understanding"** – mô hình hiểu trực tiếp nội dung tài liệu từ ảnh mà không cần một bước OCR tường minh riêng biệt, dự báo trước xu hướng sau này.

## Công cụ và nền tảng thương mại/mã nguồn mở quan trọng

Bên cạnh nghiên cứu học thuật, thị trường OCR ứng dụng cũng phát triển sôi động: **Google Cloud Vision OCR**, **AWS Textract**, **Microsoft Azure Form Recognizer** (nay là **Azure AI Document Intelligence**) là các dịch vụ đám mây phổ biến cho doanh nghiệp. Về phía mã nguồn mở, **PaddleOCR** do Baidu phát hành năm 2020 nổi bật nhờ hỗ trợ đa ngôn ngữ tốt (bao gồm cả tiếng Việt), nhẹ và dễ triển khai, trở thành lựa chọn hàng đầu cho nhiều nhà phát triển.

## 2023-2026: Kỷ nguyên mô hình ngôn ngữ đa phương thức tổng quát

Giai đoạn gần đây nhất chứng kiến sự trỗi dậy của các **mô hình ngôn ngữ lớn đa phương thức (multimodal LLM)** như **GPT-4V**, **Gemini**, **Qwen-VL**... Những mô hình này không được huấn luyện chuyên biệt cho OCR nhưng lại có khả năng "đọc hiểu" hình ảnh chứa văn bản như một năng lực phụ trợ, thậm chí đọc được cả bảng biểu, biểu đồ, chữ viết tay trong ngữ cảnh phức tạp – điều mà OCR truyền thống thường phải xử lý qua nhiều bước riêng lẻ.

Đồng thời, các mô hình OCR/hiểu tài liệu chuyên biệt mới tiếp tục ra đời để cạnh tranh về độ chính xác và hiệu năng, như **Kosmos-2.5** (Microsoft), **Nougat** (Meta, chuyên đọc tài liệu học thuật có công thức toán), và **GOT-OCR2.0** – được quảng bá là mô hình OCR "thế hệ 2.0" thống nhất nhiều tác vụ (đọc chữ in, công thức, bảng, nốt nhạc...) trong một kiến trúc duy nhất. Xu hướng chung của giai đoạn này là **"OCR-free document understanding"** – dần xóa nhòa ranh giới giữa bước "nhận dạng ký tự" và bước "hiểu nội dung", hướng tới các hệ thống end-to-end vừa đọc vừa suy luận trên tài liệu.

## Kết luận

Từ chiếc optophone chuyển ánh sáng thành âm thanh của Fournier d'Albe hơn một thế kỷ trước, đến các mô hình đa phương thức khổng lồ hôm nay có thể "đọc" và "hiểu" tài liệu cùng lúc, OCR đã đi một chặng đường dài phản chiếu chính lịch sử phát triển của trí tuệ nhân tạo: từ cơ khí – quang học, qua nhận dạng mẫu và thống kê, đến học sâu và cuối cùng là các mô hình nền tảng tổng quát. Điều thú vị là mục tiêu ban đầu và nhân văn nhất của công nghệ này – giúp người khiếm thị "đọc" được chữ viết – vẫn luôn là sợi chỉ xuyên suốt, từ Kurzweil Reading Machine năm 1976 cho đến các trợ lý AI đa phương thức có thể mô tả và đọc văn bản trong ảnh ngày nay.