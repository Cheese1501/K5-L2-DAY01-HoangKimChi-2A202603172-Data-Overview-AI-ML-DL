# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:**  Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id` = 468; `class_name` = cab; `rank` = 1; `score` = 0.510915; `taxonomy_name` = ImageNet-1K
2. Record này là dự đoán cho toàn bộ bức ảnh, không phải nhãn riêng cho một vật thể. Mô hình phân loại ảnh xếp hạng các lớp có thể mô tả ảnh tổng thể; record không có bounding box hay polygon xác định vị trí một vật thể cụ thể
3. Danh sách lớp đến từ taxonomy đi kèm checkpoint đã huấn luyện, ở đây là ImageNet-1K. Mô hình không tự tạo tên lớp mới cho từng ảnh
4. Cần lưu class_id, class_name và taxonomy_name để định danh lớp rõ ràng, tránh nhầm khi tên lớp giống nhau hoặc cách viết/phiên bản taxonomy khác nhau. class_id là mã lớp, class_name là tên dễ đọc, còn taxonomy_name cho biết danh sách lớp đang được dùng
5. Guideline cần xác định chủ đề chính của ảnh, tiêu chí ưu tiên khi có nhiều vật thể, và cách xử lý ảnh có nhiều chủ đề ngang nhau hoặc không thuộc lớp phù hợp. Các annotator cần áp dụng cùng một quy tắc thay vì tự chọn theo cảm tính
6. score chỉ thể hiện mức độ mô hình nghiêng về dự đoán đó trong lần chạy này; nó không chứng minh nhãn là đúng. Ground truth phải được tạo theo guideline và bằng chứng trong ảnh, có thể cần reviewer kiểm tra

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=dining table ; `score`: 0.894425; `bbox_xyxy`=[0.0,18.65,478.73,638.83]; `bbox_width`=478.73 ; `bbox_height`=320.18
2. bbox_xyxy có dạng [x_min, y_min, x_max, y_max], tính theo pixel và lấy gốc tọa độ (0, 0) ở góc trên-trái ảnh. Góc trên-trái của hộp là (x_min, y_min), góc dưới-phải là (x_max, y_max). Chiều rộng bằng x_max - x_min, chiều cao bằng y_max - y_min. Kích thước ảnh kitchen cần được đọc từ trường image_width và image_height của JSON
3. Số prediction ở ba ngưỡng trong notebook:
- Ngưỡng 0.20: 17 prediction.
- Ngưỡng 0.35: 11 prediction.
- Ngưỡng 0.60: 6 prediction.
Hạ ngưỡng từ 0.60 xuống 0.20 giữ lại nhiều prediction hơn; tăng ngưỡng sẽ loại bỏ các prediction có score thấp hơn ngưỡng.
4. Khi hạ ngưỡng, reviewer phải kiểm tra thêm các hộp có khả năng sai, trùng lặp, gán nhầm lớp, hoặc chỉ bao một phần vật thể. Reviewer cũng cần rà soát vật thể bị bỏ sót; không được mặc định rằng mọi hộp model tạo ra đều đúng.
5. Quy tắc box đề xuất: Vẽ một hộp chữ nhật nhỏ nhất có thể bao trọn phần vật thể nhìn thấy, không cố ý lấy thêm nền; nếu vật thể bị cắt ở mép ảnh thì hộp dừng tại mép ảnh, và áp dụng nhất quán cho mọi ảnh theo guideline.
6. Ở mép trái ảnh kitchen, chỉ nhìn thấy một phần bàn tay/cánh tay, còn phần lớn cơ thể nằm ngoài khung hình. Model dự đoán person với score khoảng 0.61. Nếu guideline chưa quy định cách gán nhãn người bị cắt mép, tôi sẽ hỏi Lab Coach (escalation) để thống nhất có gán nhãn hay không và cách vẽ box, tránh tự suy đoán phần cơ thể ngoài ảnh.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-003 ; `class_name`=bowl ; `score`=0.698698 ; số điểm polygon=`polygon_point_count`=70 ; 3–5 điểm đầu của `polygon_xy`=[84.0,289.0],[83.0,290.0],[81.0,290.0]
2. Bounding box chỉ cho biết hình chữ nhật bao quanh vật thể. Polygon cung cấp các điểm tọa độ bám theo đường biên, nên mô tả hình dạng và phạm vi vật thể sát hơn, đặc biệt khi vật thể không có hình chữ nhật hoặc có đường viền không đều.
3. instance_id phân biệt từng đối tượng riêng trong output của bài lab, ví dụ kitchen-001. Nó không phải class_id vì nhiều instance có thể cùng một lớp; cũng không phải tracking ID vì bài lab xử lý ảnh tĩnh, không theo dõi cùng một đối tượng qua nhiều khung hình/video.
4. Quy tắc mask đề xuất: Vẽ polygon theo đường biên của phần vật thể nhìn thấy, dùng đủ điểm để theo sát hình dạng nhưng không thêm nền; không tự suy diễn phần bị che khuất. Áp dụng cùng tiêu chí cho các ảnh và chuyển trường hợp biên không rõ sang reviewer/escalation.
5. Ở chiếc bát có nhãn bowl 0.74 phía trước bàn, mép dưới của bát tiếp giáp với mặt bàn. Đây là vị trí cần phân biệt rõ vùng thuộc chiếc bát và vùng thuộc mặt bàn để tránh mask lấy thừa nền. Nếu guideline đã quy định vẽ theo biên nhìn thấy, tôi áp dụng quy tắc đó. Nếu một đoạn biên không đủ rõ và guideline chưa hướng dẫn cách xử lý, tôi sẽ hỏi Lab Coach (escalation) để thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một `class_id`/`class_name` cho toàn bộ ảnh, theo taxonomy và guideline đã thống nhất. | Ảnh `traffic` có top-1 là `cab` với score `0.510915`; top-1 khác với một số lớp trực quan có thể kỳ vọng. Đây là prediction của mô hình, không phải bằng chứng ground truth. | Xác định chủ đề theo guideline; dùng taxonomy cố định; đánh dấu hoặc escalation nếu ảnh không phù hợp lớp nào hay có nhiều chủ đề khó phân định. | Kiểm tra lớp có phù hợp toàn ảnh không, taxonomy có đúng không, và quy tắc ưu tiên chủ đề có được áp dụng nhất quán không. |
| Phát hiện vật thể | Một lớp và một bounding box `xyxy` theo pixel cho mỗi vật thể cần gán nhãn. | Với ảnh `kitchen`, số prediction thay đổi từ 17 ở ngưỡng 0.20 xuống 6 ở ngưỡng 0.60; ngưỡng thấp có thể giữ nhiều hộp sai hơn, ngưỡng cao có thể bỏ sót vật thể. | Gán nhãn mọi vật thể thuộc phạm vi theo guideline, không dùng ngưỡng score của model làm quy tắc tạo ground truth; kiểm tra vật thể nhỏ, bị che hoặc cắt mép. | Kiểm tra lớp, vị trí/kích thước hộp, hộp trùng, vật thể bị bỏ sót và tính nhất quán giữa các ảnh. |
| Instance segmentation | Một lớp, một `instance_id` và một polygon/mask riêng cho từng vật thể. | Biên vật thể có thể mờ, hai vật có thể chạm nhau hoặc một phần bị che; polygon của model chỉ là prediction để tham khảo. | Vẽ biên phần nhìn thấy theo guideline; giữ từng instance riêng, kể cả khi cùng lớp; escalation nếu không xác định được ranh giới. | Kiểm tra polygon bám biên, không lấy nhiều nền, không bỏ phần nhìn thấy, instance riêng biệt và `instance_id` duy nhất. |

## 5. An toàn dữ liệu

1. Chỉ sử dụng ảnh công khai được chỉ định trong bài lab, không tải lên hoặc đưa vào output dữ liệu cá nhân, dữ liệu nội bộ hay thông tin nhạy cảm
2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi, em sẽ dừng xử lý và báo Lab Coach/mentor phụ trách để được hướng dẫn, không tự ý tiếp tục hoặc chia sẻ dữ liệu

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
