# Thiết kế P4 — Track B presence–background tại TPHCM cũ

Ngày: 2026-11-03. Trạng thái: Hướng kiến trúc đã được người dùng duyệt; bản design spec chờ review.

## Mục tiêu

Thêm một thử nghiệm presence–background có thể tái lập để xếp hạng tương đối các đoạn đường theo độ phù hợp với các positive đã được review. Thử nghiệm chỉ áp dụng cho vùng TPHCM cũ và dùng chung các đầu vào, spatial partitions và capture evaluation của baseline MCDA hiện tại.

Đầu ra là `relative_score` phục vụ nghiên cứu và so sánh. Kết quả không phải xác suất ngập, tỷ lệ ngập dân số hay nhãn phân loại ngập/không ngập.

Hiện chưa có reviewed positive trong dữ liệu thật. Có thể hoàn thành module, CLI và kiểm chứng bằng synthetic fixtures trước; chỉ fit dữ liệu thật khi gate review reference đạt yêu cầu tại mục **Gate dữ liệu**.

## Phương án đã chọn

Chọn Logistic Regression theo kiểu sampled presence–background, với một nhóm background lấy từ các road segment đang mang trạng thái `unknown`. Cách này phù hợp với pipeline hiện có, dễ tái lập theo seed và so sánh trực tiếp với MCDA.

Các phương án đã cân nhắc:

1. **Sampled-background Logistic Regression — chọn:** phương pháp đơn giản, kết quả xếp hạng có thể kiểm tra và phù hợp với dữ liệu hiện tại.
2. **nnPU hoặc PU risk estimator:** cần giả định thêm về tỷ lệ positive và cơ chế phát hiện/báo cáo sự kiện; dữ liệu hiện có chưa hỗ trợ bảo vệ các giả định đó.
3. **Positive-only ranking:** giảm phụ thuộc vào tập background nhưng chưa phù hợp bằng cho pilot cần đối chiếu với các baseline đang có.

Background chỉ là nhóm tham chiếu được lấy mẫu cho phép fit. Road trong nhóm này tiếp tục là `unknown`; không ghi `label_binary=0`, không chuyển thành `verified_negative` và không diễn giải đầu ra của Logistic Regression như xác suất.

## Hợp đồng dữ liệu và chống rò rỉ

- Feature set đầu tiên cố định bằng đúng năm feature của MCDA: `elevation_median_m`, `relative_elevation_m`, `slope_mean_deg`, `dist_water_m`, `builtup_ratio_250m`. Việc dùng chung feature giữ phép so sánh giữa phương pháp học và baseline dễ diễn giải.
- Positive fit chỉ lấy từ road có `label_status=positive`, provenance review hợp lệ và thuộc partition `reference`.
- Background candidates chỉ lấy từ road có `label_status=unknown`, đủ năm feature hữu hạn, score-eligible và thuộc partition `reference`.
- Loại khỏi cả hai tập các road ngoài reference, guard/excluded, thiếu feature bắt buộc, bị loại theo structural score policy, hoặc nằm trong pool label khác `positive`/`unknown`.
- Holdout positive chỉ dùng cho đánh giá. Holdout unknown không được lấy làm background. Spatial block map và partition hiện hành được giữ nguyên; không chọn lại block theo số nhãn hoặc kết quả mô hình.
- Các nguồn cột nhãn/evidence như `label_*`, `flood_*`, `source_count`, event/observation IDs không được đưa vào X.
- Scaling và mọi preprocessing được fit chỉ trên tập positive/background của từng lần fit ở reference. Không tuning tham số theo holdout.

## Lấy mẫu background và fit

- Đơn vị fit và lấy mẫu là `road_segment_id`.
- Mỗi lần fit lấy background không hoàn lại, phân tầng theo `block_id` trong các block reference. Quota block được xác định tất định theo kích thước pool background đủ điều kiện; seed và số mẫu từng block được ghi vào manifest.
- Tỷ lệ chính là 1 positive : 10 background. Phân tích nhạy cảm dùng 1:5 và 1:20; mỗi tỷ lệ chạy 20 seed liên tiếp từ 42 đến 61. Nếu pool không đủ số road duy nhất cho một tỷ lệ, lần fit đó không tạo model và báo `background_pool_insufficient`.
- Model cố định là `StandardScaler` rồi `LogisticRegression` trong cùng sklearn Pipeline. Feature order, tỷ lệ, seed và tham số model nằm trong config snapshot. Không tìm siêu tham số trên holdout.
- Dùng `decision_function` làm `relative_score`; không cần phát hành `predict_proba`.
- Mỗi fit lưu run identity, feature list/hash, seed, positive/background counts, block quotas, ID/hash của input, partition fingerprint, tham số và phiên bản thư viện. Artifact/model outputs tuân theo chính sách ignore và giới hạn giấy phép dữ liệu của repo.

## Tích hợp và đánh giá

Thêm module training và CLI riêng để chạy Track B trên artifacts đã tạo bởi pipeline hiện hữu. Chạy Track B không làm thay đổi các stage bắt buộc của Track A/MCDA. Model scores được đưa vào evaluator dưới dạng các phương pháp có tên; evaluator dùng chung quy tắc và cùng score-eligible holdout road population cho Track B, MCDA, elevation-only và random.

Đánh giá dùng frozen spatial holdout và complete reviewed affected geometry như capture evaluator hiện tại. Báo cáo:

- held-out event capture theo top-k% chiều dài mạng đường, dùng các ngưỡng đã cấu hình trong baseline;
- lift so với random và capture so với elevation-only/MCDA;
- phân bố capture/lift và overlap top-k giữa seed và background ratio;
- số event/block, score coverage, seed, ratio, counts, exclusions, input hashes và trạng thái gate.

Chỉ đưa các event có provenance review hợp lệ và affected geometry đã được review đầy đủ vào capture denominator. Báo riêng số event bị loại và lý do. Không báo cáo accuracy, precision, recall, F1, ROC-AUC hay calibration như phân loại ngập-vs-không-ngập trên dữ liệu positive–unlabeled. Không dùng holdout để chọn feature, ratio hay model.

## Gate dữ liệu

- **Gate fit dữ liệu thật:** tối thiểu 20 `event_group_id` độc lập, được review và thuộc reference; mỗi event có ít nhất một positive road đủ feature sau khi áp dụng eligibility policy, và các nhóm phải qua audit partition hiện có. Khi chưa đạt, không tạo model thật và trả trạng thái `pending_positive_review` cùng các thiếu hụt.
- **Gate capture:** giữ nguyên gate evaluator hiện tại: tối thiểu 20 event reference, 20 event holdout, ít nhất 3 holdout blocks với ít nhất 5 event/block; tất cả event đưa vào denominator phải có affected geometry được review và complete, cùng coverage score bắt buộc của evaluator.
- Gate capture là điều kiện để mở báo cáo capture có điều kiện; mọi metric vẫn được ghi là exploratory và không đại diện tỷ lệ ngập toàn dân số. Đạt gate fit không đồng nghĩa gate capture đã đạt.
- Không tạo hoặc suy diễn verified-negative trong bất kỳ gate nào. Track C supervised classification nằm ngoài phạm vi P4.

## Kiểm chứng và nghiệm thu

Implementation plan sẽ thêm kiểm chứng offline trên fixtures để xác nhận:

- unknown chỉ được dùng như background trong lần fit và nhãn gốc không đổi;
- holdout/guard/excluded và verified-negative không lọt vào tập fit hoặc background;
- block sampling và model fit tái lập theo cùng seed/config;
- scaler chỉ fit trên mẫu reference, không đọc thống kê holdout;
- thiếu positive hoặc background pool trả trạng thái pending/failure có lý do, không xuất model/metric dở dang;
- điểm `decision_function` được xuất dưới tên relative score và evaluator giữ nguyên gate, coverage, random, elevation-only, MCDA cùng capture rules.

Không yêu cầu download thêm dữ liệu hay chạy fit thật để hoàn tất code path. Kết quả thực nghiệm trên dữ liệu thật vẫn phụ thuộc việc review đủ positive.

## Ngoài phạm vi

- TPHCM sau mở rộng địa giới, realtime hoặc cảnh báo vận hành.
- Tạo verified-negative, Track C supervised classification hoặc nhãn âm giả.
- nnPU, neural network, tree boosting, thêm feature ngoài allowlist hoặc tìm feature/siêu tham số theo holdout.
- Hiệu chỉnh xác suất, tuyên bố causal, hoặc suy rộng từ positive báo chí thành tỷ lệ ngập toàn thành phố.
- Thay đổi AOI, segmentation, frozen blocks, baseline MCDA hoặc review gate hiện tại.
