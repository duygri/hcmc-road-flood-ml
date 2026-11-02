# Thiết kế mốc đầu tiên — susceptibility tại TP.HCM cũ

Ngày: 2026-11-02. Trạng thái: ĐÃ DUYỆT (Approved) - Sẵn sàng lập Implementation Plan.

## Mục tiêu và quyết định đã có

Người dùng chọn TP.HCM trước mở rộng địa giới năm 2025. Mục tiêu là pipeline GIS có thể chạy lại, từ nguồn thô tới bảng feature/nhãn theo đoạn đường, baseline MCDA và báo cáo chất lượng/đánh giá không gian. SPEC.md do người dùng cung cấp là yêu cầu gốc; tài liệu này cụ thể hóa mốc đầu tiên. Các lựa chọn mặc định bên dưới do người thiết kế đề xuất và đã được người dùng duyệt cùng tài liệu.

Nguồn ranh giới: data/raw/HCMC_new_limits.gpkg, chọn duy nhất giá trị GhiChu = "TP. Ho CHi Minh"; kiểm tra duy nhất một bản ghi trước khi dùng. Không chọn theo thứ tự dòng. Lưu checksum nguồn và bộ lọc trong config. Hình học hiện hợp lệ, diện tích chiếu khoảng 2112.12 km²; đây là diện tích tính từ bản đồ nguồn, không phải xác nhận diện tích hành chính chính thức. 418/425 điểm IRD được polygon covers; 7 điểm ngoài được giữ nguyên, gắn cờ và đưa vào báo cáo review. Không dịch tọa độ hay mở rộng AOI để thu nhận các điểm này.

## Cách tiếp cận

Theo SPEC đã chọn pipeline Python với CLI, cấu hình YAML và các bảng trung gian có hợp đồng rõ ràng. Logic nằm trong src/flood_ml; các script chỉ đọc config, gọi module và ghi kết quả. Notebook dùng khảo sát và trình bày kết quả. Một notebook chứa toàn bộ pipeline sẽ khó chạy lại và kiểm tra; bắt đầu classifier ngay không phù hợp với nguồn chỉ có bằng chứng ngập. Vì vậy mốc này giữ Track A bắt buộc; Track B/C là các mốc kế tiếp có điều kiện.

## Phạm vi

Bao gồm môi trường Python >=3.11 với lock file; cấu hình nguồn/AOI; xác thực nguồn; chuẩn hóa đường và điểm ngập; tạo ít nhất năm nhóm feature theo SPEC; nhãn có provenance và hàng đợi review; MCDA, độ nhạy trọng số, báo cáo QC và đánh giá không gian khi có đủ nhãn được duyệt.

Không có API, frontend, realtime hoặc dự báo ngập theo giờ. Điểm MCDA 0–100 là thứ hạng tương đối. Không tạo nhãn âm từ thiếu báo cáo; không dùng flood_count hoặc thuộc tính sinh từ nguồn nhãn làm X; không tối ưu trọng số trên nhãn test.

## Dữ liệu và không gian

Giữ nguyên data/raw. Mọi đường dẫn trong config là tương đối với repo; fail khi thiếu file, sai checksum, thiếu CRS hoặc CRS không có đơn vị mét trước phép đo. CRS làm việc đề xuất là EPSG:9210, cùng hệ của điểm ngập và ranh giới nguồn. Raster WGS84 được xử lý theo grid chiếu mét đã công bố; categorical raster dùng nearest-neighbor, DEM dùng nội suy đã cấu hình. Lưu resolution, transform, resampling và vertical datum; không diễn giải DEM thành cao độ khảo sát mặt đường.

AOI dùng polygon thật, không dùng bounding box thay ranh giới. Các đoạn đầu ra bị clip theo polygon; dữ liệu bối cảnh feature được lấy thêm halo bằng bán kính lân cận lớn nhất. Bốn tile FABDEM hiện có và tile WorldCover được kiểm tra footprint/valid mask cho AOI cộng halo trước khi tính. Tính slope trên grid DEM có halo, rồi lấy thống kê theo đoạn.

Lớp nước IRD không được mặc định đầy đủ ngoài ranh giới. Những buffer cần bối cảnh ngoài phạm vi nguồn được gắn water_context_incomplete; feature nước liên quan được coi là thiếu, không coi diện tích trống là tỷ lệ nước bằng 0. Báo cáo coverage và số đường không chấm điểm vì thiếu dữ liệu. Nếu thiếu nhiều, bổ sung thủy hệ từ OSM là quyết định dữ liệu kế tiếp có provenance riêng; không ghép âm thầm trong mốc này.

Manifest hiện có cần chuẩn hóa thành một schema nhất quán, tách rõ D04 thủy hệ dù cùng dataset IRD. Mọi tài sản có SHA-256 được tính lại từ file, không sao chép chuỗi hash bằng tay. Các ghi chú giấy phép IRD mâu thuẫn được giữ; raw/model artifact hạn chế giấy phép vẫn ở local.

## Đường và feature

Đề xuất giữ đường cho giao thông mặt đất có highway thuộc motorway/trunk/primary/secondary/tertiary, các *_link tương ứng, residential, unclassified, living_street, service và road. Loại proposed/construction, footway, steps, path, cycleway và đối tượng không phải đường. Bridge/tunnel được giữ provenance và gắn cờ; mặc định chưa chấm MCDA cho các đoạn này vì DEM không mô tả mặt cầu/hầm. Config ghi rõ chính sách này.

Giữ mapping osm_way_id và topology theo OSM node; không nối khác mức chỉ vì đường cắt nhau trên bản đồ. Tách tại nút giao đã biết, clip AOI, rồi chia đoạn mục tiêu 150 m, dài tối đa 200 m; giữ các đoạn ngắn do giao lộ/ranh giới. ID ổn định trong cùng snapshot và cùng config; thay snapshot/config không hứa giữ nguyên ID. Thuật toán tạo ID và hướng hình học canonical phải được tài liệu hóa.

Feature đầu tiên: median/p10 cao độ, cao độ tương đối với lân cận đề xuất 500 m, slope mean, khoảng cách tới nước và tỷ lệ built-up trong 250 m. Tính thêm water/vegetation ratio khi có coverage. Đề xuất buffer lấy mẫu DEM 30 m; mọi bán kính có thể cấu hình. Các tỷ lệ dùng số pixel hợp lệ làm mẫu số, đi kèm valid_fraction. Một dòng/road_segment_id, join một-một, tách feature khỏi label/provenance.

## Quan sát và nhãn

Chuẩn hóa tên cột, khoảng trắng và các biến thể mã; giữ giá trị nguồn cùng cờ sửa, không tự sửa dữ kiện vị trí/thời gian. Giữ độ chính xác ngày/năm; thiếu ngày không được gán ngày giả. Dedup chỉ tạo nhóm ứng viên theo vị trí/thời gian/nguồn, cho review trường hợp mơ hồ; không gộp mọi báo cáo cùng đường thành một sự kiện.

Bảng matching lưu tất cả ứng viên trong ngưỡng cấu hình, khoảng cách, tên đường, spatial precision, phương pháp và review_required. Điểm mức đường/khu vực không tự thành positive cho một segment gần nhất. Mặc định nhãn unknown cho tới khi ghép được xác nhận có bằng chứng; bảng review do người dùng/người có chuyên môn xác nhận mới cho phép chuyển positive. Không có verified_negative từ nguồn hiện tại. Lưu source ID, event_group_id, lý do và quyết định review để truy vết.

## MCDA và đánh giá

Công bố config trọng số và chiều rủi ro trước khi đánh giá. Đề xuất bộ chính dùng năm nhóm: relative elevation, absolute elevation, slope, distance to water và built-up, mỗi nhóm 0.2; p10/median không được tính đôi cùng nhóm cao độ. Đây là giả thuyết baseline, không phải quan hệ vật lý đã kiểm chứng. Chuẩn hóa theo phân vị trên vùng tham chiếu train đã cố định, lưu tham số, áp dụng sang holdout; score 0–100 và percentile_rank là hai cột khác nhau.

Giữ nguyên phân đoạn không đủ feature với score null và lý do, không tự bù hoặc tái phân phối trọng số từng dòng. So sánh độ nhạy bằng các bộ trọng số công bố trước, trên cùng tập segment đủ dữ liệu và cùng normalization; báo cáo rank correlation và top-k overlap.

Spatial blocks tách train/reference và holdout theo config, có khoảng đệm chống rò rỉ từ feature neighborhood; nhóm sự kiện/nhóm nguồn trùng không tràn qua partitions. Khi review chưa tạo đủ positive độc lập, báo cáo evaluation_status = pending_label_review, kèm số lượng thiếu; vẫn có thể xuất baseline và sensitivity. Khi đủ, đánh giá held-out event capture/top-k% chiều dài đường, denominator chỉ gồm tập đủ coverage đã công bố; so với random và elevation-only. Không báo Accuracy/F1/ROC-AUC như phân loại trên positive/unknown. Không gọi hoàn tất đánh giá nếu chưa có nhãn được duyệt.

## Đầu ra và tái lập

configs/{sources,study_area,features,labeling,experiments}.yaml; data/interim/{study_area,road_segments,flood_observations}.gpkg; road_flood_matches và review_candidates; data/processed/{road_features,road_labels,analysis_dataset,road_susceptibility}.parquet. Các file hình học Parquet phải có metadata GeoParquet nếu chứa geometry.

reports chứa source/QC/coverage, bảng review, bản đồ offline, sensitivity và evaluation status; artifacts chứa rule config và normalization parameters. Mỗi run lưu run_id, config snapshot, seed, commit/worktree dirty state, package lock và checksum inputs. Ghi output theo thư mục run, không ghi đè raw; lỗi trả exit code khác 0 và không để file output dở được coi là hoàn tất.

## Kiểm chứng và điều kiện nghiệm thu

Dùng fixture GIS nhỏ chạy offline cho CRS/unit, segmentation tại giao lộ, raster nodata/halo, matching mơ hồ và join cardinality. Kiểm tra ổn định ID, không gán 0 cho unknown, feature allowlist chống leakage, preprocessing train-only và splits không chồng nhóm. CI chạy format/lint/test. Kiểm tra dữ liệu thật với source checksum, coverage, ratio [0,1], unique IDs, geometry validity và số điểm ngoài AOI.

Mốc pipeline đạt khi chạy lại được từ nguồn đã tải với config AOI cũ, xuất feature/labels/QC và MCDA có sensitivity; trạng thái evaluation phải trung thực về review còn thiếu. Mốc đánh giá khoa học chỉ đạt khi đủ positive đã review để chạy holdout. Python launcher hiện chỉ có 3.8/2.7, nên cần môi trường >=3.11 riêng và không thay Python hệ thống.

## Để người triển khai quyết định

Phiên bản package tương thích và lock, kích thước spatial blocks/guard theo dữ liệu và bán kính feature, ngưỡng candidate matching thận trọng, cấu trúc helper nội bộ, batch size và runtime tuning. Các lựa chọn phải nằm trong config/report; không thay đổi AOI, chính sách nhãn hoặc diễn giải score khi tối ưu.

## Làm rõ sau builder review

Đề xuất gate vận hành để chạy báo cáo đánh giá: ít nhất 20 event_group_id độc lập đã review trong holdout, trải trên ít nhất 3 spatial blocks với tối thiểu 5 event/block, và tập reference có ít nhất 20 sự kiện độc lập ở các blocks khác. Đây là mức tối thiểu cho một báo cáo thăm dò, không phải bảo đảm đủ lực thống kê; mọi báo cáo vẫn ghi exploratory, công bố số event/block, coverage, uncertainty theo bootstrap nhóm sự kiện/block. Nếu không đạt, giữ pending_label_review và báo thiếu theo từng điều kiện; không coi phần đánh giá là nghiệm thu. Không dùng nguồn ID báo chí làm định nghĩa sự kiện độc lập; reviewer xác nhận event_group bằng chứng vị trí/thời gian.

Định nghĩa capture chính: mỗi event đã review có tập đoạn ảnh hưởng được xác nhận và chiều dài hợp của chúng. Event được capture tại top-k nếu ít nhất 50% chiều dài ảnh hưởng xác nhận nằm trong tập đường được chọn. Báo thêm sensitivity với ngưỡng any/100%; bản ghi chỉ có địa điểm mơ hồ hoặc thiếu tập đoạn xác nhận không thuộc denominator, phải báo số bị loại. Nhiều lần ngập thực sự khác nhau trên cùng đường có thể là các event riêng, nhưng các nhóm cùng vị trí/đường trùng phải thuộc cùng partition để không rò rỉ train-test. Không tính nhiều nguồn đưa tin cùng sự kiện thành nhiều event. Công bố tie-break theo road_segment_id và tỷ lệ chiều dài thực tế được chọn nếu ngân sách top-k cắt qua một segment.

