# TECHNICAL SPECIFICATION — HCMC Road Flood Susceptibility ML

**Document ID:** HCMC-FLOOD-ML-SPEC-001  
**Version:** 1.0.0 (MVP research specification)  
**Date:** 2026-11-02  
**Repository:** `hcmc-road-flood-ml`  
**Status:** Proposed — implementation not started  
**Language:** Tiếng Việt  

> **Tuyên bố phạm vi:** Repo này nghiên cứu và xây dựng mô hình đánh giá **mức độ dễ ngập** (flood susceptibility) của từng **đoạn đường** tại TP.HCM từ dữ liệu GIS, địa hình và lịch sử ghi nhận ngập. **Không** khẳng định dự báo chính xác vị trí/thời điểm ngập trước 1 giờ; **không** cung cấp xác suất ngập thực tế khi chưa hiệu chỉnh và xác thực nhãn.

---

## 1. Tổng quan

### 1.1. Bối cảnh

Dữ liệu ngập có thời điểm theo giờ và nhãn không ngập đã xác minh ở TP.HCM hiện chưa đủ để huấn luyện mô hình dự báo trước 60 phút. Phiên bản hiện tại chuyển sang đánh giá **nguy cơ ngập tương đối giữa các đoạn đường**, dùng các biến địa lý tương đối ổn định và bằng chứng ngập trong quá khứ. Kết quả phục vụ nghiên cứu, không thay thế thông tin cảnh báo chính thức.

### 1.2. Mục tiêu

1. Xây dựng pipeline có thể chạy lại từ file dữ liệu gốc tới bảng feature theo đoạn đường.
2. Chuẩn hóa, ghép không gian các điểm ngập lịch sử và ghi nhận chất lượng nhãn.
3. Tạo baseline GIS đa tiêu chí (MCDA), có phân tích độ nhạy trọng số.
4. Thử nghiệm ranking bằng presence–background hoặc positive–unlabeled (PU) **nếu** dữ liệu phù hợp; đánh giá đúng giới hạn phương pháp.
5. Chỉ huấn luyện Random Forest/XGBoost dạng phân loại có giám sát khi có mẫu âm được xác minh độc lập.
6. Xuất dữ liệu, điểm nguy cơ tương đối, báo cáo, cấu hình và artifact model có thể tái lập.

### 1.3. Ngoài phạm vi (v1)

- Dự báo ngập theo giờ hoặc khẳng định sẽ ngập trong 60 phút.
- Dự báo độ sâu nước, thời gian rút nước, dẫn đường tránh ngập.
- Backend, REST API, frontend, app di động, hệ thống cảnh báo thời gian thực.
- Thu thập/lưu trữ dữ liệu người dùng hoặc triển khai production.

### 1.4. Đơn vị phân tích

- **Đơn vị:** `road_segment_id` (một đoạn đường), mục tiêu chiều dài 100–200 m; giữ nguyên các phân đoạn ngắn tại giao lộ khi cần.
- **Hình học:** `LineString`/`MultiLineString`, cắt bởi ranh giới nghiên cứu.
- **Không gian nghiên cứu:** phần TP.HCM được lựa chọn và mô tả trong cấu hình; bắt đầu với khu vực/đoạn có khả năng đối chiếu dữ liệu ngập, không mặc định toàn bộ địa giới hành chính mở rộng năm 2025.
- **Phân giải:** không tuyên bố độ chính xác vị trí tốt hơn độ phân giải nguồn DEM (~30 m), độ chính xác điểm ngập, hoặc độ phủ OSM.

## 2. Nguồn dữ liệu

| ID | Dataset | Định dạng / tính chất | Vai trò | Mức ưu tiên |
|---|---|---|---|---|
| D01 | Historical database of urban flooding in HCMC (IRD/DataSuds; 425 ghi nhận 2002–2026) | GPKG, TSV; điểm ngập và thuộc tính nguồn | Positive evidence, phân tích điểm nóng | P0 |
| D02 | OpenStreetMap Vietnam (Geofabrik) | `.osm.pbf` hoặc `.gpkg.zip` | Đường và phân đoạn đường | P0 |
| D03 | FABDEM Vietnam | Raster ~30 m; DEM đã giảm sai lệch do nhà/cây | Cao độ, độ dốc, địa hình tương đối | P0 |
| D03b | Copernicus DEM GLO-30 | DSM ~30 m | DEM dự phòng / so sánh độ nhạy | P1 |
| D04 | `HCMC_water.gpkg` thuộc bộ IRD | GPKG, sông/kênh/vùng nước | Khoảng cách tới thủy hệ | P0 |
| D05 | ESA WorldCover 2021 | GeoTIFF 10 m | Land cover, chỉ số built-up | P0 |
| D06 | JRC Global Surface Water | Raster 30 m, occurrence/seasonality | Đặc trưng mặt nước lịch sử (không phải nhãn ngập đường) | P1 |
| D07 | Open-Meteo Historical Weather | Dữ liệu mưa theo giờ từ reanalysis | Khí hậu mưa tổng hợp, phân tích phụ | P2 |

**Nguồn chính thức / trang truy cập:**

- IRD: https://dataverse.ird.fr/dataset.xhtml?persistentId=doi:10.23708/8Y16HU
- OSM Geofabrik: https://download.geofabrik.de/asia/vietnam.html
- FABDEM: https://research-information.bris.ac.uk/en/datasets/fabdem-2/
- Copernicus DEM: https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM
- ESA WorldCover: https://esa-worldcover.org/en/data-access
- JRC GSW v1.4: https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater
- Open-Meteo Historical: https://open-meteo.com/en/docs/historical-weather-api

**Bắt buộc:** mỗi dataset lưu `source_url`, phiên bản, ngày tải, thời kỳ dữ liệu, CRS, giấy phép, attribution, độ phân giải, phạm vi, checksum SHA-256 và ghi chú giới hạn trong `data/raw/manifest.yaml`. Không tự động phân phối lại file thô khi giấy phép không cho phép.

**Lưu ý bản quyền:** IRD: CC BY-NC 4.0; FABDEM: CC BY-NC-SA 4.0; OSM: ODbL và yêu cầu ghi công, xem xét điều kiện chia sẻ với dữ liệu phái sinh. Repo mặc định nghiên cứu/phi thương mại. Không mặc định các nguồn có giấy phép giống nhau.

## 3. Thiết kế dữ liệu

### 3.1. File trung gian & đầu ra

| File | Khóa | Nội dung |
|---|---|---|
| `data/interim/road_segments.gpkg` | `road_segment_id` | Đoạn đường chuẩn hóa, hình học, metadata OSM |
| `data/interim/flood_observations.gpkg` | `observation_id` | Điểm ngập IRD đã chuẩn hóa, nguồn, độ chính xác |
| `data/interim/road_flood_matches.parquet` | `observation_id`, `road_segment_id` | Bảng ghép và độ tin cậy ghép |
| `data/processed/road_features.parquet` | `road_segment_id` | Feature không bao gồm nhãn/nguồn nhãn |
| `data/processed/road_labels.parquet` | `road_segment_id` | Positive / verified-negative / unknown |
| `data/processed/analysis_dataset.parquet` | `road_segment_id` | Feature + nhãn + provenance (không được lẫn vào X) |
| `reports/metrics/*.json` | `run_id` | Metrics, config, seed, thông tin split |
| `reports/figures/*` | — | Bản đồ kiểm định, phân bố feature, biểu đồ hiệu năng |
| `artifacts/` | — | Baseline rules / model + metadata xuất ra |

### 3.2. Schema: `road_segments`

| Field | Type | Bắt buộc | Ý nghĩa |
|---|---|---|---|
| `road_segment_id` | string | Có | ID ổn định, không trùng trong snapshot |
| `osm_way_id` | string/null | Có | ID gốc nếu có |
| `road_name` | string/null | Không | Tên đường |
| `highway_type` | category | Có | Phân loại OSM |
| `length_m` | float | Có | Chiều dài theo CRS mét |
| `geometry` | LineString | Có | Đường dạng vector |
| `geometry_crs` | string | Có | CRS của file/snapshot |
| `osm_snapshot_date` | date | Có | Phiên bản dữ liệu OSM |
| `study_area_id` | string | Có | Vùng nghiên cứu |

### 3.3. Schema: `flood_observations`

| Field | Type | Ý nghĩa |
|---|---|---|
| `observation_id` | string | Khóa bản ghi / hash ổn định |
| `observed_date` | date/null | Ngày ngập theo nguồn (độ chính xác có thể chỉ theo ngày) |
| `geometry` | Point | Tọa độ đã chuẩn hóa |
| `location_accuracy_m` | float/null | Sai số định vị nếu nguồn hỗ trợ |
| `spatial_precision` | string/null | Độ chính xác mô tả của nguồn |
| `source_id` | string | ID nguồn gốc |
| `source_type` | category | Báo chí / báo cáo / quan trắc / khác |
| `cause` | string/null | Mưa / triều / kết hợp / không rõ |
| `depth_min_cm`, `depth_max_cm` | float/null | Độ sâu được báo cáo (nếu có) |
| `event_group_id` | string | Nhóm nguồn/bản ghi trùng cùng sự kiện |
| `quality_flag` | category | high / medium / low / review |

Không suy ra `location_accuracy_m` từ độ dài chữ số tọa độ. Khi nguồn chỉ xác định vị trí mức tên đường, không ép điểm đó thành nhãn chính xác cho 1 đoạn 100 m.

### 3.4. Feature v1 (dự kiến)

| Feature | Type | Nguồn | Ghi chú |
|---|---|---|---|
| `elevation_median_m` | float | FABDEM | Median mẫu trên buffer đoạn |
| `elevation_p10_m` | float | FABDEM | Phân vị thấp, đánh dấu nhiễu/nodata |
| `relative_elevation_m` | float | FABDEM | Chênh lệch với lân cận trong bán kính cấu hình |
| `slope_mean_deg` | float | FABDEM | Tính từ raster, không coi là độ dốc khảo sát mặt đường |
| `dist_water_m` | float | IRD water | Khoảng cách hình học ngắn nhất, CRS mét |
| `water_area_ratio_250m` | float [0,1] | IRD water | Tỷ lệ diện tích nước trong buffer 250 m |
| `builtup_ratio_250m` | float [0,1] | WorldCover | Tỷ lệ pixels class built-up |
| `vegetation_ratio_250m` | float [0,1] | WorldCover | Tỷ lệ lớp phủ thực vật |
| `water_occurrence_mean` | float/null | JRC (P1) | Đặc trưng phụ, không phải bằng chứng ngập đường |
| `road_length_m` | float | OSM | Đặc trưng để kiểm soát hình học |
| `highway_type` | category | OSM | Xử lý rare classes |

**Không dùng làm feature:** `flood_count`, `flood_observed`, `source_count` được tính từ chính bộ IRD dùng tạo nhãn; đây là leakage/target proxy. Feature theo thời kỳ cần mô tả `as_of_date`, nhất là khi đánh giá khái quát về quá khứ.

### 3.5. Label schema và quy tắc

- `label_status` ∈ `{positive, verified_negative, unknown}`.
- `label_binary`: `1` khi positive, `0` khi verified_negative, `null` khi unknown.
- `label_evidence`: chuỗi chỉ chứng cứ, ID nguồn, cách ghép, ghi chú kiểm tra.
- `label_confidence`: high / medium / low, không giả làm xác suất thống kê.
- `match_distance_m`, `match_method`, `review_required` lưu ở bảng join.

**Positive**: có sự kiện ngập đã được xác thực và ghép được với đoạn đường ở độ chính xác phù hợp. **Verified-negative**: cần quan trắc/khảo sát/nguồn độc lập xác nhận không ngập trong **phạm vi và giai đoạn quan sát được định nghĩa rõ**; không lấy ngẫu nhiên đường không có báo cáo làm 0. **Unknown**: mọi trường hợp còn lại.

**Matching:** thực hiện nearest/spatial join có ngưỡng cấu hình, xét tên đường, độ chính xác nguồn và khả năng một sự kiện ảnh hưởng nhiều đoạn. Mặc định bản ghi mơ hồ `review_required = true`; tuyệt đối không gán tự động cho một đoạn bất kỳ chỉ vì gần nhất.

**Dedup:** gộp các nguồn đưa tin cùng một lần ngập qua vị trí/thời gian/mô tả; giữ provenance nhiều nguồn. Không coi nhiều bài viết về cùng một sự kiện là những quan sát độc lập.

## 4. Tiền xử lý không gian

1. **Ingest & validate:** tải file thủ công/qua script, kiểm tra schema, checksum, codec và số dòng.
2. **Chọn AOI:** cắt lớp GIS theo vùng nghiên cứu cố định; ghi phiên bản ranh giới trước/sau cải cách hành chính 2025.
3. **CRS:** chuẩn hóa về CRS đo khoảng cách theo mét thích hợp cho HCMC; IRD chính là EPSG:9210, ancillary như thủy hệ có EPSG:4326. Trước mỗi phép `buffer`, `distance`, `length` bắt buộc xác nhận unit mét.
4. **Road segmentation:** làm sạch geometry OSM, bỏ đối tượng không thuộc đường cần khảo sát, chia 100–200 m có kiểm soát nút giao, giữ mapping về way gốc.
5. **Raster sampling:** tạo buffer có kích thước cấu hình; xử lý nodata, mask lớp raster và ghi `raster_valid_fraction`.
6. **Spatial labels:** ghép điểm, deduplicate, giữ độ chính xác và quyết định manual review.
7. **Feature assembly:** một dòng/`road_segment_id`, join một-một, không để nhãn trong danh sách feature.
8. **Quality report:** thống kê coverage, missing, distribution, anomalies theo từng nguồn và khu vực.

## 5. Phương pháp mô hình hóa

### Track A — Baseline hợp lệ khi chưa có nhãn âm (BẮT BUỘC)

**GIS MCDA susceptibility index:** chuẩn hóa các feature nguy cơ hợp lý (`relative_elevation`, `dist_water`, `builtup_ratio`, v.v.), xác định trọng số bằng quy tắc công bố trước, tính điểm [0,100] để **xếp hạng tương đối**, không gọi là xác suất. Lưu các biến thành phần và chạy độ nhạy với bộ trọng số khác. Không hiệu chỉnh trọng số theo chính test labels.

### Track B — Research-only presence–background / PU (CÓ ĐIỀU KIỆN)

Dùng điểm `positive` và điểm nền lấy mẫu có kiểm soát theo không gian. Các điểm nền **không mang ý nghĩa xác nhận không ngập**. Mục tiêu chỉ là so sánh xếp hạng, khả năng tìm lại positive held-out, và sự ổn định theo không gian. Kết quả `predict_proba` nếu dùng classifier với pseudo-negative chỉ là score tương đối theo cơ chế lấy mẫu, **không được gọi** là xác suất ngập đã được kiểm chứng. Báo cáo phương pháp lấy mẫu, seed, tỷ lệ positive/background và phân tích nhạy cảm.

### Track C — Supervised classification (CHỈ KHI đủ verified-negative)

- Model: `DummyClassifier` / Logistic Regression / Random Forest / XGBoost.
- Input: các feature không rò rỉ label; `y` chỉ gồm positive/verified_negative.
- Split: spatial blocks hoặc group theo khu vực độc lập; dành riêng hold-out test, không để đoạn lân cận gần như trùng nhau tràn qua folds; tách nhóm sự kiện trùng trong mọi phép thử có dựa trên sự kiện.
- Imputation, encoding, scaling (nếu cần): `fit` chỉ ở train fold.
- Metrics: PR-AUC, precision, recall, F1, balanced accuracy, confusion matrix, reliability/calibration khi có đủ mẫu đại diện.
- Không có mục tiêu Accuracy cố định; ưu tiên độ tin cậy khoa học.

### Đánh giá Track A/B khi không có nhãn âm

- Held-out event/positive recall theo ngưỡng top-k% chiều dài đường hoặc diện tích nghiên cứu.
- Capture/lift so với baseline ngẫu nhiên và baseline địa hình đơn biến.
- Tính ổn định khi thay seed, background sampling, spatial holdout, trọng số MCDA.
- Nếu có khảo sát độc lập, báo cáo riêng kết quả trên bộ đó.
- **Không** báo cáo Accuracy/F1/ROC-AUC như đánh giá phân loại ngập-vs-không ngập trên bộ positive–unlabeled.

## 6. Thiết kế repository

```text
hcmc-road-flood-ml/
├── configs/
│   ├── sources.yaml
│   ├── study_area.yaml
│   ├── features.yaml
│   ├── labeling.yaml
│   └── experiments.yaml
├── data/
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   └── processed/.gitkeep
├── src/flood_ml/
│   ├── ingestion/
│   ├── gis/
│   ├── labeling/
│   ├── features/
│   ├── baseline/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── scripts/
│   ├── validate_sources.py
│   ├── prepare_roads.py
│   ├── build_features.py
│   ├── build_labels.py
│   ├── run_baseline.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
├── tests/
├── docs/
├── artifacts/.gitkeep
├── reports/figures/.gitkeep
├── reports/metrics/.gitkeep
├── pyproject.toml
├── .gitignore
├── README.md
└── SPEC.md
```

**Stack:** Python >=3.11, `pandas`, `numpy`, `geopandas`, `shapely`, `pyproj`, `rasterio`, `pyarrow`, `scikit-learn`, `xgboost` (track C), `matplotlib`, `pyyaml`, `pytest`, `ruff`. Dùng `uv` hoặc `pip` với lock file và môi trường ảo. Không bắt buộc GPU.

## 7. CLI dự kiến (contract, chưa triển khai)

```bash
python scripts/validate_sources.py --config configs/sources.yaml
python scripts/prepare_roads.py --config configs/study_area.yaml
python scripts/build_features.py --config configs/features.yaml
python scripts/build_labels.py --config configs/labeling.yaml
python scripts/run_baseline.py --config configs/experiments.yaml
python scripts/evaluate.py --method mcda
# Chỉ khi đáp ứng gate mẫu âm xác thực:
python scripts/train.py --model xgboost --config configs/experiments.yaml
```

Mỗi lệnh trả mã thoát khác 0 khi lỗi, ghi log, ghi config snapshot/seed/checksum và không sửa `data/raw`. Các đường dẫn phải đọc từ config, tránh hardcode máy cá nhân.

## 8. Kiểm thử và yêu cầu phi chức năng

- **Unit tests:** CRS, distance/length unit, raster nodata, buffer, segmentation, schema, dedup, label trạng thái.
- **Integration tests:** tập GIS fixture nhỏ có expected outputs và có thể chạy offline.
- **Data tests:** không trùng `road_segment_id`, geometry hợp lệ, ratio [0,1], elevation sane range theo AOI, tọa độ nằm AOI (hoặc flagged), không mất provenance.
- **Leakage test:** danh sách X không chứa nhãn hay derived features từ nguồn label; fit preprocessing chỉ trên train.
- **Reproducibility:** lưu seed, package lock, data checksum, git commit, config snapshot; tái chạy ra cùng outputs với cùng input trong dung sai xác định.
- **Documentation:** README hướng dẫn setup, download, quy định giấy phép, phương pháp và giới hạn; notebook không được là nơi duy nhất chứa logic production.
- **Performance:** xử lý AOI mẫu trên CPU phổ thông; benchmark runtime/memory và tối ưu khi có số liệu thật, chưa đặt SLA không có cơ sở.

## 9. Các giai đoạn và deliverables

| Phase | Công việc | Deliverables | Điều kiện hoàn thành |
|---|---|---|---|
| P0 — Setup | Repo, config, môi trường, license inventory | `README.md`, `pyproject.toml`, manifest template | Cài môi trường và chạy tests skeleton |
| P1 — Data foundation | Ingest IRD/OSM/FABDEM/water/WorldCover, AOI/CRS | raw manifest, road segments, flood observations | Dữ liệu nguồn truy vết, hình học hợp lệ |
| P2 — Features & labels | Spatial features + evidence-based labels | `road_features.parquet`, `road_labels.parquet`, QC report | Không có âm giả do thiếu ghi nhận; join có review flags |
| P3 — Baseline | MCDA + sensitivity analysis | `road_susceptibility.parquet`, bản đồ offline, methodology | Score có provenance, không gọi xác suất |
| P4 — ML research | PU/ranking hoặc supervised có điều kiện | Experiment configs, model files, evaluation report | Spatial holdout, công bố giới hạn và so sánh baseline |
| P5 — Release | Tài liệu tái lập | Model card, data card, reproducibility report | Người khác có thể chạy lại sau khi tải nguồn đúng giấy phép |

## 10. Acceptance criteria cho repo v1.0

- [ ] Repo chạy được trên môi trường Python sạch từ `README`.
- [ ] Mỗi nguồn có link, license, snapshot/version, checksum và phương pháp lấy dữ liệu.
- [ ] Xây dựng được bảng đoạn đường, thống kê số lượng và tỷ lệ geometry hợp lệ.
- [ ] Tính được ít nhất 5 feature: elevation, relative elevation, slope, distance to water, built-up ratio; có báo cáo độ thiếu.
- [ ] Mỗi đoạn có label trạng thái positive / verified_negative / unknown và bằng chứng nguồn.
- [ ] Không có nhãn âm tạo chỉ từ việc không có báo cáo ngập.
- [ ] Hoàn thành baseline MCDA và phân tích độ nhạy.
- [ ] Có đánh giá không gian, baseline so sánh và báo cáo giới hạn.
- [ ] Mọi run có config, seed, mã commit và nguồn dữ liệu.
- [ ] Có unit/integration tests, CI kiểm tra format, lint, test.
- [ ] Không commit file dữ liệu lớn, secrets, hoặc artifact bị hạn chế giấy phép.
- [ ] Nếu chưa có verified-negative: release là **susceptibility ranking**, không tuyên bố mô hình phân loại đã được xác thực.

## 11. Rủi ro và hướng xử lý

| Rủi ro | Tác động | Giải pháp |
|---|---|---|
| Thiếu verified-negative | Không đủ căn cứ train classifier chuẩn | Track A; tìm khảo sát độc lập; nghiên cứu PU có giới hạn |
| Thiên lệch điểm ngập do báo chí | Positive tập trung tại đường nổi tiếng | Phân tích coverage, nhóm nguồn, đối chiếu dữ liệu địa phương |
| Điểm ngập mơ hồ | Gán nhầm đoạn đường | Manual review, confidence, unknown label |
| DEM 30 m và DSM | Không nhận ra chỗ trũng cục bộ | FABDEM ưu tiên, so sánh Copernicus, không tuyên bố quá độ phân giải |
| Không tương thích CRS / pixel | Sai khoảng cách, lệch lớp | Thống nhất CRS đo mét, test hình học |
| Leakage khi đánh giá | Metrics ảo | Cấm flood-derived features dùng train cùng nhãn, spatial split |
| Thời kỳ dữ liệu không đồng nhất | Feature hiện tại không phản ánh thời điểm ngập cũ | Ghi year/vintage, tách mục tiêu susceptibility hiện tại và historical analysis |
| Giấy phép phi thương mại / ODbL | Hạn chế chia sẻ/ứng dụng sau này | Data license inventory, review trước phát hành sản phẩm |

## 12. Các quyết định còn mở

1. AOI cụ thể: các quận/khu vực cũ hay ranh giới TP.HCM hiện hành; quyết định bằng `study_area.yaml`.
2. Cơ chế thu thập dữ liệu verified-negative: cảm biến, khảo sát hoặc cơ quan chuyên môn; nếu chưa có, v1 chỉ ranking.
3. Mức độ tin cậy nào cho phép tự động map positive sang segment; xác định qua kiểm tra mẫu thủ công.
4. Feature và phạm vi thời kỳ: đánh giá susceptibility hiện tại, hay dự đoán hồi cứu theo các mốc năm (cần dữ liệu có vintage tương ứng).
5. Xuất artifact dạng parquet/GeoPackage và quy định redistributable vs restricted theo giấy phép.

## 13. Tham khảo và ghi công

- Hardy, S.; Radziute, G. (2026). *Historical database of urban flooding in Ho Chi Minh City, Vietnam (2002–2026)*. DataSuds. DOI: https://doi.org/10.23708/8Y16HU (CC BY-NC 4.0).
- OpenStreetMap contributors. Data licensed under ODbL. https://www.openstreetmap.org/copyright
- Hawker, L. et al. FABDEM. University of Bristol. https://research-information.bris.ac.uk/en/datasets/fabdem-2/ (CC BY-NC-SA 4.0).
- ESA WorldCover. https://esa-worldcover.org/en/data-access
- Copernicus DEM. https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM
- EC JRC/Google, Global Surface Water v1.4. https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater (v1.4: 1984–2021; các phiên bản mới hơn cần đối chiếu riêng).
- Open-Meteo Historical Weather. https://open-meteo.com/en/docs/historical-weather-api

---

**Definition of done:** Repo có một pipeline GIS tái lập, bảng feature theo đoạn đường, bộ nhãn có provenance/unknown rõ ràng, baseline xếp hạng nguy cơ, báo cáo đánh giá không gian và tài liệu phương pháp. ML classifier chỉ được coi là hoàn thành khi có kiểm chứng trên dữ liệu âm/dương đáng tin cậy.
