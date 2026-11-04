# Old HCMC Flood Susceptibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chạy lại được pipeline GIS cho TP.HCM trước mở rộng 2025, xuất feature/nhãn có provenance, baseline MCDA và đánh giá có điều kiện review.

**Architecture:** Python package chứa logic; CLI mỏng đọc cấu hình và ghi artifact theo run. Đường, quan sát, feature, review và score là những bảng riêng có khóa ổn định. Pipeline chạy được khi chưa có positive được duyệt; phần evaluation khi đó xuất trạng thái pending và số mẫu thiếu.

**Tech Stack:** Python >=3.11; môi trường đề xuất Python 3.12 do uv quản lý; pandas, numpy, geopandas, shapely, pyproj, pyogrio, osmium, rasterio, pyarrow, scikit-learn, matplotlib, pyyaml; pytest và ruff. Chưa cần XGBoost/GPU.

**Spec:** `SPEC.md`; thiết kế đã duyệt tại `docs/superpowers/specs/2026-11-02-old-hcmc-baseline-design.md`. Quyết định AOI và làm rõ sau builder review trong thiết kế cụ thể hóa các điểm còn mở của SPEC.

## Global Constraints

- Python >=3.11; dữ liệu thô đã tải phải giữ nguyên bytes. Chỉ metadata manifest được chuẩn hóa và tính lại từ file.
- AOI: chọn đúng một polygon có `GhiChu = "TP. Ho CHi Minh"` trong `HCMC_new_limits.gpkg`; CRS làm việc EPSG:9210.
- Đo khoảng cách/chiều dài/buffer trong CRS mét. Dùng polygon AOI, giữ halo cho ngữ cảnh feature.
- Đoạn mục tiêu 150 m, tối đa 200 m; giữ đoạn ngắn tại giao lộ/ranh giới. Không tạo nút giao chỉ từ giao cắt hình học khác mức.
- Nhãn mặc định unknown/null; positive cần review có bằng chứng. Chưa có verified-negative từ IRD. Không suy âm từ thiếu báo cáo.
- Không dùng feature sinh từ chính bằng chứng ngập. Fit preprocessing/normalization chỉ trên reference partition.
- MCDA 0–100 là score tương đối; `percentile_rank` là cột riêng. Đoạn thiếu feature/bridge/tunnel giữ score null kèm lý do.
- Mọi run lưu seed, config, input hashes, lock hash, commit và dirty state; mọi stage ghi atomically và có completion record.
- Output geometry dạng Parquet dùng GeoParquet. Raw, interim, processed, reports kết quả và artifact model hạn chế giấy phép ở local.
- Evaluation cần >=20 event reference và >=20 event holdout đã review, holdout >=3 blocks, mỗi block >=5 event. Đây chỉ là gate báo cáo exploratory, không bảo đảm lực thống kê.
- Capture chính yêu cầu >=50% chiều dài ảnh hưởng đã review; báo thêm any/100%. Không tính nhiều bản tin cùng sự kiện thành nhiều event.

## Review Focus

1. Config sai đường dẫn/CRS hoặc bytes nguồn đổi: fail rõ trước xử lý; Task 1–3 kiểm tra.
2. Nút OSM chung so với đường cắt nhau khác mức và way chạy qua biên AOI: Task 4 kiểm tra topology, clip và ID.
3. Pixel nodata, half-pixel offset và buffer qua rìa lớp nước: Task 6–7 kiểm tra validity và missing; không đổi thành 0.
4. Review trùng/mâu thuẫn, tọa độ chỉ mức tên đường, nhóm sự kiện kéo qua holdout: Task 5, 8–9 kiểm tra unknown/provenance và partition isolation.
5. Không có nhãn được duyệt, reference rỗng, hoặc stage bị ngắt: Task 10–12 kiểm tra pending/failure và không phát hành artifact dở.

## File map và quy ước chung

| Files | Trách nhiệm |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `README.md` | Môi trường, cài đặt, dữ liệu local |
| `configs/{sources,study_area,features,labeling,experiments}.yaml` | Một bộ cấu hình mặc định AOI cũ; override local trong `configs/local/` được ignore |
| `src/flood_ml/config.py`, `utils/run.py` | Config, paths, run provenance, atomic write |
| `ingestion/sources.py`, `gis/study_area.py` | Checksum và chọn AOI |
| `gis/roads.py`, `ingestion/osm.py` | OSM node topology và segmentation |
| `labeling/observations.py`, `matching.py`, `reviews.py` | Nguồn IRD, ứng viên và quyết định có người review |
| `features/{raster,water,assembly}.py` | Feature theo segment và valid fractions |
| `evaluation/{splits,capture}.py`, `baseline/mcda.py` | Spatial isolation, MCDA, sensitivity và metrics |
| `pipeline.py`, `reporting.py`, `scripts/*.py` | Điều phối stage và báo cáo offline |
| `tests/{unit,integration}/`, `tests/fixtures/` | Fixture tự tạo, offline; không đưa dữ liệu IRD vào Git |
| `docs/{methodology,data_card,model_card,reproducibility}.md` | Phương pháp, nguồn, giới hạn và chạy lại |
| `.github/workflows/ci.yml` | Format, lint, tests với nguồn synthetic |

Thêm `__init__.py` cho các package thực sự dùng; không tạo sẵn training module rỗng. Mọi path dưới đây là tương đối với repo mới, không phải repo đồ án cũ.

Public types tại `config.py`: `PipelineError(ValueError)`; frozen `Settings(repo_root: Path, sources: dict, study_area: dict, features: dict, labeling: dict, experiments: dict, seed: int)`.
`RunContext` tại `utils/run.py` có `repo_root: Path, run_id: str, seed: int, config_hash: str, interim_dir: Path, processed_dir: Path, reports_dir: Path, artifacts_dir: Path`.
GDF/DF trong các Interfaces là `geopandas.GeoDataFrame` / `pandas.DataFrame`. Những schema của SPEC giữ nguyên tên khóa; thêm cột QC không được đưa vào feature allowlist.

Run layout: `data/interim/<run_id>/`, `data/processed/<run_id>/`, `reports/runs/<run_id>/`, `artifacts/<run_id>/`. CLI thêm `--run-id`; nếu bỏ qua thì đọc `experiments.run_id`. Config/source fingerprint thay đổi yêu cầu run_id mới. Stage thành công có manifest của input/output hashes; file tồn tại đơn thuần chưa phải thành công.

Thực thi theo thứ tự Task 1–12. Trước code, dùng skill using-git-worktrees để chọn workspace cách ly. Giữ nguồn 480 MiB tại vị trí đã tải; nếu dùng worktree khác, cấu hình raw_root trong local sources override đọc nguồn hiện có, không copy dữ liệu lớn và không ghi đè dữ liệu. Đọc source ngoài repo chỉ qua raw_root khai báo; output luôn ở trong workspace.

## Task 1: Môi trường, config và run provenance

**Files:** tạo `pyproject.toml`, `uv.lock`, `.python-version`, `src/flood_ml/{__init__,config}.py`, `src/flood_ml/utils/{__init__,run}.py`, `configs/*.yaml`; sửa `.gitignore`, `README.md`; test `tests/unit/test_config_run.py`.

**Interfaces:** `load_settings(config_path: Path, *, repo_root: Path) -> Settings`; `resolve_input(raw_root: Path, asset_path: str) -> Path`; `resolve_output(repo_root: Path, relative_path: str) -> Path`; `start_run(settings: Settings, run_id: str) -> RunContext`; `write_json_atomic(path: Path, payload: dict) -> None`.

Ví dụ test bắt buộc trong test_config_run.py:
```python
def test_output_cannot_escape_workspace(tmp_path):
    with pytest.raises(PipelineError):
        resolve_output(tmp_path, "../outside")
```

- [ ] Cài uv nếu thiếu bằng phương thức chính thức; `uv python install 3.12`, tạo pyproject có requires-python >=3.11, lock và môi trường riêng. Không đổi Python hệ thống. Lock phiên bản giải được trên Windows; chưa cài XGBoost.
- [ ] Viết tests: config paths độc lập CWD; thiếu seed/section hoặc output `../outside` raise PipelineError; asset traversal ra ngoài raw_root bị từ chối; raw_root explicit bên ngoài workspace chỉ được đọc. Assert `resolve_output(tmp_path, 'data/x') == tmp_path / 'data/x'`.
- [ ] Chạy `uv run pytest tests/unit/test_config_run.py -q`; xác nhận lỗi import/hành vi chưa có.
- [ ] Implement Interfaces. YAML safe_load và kiểm tra type/range; tất cả configs liên kết về cùng run/seed. Run manifest có config snapshot, UTC timestamp, git commit/dirty, package lock hash, input hashes. Atomic temp file đặt cùng filesystem với đích.
- [ ] Thêm test simulate lỗi write: không có completion marker và không có output cuối hợp lệ; run_id đổi config bị từ chối. Chạy lại tests trên, `uv run python --version`, `uv lock --check`.
- [ ] Commit các file Task 1 cùng SPEC và docs đã duyệt/kế hoạch: `chore: establish reproducible flood research runtime`. Stage rõ file; raw binaries phải bị ignore. README có lệnh Windows và thư mục dữ liệu.

## Task 2: Inventory nguồn và checksum có thể kiểm chứng

**Files:** tạo `src/flood_ml/ingestion/{__init__,sources}.py`, `scripts/validate_sources.py`, `tests/unit/test_sources.py`; sửa `configs/sources.yaml`, `data/raw/manifest.yaml`.

**Interfaces:** `validate_sources(settings: Settings) -> DF`; `build_manifest(settings: Settings) -> dict`. Audit columns: dataset_id, asset_id, path, bytes, expected_sha256, actual_sha256, status. Mismatch/missing raise PipelineError sau khi lưu audit thất bại.

Fixture configured_sources trỏ một asset b'abc' trong tmp_path và hash kỳ vọng đã nêu.
```python
def test_published_checksum_is_validated(configured_sources):
    audit = validate_sources(configured_sources)
    assert audit["status"].tolist() == ["PASS"]
    assert audit["actual_sha256"].tolist() == audit["expected_sha256"].tolist()
```

- [ ] Tạo file synthetic `b'abc'`; assert SHA-256 đúng `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`; test thiếu file, hash sai, asset trùng và D03 nhiều tile. Chạy `uv run pytest tests/unit/test_sources.py -q`, phải fail trước implementation.
- [ ] Chuẩn hóa schema dataset/assets: dataset_id, source_url, version, downloaded_on, period, license/attribution, crs/resolution, scope/coverage_assumption, asset local path/bytes/SHA-256. D04 là thủy hệ riêng dù cùng DOI IRD; boundary/docs là supporting assets. Manifest đã tải chỉ là dữ liệu đầu vào để migration.
- [ ] Implement hash streaming và audit; chỉ refresh manifest khi có tùy chọn `--refresh-manifest`, không thay hash kỳ vọng âm thầm khi validate. Giữ nguyên mọi binary và ghi chú license mâu thuẫn. Metadata thiếu phải được ghi unknown có lý do, không đoán vintage.
- [ ] Chạy tests tới pass; chạy `uv run python scripts/validate_sources.py --config configs/sources.yaml --run-id source_audit_001` trên dữ liệu local: mỗi asset phải có PASS/FAIL rõ và source bytes bất biến.
- [ ] Commit `feat: validate source inventory and asset integrity`.

## Task 3: Polygon TP.HCM cũ và CRS mét

**Files:** tạo `src/flood_ml/gis/{__init__,study_area}.py`, `tests/unit/test_study_area.py`; sửa `configs/study_area.yaml`.

**Interfaces:** `select_aoi(boundaries: GDF, *, field: str, value: str, target_crs: str) -> GDF`; `require_metric_crs(frame: GDF) -> None`; `make_context(aoi: GDF, radius_m: float) -> GDF`.

Fixture geographic_frame là polygon EPSG:4326.
```python
def test_geographic_coordinates_are_rejected(geographic_frame):
    with pytest.raises(PipelineError):
        require_metric_crs(geographic_frame)
```

- [ ] Tạo fixture hai polygon EPSG:9210 có tên khác; đảo thứ tự dòng vẫn chọn đúng. Assert một hàng AOI; selection 0/2 hàng, CRS thiếu, geometry invalid/empty đều raise PipelineError; `require_metric_crs` từ chối EPSG:4326. Chạy `uv run pytest tests/unit/test_study_area.py -q`, xác nhận fail.
- [ ] Implement select theo giá trị chính xác đã duyệt; không dùng index dòng, bbox hoặc buffer để sửa ranh giới. Geometry repair nếu cần chỉ qua nhánh cấu hình có audit; mặc định fail.
- [ ] Config study_area_id `hcmc_pre_2025`, EPSG:9210, selector GhiChu/TP. Ho CHi Minh; context radius lấy max bán kính feature cộng stencil raster, không hardcode một halo nhỏ hơn feature.
- [ ] Chạy tests tới pass; kiểm tra nguồn thật: một AOI hợp lệ, khoảng 2112.12 km² (QC so sánh, không tự điều chỉnh polygon cho khớp số này). Ghi CRS nguồn/đích và boundary hash.
- [ ] Commit `feat: establish pre-expansion HCMC study area`.

## Task 4: Đọc OSM có node topology và chia đoạn đường

**Files:** tạo `src/flood_ml/ingestion/osm.py`, `src/flood_ml/gis/roads.py`, `scripts/prepare_roads.py`, `tests/fixtures/osm/intersections.osm`, `tests/unit/test_roads.py`; sửa study_area config.

**Interfaces:** `read_osm_roads(path: Path, context: GDF, allowed_highways: list[str]) -> GDF` trả osm_way_id/node_ids/highway_type/name/bridge/tunnel/layer/geometry; `segment_roads(ways: GDF, aoi: GDF, *, snapshot_date: str, target_m: float, maximum_m: float) -> GDF` trả schema road_segments của SPEC, thêm structure flags và chainage.

Fixtures ways_case/aoi chứa các trường hợp OSM nêu trên, CRS EPSG:9210.
```python
def test_segmentation_clips_and_caps_length(ways_case, aoi):
    result = segment_roads(ways_case, aoi, snapshot_date="2026-11-02",
                          target_m=150.0, maximum_m=200.0)
    assert result["road_segment_id"].is_unique
    assert result["length_m"].max() <= 200.0 + 1e-6
    assert result.geometry.difference(aoi.geometry.union_all()).length.max() <= 1e-6
```

- [ ] Fixture OSM tự tạo: nút giao T thật, hai way cắt hình học nhưng không chung node, bridge khác mức, một way đi xuyên AOI, một way 450 m và một phần ngắn 40 m. Test giao T được tách, khác mức không nối, mọi output nằm trong AOI, `length_m <= 200 + 1e-6`, tổng chiều dài bảo toàn trong dung sai 1e-6 m trên fixture chiếu.
- [ ] Test cùng snapshot/config và input đảo thứ tự/reverse geometry cho cùng ID set; snapshot khác được phép đổi ID. Chạy `uv run pytest tests/unit/test_roads.py -q`, xác nhận fail.
- [ ] Dùng osmium đọc node refs/tags từ PBF, không suy topology từ OGR lines bị mất node IDs. Giữ nguyên way source refs, lọc allowlist từ thiết kế. Stage input feature envelope chỉ để giảm đọc; clip bằng polygon thật sau reprojection.
- [ ] Tách shared OSM nodes cùng mức, canonicalize hướng, rồi chia từng chain bằng số phần ceil(length/150) và chiều dài đều nhau; giữ short chain do nút giao/AOI. ID hash từ snapshot, way ID, canonical chain geometry/chainage và segmentation config digest; tài liệu hóa precision canonical trong code/test.
- [ ] Chạy tests tới pass và CLI trên nguồn thật; QC số ways/segments theo loại, bridge/tunnel, short đoạn và geometry hợp lệ. Không diễn giải count đã khảo sát qua bbox thành số segment cuối.
- [ ] Commit `feat: derive stable road segments from OSM topology`.

## Task 5: Quan sát IRD và ứng viên nhóm sự kiện

**Files:** tạo `src/flood_ml/labeling/{__init__,observations}.py`, `tests/unit/test_observations.py`; sửa `configs/labeling.yaml`.

**Interfaces:** `normalize_observations(raw: GDF, aoi: GDF) -> GDF`; `propose_event_groups(observations: GDF, config: dict) -> DF`. Output giữ observation_id/source_id/raw values, normalized fields, date_precision, outside_aoi, quality flags. Nhóm ứng viên chưa phải event đã được người review xác nhận.

Fixture raw_year_only có một observation, Year=2025, Date=null; aoi chứa điểm đó.
```python
def test_year_precision_does_not_invent_date(raw_year_only, aoi):
    result = normalize_observations(raw_year_only, aoi)
    assert result["observed_date"].isna().all()
    assert result["date_precision"].tolist() == ["year"]
```

- [ ] Fixture có Date thiếu/Year 2025, biến thể `Low `, `Hgh`, `Hugh`, `Exact_point`, ngày nguồn ISO không chuẩn, điểm ngoài AOI và hai nguồn đưa cùng sự kiện. Assert observed_date null khi chỉ có năm; raw field giữ nguyên; precision canonical có cờ sửa; điểm ngoài vẫn còn; cùng tên đường khác ngày không tự gộp.
- [ ] Test observation_id trùng fail; tọa độ nhiều chữ số không tạo location_accuracy_m. Chạy `uv run pytest tests/unit/test_observations.py -q`, phải fail.
- [ ] Implement deterministic source ID mapping, parse ngày có audit, chỉ tạo candidate-group edges theo thời gian/vị trí cấu hình (đề xuất 1 ngày/100 m khi có ngày; thiếu ngày cần review). Giữ tất cả source links và sự khác biệt độ chính xác.
- [ ] Chạy tests tới pass; QC nguồn hiện tại kỳ vọng 425 records, 418 covered/7 outside (mismatch phải báo chứ không ép số). Xuất review candidate groups.
- [ ] Commit `feat: preserve flood observation precision and provenance`.

## Task 6: Raster grid, halo và feature địa hình/lớp phủ

**Files:** tạo `src/flood_ml/features/{__init__,raster}.py`, `tests/fixtures/raster_factory.py`, `tests/unit/test_raster_features.py`; sửa `configs/features.yaml`.

**Interfaces:** `compute_raster_features(roads: GDF, aoi: GDF, dem_paths: list[Path], landcover_path: Path, config: dict) -> DF`. Output có road_segment_id, elevation_median_m/elevation_p10_m/relative_elevation_m/slope_mean_deg/builtup_ratio_250m/vegetation_ratio_250m và valid_fraction riêng từng feature.

Fixture flat_raster_case cung cấp args đúng signature từ DEM phẳng 7 m và landcover đối xứng theo mask đã nêu.
```python
def test_flat_dem_has_zero_relative_height(flat_raster_case):
    result = compute_raster_features(**flat_raster_case)
    assert result["elevation_median_m"].iloc[0] == pytest.approx(7.0)
    assert result["relative_elevation_m"].iloc[0] == pytest.approx(0.0)
    assert result["slope_mean_deg"].iloc[0] == pytest.approx(0.0)
```

- [ ] Tạo GeoTIFF synthetic chiếu mét: mặt phẳng z=x cho slope 45°, DEM phẳng 7 m cho median=7/relative=0, landcover mask đối xứng đúng nửa class50 cho builtup=0.5. Test nodata trong stencil làm slope thiếu, all-nodata buffer cho feature null, không đổi thành zero.
- [ ] Test tile có half-pixel offset/mosaic seam và footprint không phủ đủ AOI+halo được phát hiện; landcover resampling không tạo class mới. Chạy `uv run pytest tests/unit/test_raster_features.py -q`, phải fail.
- [ ] Warp DEM bilinear tới EPSG:9210/30 m, landcover nearest tới EPSG:9210/10 m; grid origin snap về bội của resolution. Mask/stencil validity riêng, không clip raster về AOI trước neighborhood calculations.
- [ ] Tính slope Horn 3x3 theo khoảng cách mét; DEM road buffer30 m; relative = median trong road-buffer30 m trừ median trong road-buffer500 m; lớp phủ dùng road-buffer250 m. Dùng pixel centers (`all_touched=False`); tỷ lệ chỉ trên pixels valid, threshold valid_fraction mặc định0.9. Lưu assumptions/grid/vertical datum.
- [ ] Chạy tests tới pass và một tập roads thật theo chunk; QC histogram/missing, halo và thời gian/bộ nhớ. Không load toàn bộ raster/OSM vào nhiều bản sao DataFrame cùng lúc.
- [ ] Commit `feat: extract terrain and land-cover features with coverage`.

## Task 7: Thủy hệ, coverage và feature assembly

**Files:** tạo `src/flood_ml/features/{water,assembly}.py`, `scripts/build_features.py`, `tests/unit/test_water_assembly.py`; sửa sources/features configs.

**Interfaces:** `compute_water_features(roads: GDF, water: GDF, coverage_domain: GDF, config: dict) -> DF`; `assemble_features(roads: GDF, raster_features: DF, water_features: DF, feature_columns: list[str]) -> DF`.

Fixture edge_water_case có buffer250 m vượt domain.
```python
def test_incomplete_water_context_stays_missing(edge_water_case):
    result = compute_water_features(**edge_water_case)
    assert result["water_context_incomplete"].iloc[0]
    assert pd.isna(result["water_area_ratio_250m"].iloc[0])
```

- [ ] Fixture có road cách polygon nước10 m; assert dist_water_m=10 trong dung sai1e-6. Buffer250 m vượt coverage_domain => water ratio null/context_incomplete. Nếu nearest nước trong domain xa hơn biên domain thì distance null: không thể loại trừ nước gần hơn bên ngoài.
- [ ] Test buffer hoàn toàn trong domain mà không có polygon nước cho ratio0, không phải missing; mất coverage khác với vắng hình học. Join duplicate key/thiếu segment và feature allowlist chứa flood_count/source_count/label_binary phải fail. Chạy `uv run pytest tests/unit/test_water_assembly.py -q`, phải fail.
- [ ] Config D04 coverage_domain dùng AOI nguồn đã nêu và ghi coverage_assumption chưa được xác minh độc lập; không suy domain từ convex hull/bbox các polygon nước. Outside domain unknown. Distance chỉ đầy đủ nếu nearest distance nhỏ hơn khoảng cách tới domain boundary; tie coi incomplete.
- [ ] Implement polygon union trước area ratio để tránh đếm trùng, metric spatial index, feature assembly one-to-one bảo toàn mọi road ID. Loại bridge/tunnel khỏi score eligibility, vẫn giữ features/QC. Chạy tests tới pass và feature CLI trên data thật.
- [ ] Commit `feat: assemble leakage-safe features with water coverage flags`.

## Task 8: Matching và nhãn từ quyết định review

**Files:** tạo `src/flood_ml/labeling/{matching,reviews}.py`, `scripts/build_labels.py`, `tests/unit/test_matching_reviews.py`; sửa labeling config.

**Interfaces:** `match_candidates(observations: GDF, roads: GDF, config: dict) -> DF`; `apply_reviews(roads: GDF, observations: GDF, candidates: DF, match_reviews: DF, event_reviews: DF) -> tuple[DF, DF]` trả road_labels và confirmed_event_segments.

Fixture no_review_case có đủ roads/observations/candidates cùng hai review tables rỗng đúng schema.
```python
def test_no_review_never_creates_negatives(no_review_case):
    labels, links = apply_reviews(**no_review_case)
    assert set(labels["label_status"]) == {"unknown"}
    assert labels["label_binary"].isna().all()
    assert links.empty
```

- [ ] Fixture điểm ở giữa hai đường gần nhau phải có cả hai candidates/review_required; Street point không tự gán nearest positive. Không có review => tất cả label_status=unknown và label_binary.isna().all(). Bản ghi ngoài AOI vẫn được liệt kê lý do review, không tự match road trong AOI.
- [ ] Test review xác nhận positive có provenance hợp lệ => 1; reject candidate vẫn unknown; review mâu thuẫn/trùng, thiếu evidence/reviewer/time hoặc segment/event ID không tồn tại phải fail. Không chấp nhận negative chỉ vì no_report. Chạy `uv run pytest tests/unit/test_matching_reviews.py -q`, phải fail.
- [ ] Search radius mặc định100 m chỉ để tạo candidates, không diễn giải location accuracy; config cho phép đổi. Tạo templates local `review_decisions.csv` (observation_id, road_segment_id, decision, evidence, reviewed_by, reviewed_at) và `event_reviews.csv` (observation_id, event_group_id, location_group_id, evidence, reviewed_by, reviewed_at).
- [ ] Implement normalization/read CSV và validation; accepted/rejected/needs_info không tạo bằng chứng âm. Schema label cho phép positive/verified_negative/unknown nhưng nguồn hiện tại chỉ sinh positive/unknown. Giữ source IDs nhiều nguồn, decision history, event-to-segment links; không tự điền reviewer hoặc xác nhận thay người dùng.
- [ ] Chạy tests tới pass. Với sources thật chưa có review, CLI thành công với unknown và hàng đợi review; không báo hoàn tất xác thực nhãn.
- [ ] Commit `feat: require evidence-backed review for positive road labels`.

## Task 9: Spatial reference/holdout không rò rỉ

**Files:** tạo `src/flood_ml/evaluation/{__init__,splits}.py`, `tests/unit/test_splits.py`; sửa experiments config.

**Interfaces:** `assign_spatial_partitions(roads: GDF, confirmed_event_segments: DF, config: dict) -> DF`. Columns road_segment_id, block_id, partition (reference/holdout/guard/excluded), exclusion_reason. Return giữ mọi road.

Fixture split_case có roads, confirmed_event_segments và config đã nêu.
```python
def test_spatial_split_is_deterministic(split_case):
    first = assign_spatial_partitions(**split_case).sort_values("road_segment_id")
    changed = dict(split_case)
    changed["roads"] = changed["roads"].iloc[::-1]
    second = assign_spatial_partitions(**changed).sort_values("road_segment_id")
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
```

- [ ] Fixture lưới blocks, đường sát biên và event/location group vượt partitions. Assert `set(reference_event_ids).isdisjoint(holdout_event_ids)` và tương tự location groups; reference feature-support không chồng holdout-support. Đảo input và giữ seed cho cùng assignments.
- [ ] Test thay labels/score không thay holdout block selection; group crossing bị excluded có lý do, không chuyển blocks để đạt quota. Chạy `uv run pytest tests/unit/test_splits.py -q`, phải fail.
- [ ] Config khởi đầu blocks5 km, holdout fraction0.2 chọn bằng hash(block_id,seed42) không dựa nhãn; guard1000 m tương ứng hai neighborhood500 m, tăng nếu config radius tăng. Loại reference roads có footprint guard overlap holdout; xử lý full line geometry, không chỉ centroid.
- [ ] Implement group connected components theo confirmed event/location IDs; group vượt partitions bị loại khỏi denominator đánh giá, giữ trong output/audit. Baseline vẫn có thể score guard/excluded nếu đủ feature, nhưng reference fit và evaluation masks độc lập.
- [ ] Chạy tests tới pass, xuất split map/count/coverage. Freeze split/config trước đánh giá; thiếu quota không được chọn lại holdout theo kết quả.
- [ ] Commit `feat: isolate spatial partitions and reviewed event groups`.

## Task 10: MCDA, normalizer và sensitivity

**Files:** tạo `src/flood_ml/baseline/{__init__,mcda}.py`, `scripts/run_baseline.py`, `tests/unit/test_mcda.py`; sửa experiments config.

**Interfaces:** `fit_normalizer(reference_features: DF, feature_spec: dict) -> dict`; `score_mcda(features: DF, normalizer: dict, weights: dict[str, float]) -> DF`; `compare_weight_sets(features: DF, normalizer: dict, weight_sets: dict) -> dict`.

Fixture missing_mcda_case có một road thiếu một nhóm, normalizer và weights hợp lệ.
```python
def test_missing_feature_keeps_score_null(missing_mcda_case):
    result = score_mcda(**missing_mcda_case)
    assert pd.isna(result["score"].iloc[0])
    assert result["score_reason"].iloc[0] == "missing_required_feature"
```

- [ ] Test risk-normalized all0/all1 cho score0/100; thiếu1 nhóm => score null và reason; bridge/tunnel => score null. Feature constant trong reference dùng contribution0.5 và flag, không chia0. Reference rỗng phải fail.
- [ ] Test đổi cực trị holdout không đổi normalizer, weight invalid/feature label column fail; cùng eligible set cho mọi sensitivity run. Chạy `uv run pytest tests/unit/test_mcda.py -q`, phải fail.
- [ ] Config5 nhóm dùng elevation_median_m, relative_elevation_m, slope_mean_deg, dist_water_m, builtup_ratio_250m; bốn nhóm đầu lower-is-higher-risk, builtup higher-is-higher-risk. Fit q05/q95 reference-only, linear clip [0,1], lưu quantiles; đây là baseline hypothesis. elevation_p10 chỉ diagnostic, không tính đôi.
- [ ] Bộ chính mỗi nhóm0.2; sensitivity terrain-heavy=(0.15,0.4,0.15,0.15,0.15), water-heavy=(0.15,0.15,0.15,0.4,0.15), builtup-heavy=(0.15,0.15,0.15,0.15,0.4), thứ tự như danh sách5 nhóm. Không tune theo labels.
- [ ] Implement score và percentile rank riêng; giữ all roads/score_reason. Sensitivity Spearman rank correlation và length-based top-k overlap, tie deterministic road_segment_id. Chạy tests tới pass, lưu normalizer/rules/config hashes và baseline outputs.
- [ ] Commit `feat: publish reproducible MCDA scores and weight sensitivity`.

## Task 11: Event capture, baseline so sánh và gate review

**Files:** tạo `src/flood_ml/evaluation/capture.py`, `scripts/evaluate.py`, `tests/unit/test_capture.py`; sửa experiments config.

**Interfaces:** `evaluation_gate(events: DF, partitions: DF) -> dict`; `evaluate_capture(scores: DF, roads: GDF, confirmed_event_segments: DF, partitions: DF, config: dict) -> dict`.

Fixture empty_event_gate_case có event table rỗng đúng schema và partitions hợp lệ.
```python
def test_empty_review_reports_pending(empty_event_gate_case):
    result = evaluation_gate(**empty_event_gate_case)
    assert result["status"] == "pending_label_review"
    assert result["missing_reference_events"] == 20
    assert result["missing_holdout_events"] == 20
```

- [ ] Không review => `evaluation_gate(...)["status"] == "pending_label_review"`; reference19/holdout20 vẫn pending; holdout20 chỉ2 blocks hoặc một block4 cũng pending. Gate pass khi reference20/holdout20 với phân bố holdout5/5/10 và group independence.
- [ ] Fixture event có hai segment dài10/90 m: chỉ segment10 m trong top budget => không capture50%; segment90 m có => capture. Trùng nguồn cùng event không tăng denominator. Event không đủ reviewed geometry/score coverage bị loại, có exclusion counts.
- [ ] Test ties và whole-segment budget overshoot deterministic; denominator cùng score-eligible holdout road length cho MCDA/elevation/random. Chạy `uv run pytest tests/unit/test_capture.py -q`, phải fail.
- [ ] Implement top-k chiều dài tại1/5/10/20%, primary threshold0.5, sensitivity >0/1.0; actual selected fraction được báo vì lấy nguyên segment. Tính hợp ảnh hưởng event tránh đếm trùng; không coi toàn bộ candidate segments là reviewed affected segments.
- [ ] Gate pending vẫn ghi JSON status + quota thiếu, không sinh metrics giả. Khi pass: random100 lần seed42..141, elevation-only trên cùng tập đường, capture/lift và bootstrap1000 lần theo block rồi event; report exploratory cùng denominator, CI và các giới hạn source bias. Trường hợp bootstrap denominator0 báo undefined, không ghi0.
- [ ] Chạy tests tới pass; hiện data thật có thể pending đúng thiết kế. Không gọi supervised metrics Accuracy/F1/ROC-AUC trên unknown.
- [ ] Commit `feat: evaluate reviewed event capture with explicit data gates`.

## Task 12: Pipeline end-to-end, báo cáo và CI

**Files:** tạo `src/flood_ml/{pipeline,reporting}.py`, `scripts/{prepare_observations,run_pipeline}.py`, `tests/integration/test_pipeline.py`, `.github/workflows/ci.yml`, `docs/{methodology,data_card,model_card,reproducibility}.md`; hoàn thiện CLI scripts/README/.gitignore.

**Interfaces:** `run_pipeline(settings: Settings, context: RunContext, *, stages: list[str] | None = None) -> dict`; `build_report(context: RunContext, stage_results: dict) -> dict`. Các script expose main(argv: list[str] | None = None) -> int và trả exit nonzero khi PipelineError.

Fixture synthetic_pipeline có Settings/RunContext cùng nguồn tự tạo và review rỗng.
```python
def test_full_pipeline_is_honest_without_review(synthetic_pipeline):
    settings, context = synthetic_pipeline
    summary = run_pipeline(settings, context)
    assert summary["pipeline_status"] == "complete"
    assert summary["evaluation_status"] == "pending_label_review"
    labels = pd.read_parquet(context.processed_dir / "road_labels.parquet")
    assert labels["label_binary"].isna().all()
```

- [ ] Integration fixture offline tự tạo: AOI, OSM XML có node IDs, point evidence, water, DEM, WorldCover. Chạy đầy đủ với review trống: có mọi bảng output, unique road IDs, features/labels tách biệt, score có lý do missing, evaluation pending. Trường hợp fixture được review đủ quota chạy nhánh evaluation pass.
- [ ] Test từ working directory khác; raw hashes trước/sau không đổi; lỗi giữa stage không để completion marker; run_id cùng fingerprint reuse stage đã kiểm chứng, config khác từ chối; source/output bị sửa vô hiệu stage và downstream. Chạy `uv run pytest tests/integration/test_pipeline.py -q`, xác nhận fail trước điều phối.
- [ ] Implement stage dependency graph validate -> AOI -> roads/observations -> raster/water/assembly & matching/review -> split -> MCDA -> evaluation/report. Ghi atomic GPKG/Parquet/JSON, GeoParquet metadata khi có geometry; các output theo tên SPEC trong run directories. Chỉ checkpoint stage sau hash/validation output thành công.
- [ ] Reports offline có AOI/roads, source coverage, observations ngoài AOI, review queue, null/score reasons, feature distributions, split map, baseline map, sensitivity và evaluation status. Data/model cards ghi source vintages không đồng nhất, 30 m DEM, media bias, score-relative, license conflict, unknown labels. Reproducibility doc có hardware/runtime/memory quan sát thực, không đặt SLA.
- [ ] CI Windows Python3.12 dùng uv lock: `uv sync --locked --group dev`; `uv run ruff check .`; `uv run ruff format --check .`; `uv run pytest -q`. Tests dùng synthetic fixtures, không tải data nghiên cứu hoặc gọi internet khi chạy.
- [ ] Chạy pipeline thật một run mới rồi chạy lại cùng inputs để kiểm tra deterministic IDs/scores với atol1e-6 và thống kê QC. Ghi lại command, timings, memory, hashes và review gate. Không bắt review được duyệt để hoàn thành mốc baseline; không gọi mốc evaluation hoàn tất khi pending.
- [ ] Commit `feat: deliver reproducible old-HCMC susceptibility pipeline` sau khi kiểm tra Git staged files không có raw/interim/processed/artifact local. Theo execution skill thực hiện whole-branch review trước bàn giao.

## Commands nghiệm thu ở repo root

```powershell
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python scripts/run_pipeline.py --config configs/experiments.yaml --run-id hcmc_old_v1_001
uv run python scripts/evaluate.py --method mcda --config configs/experiments.yaml --run-id hcmc_old_v1_001
```

Các CLI của SPEC vẫn dùng được với --config hiện có; --run-id là tùy chọn bổ sung. `prepare_roads` gọi AOI prerequisite, `build_features` kiểm tra prerequisites, `build_labels` gọi observation preparation nếu cần; lệnh lỗi phải nêu stage còn thiếu. `run_pipeline` là cách chạy đầy đủ được README khuyến nghị.

## Self-review đã thực hiện

- Coverage: AOI/source/CRS Task2–3; topology Task4; schema/time/dedup Task5/8; raster/halo/water Task6–7; leakage/splits Task7/9–10; MCDA/sensitivity Task10; gate/capture/uncertainty Task11; reproducibility/QC/CI/cards Task1/12. Track B/C được để ngoài mốc này theo thiết kế đã duyệt.
- Interfaces: mọi signature cross-task được định nghĩa ở task chủ; schema features/labels dùng SPEC và cột QC có allowlist. Labels không là prerequisite cho fit theo kiểu supervised; baseline normalization chỉ cần reference features.
- Review Focus: cả5 lớp lỗi được gắn tests cụ thể ở các task đã chỉ ra.
- Execution: chưa viết product code hoặc tạo môi trường. Tài liệu kế hoạch phải được người dùng review và chọn Native/Subagent-driven theo writing-plans trước khi thực thi.

