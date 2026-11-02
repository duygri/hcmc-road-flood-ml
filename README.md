# HCMC road flood susceptibility research

Pipeline nghiên cứu **mức độ dễ ngập tương đối** của đoạn đường trong TP.HCM
trước mở rộng địa giới 2025. Mốc hiện tại chỉ thiết lập môi trường, cấu hình và
run provenance. Các CLI xử lý GIS và MCDA sẽ được bổ sung theo kế hoạch đã duyệt.
Score MCDA không phải xác suất ngập và không dự báo thời điểm ngập.

## Windows / PowerShell

Cài [uv bằng installer chính thức](https://docs.astral.sh/uv/getting-started/installation/)
nếu chưa có, sau đó mở PowerShell tại thư mục repo:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Nếu terminal hiện tại chưa nhận PATH mới:
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
# Python riêng của repo; không thay Python hệ thống.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.uv-python'
uv python install 3.12
uv sync --locked --group dev
uv run python --version
uv lock --check
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Repo yêu cầu Python >=3.11, pin nhánh 3.12 bằng `.python-version`, tạo `.venv`
riêng và khóa dependency trong `uv.lock`. Chưa cài XGBoost. Không cần GPU.

## Cấu hình và dữ liệu local

`configs/experiments.yaml` là nơi duy nhất khai báo seed/run_id và liên kết bốn
section còn lại. Mọi path cấu hình tương đối được giải theo `repo_root`, độc lập
với working directory. Có thể vào `load_settings` qua bất kỳ file nào trong năm
file YAML; tất cả cùng đọc bundle và seed đó. Mỗi file có một mapping theo tên
section (`sources`, `study_area`, `features`, `labeling`, `experiments`).

Mặc định `sources.raw_root: data/raw`. Để đọc dữ liệu đã tải từ một thư mục khác,
tạo `configs/local/sources.yaml` (được Git ignore), dùng path tuyệt đối riêng máy:

```yaml
sources:
  raw_root: 'E:/research-data/raw'
```

Local override chỉ overlay các khóa trong section sources. Không ghi path máy
cá nhân vào config tracked. Task 2 sẽ thêm source inventory vào `sources.assets`;
danh sách hiện rỗng có chủ đích, không có nghĩa nguồn đã được xác thực.

Raw được đọc qua `resolve_input(raw_root, asset_path)`, không tạo/copy/sửa file
nguồn. `resolve_output(repo_root, relative_path)` từ chối path tuyệt đối, traversal
và symlink/junction thoát root. Output config không được chồng raw root. Các caller
phải dùng resolver đúng vai trò; resolver trả `Path`, không phải sandbox của OS.

Output mỗi run ở `data/interim/<run_id>/`, `data/processed/<run_id>/`,
`reports/runs/<run_id>/`, `artifacts/<run_id>/`. Raw binaries và output local
không đưa vào Git. Các metadata nguồn hiện có chưa được migrate ở Task 1.

## Run provenance

```python
from pathlib import Path
from flood_ml.config import load_settings
from flood_ml.utils.run import start_run

root = Path.cwd()  # truyền root repo rõ ràng nếu chạy từ thư mục khác
settings = load_settings(Path('configs/experiments.yaml'), repo_root=root)
context = start_run(settings, settings.experiments['run_id'])
```

`reports/runs/<run_id>/run.json` chứa snapshot cấu hình, seed, thời gian UTC,
git commit/dirty state, SHA-256 lock và SHA-256 của từng input asset cấu hình.
Không scan raw root tự động. Run ID cũ chỉ reuse khi fingerprint cấu hình, inputs
và lock không đổi; nếu thay đổi phải chọn ID mới. Git state lưu tại lúc khởi tạo.
Chưa có Git metadata thì ghi null rõ ràng.

Run record có `status: started`, **không** phải completion marker. Atomic JSON
ghi temp cùng thư mục, flush/fsync và replace; lỗi không xuất file dở và giữ file
cũ nguyên vẹn. Khởi tạo đồng thời cùng run ID bị từ chối qua lock độc quyền;
nếu process bị kill và để `.start.lock`, cần kiểm tra không còn process sử dụng
run trước khi bỏ stale lock. Stage checkpoints và completion sẽ thuộc Task 12.

Nguồn và giới hạn giấy phép được mô tả trong [SPEC](SPEC.md) và
[thiết kế đã duyệt](docs/superpowers/specs/2026-11-02-old-hcmc-baseline-design.md).
IRD/FABDEM có điều kiện phi thương mại; OSM cần attribution và tuân thủ ODbL.
