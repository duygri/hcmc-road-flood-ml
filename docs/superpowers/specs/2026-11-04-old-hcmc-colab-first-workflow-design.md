# Thiết kế — Colab-first workflow cho pipeline HCMC road flood ML

Ngày: 2026-11-04. Trạng thái: **Design đã được người dùng duyệt; spec cập nhật theo review comments, chờ người dùng review.**

## Mục tiêu

Dùng Google Colab làm môi trường chính để chạy pipeline từ preprocessing đến training và evaluation bằng một Jupyter notebook. Google Drive lưu raw dataset, review files, model và kết quả run. GitHub giữ source code và notebook có version control. Local Python chỉ được dùng ở bước cuối để tái chạy và xác minh run đã thực hiện trên Colab.

Track B tiếp tục dùng presence–background Logistic Regression đã được duyệt. MaxEnt có thể được thêm sau như một phương pháp so sánh riêng, với spec và plan riêng; thay đổi này không thay model hiện tại.

## Quyết định thiết kế

Chọn một **notebook điều phối mỏng** trong repository. Notebook gọi các CLI và module hiện có thay vì chép lại logic preprocessing, fit hoặc evaluation vào các cell.

Các lựa chọn đã cân nhắc:

| Phương án | Ưu điểm | Chi phí và rủi ro |
|---|---|---|
| Notebook Colab gọi CLI hiện có — **chọn** | Cùng logic với local; provenance và gate tiếp tục do code đã kiểm thử quản lý. | Cần cấu hình đường dẫn Drive và bước publish kết quả từ Colab workspace sang Drive. |
| Chép pipeline vào notebook | Người dùng có thể đọc hết logic ngay trong notebook. | Tạo bản logic thứ hai, dễ lệch khỏi CLI và khó test đầy đủ. |
| Viết pipeline riêng cho Colab | Có thể tối ưu riêng cho runtime hosted. | Tăng code cần bảo trì, tạo thêm một đường chạy cần so sánh và kiểm chứng. |

## Kiến trúc và luồng dữ liệu

1. Notebook template được lưu và quản lý cùng source trên GitHub. Mỗi run chọn một Git commit cụ thể, clone checkout đó vào vùng làm việc Colab và ghi commit SHA vào run manifest.
2. Notebook mount Google Drive. Raw assets và hai review CSV được đọc từ các thư mục Drive do người dùng cấu hình. Expected source manifest/checksum được đọc từ commit GitHub đang pin. Review CSV được chép vào `configs/local/` trong checkout Colab; các file này tiếp tục bị git ignore và không đi vào commit.
3. Notebook kiểm tra runtime Python/dependency lock, đặt `UV_PROJECT_ENVIRONMENT=/content/.venv`, cài dependency bằng `uv sync --locked` vào venv riêng thay vì system Python của Colab, kiểm tra run ID, các file input và hash khai báo trước khi gọi pipeline.
4. Pipeline chạy trong filesystem local của Colab (`/content`) để các bước ghi file và kiểm tra output dùng filesystem thông thường. Notebook gọi theo thứ tự:
   - `scripts/run_pipeline.py` để chạy preprocessing, labels, partitions và baseline scores;
   - `scripts/train.py` để chạy Track B theo fit gate;
   - `scripts/evaluate.py` để so sánh baseline và Track B theo capture gate.
5. Sau khi command kết thúc, notebook kiểm tra report/status và hash đầu ra, sau đó chép thư mục run sang Drive. Processed dataset, Track B scores, model artifacts, evaluation/training reports, manifest và logs là các artifact bắt buộc phải lưu. `publish_interim` mặc định là `false`; `data/interim/<run_id>` chỉ được chép khi người dùng bật tùy chọn này để debug.
6. Bước local Python cuối dùng cùng commit, lockfile, run ID và input đã đồng bộ từ Drive để tái chạy hoặc xác minh report/hash Colab. Bước này không tự mở gate review và không đổi model.

Notebook là entrypoint điều phối; các hàm trong package và CLI vẫn là nguồn logic chuẩn. Notebook không tạo feature, label, partition hoặc metric riêng.

## Lockfile và môi trường Colab

- `uv.lock` của uv project interface dùng universal resolution: khóa dependency theo Python/platform markers và có thể dùng trên các hệ điều hành được hỗ trợ. Wheel tags vẫn phụ thuộc OS/CPU; universal lock không bảo đảm mọi package đều có binary wheel phù hợp cho Windows và Linux.
- Giữ một lockfile dùng chung cho Windows x86_64 và Linux x86_64 glibc của Colab. Cả hai môi trường phải chạy `uv sync --locked` để phát hiện thiếu wheel hoặc yêu cầu build dependency.
- Dùng `uv lock --check` để xác nhận lock khớp `pyproject.toml`. Chỉ thêm `tool.uv.required-environments` nếu kiểm tra dependency cho thấy cần buộc wheel coverage cho package không có source distribution; package cần build từ source phải có system libraries/build tools được khai báo trong notebook.
- Đặt `UV_PROJECT_ENVIRONMENT=/content/.venv` trước khi sync trên Colab; không cài dependency dự án vào system Python.
- Mọi thay đổi lockfile phải được xác minh bằng `uv sync --locked` trên Windows local và Linux Colab trước khi merge/publish source.

## Cấu trúc lưu trữ trên Google Drive

Notebook dùng một thư mục gốc do người dùng cấu hình, với bố cục đề xuất:

```text
MyDrive/hcmc-road-flood-ml/
  raw/                 # raw assets đối chiếu với manifest được version trong GitHub
  review/              # match review CSV và event review CSV
  runs/<run_id>/       # processed data, scores, models, reports, logs, manifest
```

Drive là nơi lưu trữ bền vững. Trong lúc tính toán, run được tạo trong checkout Colab. Kết quả được chép vào thư mục staging riêng trên Drive; notebook ghi completion manifest cuối cùng chỉ khi mọi file bắt buộc đã chép và hash khớp. Folder thiếu completion manifest được xem là run chưa publish hoàn tất.

Mặc định notebook chỉ publish processed dataset, scores, models, reports, logs và manifests. Biến cấu hình `publish_interim: false` kiểm soát việc chép `data/interim/<run_id>`; chỉ bật khi cần debug. Quota phụ thuộc tài khoản và có thể thay đổi, vì vậy notebook không hard-code một mức dung lượng cố định. Notebook không tự xóa các run hoàn tất trước đó. Ước lượng dung lượng cần publish trước khi copy, bắt lỗi quota/write failure và để run staging thiếu completion manifest để người dùng xử lý.

Không lưu credentials hoặc token trong notebook, config được commit, hoặc GitHub. Raw data và review decisions không được commit lên GitHub. `configs/local/sources.yaml` chỉ được sinh trong checkout runtime để trỏ `raw_root` về Drive; các review CSV được đặt dưới `configs/local/` theo cách tương tự.

## Git hygiene cho notebook

- File `.ipynb` trong GitHub là template sạch: tất cả cell có `execution_count: null`, `outputs: []`; không có widget state, data output nhúng, runtime path hoặc run-specific Colab metadata.
- Kết quả notebook đã chạy được xuất thành Markdown/HTML hoặc report artifacts trên Drive. Không commit notebook đã chạy hoặc output data nặng về GitHub.
- CI kiểm tra JSON notebook, execution counts, outputs, widget state và metadata allowlist để ngăn notebook output hoặc thông tin runtime lọt vào source.

## Trạng thái và lỗi

- Thiếu file, sai checksum, config lỗi hoặc không đủ dung lượng local scratch ước tính: dừng trước khi chạy stage có tác động, ghi log chẩn đoán vào run staging nếu đã có run ID. Lỗi quota hoặc write failure trên Drive trong lúc publish không tạo completion manifest.
- Fit gate chưa đạt `minimum_reference_events`: coi `pending_positive_review` là kết quả hợp lệ của pipeline; không tạo model hoặc score giả, lưu report pending lên Drive và hiển thị thiếu hụt rõ ràng.
- Capture gate hoặc Track B score coverage chưa đủ: lưu trạng thái pending, không hiển thị capture metrics như hoàn tất.
- Lỗi CLI, I/O hoặc lỗi khi copy/hash artifact: notebook dừng run; không ghi completion manifest. Staging chưa hoàn tất được giữ riêng để chẩn đoán và không được xem là kết quả đã publish.
- Run ID phải định danh duy nhất một cấu hình/input snapshot. Muốn chạy với dữ liệu, review hoặc config mới thì tạo run ID mới theo quy tắc hiện hành.

Các gate và trạng thái do CLI hiện tại quyết định. Notebook không được bỏ qua review gate hoặc biến background `unknown` thành verified negative.

## Yêu cầu chức năng

- **FR-1:** Notebook phải mount Drive và nhận đường dẫn gốc chứa raw data, review inputs và run outputs.
- **FR-2:** Notebook phải kiểm tra và ghi Git commit SHA, Python/dependency versions, run ID, config snapshot/hash, input hashes và output hashes.
- **FR-3:** Notebook phải gọi CLI từ đúng checkout và lockfile; notebook cell không được chứa implementation song song của model hoặc evaluator.
- **FR-4:** Raw inputs và review CSV phải nằm ngoài versioned source; notebook phải xác nhận chúng không được stage hoặc commit.
- **FR-5:** Notebook phải hỗ trợ đầy đủ hai kết quả training: `ready` và `pending_positive_review`/pool-insufficient, đồng thời giữ đúng artifact contract của CLI.
- **FR-6:** Notebook chỉ publish run lên Drive sau khi xác minh file/hash; ghi completion manifest là thao tác cuối.
- **FR-7:** Notebook phải lưu stdout/stderr hoặc log tương đương và summary dễ đọc cho run ID, gate statuses, artifact paths, counts và hashes.
- **FR-8:** Local Python final run phải dùng cùng source commit, lockfile, run ID và input snapshot. Git SHA, lock/source-manifest/input hashes, normalized config hash, row keys, schemas, labels, partitions, order và null masks phải khớp chính xác. Khi hash effective config, runtime-specific absolute roots phải được chuẩn hóa thành logical tokens như `<RAW_ROOT>`, `<REVIEW_ROOT>`, `<RUN_ROOT>`. Với numeric columns của feature, score và metric tables, so sánh bằng `numpy.allclose(atol=1e-5, rtol=0)`; báo rõ cột/dòng vượt tolerance. Binary hash của Parquet/GPKG giữa Linux/Windows được lưu để truy vết nhưng không dùng làm tiêu chí duy nhất cho numeric equivalence.
- **FR-9:** Future MaxEnt comparison không thuộc notebook release này; notebook phải để cấu trúc run/method manifest có thể mở rộng nhưng không cài hoặc gọi MaxEnt implementation ở P4 này.
- **FR-10:** Runtime Linux Colab và Windows local phải cài được từ cùng `uv.lock` với `uv sync --locked`; Colab dùng venv `/content/.venv`. Mọi thay đổi lockfile phải qua kiểm tra cài đặt trên Linux và Windows.
- **FR-11:** Notebook cung cấp `publish_interim: false` làm mặc định; khi false, không copy interim outputs lên Drive. Khi true, chép interim vào run staging và thêm hashes vào completion manifest.
- **FR-12:** Notebook template commit không có cell outputs/execution counts/widget state hoặc transient Colab metadata. Executed report được lưu trên Drive dưới dạng Markdown/HTML/artifact riêng.

## Yêu cầu phi chức năng

- **NFR-1 Reproducibility:** cùng source commit, lockfile, run ID, config và input hashes phải tái tạo cùng manifest và deterministic outputs theo contract hiện có. So sánh cross-platform tuân theo exact/tolerance rule của FR-8.
- **NFR-2 Data integrity:** Drive publication không được gắn nhãn hoàn tất cho output thiếu hoặc hash sai.
- **NFR-3 Privacy:** raw data và human review decisions không được commit lên GitHub hoặc notebook output đã commit. Chúng được giữ trên Drive; runtime Colab và local final workspace có thể đọc/cached chúng theo đường dẫn local cấu hình.
- **NFR-4 Failure clarity:** lỗi setup, input, run gate và copy/publish phải được phân biệt; gate pending không bị trình bày như exception hoặc successful fit.
- **NFR-5 Portability:** package/CLI hiện có vẫn chạy ở local Python; Colab không trở thành runtime duy nhất được hỗ trợ.

## Kiểm chứng và nghiệm thu

- Offline tests dùng synthetic fixtures để kiểm tra thứ tự command, cấu hình Drive path, pending/ready statuses, hash manifest, copy failure và không stage dữ liệu private.
- CI matrix cài dependencies bằng `uv sync --locked` trên Windows x86_64 và Linux x86_64 glibc; Colab smoke run xác minh cùng lock hoạt động trong runtime hosted.
- Notebook được kiểm tra cấu trúc cell và syntax trong CI; checks xác nhận outputs rỗng, execution counts null, widget state vắng mặt và metadata theo allowlist. Các helper orchestration quan trọng phải nằm trong code có thể unit test hoặc được gọi trực tiếp từ CLI hiện hữu.
- Colab smoke run dùng synthetic data trước. Preprocessing trên raw data thật có thể chạy khi review gate còn pending; Track B chỉ fit sau khi fit gate positive review đạt yêu cầu, và evaluator giữ trạng thái pending khi capture gate chưa đạt.
- Local final verification xác nhận cùng source SHA, lock hash, input hashes, run ID và kết quả/hash theo tolerance đã định nghĩa.
- Không yêu cầu train dữ liệu thật để chấp nhận code path; hiện thực nghiệm trên dữ liệu thật chỉ được báo sau khi fit/capture gates tương ứng đạt.

## Ngoài phạm vi

- Thay Logistic Regression bằng một MaxEnt estimator khác trong Track B hiện tại.
- Train thật trước khi có tối thiểu 20 event groups reviewed reference đủ feature.
- Lưu raw/review data lên GitHub, đưa secret vào notebook, hoặc tự động thay đổi review decisions.
- Viết lại pipeline logic bên trong notebook hoặc thay đổi Track A/MCDA, AOI, segmentation, feature set, partitions hay review rules.
- Dùng Drive mount như filesystem transaction store; completion manifest là căn cứ duy nhất để nhận biết run đã publish hoàn chỉnh.
- Realtime service, deployment, alerting hoặc full-network inference.

## Câu hỏi được chốt

- Track B hiện tại giữ Logistic Regression presence–background theo spec đã duyệt.
- MaxEnt có thể được thêm về sau như phương pháp so sánh riêng, sau khi có thiết kế, spec và plan được review.
- Colab là môi trường xử lý/train chính; Jupyter notebook là entrypoint; Drive lưu data và artifacts; GitHub quản lý source; local Python chạy cuối để tái chạy/xác minh.

## Tài liệu kỹ thuật tham chiếu

- [uv resolution and platform support](https://docs.astral.sh/uv/concepts/resolution/)
- [uv project environment configuration](https://docs.astral.sh/uv/concepts/projects/config/)
