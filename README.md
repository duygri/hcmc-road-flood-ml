# HCMC road flood susceptibility research

This repository builds a reproducible, reviewable baseline for relative flood
susceptibility of road segments inside pre-2025 Ho Chi Minh City. It retains
unreviewed flood reports as unknown evidence and does not interpret them as
verified negatives. MCDA scores are relative indices, not flood probabilities
or forecasts. The current milestone is a complete baseline pipeline; evaluation
stays pending until its independent human review quotas are met.

## Start in Colab

Open [`notebooks/colab_first_workflow.ipynb`](notebooks/colab_first_workflow.ipynb)
in Colab and mount Google Drive at `/content/drive` using the Files panel. Use a
fresh runtime and run the standard-Python cells top to bottom. Set these notebook
parameters first:

```python
REPO_URL = "https://github.com/duygri/hcmc-road-flood-ml.git"
SOURCE_COMMIT = ""  # Required: the full 40-character source commit SHA.
RUN_ID = ""  # Required: a fresh run ID.
DRIVE_ROOT = Path("/content/drive/MyDrive/hcmc-road-flood-ml")
PUBLISH_INTERIM = False
OUTPUT_RESERVE_GIB = 20  # Increase to cover expected intermediate/output space.
```

The notebook imports `Path` before this parameter block. The selected commit
must be available in the public repository; both Colab and the final local
replay must use that exact SHA and the same raw/review snapshot.

Prepare this layout in Drive:

```text
hcmc-road-flood-ml/
  raw/                         # Assets at their manifest-relative paths
  review/                      # Optional human review inputs
    review_decisions.csv       # Optional observation-to-road decisions
    event_reviews.csv          # Optional event/location/extent reviews
  runs/<run_id>/               # Created by the publisher
    data/processed/<run_id>/
    artifacts/<run_id>/
    reports/runs/<run_id>/
    data/interim/<run_id>/     # Only with PUBLISH_INTERIM = True
    completion_manifest.json
```

The tracked `data/raw/manifest.yaml` supplies source sizes, SHA-256 hashes and
metadata; it is not the raw dataset. Put the matching binaries and supporting
files in Drive `raw/`, preserving subdirectories. The notebook verifies every
manifest asset and checks `/content` free space against the manifest input sizes
plus the output reserve. Raw/review files remain off GitHub.

Bootstrap installs standalone uv **0.13.6** from its versioned Astral installer,
selects Python **3.12**, and sets `UV_PROJECT_ENVIRONMENT=/content/.venv`. Locked
dependencies are installed with `uv sync --locked --group dev`; project commands
use `uv run --project <checkout> --locked` in that environment. Project packages
are never installed into Colab's system kernel.

The notebook runs the synthetic Track B smoke test before real-data processing,
then pipeline → training → evaluation → publication under one run ID. Present
review CSVs are copied into ignored `configs/local/`; absent CSVs stay absent.
Without reviewed positives, Track B remains `pending_positive_review` and
capture evaluation remains `pending_label_review`. Other unmet fit/capture gates
also remain explicit. Only `track_b.json.status == "ready"` permits evaluation
with Track B scores; background roads remain unknown.

Publication uses `scripts/publish_run.py` and defaults to `publish_interim=false`
(`PUBLISH_INTERIM = False` in the notebook; omit `--publish-interim` on the CLI).
It copies run-scoped outputs, reports and preceding command logs, verifies their
hashes, then writes `completion_manifest.json` last. A missing or invalid marker
means the bundle is incomplete. Existing published run IDs are not overwritten;
choose a new ID for changed source/configuration or Track B inputs.

For final verification, sync the completed bundle and the same raw/review
snapshot to Windows/local storage, create a separate checkout at the same SHA,
and execute the notebook's final local replay script. It independently produces
the local outputs and report/log counterparts before calling
`scripts/verify_colab_run.py`. The verifier compares existing runs; it does not
launch replay. See [the exact publication and verification commands](docs/reproducibility.md#publication-and-final-local-comparison)
and the [full reproducibility procedure](docs/reproducibility.md).

## Local Windows environment

Install standalone uv 0.13.6, then open PowerShell at the pinned checkout root:

```powershell
uv --version
uv python install 3.12
uv lock --check
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Confirm `uv --version` reports `uv 0.13.6`. Commands and tests work offline after
dependencies and research data are present. CI configures locked installs and
synthetic tests on `ubuntu-latest` and `windows-latest`; it does not download
research data. Live Colab execution, hosted CI results and final real-data
comparison must be recorded separately when run.

## Configure local source paths

`configs/experiments.yaml` links the shared configuration sections and owns the
run ID, seed, output roots, split, MCDA and evaluation rules. Configured paths
resolve from the repository root, independent of the current working directory.
The tracked default reads `data/raw`. To use the downloaded raw files outside
the checkout, edit the ignored `configs/local/sources.yaml`:

```yaml
sources:
  raw_root: 'D:/hcmc-road-flood-ml/data/raw'
```

The local path and human review CSVs are never committed. Raw source binaries
are read-only; the pipeline validates the manifest and every configured checksum
before processing.

## Run the baseline pipeline locally

This command runs the baseline pipeline. The notebook adds Track B training,
gated evaluation, runtime/log recording and Drive publication around it.

```powershell
uv run python scripts/run_pipeline.py --config configs/experiments.yaml --run-id hcmc_old_v1_001
```

Each run writes under its own folders:

- `data/interim/<run_id>/`: AOI, OSM road segments, normalized observations,
  and raster/water intermediate features.
- `data/processed/<run_id>/`: road features, nullable labels, review links,
  spatial partitions, analysis dataset and susceptibility scores.
- `reports/runs/<run_id>/`: source/stage audits, offline report, sensitivity,
  evaluation gate and completion record.
- `artifacts/<run_id>/`: normalization rules and offline maps.

The completion record is written only after every required output is present
and hashed. Stage outputs are reused only when their run fingerprint, dependency
hashes and output hashes still match. A changed input requires a new `--run-id`;
modified outputs are rebuilt with their dependent stages.

You can run only selected stages and their prerequisites:

```powershell
uv run python scripts/run_pipeline.py --config configs/experiments.yaml --run-id hcmc_old_v1_002 --stage observations
```

To prepare normalized observations separately:

```powershell
uv run python scripts/prepare_observations.py --config configs/labeling.yaml --run-id hcmc_old_v1_003
```

## Human review

The optional local review files are:

- `configs/local/review_decisions.csv`: observation-to-road decisions.
- `configs/local/event_reviews.csv`: event and location group memberships,
  plus `affected_geometry_complete`.

Only the explicit `accepted` decision creates a positive road label, and it
requires both match and event review. Blank completeness values mean false.
Setting `affected_geometry_complete=true` attests that the accepted full road
segments form the entire affected extent. Candidate matches are never labels.
Review instructions and the meaning of partial geometry are documented in
[`docs/methodology.md`](docs/methodology.md).

The pipeline can produce a baseline while those files are absent or header-only.
In that state `road_labels.label_binary` remains null and `evaluation_status` is
`pending_label_review`; no capture metrics are presented as completed.

## Research limits and data use

The data, model and reproducibility cards describe source vintages, spatial
resolution, media bias, coverage assumptions, unknown labels and license terms.
IRD and FABDEM have noncommercial restrictions in the current inventory; OSM
requires attribution and ODbL handling. Review source conditions before sharing
derived products.

See [SPEC.md](SPEC.md) for the approved analysis contract,
[`docs/methodology.md`](docs/methodology.md) for the processing choices,
[`docs/data_card.md`](docs/data_card.md), [`docs/model_card.md`](docs/model_card.md),
and [`docs/reproducibility.md`](docs/reproducibility.md) for run evidence.
