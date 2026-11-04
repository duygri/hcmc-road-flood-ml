# Reproducibility record

## Colab-first execution

Use [`notebooks/colab_first_workflow.ipynb`](../notebooks/colab_first_workflow.ipynb)
as the preprocessing, Track B training, evaluation and publication entrypoint.
It coordinates existing CLIs with standard Python and `subprocess`; it contains
no shell magics or saved execution output. Run it top to bottom in a fresh Colab
runtime after mounting Drive at `/content/drive` through Colab's Files panel.
The same cells require an equivalent Linux `/content` layout when used in Jupyter.

Set `REPO_URL=https://github.com/duygri/hcmc-road-flood-ml.git`, a required full
40-character `SOURCE_COMMIT`, a fresh `RUN_ID`, and
`DRIVE_ROOT=/content/drive/MyDrive/hcmc-road-flood-ml`. The notebook checks out the
specified SHA in `/content/hcmc-road-flood-ml` and checks `git rev-parse HEAD`
against it. Use a commit available in the public repository; keep this SHA and
raw/review snapshot fixed through the final local comparison.

### Manifest, Drive inputs and scratch space

The tracked `data/raw/manifest.yaml` contains the expected sizes, SHA-256 digests
and source metadata. It does not contain the raw binaries, and changing
`sources.raw_root` does not relocate the tracked manifest. Place every manifest
asset, including supporting documents, under Drive `raw/` using its exact
manifest-relative path, such as `raw/fabdem_tiles/<tile>.tif`.

```text
/content/drive/MyDrive/hcmc-road-flood-ml/
  raw/                         # Matching assets, with manifest subdirectories
  review/                      # Optional; both CSVs may be absent
    review_decisions.csv
    event_reviews.csv
  runs/<run_id>/               # A completed published bundle
    data/processed/<run_id>/
    artifacts/<run_id>/
    reports/runs/<run_id>/
    data/interim/<run_id>/     # Optional; default omitted
    completion_manifest.json
```

The notebook verifies every manifest asset's size and SHA-256 before pipeline
execution. It also requires free `/content` scratch bytes at least equal to the
sum of manifest input sizes plus `OUTPUT_RESERVE_GIB` (default 20 GiB). Increase
this reserve for expected intermediate/output sizes; it is a preflight estimate,
not a promise that processing will fit. Raw files stay read-only on Drive;
derived work is written to the Colab checkout before publication.

Present `review/review_decisions.csv` and `review/event_reviews.csv` files are
copied into ignored `configs/local/`. The notebook creates ignored
`configs/local/sources.yaml` with `sources.raw_root` set to Drive `raw/`. It passes
`--match-reviews configs/local/review_decisions.csv` and
`--event-reviews configs/local/event_reviews.csv` to the pipeline only for files
that exist. Preserve absence on replay; do not substitute synthetic review
labels or convert background roads into negatives. Both match and event review
are needed to establish confirmed positive evidence.

### Isolated locked environment and execution order

The bootstrap downloads `https://astral.sh/uv/0.13.6/install.sh` and passes its
bytes to `subprocess.run(["sh"], input=installer, env=ENV, check=True)`. It installs
a standalone user-owned binary in `/content/uv-bin`, without sudo, and verifies
`uv --version` is exactly `uv 0.13.6`. uv installs/selects Python 3.12. With
`UV_PROJECT_ENVIRONMENT=/content/.venv` and `UV_PYTHON=3.12`, the notebook runs
`uv sync --project <checkout> --locked --group dev`; every project command runs
through `uv run --project <checkout> --locked`. The Colab kernel imports only
standard-library modules; project dependencies stay in `/content/.venv`.

Validate the lock without rewriting it using `uv lock --check` from the pinned
checkout. The notebook's `uv sync --locked` and `uv run --locked` fail if the
lock cannot be used as committed. Do not regenerate it as part of a replay.
The notebook performs these steps:

1. Validate the full source SHA and Drive paths; clone and check out that SHA.
2. Verify source-manifest hashes and scratch capacity, then prepare the local
   source override and copy present review inputs.
3. Run the synthetic smoke command before raw-data execution:
   `uv run --project <checkout> --locked pytest tests/integration/test_track_b.py -q`.
4. Run `scripts/run_pipeline.py --config configs/experiments.yaml --repo-root <checkout> --run-id <run_id>`
   with the present review flags. This produces baseline features, labels,
   confirmed links, frozen partitions and susceptibility scores.
5. Run `scripts/train.py` with the same `--config`, `--repo-root` and `--run-id`,
   plus `--features data/processed/<run_id>/road_features.parquet`,
   `--labels data/processed/<run_id>/road_labels.parquet`,
   `--confirmed-event-segments data/processed/<run_id>/confirmed_event_segments.parquet`
   and `--partitions data/processed/<run_id>/road_partitions.parquet`.
6. Read `reports/runs/<run_id>/track_b.json`. Run `scripts/evaluate.py` with the
   common flags, `--scores data/processed/<run_id>/road_susceptibility.parquet`,
   `--roads data/interim/<run_id>/road_segments.gpkg`,
   `--confirmed-event-segments data/processed/<run_id>/confirmed_event_segments.parquet`
   and `--partitions data/processed/<run_id>/road_partitions.parquet`. Add
   `--track-b-scores data/processed/<run_id>/track_b_scores.parquet` only when
   Track B status is `ready`.
7. Write structured `runtime.json`, completed command stdout/stderr logs and
   `colab_summary.md` into `reports/runs/<run_id>/`, then publish.

Without reviewed positives, Track B reports `pending_positive_review`; a fitted
model is not produced. Insufficient background reports
`background_pool_insufficient`. Missing capture-review quotas report
`pending_label_review` and omit completed capture metrics. A fit or capture gate
can remain pending even when other stages complete. In these cases the notebook
evaluates the baseline without an unavailable Track B score file and publishes
the pending reports. Changed source/configuration needs a new run ID; repeated
training with changed derived input paths/hashes is also rejected before
existing Track B results can be overwritten.

## Publication and final local comparison

The notebook invokes `scripts/publish_run.py` after all required run reports
exist, including `runtime.json` with Python version, uv version and platform.
This is the invocation shape using the notebook's existing variables; the
notebook already performs it, so do not publish the same run twice:

```python
publish_args = [
    UV, "run", "--project", str(CHECKOUT), "--locked", "python",
    "scripts/publish_run.py",
    "--config", "configs/experiments.yaml",
    "--repo-root", str(CHECKOUT),
    "--run-id", RUN_ID,
    "--drive-root", str(DRIVE_ROOT),
]
if PUBLISH_INTERIM:
    publish_args.append("--publish-interim")
subprocess.run(publish_args, cwd=CHECKOUT, env=ENV, check=True)
```

The default is `publish_interim=false`: `PUBLISH_INTERIM = False` in the notebook
means omit the optional `--publish-interim` flag. Enable it explicitly only when
interim artifacts should also be preserved in Drive. Processed outputs, models, reports,
runtime provenance, the summary and preceding command logs preserve their
repository-relative layout inside `<drive_root>/runs/<run_id>/`.

The publisher copies and verifies sizes/hashes, then writes
`completion_manifest.json` last. The manifest records publication status, run ID,
source SHA, runtime, lock/config/input/review hashes, gate statuses, interim
policy and a file inventory. A valid completion manifest and matching file hashes
are required before treating the bundle as complete; a partial copy or failed
verification must not have a valid marker. Existing completed runs are never
automatically deleted or overwritten. Publisher stdout/stderr are saved locally
after publication, outside the just-finalized immutable bundle; preceding
command logs are inside the bundle.

For final Windows/local verification:

1. Sync the complete published bundle and the exact same raw/review snapshot
   from Drive. Keep the published bundle unchanged and separate from local
   replay output. Preserve missing review CSVs as missing.
2. Use the notebook's final **Windows/local replay in a separate checkout**
   Python script (`py replay.py` on Windows after saving and configuring it).
   It clones the public repository into a fresh checkout, detaches at the same
   full SHA, verifies HEAD, selects Python 3.12 and uv 0.13.6, and uses a local
   `.venv`. Point `LOCAL_SNAPSHOT` to the synced Drive root and `REPLAY` to a new
   checkout path; keep `RUN_ID` identical to Colab's.
3. The script verifies the raw snapshot and scratch capacity, writes the local
   source override, copies present reviews, runs the synthetic smoke and then
   pipeline/train/evaluate with the same gate branching. It independently writes
   runtime, summary and command-log counterparts required by the verifier.
   Do not copy Colab derived outputs into this checkout to fill missing files.
4. The script invokes the verifier against the already-produced local outputs.
   Inspect its JSON report and exit code. You can repeat just the comparison
   using the following PowerShell command after setting these paths/ID:

```powershell
$Checkout = (Resolve-Path "local-replay").Path
$Snapshot = (Resolve-Path "drive-snapshot").Path
$RunId = "REPLACE_WITH_THE_SAME_RUN_ID"
$env:UV_PROJECT_ENVIRONMENT = Join-Path $Checkout ".venv"
$env:UV_PYTHON = "3.12"
uv run --project $Checkout --locked python "$Checkout/scripts/verify_colab_run.py" --published-run "$Snapshot/runs/$RunId" --repo-root $Checkout --config configs/experiments.yaml --run-id $RunId --atol 1e-5 --output "$Checkout/reports/runs/$RunId/local_verification.json"
```

`--published-run` is the bundle root containing `completion_manifest.json`, not
its nested reports directory. `--repo-root` is the separate local checkout.
`--output` must remain inside that checkout; the default is
`reports/runs/<run_id>/local_verification.json`. The verifier does not execute
training or replay and does not modify the published bundle. It exits 0 only
when the comparison is equivalent, otherwise exits nonzero and reports concrete
differences (or a validation error).

Source commit, lock/raw/review identity, effective configuration, gate states,
table schemas/CRS, IDs, ordering, labels, partitions and null masks must match
exactly. Machine-specific roots are normalized. Numeric features, scores and
metrics use absolute tolerance `atol=1e-5`, with `rtol=0`. Derived binary hashes
remain traceability evidence; semantic equality is checked from typed tables and
reports rather than requiring identical Parquet/GeoPackage/model bytes. Matching
pending runs can verify without fitted scores/models; a ready run must include
its declared fitted artifacts. A matching pending result does not establish
real-data training or completed capture evaluation.

## Development checks and evidence limits

From the pinned checkout, run:

```powershell
uv --version
uv python install 3.12
uv lock --check
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit/test_notebook_hygiene.py -q
uv run pytest -q
```

Confirm uv reports 0.13.6. The CI matrix in `.github/workflows/ci.yml` targets
`ubuntu-latest` and `windows-latest` with Python 3.12 and uv 0.13.6, locked install,
lint/format and the full synthetic-only suite. Matrix configuration is not proof
that hosted jobs have passed; record their results when available. The synthetic
integration suite creates its own local AOI, OSM, evidence, water, DEM and
WorldCover fixtures, requires no research data and does not process real raw files.

The checked-in notebook must retain empty outputs, null execution counts and no
widget/transient Colab metadata. Keep executed notebooks, credentials, raw data,
review decisions and per-run outputs off GitHub. Store execution summaries and
run artifacts on Drive.

The procedures above document the workflow. They do not assert a live
Colab/Drive execution, a successful real-data Track B fit, a completed final local
comparison or successful hosted Windows/Linux jobs. Record those observations
with their actual run ID, SHA, runtime, gate states and comparison report when
performed. The following measurements are earlier Windows baseline evidence;
they are not observations of this Colab-first workflow.

## Prior Windows baseline execution

The recorded environment was Windows 11, Python 3.12.16, uv 0.13.6, and an AMD
Ryzen 5 5600H (6 cores, 12 logical processors). Two independent full runs used
the same source snapshot and produced 298,199 roads each. Run `hcmc_old_v1_004`
took 2,585.04 seconds (43m 5s), with a measured peak working set of
1,719,775,232 bytes. Run `hcmc_old_v1_005` took 1,748.85 seconds (29m 9s),
with a measured peak working set of 1,780,629,504 bytes. These measurements are
observations from this host, not a performance promise or SLA.

The 2 km tiled raster extraction took 1,730.32 seconds in run 004 and
1,132.88 seconds in run 005 for all 298,199 roads. The 2 km target-space tile
windows reduced source merge/reprojection work from two warps per road to two
warps per occupied tile. A strict comparison against the prior per-road path
passed on synthetic rasters, including adjacent tile boundaries and nodata.
Across the two full runs, raster features, assembled road features and MCDA
scores had identical ordered road IDs and zero numeric difference (all values
within absolute tolerance 1e-6). Their stage output hashes also match their
completion markers; the comparison and SHA-256 inventory are in the local
`reports/runs/hcmc_old_v1_005/replay_comparison.json` artifact.

Both full runs completed with 9,059 candidate links, 7 observations outside
the AOI, and 298,199 unknown labels. The split contained 226,824 reference,
41,621 holdout, and 29,754 guard roads. The baseline scored 281,781 roads from
215,197 reference-fit rows. Water context was incomplete for 7,215 roads and
water distance was incomplete for 11,684 roads. No labels were fabricated.
Evaluation remains `pending_label_review`: there are zero reviewed reference
and holdout events, leaving shortfalls of 20 reference events, 20 holdout
events, and 3 qualifying holdout blocks; no evaluation metrics were emitted.

All 13 original raw files retain the byte sizes and SHA-256 digests recorded
before processing. This includes the PBF, four DEM tiles, WorldCover, IRD
observations and metadata, water and boundary data, and source manifests/readmes.
Run outputs and the detailed comparison live in the ignored local SDD evidence
folder and the ignored per-run output directories.
