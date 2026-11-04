# Colab-first Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Google Colab/Jupyter entrypoint that runs the existing preprocessing, Track B training and evaluation CLIs, stores complete run artifacts on Google Drive, and supports final local Python verification.

**Architecture:** Keep the existing Python package and CLIs as the only implementation of preprocessing, modeling and evaluation. Add a tested Drive run publisher and a local-vs-Colab run comparator; a clean notebook template coordinates Drive mounting, pinned Git checkout, locked environment setup and CLI execution. CI checks the locked project on Linux and Windows, while the notebook keeps user data and executed outputs out of GitHub.

**Tech Stack:** Python 3.12; `uv` 0.13.6 and the existing universal `uv.lock`; Jupyter `.ipynb`; pandas/numpy/GeoPandas; standard library JSON, hashing, subprocess and copy utilities; pytest and Ruff.

**Spec:** `docs/superpowers/specs/2026-11-04-old-hcmc-colab-first-workflow-design.md`

## Global Constraints

- Colab is the primary preprocessing/training/evaluation runtime; the Jupyter notebook is its entrypoint.
- Google Drive stores raw data, review CSVs, processed datasets, models, reports, logs and run manifests. GitHub stores only code and the clean notebook template.
- Each Colab run checks out an explicit Git commit and records the source SHA, Python/uv versions, lock/config/input hashes, run ID and output hashes.
- Set `UV_PROJECT_ENVIRONMENT=/content/.venv`; install the project with `uv sync --locked` and do not install project dependencies into Colab system Python.
- Use one lockfile across Windows x86_64 and Linux x86_64 glibc. Verify `uv lock --check` and `uv sync --locked` on both platforms; add `tool.uv.required-environments` only if the locked package artifact check requires it.
- Track B remains presence–background Logistic Regression with the approved five-feature set. MaxEnt is a future separate comparison, not part of this workflow.
- Background roads stay `unknown`; missing positive review or capture coverage remains a pending gate and never becomes a successful real-data fit.
- Publish `data/interim/<run_id>` to Drive only when `publish_interim: true`; the default is false. Never automatically delete completed Drive runs.
- Write the Drive `completion_manifest.json` only after all required files copy and their hashes verify. An incomplete run has no valid completion manifest.
- Compare exact metadata, schemas, IDs, labels, partitions, ordering and null masks. Compare numeric feature, score and metric values with `numpy.allclose(atol=1e-5, rtol=0)`.
- Repository notebook templates have no saved outputs, non-null execution counts, widget state or transient Colab metadata. Executed summaries go to Drive as Markdown/HTML or run artifacts.
- Use synthetic local fixtures for automated tests and the first Colab smoke run; no secrets, raw data or review decisions are committed to GitHub.

## Review Focus

1. **Partial/failed Drive publication:** quota, interrupted copy, hash mismatch, pending Track B or an existing run must never create/overwrite a valid completion marker. Pin with Task 1's copy/hash failure, pending-artifact and collision tests.
2. **Cross-platform output drift:** different IDs/schema/null masks, duplicate composite Track B keys, or values outside `atol=1e-5, rtol=0` must fail local verification; values within tolerance may pass after stable key sorting and index reset. Pin with Task 2's tolerance-boundary, key/order, duplicate-key and pending-status tests.
3. **Notebook/runtime contamination:** runtime outputs, widget state, credentials, non-pinned code, or system-Python installs must not enter the GitHub template. Pin with Task 3's clean-state, metadata and runtime-environment tests.
4. **Platform lock gaps:** a Windows-generated lock that cannot sync on Linux, or vice versa, must fail CI before a Colab run is offered. Pin with Task 4's `test_ci_matrix_contains_windows_and_linux_and_uses_locked_commands` and the two CI jobs.
5. **Review-gate/replay errors:** missing reviews remain pending, only ready Track B scores reach evaluation, and local replay uses the same snapshot in a separate checkout. Pin with Task 3's review-gate and separate-replay notebook checks.

---

## File Map

| File | Responsibility |
|---|---|
| `src/flood_ml/utils/run_publisher.py` | Copy one completed local run to a Drive run directory, hash copied files and publish a completion manifest last. |
| `scripts/publish_run.py` | CLI wrapper for run publication, called through `uv run` from Colab. |
| `tests/unit/test_run_publisher.py` | Offline temporary-directory tests for required outputs, interim policy, hashes, pending runs and copy failures. |
| `src/flood_ml/utils/run_comparison.py` | Compare a published Colab bundle to a local run using exact metadata/schema checks and numeric tolerance. |
| `scripts/verify_colab_run.py` | Local CLI to run the comparison and write a verification report. |
| `tests/unit/test_run_comparison.py` | Exact/tolerant equality, mismatch, and pending-run tests. |
| `notebooks/colab_first_workflow.ipynb` | Clean template: Drive mount, pinned checkout, isolated uv environment, pipeline/train/evaluate/publish and final local-verification instructions. |
| `tests/unit/test_notebook_hygiene.py` | Standard-library JSON and AST checks for notebook syntax, cell order, gate branches and clean saved state. |
| `tests/unit/test_ci_workflow.py` | Check the CI platform matrix and locked-install/lint/test commands. |
| `.github/workflows/ci.yml` | Locked install, lint, formatting and synthetic test matrix for Windows and Linux. |
| `README.md`, `docs/reproducibility.md` | Colab/Drive setup, expected gate states, Drive publication policy and final local verification procedure. |

### Task 1: Publish completed runs to Google Drive

**Files:** Create `src/flood_ml/utils/run_publisher.py`, `scripts/publish_run.py`, and `tests/unit/test_run_publisher.py`.

**Interfaces:**

- `publish_run(repo_root: Path, drive_root: Path, run_id: str, *, config: Path = Path("configs/experiments.yaml"), publish_interim: bool = False) -> Path`; load configured output roots and return the completed Drive run directory.
- CLI: `python scripts/publish_run.py --config configs/experiments.yaml --repo-root <repo> --run-id <run_id> --drive-root <MyDrive/hcmc-road-flood-ml> [--publish-interim]`.
- Copy all run-scoped files while preserving their relative layout: `data/processed/<run_id>`, `artifacts/<run_id>`, and `reports/runs/<run_id>`; include `data/interim/<run_id>` only when requested. Exclude transient lock and temporary files. Require the pipeline completion marker, run provenance, pipeline/evaluation reports, and Track B report; a `ready` Track B report also requires its declared score and model artifacts, while an approved pending status does not.
- Drive target: `<drive_root>/runs/<run_id>/`; `completion_manifest.json` contains schema version, publication status, run ID, source commit SHA, Python/uv versions and platform, lock/config/input hashes, gate statuses, `publish_interim`, and a relative-path-to-size/SHA-256 file map.
- Derive provenance and source/input/review hashes from `reports/runs/<run_id>/run.json`; derive gate statuses from the pipeline, evaluation and Track B reports. The publisher status means artifact publication completed and must preserve (not upgrade) pending training/evaluation status.
- Preflight selected output byte size and available local/Drive space where the mounted filesystem reports it; catch quota and write failures. Do not assume Drive free-space reporting is authoritative.
- Expected pending statuses are publishable. Track B model/score outputs are optional only when the Track B report says the fit gate is pending or the primary background pool is insufficient.

- [ ] **Step 1: Write failing publisher tests:** `test_publisher_copies_required_run_outputs_and_records_hashes`; `test_publisher_skips_interim_by_default`; `test_publisher_includes_interim_when_enabled`; `test_publisher_accepts_pending_track_b_without_model_or_track_b_scores`; `test_publisher_requires_ready_track_b_artifacts`; `test_publisher_does_not_write_completion_manifest_after_copy_or_hash_failure`; `test_publisher_refuses_existing_run_without_overwriting_it`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_run_publisher.py -q`; confirm the tests fail because publisher code and CLI do not exist.
- [ ] **Step 3: Implement** safe run-root selection, required report/gate validation, selected-byte preflight, recursive copy into `<drive_root>/runs/<run_id>/`, post-copy hash verification and manifest-last publication. Read Python/uv/platform from the notebook's structured runtime record. Reject an existing target without modifying it; preserve incomplete folders without a completion marker for diagnosis and never delete prior completed runs.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_run_publisher.py -q`; confirm all seven publication cases pass.
- [ ] **Step 5: Commit** `feat: publish completed runs to Drive with manifest`.

### Task 2: Verify a local run against the published Colab run

**Files:** Create `src/flood_ml/utils/run_comparison.py`, `scripts/verify_colab_run.py`, and `tests/unit/test_run_comparison.py`.

**Interfaces:**

- `compare_runs(published_run: Path, local_repo_root: Path, run_id: str, *, config: Path = Path("configs/experiments.yaml"), atol: float = 1e-5) -> dict`.
- CLI: `python scripts/verify_colab_run.py --published-run <Drive-sync/runs/run_id> --repo-root <local-checkout> --config configs/experiments.yaml --run-id <run_id> [--atol 1e-5] [--output <report.json>]`.
- Compare exact semantic provenance: run ID, source commit, lock/source-manifest/raw/review input hashes, normalized effective configuration, gate statuses, table schemas, row keys/order, labels, partitions and null masks. Ignore volatile runtime fields (timestamps, elapsed time, peak memory and machine details); do not compare raw fingerprints that encode normalized-away absolute roots.
- Read GeoPackage tables through GeoPandas/pyogrio and Parquet tables through pandas/GeoPandas. Validate non-null unique row keys, then compare the original ordered key sequence and exact schema/null metadata. Before numeric comparisons, stable-sort both road-level frames by `road_segment_id` and `reset_index(drop=True)` so pandas index labels cannot misalign values. For Track B long-form scores, use the composite key `("road_segment_id", "model_id")`; compare its original ordered key sequence exactly, then sort by both columns and reset the index. Compare numeric feature, baseline/Track B score and evaluation metric values with `numpy.allclose(atol=1e-5, rtol=0)`; report each file/column and first mismatching rows.
- Treat missing Track B models/scores as valid only when both runs report the same pending fit status. Do not compare `.joblib`, Parquet or GeoPackage binary SHA as numeric-equivalence proof; keep file hashes in the report for traceability.
- Replay into a separate local checkout pinned to the same commit and a local copy of the same raw/review snapshot; never run the replay into the Colab result checkout or Drive-sync folder. Return `equivalent: true` only when all exact and tolerant checks pass. CLI exits 0 for equivalent runs and nonzero for differences; default report path is `reports/runs/<run_id>/local_verification.json`.

- [ ] **Step 1: Write failing comparison tests:** `test_compare_runs_accepts_numeric_jitter_within_atol`; `test_compare_runs_rejects_numeric_difference_above_atol`; `test_compare_runs_requires_exact_keys_schema_labels_order_and_null_masks`; `test_compare_runs_sorts_and_resets_road_tables_before_numeric_comparison`; `test_compare_runs_uses_composite_track_b_score_key`; `test_compare_runs_rejects_duplicate_row_keys`; `test_compare_runs_rejects_source_lock_or_input_hash_mismatch`; `test_compare_runs_accepts_matching_pending_track_b_without_scores`; `test_compare_runs_reports_ready_model_and_evaluation_statuses`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_run_comparison.py -q`; confirm tests fail because comparison code does not exist.
- [ ] **Step 3: Implement** canonical effective-config comparison with runtime-specific roots normalized to `<RAW_ROOT>`, `<REVIEW_ROOT>` and `<RUN_ROOT>`, typed table comparisons, recursive metric comparison and a JSON CLI report. Do not rewrite existing run provenance or treat volatile report fields as mismatches.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_run_comparison.py -q`; confirm exact checks and tolerance boundaries pass.
- [ ] **Step 5: Commit** `feat: verify local outputs against Colab runs`.

### Task 3: Add the clean Colab/Jupyter workflow notebook

**Files:** Create `notebooks/colab_first_workflow.ipynb` and `tests/unit/test_notebook_hygiene.py`.

**Interfaces:**

- Notebook parameters: public `REPO_URL=https://github.com/duygri/hcmc-road-flood-ml.git`, required full `SOURCE_COMMIT`, `RUN_ID`, `DRIVE_ROOT=/content/drive/MyDrive/hcmc-road-flood-ml`, and `PUBLISH_INTERIM=False`.
- Drive layout: `raw/`, optional `review/review_decisions.csv`, optional `review/event_reviews.csv`, and `runs/<run_id>/`.
- Bootstrap uv 0.13.6 as a standalone binary using Astral's versioned installer URL `https://astral.sh/uv/0.13.6/install.sh`, without sudo/root and without installing project packages into the Colab kernel. Keep the notebook cell in standard Python: download the installer and pass its bytes to `subprocess.run(["sh"], input=..., check=True)`; verify the installed `uv --version`. A direct standalone release binary is an alternate path only when its published checksum is verified. Install/select Python 3.12 with uv. Set `UV_PROJECT_ENVIRONMENT=/content/.venv`; run `uv sync --locked --group dev` and `uv run --project <checkout> --locked ...`. See [Astral's uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
- Code cells use standard Python and `subprocess`; do not use IPython shell magics or import project dependencies into Colab's system kernel.
- Run order: validate full commit SHA and Drive paths; clone/check out that SHA; verify every source-manifest hash and preflight `/content` scratch capacity from the manifest input sizes plus a configurable output reserve; copy present review CSVs to ignored `configs/local/` and generate ignored `configs/local/sources.yaml` with the Drive `raw_root`; run `scripts/run_pipeline.py` with optional `--match-reviews configs/local/review_decisions.csv` and `--event-reviews configs/local/event_reviews.csv`; run `scripts/train.py` with `--features data/processed/<run_id>/road_features.parquet`, `--labels data/processed/<run_id>/road_labels.parquet`, `--confirmed-event-segments data/processed/<run_id>/confirmed_event_segments.parquet`, and `--partitions data/processed/<run_id>/road_partitions.parquet`; read `reports/runs/<run_id>/track_b.json`; rerun `scripts/evaluate.py` with `--scores data/processed/<run_id>/road_susceptibility.parquet`, `--roads data/interim/<run_id>/road_segments.gpkg`, `--confirmed-event-segments data/processed/<run_id>/confirmed_event_segments.parquet`, and `--partitions data/processed/<run_id>/road_partitions.parquet`, adding `--track-b-scores data/processed/<run_id>/track_b_scores.parquet` only when Track B status is ready; write structured `runtime.json`, captured stdout/stderr and `colab_summary.md` under `reports/runs/<run_id>/`; call `scripts/publish_run.py`.
- If review CSVs are absent, preserve the no-review setup and show `pending_positive_review`; do not create synthetic real-data labels. If a Track B gate is pending, evaluate the baseline without a Track B score file and publish the pending report.
- Before raw-data execution, run the synthetic smoke test `uv run --project <checkout> --locked pytest tests/integration/test_track_b.py -q`.
- Last notebook section shows the Windows/local Python commands to create a separate checkout at the same full SHA, sync the same raw/review snapshot, replay with the same run ID without touching the Colab result, and execute `scripts/verify_colab_run.py` against the Drive-synced bundle.

- [ ] **Step 1: Write failing notebook checks:** `test_notebook_template_is_valid_json_and_python_cells_parse`; `test_notebook_template_has_empty_outputs_and_null_execution_counts`; `test_notebook_template_has_no_widget_or_transient_colab_metadata`; `test_notebook_contains_no_credentials_or_user_data_paths`; `test_notebook_runs_pipeline_train_evaluate_publish_cells_in_order`; `test_notebook_pins_source_and_uses_uv_project_environment`; `test_notebook_keeps_missing_reviews_pending_and_only_passes_ready_track_b_scores`; `test_notebook_documents_same_snapshot_replay_in_separate_checkout`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_notebook_hygiene.py -q`; confirm the new notebook is missing and the checks fail.
- [ ] **Step 3: Implement** notebook cells with the declared order, optional review inputs, fit-gate branching, log capture and final Drive summary. Call every project command through `uv run` in the isolated environment; keep cell outputs empty in the checked-in template.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_notebook_hygiene.py -q` and `uv run pytest tests/integration/test_track_b.py -q`; confirm template checks and synthetic workflow smoke pass.
- [ ] **Step 5: Commit** `feat: add Colab-first workflow notebook`.

### Task 4: Verify the lockfile and test suite on Windows and Linux

**Files:** Modify `.github/workflows/ci.yml`; create `tests/unit/test_ci_workflow.py`; modify `pyproject.toml` and `uv.lock` only if the cross-platform locked install proves an environment constraint is required.

**Interfaces:**

- CI matrix targets `windows-latest` and `ubuntu-latest`, both with Python 3.12 and uv 0.13.6.
- Each target runs `uv lock --check`, `uv sync --locked --group dev`, Ruff check/format and the full synthetic-only pytest suite. Pin the setup action's `version` input to uv 0.13.6.
- If a locked package cannot install from an available wheel/source on one target, document the missing native build requirements or add the narrowest valid `tool.uv.required-environments` constraint and regenerate the universal lock; do not silently drop a platform from the matrix.

- [ ] **Step 1: Write failing `test_ci_matrix_contains_windows_and_linux_and_uses_locked_commands`** using the existing YAML dependency; confirm current Windows-only workflow fails the check.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_ci_workflow.py -q`; confirm the new matrix check fails before the workflow update.
- [ ] **Step 3: Implement** a two-platform GitHub Actions matrix with locked Python/uv versions, `uv lock --check`, install, lint, format and full tests.
- [ ] **Step 4: Run** `uv lock --check`, `uv sync --locked --group dev`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest -q` locally; verify the GitHub Actions Windows and Linux jobs both complete successfully before merge.
- [ ] **Step 5: Commit** `ci: test locked environments on Linux and Windows`.

### Task 5: Document Colab setup and local final verification

**Files:** Modify `README.md` and `docs/reproducibility.md`.

**Interfaces:** Documentation points to `notebooks/colab_first_workflow.ipynb`, `scripts/publish_run.py`, and `scripts/verify_colab_run.py` with exact flags and Drive layout.

- [ ] **Step 1: Write/update docs** for raw manifest vs Drive raw assets, optional review CSVs, pinned Git SHA, Drive paths, `/content/.venv`, uv lock checks, `publish_interim=false`, pending fit/capture states, completion manifest and local final run.
- [ ] **Step 2: Review** every documented command against actual CLI `--help` and notebook parameters; correct any path/flag mismatch.
- [ ] **Step 3: Run final checks:** `uv lock --check`; `uv run ruff check .`; `uv run ruff format --check .`; `uv run pytest -q`; confirm CI matrix covers Windows and Linux and the notebook template has no output state.
- [ ] **Step 4: Commit** `docs: document Colab-first execution and verification`.

## Cross-task Interface Check

| Producer → consumer | Contract |
|---|---|
| Task 1 publisher → Task 2 comparer | Published root `<drive_root>/runs/<run_id>/`, with `completion_manifest.json`, preserved `data/processed`, `artifacts`, `reports`, and optional `data/interim`. |
| Task 1 publisher → Task 3 notebook | `scripts/publish_run.py --config --repo-root --run-id --drive-root [--publish-interim]`; expected pending gates are publishable, incomplete copy has no valid completion marker. |
| Task 2 comparer → Task 3 notebook/docs | `scripts/verify_colab_run.py --published-run --repo-root --run-id [--atol 1e-5] [--output]`, exit 0 only when exact checks and numeric tolerance pass. |
| Task 3 notebook → existing pipeline | `scripts/run_pipeline.py` writes `road_features.parquet`, `road_labels.parquet`, `confirmed_event_segments.parquet`, `road_partitions.parquet`, `road_susceptibility.parquet`, `road_segments.gpkg`, and run reports under configured run roots. |
| Task 3 notebook → Track B/evaluator | `scripts/train.py` receives all four required processed inputs; only `track_b.json.status == "ready"` permits passing `track_b_scores.parquet` to `scripts/evaluate.py`, which always receives baseline scores, roads, confirmed links and partitions. |
| Task 4 CI → all deliverables | Locked install and tests run on both Windows and Linux; notebook hygiene, publisher/comparer, and CI matrix tests are in the same suite. |

## Completion Checklist

- [ ] All five tasks committed and each task received its scoped independent review.
- [ ] Whole-branch review CLEAN after the final commit.
- [ ] `uv lock --check`, Ruff check/format and `uv run pytest -q` pass on Windows.
- [ ] GitHub Actions locked install and full synthetic suite pass on Windows and Linux.
- [ ] Colab synthetic smoke passes; a real-data run may remain `pending_positive_review` until reviewed-positive quotas exist.
- [ ] Drive test publication has a valid completion manifest; injected copy/hash failures do not.
- [ ] Local final comparison passes exact metadata checks and numeric tolerances, or records concrete differences.
- [ ] GitHub contains only the clean notebook template; raw/review inputs and executed notebook outputs remain off-repo.
