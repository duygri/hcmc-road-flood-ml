# Old HCMC Track B Presence–Background Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Track B presence–background Logistic Regression experiment for old HCMC roads, with honest review gates and capture comparisons against current baselines.

**Architecture:** Keep Track A/MCDA unchanged. Add a pure sampling module, a model module, and a standalone training CLI that writes local scores, model artifacts, and a provenance report. Extend the existing capture evaluator to compare each Track B draw on the same eligible holdout population as MCDA, elevation-only, and random.

**Tech Stack:** Python >=3.11; pandas, numpy, scikit-learn, joblib, pytest; existing GeoPandas/Parquet and run-provenance utilities. Declare joblib directly because the CLI imports it; it is already resolved through scikit-learn.

**Spec:** `docs/superpowers/specs/2026-11-03-old-hcmc-track-b-presence-background-design.md`

## Global Constraints

- Scope is the TPHCM old-HCMC study area only; keep the current AOI, road IDs, features, frozen partitions, and Track A/MCDA behavior unchanged.
- Use exactly `elevation_median_m`, `relative_elevation_m`, `slope_mean_deg`, `dist_water_m`, and `builtup_ratio_250m` as the first Track B feature set.
- Only reviewed positive roads in `reference` may be presence rows; fit preprocessing on each reference draw only.
- Candidate background rows must have `label_status=unknown` and be eligible reference roads; never persist them as `label_binary=0` or `verified_negative`.
- Exclude holdout, guard, excluded, structurally ineligible, incomplete-feature, and verified-negative roads from fitting and background sampling.
- Use primary ratio 1:10, sensitivity ratios 1:5 and 1:20, and seeds 42–61 for each ratio; sample background without replacement within each draw.
- Use `StandardScaler` + `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)`; publish `decision_function` as `relative_score`, never a flood probability.
- Keep the fit gate at >=20 independent reviewed reference events with at least one feature-eligible positive road per event. Keep the existing capture gate: >=20 reviewed events in reference and holdout, >=3 holdout blocks, >=5 events per qualifying holdout block, complete reviewed affected geometry, and required score coverage.
- Report Track B results as exploratory held-out event capture/lift and sampling stability; do not report accuracy, precision, recall, F1, ROC-AUC, or calibration as flood-vs-no-flood classification metrics.
- Use synthetic offline fixtures only; do not download sources or add local/raw/processed/model artifacts to Git. Keep model files under ignored `artifacts/<run_id>/`.

## Review Focus

1. **No reviewed positives or fewer than 20 eligible reference events:** sampling returns `pending_positive_review`; the CLI writes a status report and publishes no model or scores. Pin in Tasks 1 and 3.
2. **Positive/unknown/verified-negative or partition confusion:** only reviewed reference positives and unknown reference background can enter a draw; holdout/guard/excluded and verified-negative never enter. Pin in Task 1.
3. **Missing, nonfinite, or forbidden features:** affected rows are excluded or fail validation with an auditable reason; holdout statistics never affect scaling. Pin in Tasks 1 and 2.
4. **Small background pool and seed/block allocation:** no-replacement draws are deterministic, quotas are recorded, and insufficient primary background yields no partial model publication. Pin in Tasks 1 and 3.
5. **Incomplete model score coverage or event geometry:** capture metrics remain pending; when ready, all compared methods use the same holdout road-length population. Pin in Tasks 4 and 5.

## File Map

| Files | Responsibility |
|---|---|
| `src/flood_ml/training/{__init__,sampling}.py`; `tests/unit/test_presence_background_sampling.py` | Validate training population, review gate, and deterministic block-stratified background draws. |
| `src/flood_ml/training/presence_background.py`; `tests/unit/test_presence_background.py` | Fit/score the fixed sklearn Pipeline and return long-form relative scores. |
| `configs/experiments.yaml`, `scripts/train.py`; `tests/unit/test_train_cli.py` | Track B settings, input/output protection, run provenance, atomic outputs, and expected pending statuses. |
| `src/flood_ml/evaluation/capture.py`, `scripts/evaluate.py`; `tests/unit/test_capture.py` | Add named Track B methods to the existing gate and common-coverage evaluation. |
| `tests/integration/test_track_b.py`, `docs/methodology.md`, `docs/model_card.md` | Exercise the synthetic ready path and document the research-only model and current data gate. |

Before implementation, use `superpowers:using-git-worktrees` to create an isolated worktree. Execute tasks 1–5 in order; use a fresh implementer and a fresh task reviewer for each task, then perform one whole-branch review. Do not dispatch sibling tasks against shared files concurrently.

---

### Task 1: Build the reviewed reference population and background draws

**Files:** Create `src/flood_ml/training/__init__.py`, `src/flood_ml/training/sampling.py`, and `tests/unit/test_presence_background_sampling.py`.

**Interfaces:**

- `BackgroundDraw(model_id: str, background_ratio: int, seed: int, X: pd.DataFrame, y: np.ndarray, road_ids: tuple[str, ...], manifest: dict)`; `X` contains the five approved columns only; `y` is in-memory (`1` presence, `0` sampled background) and is never written to labels or score files.
- `SamplingResult(status: Literal["ready", "pending_positive_review", "background_pool_insufficient"], gate: dict, draws: tuple[BackgroundDraw, ...], failures: tuple[dict, ...])`.
- `build_background_draws(features: pd.DataFrame, labels: pd.DataFrame, confirmed_event_segments: pd.DataFrame, partitions: pd.DataFrame, config: dict) -> SamplingResult`.
- `model_id` format: `track_b_r{background_ratio}_s{seed}`. A ready result has all 20 primary-ratio draws; unavailable sensitivity draws are listed in `failures`.

- [ ] **Step 1: Write failing tests:** `test_fit_gate_requires_twenty_feature_eligible_independent_reference_events` asserts pending status and zero draws below quota; `test_background_draws_use_only_reviewed_reference_positives_and_unknowns` asserts positive/background ID membership and their internal `y`; `test_background_draws_exclude_verified_negative_and_nonreference_rows` asserts those IDs never appear; `test_incomplete_or_nonfinite_training_features_are_excluded` asserts affected road IDs and exclusion reasons; `test_draw_X_is_exact_allowlist_and_labels_are_unchanged` asserts `X.columns` equals the five feature constants and labels are unchanged; `test_background_draws_repeat_by_seed_without_replacement` asserts equal-seed IDs/quotas match and each draw has unique background IDs; `test_small_background_pool_is_audited` asserts `background_pool_insufficient` for the affected draw.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_presence_background_sampling.py -q`; confirm the new tests fail because the sampling module does not exist.
- [ ] **Step 3: Implement** the dataclasses and `build_background_draws`. Reuse `evaluation_gate()` for reviewed independent event assignments and read its `reviewed_reference_events` count; do not require the overall capture gate to be ready before fitting, because the holdout quota belongs to evaluation. Then require >=20 reference event groups with at least one complete feature-eligible positive road per group. Allocate background quotas by eligible unknown candidate counts in reference blocks, with largest-remainder ties by `block_id`; draw deterministically using `numpy.random.default_rng(seed)` and without replacement.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_presence_background_sampling.py -q`; confirm all sampling/gate tests pass.
- [ ] **Step 5: Commit** `feat: sample Track B background roads by reference block`.

### Task 2: Fit the presence–background ranker

**Files:** Create `src/flood_ml/training/presence_background.py` and `tests/unit/test_presence_background.py`.

**Interfaces:**

- `FittedTrackBModel(model_id: str, background_ratio: int, seed: int, estimator: sklearn.pipeline.Pipeline, manifest: dict)`.
- `fit_track_b(draw: BackgroundDraw, *, C: float = 1.0, max_iter: int = 1000) -> FittedTrackBModel`.
- `score_track_b(models: Sequence[FittedTrackBModel], features: pd.DataFrame) -> pd.DataFrame`, returning columns `road_segment_id`, `model_id`, `background_ratio`, `seed`, `relative_score`, and `partition`.

- [ ] **Step 1: Write failing tests:** `test_fit_track_b_uses_fixed_scaled_logistic_pipeline` asserts scaler and LogisticRegression settings; `test_scaler_is_fit_only_on_background_draw` asserts `scale.mean_ == draw.X.mean()` even when the scoring frame contains extreme holdout values; `test_score_track_b_uses_decision_function` asserts finite relative scores without calling `predict_proba`; `test_score_track_b_excludes_missing_nonfinite_or_ineligible_rows` asserts no structurally ineligible or incomplete road receives a score; `test_scores_are_long_form_unique_and_repeatable` asserts the declared schema, stable ordering, all rows belong to the supplied holdout frame, and unique `(road_segment_id, model_id)` keys.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_presence_background.py -q`; confirm the tests fail before implementation.
- [ ] **Step 3: Implement** `fit_track_b` as `Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter))])`; fit only `draw.X/draw.y`. Implement `score_track_b` using the model's `decision_function` and the same five-feature eligibility policy. Do not add or call `predict_proba`.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_presence_background.py -q`; confirm scaling, scoring, and schema tests pass.
- [ ] **Step 5: Commit** `feat: fit relative presence-background road scores`.

### Task 3: Add Track B configuration and training CLI

**Files:** Modify `configs/experiments.yaml`, `pyproject.toml`, and `uv.lock`; create `scripts/train.py` and `tests/unit/test_train_cli.py`.

**Interfaces:**

- `experiments.track_b`: `feature_columns` (the exact five-column list), `primary_background_ratio: 10`, `sensitivity_background_ratios: [5, 20]`, `seed_start: 42`, `seed_repetitions: 20`, `minimum_reference_events: 20`, and `logistic_regression: {C: 1.0, max_iter: 1000}`.
- CLI: `python scripts/train.py --config ... --run-id ... --features ... --labels ... --confirmed-event-segments ... --partitions ...` with repository-relative derived input paths, matching existing standalone CLI conventions.
- Outputs: `data/processed/<run_id>/track_b_scores.parquet` (long-form schema from Task 2, eligible holdout roads only), `artifacts/<run_id>/track_b_models/<model_id>.joblib`, and `reports/runs/<run_id>/track_b.json`. Full-network inference is outside this evaluation step; saved model artifacts can support it later.

- [ ] **Step 1: Write failing tests:** `test_track_b_config_rejects_invalid_ratio_and_seed_values` asserts malformed config raises `PipelineError`; `test_train_cli_without_review_writes_pending_only` asserts status `pending_positive_review` and no model/score files; `test_train_cli_primary_background_shortage_publishes_no_partial_outputs` asserts expected pending status and no partial outputs; `test_train_cli_ready_writes_scores_models_and_provenance` asserts paths/schema/run/input hashes; `test_train_cli_rejects_input_output_alias` asserts alias protection before writes.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_train_cli.py -q`; confirm the CLI/config tests fail before implementation.
- [ ] **Step 3: Implement** the config validation and CLI. Declare joblib directly in `pyproject.toml` and refresh the lock using the already resolved scikit-learn dependency. Mirror `scripts/run_baseline.py` for input hashing, preflight path checks, `start_run`, and atomic Parquet publication. Score only eligible `holdout` roads for the 60 repeated comparison draws to keep the evaluation artifact bounded. Save each fitted sklearn Pipeline via a same-directory temporary file and `os.replace`; keep those files under ignored artifacts. The report records config/input/partition hashes, run/git/lock provenance, counts, quotas, exclusions, per-draw status, and model versions. Expected gates (`pending_positive_review`, `background_pool_insufficient`) write the report and return success without publishing model or score files; malformed inputs and I/O errors return nonzero.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_train_cli.py -q`; confirm pending and ready paths pass and no failed run leaves final artifacts.
- [ ] **Step 5: Commit** `feat: add Track B training CLI and artifacts`.

### Task 4: Compare Track B runs in the existing capture evaluator

**Files:** Modify `src/flood_ml/evaluation/capture.py`, `scripts/evaluate.py`, and `tests/unit/test_capture.py`.

**Interfaces:**

- Extend `evaluate_capture(scores, roads, confirmed_event_segments, partitions, config, *, track_b_scores: pd.DataFrame | None = None) -> dict`.
- Add optional CLI input `--track-b-scores`; hash it with other inputs and pass its long-form rows to `evaluate_capture`.
- `report["methods"]["track_b"][model_id]` contains top-k capture, capture-any/all, lift vs random, comparison to elevation-only/MCDA, and the same event/block bootstrap uncertainty used by other ranked methods.
- `report["track_b_stability"]` reports median/IQR metrics by background ratio and mean pairwise Jaccard overlap of selected road IDs across seeds at each top-k fraction.

- [ ] **Step 1: Write failing tests:** `test_capture_without_track_b_keeps_legacy_methods` asserts existing keys/results remain unchanged; `test_capture_reports_each_track_b_draw_on_common_population` asserts every method has the same eligible road length and each draw has capture/lift; `test_capture_missing_track_b_score_returns_pending_coverage` asserts no metric is published when a model misses an eligible holdout road; `test_capture_geometry_gate_withholds_track_b_metrics` asserts incomplete geometry returns pending; `test_capture_rejects_duplicate_or_nonfinite_track_b_scores` asserts malformed long-form rows raise `PipelineError`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_capture.py -q`; confirm existing tests remain green and new Track B tests fail.
- [ ] **Step 3: Implement** the optional long-form score join, named per-model rankings, common eligible holdout coverage check, ratio/seed stability summary, and CLI loading/hash support. Preserve current event geometry gates, top-k rules, random repetitions, and bootstrap settings. If the gate or Track B coverage is incomplete, return pending without Track B metrics.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_capture.py -q`; confirm both legacy and Track B cases pass.
- [ ] **Step 5: Commit** `feat: evaluate Track B scores against baselines`.

### Task 5: Verify the synthetic experiment and document its limits

**Files:** Create `tests/integration/test_track_b.py`; modify `docs/methodology.md` and `docs/model_card.md`.

**Interfaces:** Integration fixture provides only synthetic `road_features`, `road_labels`, confirmed event links, partitions, roads, and Track B config; it must not use local research data or network access.

- [ ] **Step 1: Write failing integration tests:** `test_track_b_integration_without_review_is_pending` asserts no model/score artifact and no capture metrics; `test_track_b_train_then_capture_uses_all_gates` builds >=20 feature-eligible reference event groups, >=20 complete reviewed reference events, and >=20 complete holdout events (>=3 holdout blocks, >=5 per qualifying block), then asserts all configured draws and comparable exploratory metrics are emitted on one denominator.
- [ ] **Step 2: Run** `uv run pytest tests/integration/test_track_b.py -q`; confirm the ready/pending end-to-end tests fail before integration changes.
- [ ] **Step 3: Implement** the fixture wiring for sampler → model → scores → capture evaluator. Assert raw/config inputs remain unchanged and each method uses the same holdout road length. Update methodology/model card with Track B's feature list, background semantics, gates, ranking interpretation, sampling sensitivity, reporting-bias limitation, and current real-data status `pending_positive_review`.
- [ ] **Step 4: Run** `uv run pytest tests/integration/test_track_b.py -q`, then `uv run pytest -q`, `uv run ruff check .`, and `uv run ruff format --check .`; resolve failures before completion.
- [ ] **Step 5: Commit** `docs: document Track B experiment and review gates`.

## Plan Self-Review

- Spec coverage: sampling/unknown semantics and reference gate are Task 1; fixed model and train-only scaling are Task 2; reproducible outputs and pending states are Task 3; same-coverage capture and stability are Task 4; offline ready/pending verification and documentation are Task 5.
- Interface consistency: Task 1's `BackgroundDraw` is Task 2's fit input; Task 2's score schema is Task 3's Parquet output and Task 4's evaluator input; Task 4's output is verified in Task 5.
- Review Focus: each of the five failure/input classes above is assigned to named tests in the owning task.
- Execution: tasks are ordered by interface dependency. Use a new implementer and reviewer per task, commit each reviewed task, then run the required whole-branch review before handoff.
