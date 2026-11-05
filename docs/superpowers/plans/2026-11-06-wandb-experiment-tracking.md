# W&B Experiment Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log Track B training and evaluation summaries to the existing W&B project for each Colab-first run without sending raw data or changing local/Drive artifacts.

**Architecture:** Add one small optional W&B tracking helper used by the existing `scripts/train.py` and `scripts/evaluate.py` CLIs. Each CLI creates a stage run named from the shared `RUN_ID` and groups it under that `RUN_ID`; the Colab notebook reads the API key from Colab Secrets into the subprocess environment without printing or persisting it.

**Tech Stack:** Python 3.12 in Colab's uv-managed project environment, `wandb`, existing CLI/report JSON contracts, pytest synthetic fixtures, Jupyter notebook JSON.

**Spec:** `docs/superpowers/specs/2026-11-06-wandb-experiment-tracking-design.md`

## Global Constraints

- Keep Colab as the primary runtime and use the existing `/content/.venv` plus `uv sync --locked` workflow.
- Use W&B entity `duygri-saigon-university-sgu-` and project `hcmc-road-flood-ml` by default, with environment overrides for portability.
- Read credentials only from `WANDB_API_KEY`; the Colab notebook reads this value from Colab Secrets and passes it through the subprocess environment without printing it.
- Log Track B and evaluation as separate stage runs with `group=<RUN_ID>`; do not create a synthetic run when a gate is pending.
- Do not upload raw datasets, review CSVs, feature tables, model binaries, input paths, or other Drive artifacts to W&B.
- Missing credentials or a W&B service failure must not change model gates, local report contents, or Drive publication; make skipped/failed tracking visible.
- Keep synthetic fixtures for automated verification. Do not start or resume real-data training in this implementation task.

## Review Focus

1. Missing `WANDB_API_KEY` skips W&B initialization and preserves existing CLI/report behavior; pin this in the tracking-helper test.
2. Invalid credentials or service/network errors are reported clearly without deleting or corrupting locally published reports; pin the helper's failure result and CLI's non-blocking behavior.
3. `pending_positive_review` and background-pool-insufficient outcomes are logged as pending without fabricated fit metrics or model artifacts; pin this in Track B CLI tests.
4. Pending capture evaluation records gate/status counts but no fabricated capture metrics; pin this in evaluation CLI tests.
5. Nested evaluation reports contain strings, nulls, lists and detailed provenance; only finite numeric summary metrics are sent, and raw paths/review content never are; pin this in helper/evaluation tests.

---

### Task 1: Add the shared W&B stage logger and dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/flood_ml/utils/wandb_tracking.py`
- Create: `tests/unit/test_wandb_tracking.py`

**Interfaces:**
- Produces `log_stage_report(*, run_id: str, stage: str, config: Mapping[str, object], summary: Mapping[str, object], metrics: Mapping[str, object]) -> str`; the helper accepts runtime JSON values and logs only finite numeric metric leaves, excluding booleans.
- Allowed stages are `track-b` and `evaluation`; return values are `logged`, `skipped_no_api_key`, or `failed`.
- Read `WANDB_ENTITY` and `WANDB_PROJECT` from the environment, defaulting to `duygri-saigon-university-sgu-` and `hcmc-road-flood-ml`.
- Create a W&B run named `<run_id>-<stage>`, with `group=run_id` and `job_type=stage`; log config and summary separately from numeric metrics. Import `wandb` lazily after confirming credentials exist.

- [ ] **Step 1: Write focused helper tests** for no-key skip, successful run naming/group/config/metric logging, filtering of non-finite or non-numeric metrics, and service-error reporting with local-safe return status.
- [ ] **Step 2: Run the helper tests and confirm they fail** for the missing helper behavior.
- [ ] **Step 3: Add `wandb` to the project dependency list and implement `log_stage_report`** with lazy import, a `try/finally` around run finalization, and no exception propagation for remote logging failures.
- [ ] **Step 4: Refresh and validate the universal lockfile** with the project's pinned uv version; confirm `uv lock --check` succeeds.
- [ ] **Step 5: Run the focused helper tests** and confirm all cases pass.
- [ ] **Step 6: Commit** as `feat: add optional wandb stage tracking`.

### Task 2: Log Track B fit results

**Files:**
- Modify: `scripts/train.py`
- Modify: `tests/unit/test_train_cli.py`

**Interfaces:**
- Consumes Task 1's `log_stage_report`.
- Produces one W&B stage run after the existing Track B report and model/score artifacts have been successfully published.
- W&B config includes only Track B rules, run ID, source commit, config/lock/code hashes and model-version metadata. Summary contains Track B status and gate counts. Numeric metrics contain fit/scoring/draw counts and numeric logistic-regression parameters.
- Does not add W&B fields to `track_b.json`, preserving the current local report and replay contracts.

- [ ] **Step 1: Write CLI tests** for ready, pending-review, and insufficient-background outcomes; assert the logger receives the correct status and summary, pending states carry no fitted metrics, and report/artifact behavior remains unchanged when tracking is skipped or fails.
- [ ] **Step 2: Run the focused Track B CLI tests and confirm the new tracking expectations fail** before implementation.
- [ ] **Step 3: Call `log_stage_report` after `_publish`** using sanitized config/summary/numeric metrics; include the returned tracking status in the CLI's human-readable stdout without changing `track_b.json`.
- [ ] **Step 4: Run the focused Track B CLI tests** and confirm all cases pass using synthetic fixtures only.
- [ ] **Step 5: Commit** as `feat: log track b run summaries to wandb`.

### Task 3: Log capture-evaluation results

**Files:**
- Modify: `scripts/evaluate.py`
- Modify: `tests/unit/test_capture.py` or a new `tests/unit/test_evaluate_cli.py`

**Interfaces:**
- Consumes Task 1's `log_stage_report`.
- Produces one W&B stage run after the existing `evaluation.json` report has been written.
- Config contains the run ID, evaluation rules, config hash, and source commit metadata. Summary contains evaluation status, `evaluation_performed`, and gate counts. Metrics flatten method/top-k numeric capture, lift, and uncertainty values; omit null and non-finite values.
- Does not log `eligible_event_ids`, event geometries, local input paths, review evidence, or file contents.

- [ ] **Step 1: Write CLI tests** for ready evaluation with Track B metrics, pending evaluation with no fabricated metrics, and remote logging failure that leaves the JSON report intact.
- [ ] **Step 2: Run the focused evaluation tests and confirm the new logging expectations fail** before implementation.
- [ ] **Step 3: Flatten only allowed numeric metrics and call `log_stage_report` after writing `evaluation.json`**; report the returned tracking status without changing the evaluation JSON schema.
- [ ] **Step 4: Run the focused evaluation tests** using synthetic fixtures and confirm both ready and pending behavior.
- [ ] **Step 5: Commit** as `feat: log evaluation summaries to wandb`.

### Task 4: Pass the Colab Secret to uv subprocesses and document setup

**Files:**
- Modify: `notebooks/colab_first_workflow.ipynb`
- Modify: `README.md`
- Modify: `tests/unit/test_notebook_hygiene.py`

**Interfaces:**
- Consumes Colab Secret `WANDB_API_KEY` after notebook access is enabled.
- Adds `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT` only to the in-memory `ENV` passed to `uv run` child processes; never prints or writes the API key to a file, report, notebook output, or Drive.
- If the Secret is missing, the notebook emits a clear tracking-skipped note and continues with the existing local/Drive pipeline behavior.

- [ ] **Step 1: Write notebook-hygiene assertions** that the template reads the secret without displaying it, passes the environment to subprocesses, contains no literal credential, and keeps outputs stripped.
- [ ] **Step 2: Run notebook-hygiene tests and confirm the new assertions fail** before implementation.
- [ ] **Step 3: Read Colab `userdata` in the setup cell**, set entity/project defaults in `ENV`, and document where to add the key and enable notebook access in README.
- [ ] **Step 4: Run notebook-hygiene tests and verify the notebook parses as valid JSON** with no outputs or credential value.
- [ ] **Step 5: Commit** as `docs: configure colab secrets for wandb tracking`.

### Task 5: Cross-platform and integration review

**Files:**
- Review: `pyproject.toml`, `uv.lock`, `src/flood_ml/utils/wandb_tracking.py`, `scripts/train.py`, `scripts/evaluate.py`, `notebooks/colab_first_workflow.ipynb`, and `README.md`

- [ ] **Step 1: Run the full synthetic-only test suite** and confirm the existing fit/capture gates, CLI reports, notebook hygiene, and Drive publication behavior remain unchanged.
- [ ] **Step 2: Run static checks and lock validation** with `uv lock --check`, `uv sync --locked --group dev`, `uv run ruff check .`, and `uv run ruff format --check .` on Windows.
- [ ] **Step 3: Confirm Linux/Colab installation** with the same `uv.lock` and `/content/.venv`; do not run real-data training as part of this task.
- [ ] **Step 4: Inspect the final diff** for leaked credential values, logged local paths/review content, altered report schemas, or gate behavior changes.
- [ ] **Step 5: Commit** any final review fixes locally; do not merge or launch real training until separately requested.

## Task Dependencies

| Task | Depends on | Can run in parallel after dependency |
|---|---|---|
| 1. Shared logger and dependency | None | No |
| 2. Track B integration | Task 1 | Yes, independent from Tasks 3 and 4 |
| 3. Evaluation integration | Task 1 | Yes, independent from Tasks 2 and 4 |
| 4. Colab secret and docs | Task 1 interface | Yes, independent from Tasks 2 and 3 |
| 5. Final review | Tasks 1–4 | No |

## Execution Notes

- Use the existing isolated worktree on branch `codex/old-hcmc-ml-implementation`.
- Execute task-by-task with the user's requested subagent-driven workflow; implementers and reviewers must use synthetic fixtures only.
- Do not expose or request the API key in chat. The user already stored it in Colab Secrets.
- The notebook pins a Git commit; after the source change is pushed, Colab must use the new full commit SHA for the logger to be present.
- The current Colab run state was not confirmed after the browser tab became unavailable. Do not restart or claim a real run completed during this implementation.
