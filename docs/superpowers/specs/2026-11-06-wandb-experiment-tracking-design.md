# W&B Experiment Tracking Design

## Intent

Record comparable Track B training and capture-evaluation outcomes in the existing Weights & Biases project `hcmc-road-flood-ml`. This supplements the versioned reports and artifacts already published to Google Drive; it does not replace them.

## User-selected scope

For each `RUN_ID`, track both Track B and evaluation so model configurations and evaluation outcomes can be compared over time.

## Goals

- Record Track B status, configuration, fit-gate counts, draw settings, model versions, and scored-row counts.
- Record evaluation status, gate counts, and the available capture/lift metrics and uncertainty summaries by method and top-k fraction.
- Keep both stage runs discoverable under one W&B group named by `RUN_ID`.
- Use the existing team `duygri-saigon-university-sgu-` and project `hcmc-road-flood-ml`.
- Keep Colab as the primary runtime while allowing the same CLI instrumentation to work locally when W&B credentials are configured.
- Read `WANDB_API_KEY` from the process environment. The Colab notebook may obtain it from Colab Secrets and pass it only to child processes.

## Non-goals

- Upload raw datasets, review CSVs, feature tables, model binaries, or other Drive artifacts to W&B.
- Change the labeling rules, fit/evaluation gates, model behavior, or Drive publication contract.
- Start a training or evaluation run as part of adding this integration.

## Approaches considered

1. **Notebook-only logging:** one Colab run reads the CLI reports and logs them. This keeps W&B out of the project package but only tracks the notebook path and does not cover direct CLI/local runs.
2. **One W&B run per CLI stage, grouped by `RUN_ID` (recommended):** `scripts/train.py` and `scripts/evaluate.py` log their own final reports. This follows the existing split between the two commands and supports both Colab and local use. The trade-off is that each project run appears as two related W&B runs.
3. **One W&B run spanning all subprocesses:** this would require a new orchestration protocol to resume one run across separate CLI processes and introduces more lifecycle coupling than the current pipeline needs.

## Proposed design

- Add `wandb` as a project dependency and update the universal lockfile so the existing `uv sync --locked` workflow installs it in Colab's `/content/.venv` and in local environments.
- Add a small tracking helper that starts a stage run only when `WANDB_API_KEY` is present, sets `entity`, `project`, `name`, `group`, `job_type`, and sanitized config, and logs JSON-compatible scalar summaries.
- Use names `<RUN_ID>-track-b` and `<RUN_ID>-evaluation`, with `group=<RUN_ID>` so the W&B UI can compare both stages for each end-to-end run.
- Log only reproducibility metadata and summary results: source commit, config/lock hashes, run ID, gate statuses/counts, model parameters/versions, and evaluation metrics. Do not log local input paths or raw review/data contents.
- In Colab, retrieve `WANDB_API_KEY` using `google.colab.userdata` after notebook access has been enabled, then pass it through the environment used by `uv run`. Do not print, serialize, publish, or commit the key.
- W&B is an optional external reporting sink: missing credentials leave the existing pipeline behavior unchanged and produce a clear tracking-skipped message. A W&B service error is surfaced clearly while preserving the local JSON reports and Drive artifacts as the source of truth.
- Keep existing gate semantics: a pending Track B or evaluation gate is logged as pending, without fabricated metrics or an implied successful fit.

## Constraints

- The notebook pins a Git commit and installs from `uv.lock`; W&B source changes must be committed before a Colab run can use them.
- The W&B integration must not change existing report hashes or the comparison/replay contract for local verification.
- No API key or private review/data content may enter GitHub or the Drive run bundle.

## Acceptance criteria

- A Track B CLI run with configured W&B credentials creates a `<RUN_ID>-track-b` run in the selected project and logs its final status/config/counts.
- An evaluation CLI run creates a `<RUN_ID>-evaluation` run in the same group and logs the final status/gates and available metrics.
- With no API key, both CLIs keep their current local behavior and explicitly report that W&B tracking was skipped.
- Pending gates remain pending in W&B and in the existing JSON reports.
- The Colab notebook reads the secret without displaying it and passes it to the project subprocess environment.
- Existing Drive publication continues to work and contains no W&B credential.
