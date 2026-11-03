# HCMC road flood susceptibility research

This repository builds a reproducible, reviewable baseline for relative flood
susceptibility of road segments inside pre-2025 Ho Chi Minh City. It retains
unreviewed flood reports as unknown evidence and does not interpret them as
verified negatives. MCDA scores are relative indices, not flood probabilities
or forecasts. The current milestone is a complete baseline pipeline; evaluation
stays pending until its independent human review quotas are met.

## Windows setup

Install `uv`, then open PowerShell at the repository root:

```powershell
uv python install 3.12
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Python 3.12 and the dependency lock are pinned for reproducible runs. Commands
and tests work offline after dependencies and research data are present. CI uses
synthetic fixtures only and does not download research data.

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

## Run the full pipeline

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

The local review files are header-only templates:

- `configs/local/review_decisions.csv`: observation-to-road decisions.
- `configs/local/event_reviews.csv`: event and location group memberships,
  plus `affected_geometry_complete`.

Only the explicit `accepted` decision creates a positive road label, and it
requires both match and event review. Blank completeness values mean false.
Setting `affected_geometry_complete=true` attests that the accepted full road
segments form the entire affected extent. Candidate matches are never labels.
Review instructions and the meaning of partial geometry are documented in
[`docs/methodology.md`](docs/methodology.md).

The pipeline can produce a baseline while those files are empty. In that state
`road_labels.label_binary` remains null and `evaluation_status` is
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
