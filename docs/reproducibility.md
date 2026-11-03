# Reproducibility record

## Environment

Runs use the repository's locked dependency set with Python 3.12, Windows and
`uv`. The exact runtime, CPU, memory, stage durations, peak process working set,
run ID, commit and artifact hashes are recorded below after the full-data run.
These are observations from one host, not a performance promise or SLA.

## Commands

```powershell
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python scripts/run_pipeline.py --config configs/experiments.yaml --run-id hcmc_old_v1_004
uv run python scripts/run_pipeline.py --config configs/experiments.yaml --run-id hcmc_old_v1_005
```

The synthetic integration suite writes its own local AOI, OSM, evidence,
water, DEM and WorldCover fixtures. It requires no Internet and does not read
the research raw directory. The real pipeline validates source checksums and
uses configured files without changing their bytes.

## Full-data execution

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
completion markers; the comparison and SHA-256 inventory are in
`.superpowers/sdd/2026-11-02-old-hcmc-baseline/task-12-real-replay-qc.json`.

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
