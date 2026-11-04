# Methodology

## Study target

The study area is the exact polygon selected by `GhiChu = TP. Ho CHi Minh` from
the pre-2025 administrative boundary file and projected to EPSG:9210. Road
centerlines are clipped to that polygon. OSM node references retain junction
topology; crossings without a shared node are not joined. Roads are split around
known junctions into 150 m target segments, with a 200 m maximum except where
topology or the boundary requires shorter pieces. Segment IDs include the OSM
snapshot, source traversal and segmentation configuration; changing the source
snapshot can change IDs.

## Features and spatial support

FABDEM tiles are mosaicked on their native lattice and sampled on a 30 m
EPSG:9210 grid. The configured outputs include median and tenth-percentile
elevation, relative elevation against a 500 m neighborhood and mean Horn 3×3
slope. A 10 m ESA WorldCover grid supplies built-up and vegetation ratios within
250 m. Pixel-center sampling and per-feature valid fractions are retained.
Heights are assumed to be metres in FABDEM's native vertical reference; no
vertical datum conversion or road-surface correction is applied.

Water features use the supplied IRD water geometry inside the old-HCMC AOI under
an explicit completeness assumption that has not been independently verified.
Outside the declared coverage domain, values remain unknown. A water buffer
that reaches beyond the domain is flagged incomplete, rather than treated as
zero water. The water source is projected and spatially indexed once for the
full road population in each feature stage.

Spatial holdout blocks are frozen from the configured metric grid and seed.
Roads receive one stable block through their normalized-length midpoint. Full
geometry guards against selected holdout blocks and holdout roads. The 1,000 m
guard separates declared target-space sampling neighborhoods. It does not prove
that raw DEM pixels used by bilinear interpolation are disjoint across blocks;
future methodology requiring source-pixel independence must increase the guard
and regenerate partitions.

## Evidence, labels and review

IRD points retain their source fields, original geometry and precision flags.
Duplicate native `Obs_id` values remain separate records keyed by GeoPackage
`OBJECTID`/`fid`. Candidate event groups and point-to-road matches only create
review queues. They do not label roads.

The local match CSV allows `accepted`, `rejected`, `needs_info` and `no_report`.
Only `accepted` can create a positive label, with explicit event membership.
Roads without accepted reviewed evidence remain `unknown`; the available source
does not provide verified negatives.

The event review CSV field `affected_geometry_complete` defaults to false when
blank or missing. A reviewer sets it to true only after confirming that all
accepted full road segments describe the entire affected extent of that event.
This assertion is used in evaluation denominators. Accepted candidate links do
not imply completeness. When only partial extents are known, the reviewer may
provide reviewed LineString or MultiLineString WKB in the extended confirmed
link input; it must be valid, lie within the accepted road, have an explicit
review assertion and carry complete-extent confirmation for capture eligibility.

## Baseline and evaluation

MCDA fits 5th and 95th percentile normalization thresholds on complete,
structurally eligible reference roads only. The five predeclared inputs are
median elevation, relative elevation, mean slope, water distance and built-up
ratio. Scores are null when a required feature is missing or a road is a bridge
or tunnel. Scores are relative 0–100 indices; they are not probabilities.
Sensitivity compares fixed equal, terrain-heavy, water-heavy and built-up-heavy
weights on identical eligible roads and a shared road-length denominator.

Evaluation measures the road length of reviewed affected geometry captured in
the top-k score-ranked road length, with the primary criterion and any/100%
sensitivity defined in configuration. Metrics are withheld until the independent
reference event, holdout event, holdout block, per-block and geometry/score
coverage quotas are met. The current empty human review means the honest status
is `pending_label_review`, with deficits reported and no capture metrics.

## Track B presence–background experiment

Track B fits a sampled presence–background Logistic Regression on the same
old-HCMC road population and frozen partitions. Its fixed feature order is
`elevation_median_m`, `relative_elevation_m`, `slope_mean_deg`, `dist_water_m`,
and `builtup_ratio_250m`. Only individually reviewed positive roads in the
reference partition enter the presence set. Background candidates are complete,
structurally eligible reference roads whose label remains `unknown`. Sampled
background can include unreported affected roads; it supplies a fitting
comparison, not verified negative evidence. The temporary 0/1 fit target stays
in memory and never changes the source labels or appears in score files.

Each draw samples background without replacement, with deterministic block
quotas proportional to the eligible reference pool. The primary ratio is one
presence to ten background roads; sensitivity ratios are 1:5 and 1:20. Each
ratio uses seeds 42–61, giving up to 60 fitted models. A draw reports
`background_pool_insufficient` when there are not enough unique candidates;
failure of any primary-ratio draw withholds all fitted model and score outputs.
Each model is a `StandardScaler` followed by `LogisticRegression(C=1.0,
solver="lbfgs", max_iter=1000)`. Scaling and fitting use only that draw's
reference presence/background rows. Holdout roads do not set preprocessing,
features, hyperparameters or the choice of ratio/model.

The fit gate requires at least 20 independent reviewed reference event groups,
each with at least one feature-eligible reviewed positive road after the
existing partition audit. It is separate from capture evaluation: capture
requires at least 20 reference and 20 holdout events, at least three holdout
blocks with at least five events each, complete reviewed affected geometry,
and full score coverage. Until the fit gate passes, `scripts/train.py` writes
`pending_positive_review` and the deficits without a model or score artifact.
The current real-data review is empty, so its Track B status is
`pending_positive_review`; the ready path has been verified with offline
synthetic fixtures only.

The exported `relative_score` is the Pipeline's `decision_function`, a relative
ranking for comparison with reviewed presences. It is not a calibrated flood
probability or a flood/non-flood classification. The training CLI publishes
long-form scores for eligible holdout roads, model Pipelines and per-draw
provenance including input/config/partition hashes, seed, ratio, counts,
exclusions and block quotas. Full-network inference is a separate future step.
`scripts/evaluate.py --track-b-scores ...` compares every available draw with
MCDA, elevation-only and random rankings on one eligible holdout road-length
denominator. Geometry or score-coverage deficits withhold metrics. Complete
reports retain the existing top-k capture rules and event/block bootstrap, and
summarize capture/lift median and IQR by ratio plus mean pairwise top-k Jaccard
overlap across seeds.

All capture results remain exploratory. Review and reporting selection may
make the model rank where floods are observed or reported, alongside physical
susceptibility. Background sampling does not remove that bias or identify a
population flood rate. Unknown roads provide no basis for accuracy, precision,
recall, F1, ROC-AUC or calibration claims as flood/non-flood classification.

## Offline report

`pipeline_report.html` and its JSON companion summarize source audit, AOI and
roads, observations outside AOI, review queue, feature nulls/distributions,
split counts, score reasons, sensitivity and evaluation gate. Four PNG files
show AOI/roads, frozen partitions, eligible baseline scores and feature
distributions. Large maps show the first stable road IDs up to the displayed
limit; tables and calculations use the complete road population.
