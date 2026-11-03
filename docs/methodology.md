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

## Offline report

`pipeline_report.html` and its JSON companion summarize source audit, AOI and
roads, observations outside AOI, review queue, feature nulls/distributions,
split counts, score reasons, sensitivity and evaluation gate. Four PNG files
show AOI/roads, frozen partitions, eligible baseline scores and feature
distributions. Large maps show the first stable road IDs up to the displayed
limit; tables and calculations use the complete road population.
