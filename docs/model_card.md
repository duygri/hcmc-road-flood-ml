# Model card: old-HCMC MCDA baseline

## Intended use

This is a research baseline for comparing relative road-segment susceptibility
under the documented old-HCMC study mask. It supports feature/QC review and
predeclared sensitivity analysis. It is not an operational flood warning,
insurance score, individual travel decision tool or probability of flooding.

## Method

The model is a fixed, transparent MCDA score over five inputs: median elevation,
relative elevation, mean slope, distance to mapped water and built-up ratio.
Reference partition roads determine 5th/95th percentile normalization. Four
fixed weight sets are compared. No flood labels are used to fit or tune weights.
Missing required inputs, bridges and tunnels yield no score rather than an
imputed or redistributed-weight score.

## Labels and evaluation

Only human-reviewed accepted observation-to-road links with explicit event
membership produce positive labels. Roads without such evidence remain unknown.
The IRD source supplies positive reports and does not supply verified negatives.
Evaluation uses independent reviewed events, complete affected geometry and
frozen spatial holdout blocks. The current review files are empty; evaluation is
therefore pending and no capture metric is reported as complete.

## Known limitations

- Media and technical-source reporting creates selection bias.
- Observation coordinates and dates vary in precision.
- Source vintages differ; current OSM and land cover do not describe one common
  date with the historical event evidence.
- FABDEM is approximately 30 m and does not model road-surface elevation,
  drainage, pumping, tidal stage or rainfall intensity.
- Water-layer completeness inside the AOI is an unverified assumption.
- MCDA scores are relative to the fitted reference distribution and current
  eligible road population; they are not probabilities.
- The spatial guard separates declared target-space feature neighborhoods but
  does not guarantee non-overlap of source pixels used by DEM interpolation.
- IRD/FABDEM noncommercial terms and OSM ODbL obligations restrict sharing and
  downstream use.

## Monitoring and change control

Every run records configuration, source/checksum and lock fingerprints. Changing
the OSM snapshot or segmentation config can change road IDs and needs a new run
and reviewed-link migration. Any extension to supervised modeling requires
independently verified negative evidence, reviewable event definitions and an
updated evaluation design before model fitting.
