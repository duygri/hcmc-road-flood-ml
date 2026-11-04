# Model card: old-HCMC MCDA and Track B research models

## Intended use

These research models compare relative road-segment susceptibility under the
documented old-HCMC study mask. They support feature/QC review and predeclared
sensitivity analysis. They are not operational flood warnings, insurance
scores, individual travel decision tools or probabilities of flooding.

## Method

The model is a fixed, transparent MCDA score over five inputs: median elevation,
relative elevation, mean slope, distance to mapped water and built-up ratio.
Reference partition roads determine 5th/95th percentile normalization. Four
fixed weight sets are compared. No flood labels are used to fit or tune weights.
Missing required inputs, bridges and tunnels yield no score rather than an
imputed or redistributed-weight score.

Track B is a sampled presence–background Logistic Regression with the same
five fixed features, in this order: `elevation_median_m`,
`relative_elevation_m`, `slope_mean_deg`, `dist_water_m`, and
`builtup_ratio_250m`. Every model fits `StandardScaler` and
`LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)` only on reviewed
reference positive roads and sampled eligible reference unknown roads. The
primary presence:background ratio is 1:10; predeclared sensitivities are 1:5
and 1:20, each with seeds 42–61. Sampling is without replacement with recorded
reference-block quotas. Holdout data do not fit preprocessing or select a model.

Track B publishes the `decision_function` as `relative_score` for ranking.
Sampled background roads retain `unknown` source labels, including null binary
labels; the temporary fitting target does not create verified negatives. Scores
are not calibrated flood probabilities or flood/non-flood predictions. The
evaluation CLI consumes eligible holdout scores; full-network inference is
outside this experiment.

## Labels and evaluation

Only human-reviewed accepted observation-to-road links with explicit event
membership produce positive labels. Roads without such evidence remain unknown.
The IRD source supplies positive reports and does not supply verified negatives.
Evaluation uses independent reviewed events, complete affected geometry and
frozen spatial holdout blocks. The current review files are empty; evaluation is
therefore pending and no capture metric is reported as complete.

Track B fitting requires at least 20 independent reviewed reference event
groups, each with a complete-feature, eligible reviewed positive road after
partition audit. The current real-data status is `pending_positive_review`;
no real-data Track B model has been fitted. Offline synthetic integration tests
exercise both pending and ready paths. Too few unique background candidates
produce `background_pool_insufficient`; a primary-ratio shortage publishes no
partial fitted artifacts.

Capture evaluation additionally requires 20 reference events, 20 holdout
events, at least three holdout blocks with at least five events per block,
complete reviewed affected geometry and full common score coverage. Each Track
B draw, MCDA, elevation-only and random use the same eligible holdout road
length. Reports include capture, capture-any/all, lift and event/block bootstrap
uncertainty, plus median/IQR and top-k Jaccard stability across ratios/seeds.
Meeting these gates permits exploratory capture comparison; it does not
establish population flood prevalence. Accuracy, precision, recall, F1,
ROC-AUC and calibration are not reported as flood/non-flood classification
metrics on these positive–unlabeled data.

## Known limitations

- Media and technical-source reporting creates selection bias.
- Track B may learn reporting and review selection as well as physical
  susceptibility. Unknown background can contain unreported positives;
  changing the sampled background ratio can change relative rankings.
- Observation coordinates and dates vary in precision.
- Source vintages differ; current OSM and land cover do not describe one common
  date with the historical event evidence.
- FABDEM is approximately 30 m and does not model road-surface elevation,
  drainage, pumping, tidal stage or rainfall intensity.
- Water-layer completeness inside the AOI is an unverified assumption.
- MCDA scores are relative to the fitted reference distribution and current
  eligible road population; they are not probabilities.
- Track B relative scores are also conditional on reviewed presences and the
  sampled reference background; they do not estimate calibrated flood risk.
- The spatial guard separates declared target-space feature neighborhoods but
  does not guarantee non-overlap of source pixels used by DEM interpolation.
- IRD/FABDEM noncommercial terms and OSM ODbL obligations restrict sharing and
  downstream use.

## Monitoring and change control

Every run records configuration, source/checksum and lock fingerprints. Changing
the OSM snapshot or segmentation config can change road IDs and needs a new run
and reviewed-link migration. Moving to Track C supervised flood/non-flood
classification requires independently verified negative evidence, reviewable
event definitions and an updated evaluation design before model fitting.
