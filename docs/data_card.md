# Data card

## Purpose

Inputs support a research baseline for relative flood susceptibility of roads
inside the pre-2025 Ho Chi Minh City administrative polygon. Source files are
read from the configured raw root, checked against the tracked manifest and
never modified by pipeline stages.

## Inputs and vintages

| Dataset | Snapshot / period | Resolution or geometry | Main use | Limits |
|---|---|---|---|---|
| IRD HCMC flood database | DataSuds v1.0, published 2026-09-09; reported events 2002–2026 | 425 point records | Positive evidence and review queue | 390 newspaper and 35 technical-report records; geocoding precision varies; no verified negatives |
| Geofabrik Vietnam OSM extract | Acquired 2026-11-02 | OSM vectors | Road topology and segment geometry | Current-map snapshot is not synchronized with historic events; urban road completeness is not independently verified |
| FABDEM v1.2 | January 2023 release | Native 1 arc-second, approximately 30 m | Elevation, relative elevation and slope | Native vertical datum EGM2008; feature values are not road-surface measurements |
| IRD water geometry | DataSuds v1.0, published 2026-09-09; mapping vintage not supplied | Vector polygons/lines | Water distance and area context | Completeness within AOI is an assumption and has not been independently verified |
| ESA WorldCover | 2021 v200 | 10 m product | Built-up and vegetation ratios | Land cover predates some observations and may not describe historical exposure |
| Pre-2025 administrative boundaries | Current repository snapshot | Vector polygons | Exact old-HCMC AOI selection | Boundary is a study mask, not a post-2025 HCMC region |

The sources have different vintages and observation periods. They do not form a
single synchronized exposure snapshot. The local manifest holds asset hashes,
metadata and inspection notes.

## Bias and missingness

Flood reports overrepresent events that reached media or technical sources.
Coordinates can represent streets, areas, wards or exact locations. Date and
location uncertainty are retained as flags. The 100 m candidate search radius
is a review aid, not a claim of coordinate accuracy. Unknown roads must not be
treated as non-flooded roads.

FABDEM and WorldCover masks, AOI-plus-halo coverage and valid sample fractions
are recorded. Water measurements outside the declared AOI domain are unknown.
The report identifies missing feature values and roads that cannot be scored.

## License and access

The inventory records IRD license metadata as conflicting: Dataverse says CC
BY-NC 4.0 while the bundled ReadMe says CC BY 4.0. The project conservatively
treats IRD as CC BY-NC 4.0 pending clarification. FABDEM is CC BY-NC-SA 4.0 and
requires the upstream Copernicus WorldDEM-30 attribution. OSM is ODbL and needs
attribution and review of derivative database sharing obligations. ESA
WorldCover is CC BY 4.0. Check current source terms before redistribution or
commercial use.
