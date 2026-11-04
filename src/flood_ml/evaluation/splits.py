"""Frozen spatial blocks with full-line feature and reviewed-group isolation.

The metric grid has origin (0, 0). A road's length midpoint determines its one
immutable block ID ``floor(x / block_m):floor(y / block_m)``. A block is holdout
iff the unsigned SHA-256 integer of UTF-8 ``seed:block_id`` is below
``holdout_fraction * 2**256``. No sorting, quota, labels or scores enter selection.

Reference lines at distance <= guard from ANY selected grid cell or holdout line
are guarded. The latter also protects against holdout lines extending outside
their anchor cell. Guard >= twice the largest configured feature buffer ensures
closed reference/holdout feature-support footprints are disjoint, including at
touching boundaries. Block buffering is deliberately conservative.

``partition`` alone controls reference fit and evaluation. Eligible guard or
excluded roads can still receive a baseline score: this function does not change
their features/structure eligibility. ``block_partition`` stays frozen for audit;
``spatial_partition`` records geometry guards before reviewed-group exclusions.
Output attrs retain the frozen block map, effective radii and every confirmed
source link with its group component, final partition and exclusion reason.
"""

import hashlib
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree, box

from flood_ml.config import PipelineError
from flood_ml.gis.study_area import require_metric_crs
from flood_ml.labeling.matching import require_ids


def _number(value, name, *, positive=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise PipelineError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return float(value)


def _settings(config):
    if not isinstance(config, dict):
        raise PipelineError("Split config must be a mapping")
    experiments = config.get("experiments", config)
    features = config.get("features", {})
    if not isinstance(experiments, dict) or not isinstance(features, dict):
        raise PipelineError("experiments and features must be mappings")
    block_m = _number(
        experiments.get("spatial_block_m", 5000), "spatial_block_m", positive=True
    )
    fraction = _number(experiments.get("holdout_fraction", 0.2), "holdout_fraction")
    if fraction > 1:
        raise PipelineError("holdout_fraction must be in [0, 1]")
    seed = experiments.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise PipelineError("seed must be an integer in [0, 2**32)")
    radius = max(
        _number(features.get(key, config.get(key, default)), key)
        for key, default in [
            ("neighborhood_radius_m", 500),
            ("dem_buffer_m", 30),
            ("landcover_buffer_m", 250),
            ("water_buffer_m", 250),
        ]
    )
    guard = max(_number(experiments.get("guard_m", 1000), "guard_m"), 2 * radius)
    return block_m, fraction, seed, radius, guard


def _is_holdout(block_id, seed, fraction):
    digest = hashlib.sha256(f"{seed}:{block_id}".encode()).digest()
    # Integer comparison avoids rounding the SHA-256 value down to a float.
    numerator, denominator = fraction.as_integer_ratio()
    return int.from_bytes(digest, "big") * denominator < numerator * 2**256


def _hits(tree, geometries, distance=None):
    """Indices with a hit, batched to bound dense-road query memory."""
    hits = np.zeros(len(geometries), dtype=bool)
    for start in range(0, len(geometries), 4096):
        chunk = geometries[start : start + 4096]
        pairs = tree.query(
            chunk,
            predicate="intersects" if distance is None else "dwithin",
            **({} if distance is None else {"distance": distance}),
        )
        hits[start + np.unique(pairs[0])] = True
    return hits


def _reviewed_groups(result, confirmed):
    required = ["road_segment_id", "event_group_id", "location_group_id"]
    if not isinstance(confirmed, pd.DataFrame) or confirmed.columns.duplicated().any():
        raise PipelineError("Confirmed links require a table with distinct columns")
    for key in required + (["observation_id"] if "observation_id" in confirmed else []):
        if (
            key not in confirmed
            or not confirmed[key]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise PipelineError(f"Confirmed links require nonblank {key}")
    if not set(confirmed.road_segment_id).issubset(set(result.road_segment_id)):
        raise PipelineError("Confirmed link references nonexistent road_segment_id")
    if confirmed.empty:
        return []
    # Tokens are typed: equal text in event and location namespaces is not an edge.
    # Shared reviewed roads also connect their memberships, avoiding ambiguous
    # multi-event rows leaking through different parts of the reviewed graph.
    parents = {}

    def find(token):
        parents.setdefault(token, token)
        root = token
        while parents[root] != root:
            root = parents[root]
        while parents[token] != token:
            following = parents[token]
            parents[token] = root
            token = following
        return root

    for row in confirmed.to_dict("records"):
        tokens = [(key, row[key]) for key in required]
        if "observation_id" in row:
            tokens.append(("observation_id", row["observation_id"]))
        roots = [find(token) for token in tokens]
        root = min(roots)
        for other in roots:
            parents[other] = root
    components = {}
    for road_id in confirmed.road_segment_id.unique():
        components.setdefault(find(("road_segment_id", road_id)), []).append(road_id)
    indexed = result.set_index("road_segment_id")
    component_ids = {}
    for members in components.values():
        members = sorted(members)
        # Canonical source-independent ID; input/review order cannot affect it.
        component_id = (
            "reviewed:"
            + hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()[:24]
        )
        component_ids.update(dict.fromkeys(members, component_id))
        anchor_partitions = set(indexed.loc[members, "block_partition"])
        spatial_partitions = set(indexed.loc[members, "spatial_partition"])
        reason = (
            "reviewed_group_crosses_partitions"
            if anchor_partitions == {"reference", "holdout"}
            else "reviewed_group_intersects_guard"
            if "guard" in spatial_partitions
            else ""
        )
        if reason:
            selected = result.road_segment_id.isin(members)
            result.loc[selected, "partition"] = "excluded"
            result.loc[selected, "exclusion_reason"] = result.loc[
                selected, "exclusion_reason"
            ].map(lambda prior, reason=reason: ";".join(filter(None, [prior, reason])))
    # Preserve ALL submitted columns/provenance. Sort by all reviewed keys for a
    # stable audit even when multiple source observations confirm one road.
    audit = confirmed.copy().sort_values(
        [key for key in ["observation_id", *required, "source_id"] if key in confirmed],
        kind="stable",
    )
    audit["group_component_id"] = audit.road_segment_id.map(component_ids)
    final = result.set_index("road_segment_id")
    for key in [
        "block_id",
        "block_partition",
        "spatial_partition",
        "partition",
        "exclusion_reason",
    ]:
        audit[key] = audit.road_segment_id.map(final[key])
    audit["evaluation_eligible"] = audit.partition.isin(["reference", "holdout"])
    return audit.to_dict("records")


def assign_spatial_partitions(
    roads: gpd.GeoDataFrame, confirmed_event_segments: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Keep every road, input order, frozen block assignment and isolation audit.

    Pass ``{'experiments': settings.experiments, 'features': settings.features}``
    to derive the guard from the actual feature radii. Flat experiment config is
    also accepted and uses default feature radii unless supplied under features.
    Invalid IDs, line geometry, metric CRS, numeric config or reviewed links raise
    PipelineError. No event quota can change the frozen holdout block selection.
    """
    require_metric_crs(roads)
    require_ids(roads, "road_segment_id")
    if (
        roads.geometry.isna().any()
        or roads.geometry.is_empty.any()
        or not roads.geometry.is_valid.all()
        or not roads.geom_type.eq("LineString").all()
        or not np.isfinite(roads.geometry.bounds.to_numpy()).all()
    ):
        raise PipelineError(
            "Spatial split requires valid nonempty finite LineString roads"
        )
    block_m, fraction, seed, radius, guard = _settings(config)
    result = pd.DataFrame(
        {"road_segment_id": roads.road_segment_id.to_numpy(copy=True)}
    )
    midpoint = roads.geometry.interpolate(0.5, normalized=True)
    result["block_id"] = [
        f"{math.floor(x / block_m)}:{math.floor(y / block_m)}"
        for x, y in zip(midpoint.x, midpoint.y)
    ]
    result["block_partition"] = [
        "holdout" if _is_holdout(block_id, seed, fraction) else "reference"
        for block_id in result.block_id
    ]
    result["partition"] = result.block_partition
    result["exclusion_reason"] = ""
    block_assignments = []
    if not roads.empty:
        minx, miny, maxx, maxy = roads.total_bounds
        counts = result.block_id.value_counts().to_dict()
        holdout_cells = []
        # Include selected empty cells around the roads: the split is a frozen
        # geographic field, not a hash of the current road population.
        for x in range(
            math.ceil((minx - guard) / block_m) - 1,
            math.floor((maxx + guard) / block_m) + 1,
        ):
            for y in range(
                math.ceil((miny - guard) / block_m) - 1,
                math.floor((maxy + guard) / block_m) + 1,
            ):
                block_id = f"{x}:{y}"
                holdout = _is_holdout(block_id, seed, fraction)
                block_assignments.append(
                    {
                        "block_id": block_id,
                        "partition": "holdout" if holdout else "reference",
                        "road_count": counts.get(block_id, 0),
                    }
                )
                if holdout:
                    holdout_cells.append(
                        box(
                            x * block_m,
                            y * block_m,
                            (x + 1) * block_m,
                            (y + 1) * block_m,
                        )
                    )
        geometries = roads.geometry.to_numpy()
        reference = result.block_partition.eq("reference").to_numpy()
        if holdout_cells:
            tree = STRtree(holdout_cells)
            contact = reference & _hits(tree, geometries)
            guarded = reference & _hits(tree, geometries, guard)
            result.loc[guarded, "partition"] = "guard"
            result.loc[guarded, "exclusion_reason"] = "holdout_block_guard"
            result.loc[contact, "exclusion_reason"] = "holdout_block_contact"
        remaining = np.flatnonzero(result.partition.eq("reference").to_numpy())
        holdout_lines = geometries[~reference]
        if len(holdout_lines) and len(remaining):
            overlaps = remaining[
                _hits(STRtree(holdout_lines), geometries[remaining], guard)
            ]
            result.loc[overlaps, "partition"] = "guard"
            result.loc[overlaps, "exclusion_reason"] = "holdout_feature_support_overlap"
    result["spatial_partition"] = result.partition
    audit = _reviewed_groups(result, confirmed_event_segments)
    result["reference_fit_eligible"] = result.partition.eq("reference")
    result["evaluation_eligible"] = result.partition.isin(["reference", "holdout"])
    result.attrs.update(
        split_algorithm="sha256_midpoint_grid_v1",
        grid_origin_m=[0, 0],
        spatial_block_m=block_m,
        holdout_fraction=fraction,
        seed=seed,
        feature_support_radius_m=radius,
        effective_guard_m=guard,
        block_assignments=block_assignments,
        confirmed_event_audit=audit,
    )
    return result
