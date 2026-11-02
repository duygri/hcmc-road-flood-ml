"""Water proximity with an explicit, assumed source coverage domain."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree, union_all

from flood_ml.config import PipelineError
from flood_ml.gis.study_area import require_metric_crs


def compute_water_features(
    roads: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    coverage_domain: gpd.GeoDataFrame,
    config: dict,
) -> pd.DataFrame:
    """Measure only within supplied coverage; unobserved outside water is unknown.

    Coverage is an assumption supplied by the caller, never inferred from water
    geometry. Distance is complete only when strictly nearer than any boundary.
    """
    require_metric_crs(roads)
    if (
        "road_segment_id" not in roads
        or roads.road_segment_id.isna().any()
        or not roads.road_segment_id.is_unique
    ):
        raise PipelineError("Road IDs must be nonnull and unique")
    if any(
        g is None
        or g.is_empty
        or not g.is_valid
        or g.geom_type not in {"LineString", "MultiLineString"}
        for g in roads.geometry
    ):
        raise PipelineError("Roads require valid nonempty line geometries")
    projected = []
    for frame in (water, coverage_domain):
        if frame.crs is None:
            raise PipelineError("Water and coverage require declared CRS")
        frame = frame.to_crs(roads.crs)
        if any(
            g is None
            or g.is_empty
            or not g.is_valid
            or g.geom_type not in {"Polygon", "MultiPolygon"}
            for g in frame.geometry
        ):
            raise PipelineError("Water and domain require valid polygon geometries")
        projected.append(frame)
    water, coverage_domain = projected
    if coverage_domain.empty:
        raise PipelineError("Coverage domain cannot be empty")
    radius = config.get("water_buffer_m", 250)
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
        raise PipelineError("water_buffer_m must be positive finite metres")
    domain = union_all(coverage_domain.geometry.array)
    # Clip source features to the declared domain before metric indexing. Union
    # only local candidates per buffer, avoiding a costly region-wide dissolve.
    in_domain = [g.intersection(domain) for g in water.geometry if g.intersects(domain)]
    in_domain = [g for g in in_domain if not g.is_empty and g.area > 0]
    tree = STRtree(in_domain)
    records = []
    for road_id, road in zip(roads.road_segment_id, roads.geometry, strict=True):
        buffer = road.buffer(radius)
        complete = domain.covers(buffer)
        distance = np.nan
        if domain.covers(road) and len(in_domain):
            nearest = in_domain[int(tree.nearest(road))]
            candidate = road.distance(nearest)
            if candidate < road.distance(domain.boundary):
                distance = candidate
        ratio = np.nan
        if complete:
            indices = tree.query(buffer, predicate="intersects")
            combined = union_all([in_domain[int(i)] for i in indices])
            ratio = min(1.0, max(0.0, combined.intersection(buffer).area / buffer.area))
        records.append(
            {
                "road_segment_id": road_id,
                "dist_water_m": distance,
                "water_area_ratio_250m": ratio,
                "water_context_incomplete": not complete,
                "water_distance_incomplete": bool(pd.isna(distance)),
            }
        )
    result = pd.DataFrame(
        records,
        columns=[
            "road_segment_id",
            "dist_water_m",
            "water_area_ratio_250m",
            "water_context_incomplete",
            "water_distance_incomplete",
        ],
    )
    result.attrs["water_metadata"] = {
        "coverage_assumption": config.get(
            "coverage_assumption",
            "Caller-supplied domain; source completeness not independently verified",
        ),
        "water_buffer_m": radius,
        "metric_crs": roads.crs.to_string(),
        "distance_rule": "nearest in-domain water strictly closer than domain boundary",
        "outside_domain": "unknown",
    }
    return result
