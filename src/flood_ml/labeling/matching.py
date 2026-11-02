"""Distance creates review candidates, never label evidence or location accuracy."""

import math

import geopandas as gpd
import pandas as pd

from flood_ml.config import PipelineError
from flood_ml.gis.study_area import require_metric_crs


def require_ids(frame, column):
    if (
        column not in frame
        or frame[column].isna().any()
        or not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all()
    ):
        raise PipelineError(f"Nonblank string {column} required")
    if frame[column].duplicated().any():
        raise PipelineError(f"Duplicate {column}")


def match_candidates(
    observations: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, config: dict
) -> pd.DataFrame:
    """Keep all roads within candidate_distance_m (100 m default), including ties.

    Null-road rows keep observations outside AOI or without a nearby road visible.
    All distances use a shared metric CRS. Numeric accuracy is copied only.
    """
    for frame, key in [(observations, "observation_id"), (roads, "road_segment_id")]:
        require_metric_crs(frame)
        require_ids(frame, key)
        if (
            frame.geometry.isna().any()
            or frame.geometry.is_empty.any()
            or not frame.geometry.is_valid.all()
        ):
            raise PipelineError("Invalid candidate geometry")
    if observations.crs != roads.crs:
        raise PipelineError("Matching requires shared metric CRS")
    if (
        not observations.geom_type.eq("Point").all()
        or not roads.geom_type.isin(["LineString", "MultiLineString"]).all()
    ):
        raise PipelineError("Matching requires point observations and line roads")
    if (
        "outside_aoi" not in observations
        or not observations.outside_aoi.map(lambda v: isinstance(v, bool)).all()
    ):
        raise PipelineError("Explicit boolean outside_aoi required")
    radius = config.get("candidate_distance_m", 100)
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
        raise PipelineError("candidate_distance_m must be finite and nonnegative")
    records = []
    for _, observation in observations.iterrows():
        hits = (
            []
            if observation.outside_aoi
            else roads.sindex.query(
                observation.geometry, predicate="dwithin", distance=radius
            )
        )
        for i in hits:
            road = roads.iloc[i]
            records.append(
                {
                    "observation_id": observation.observation_id,
                    "source_id": observation.get("source_id"),
                    "road_segment_id": road.road_segment_id,
                    "road_name": road.get("road_name"),
                    "match_distance_m": observation.geometry.distance(road.geometry),
                    "match_method": "within_candidate_radius",
                    "review_required": True,
                    "review_reason": "candidate_requires_review",
                    "spatial_precision": observation.get("spatial_precision"),
                    "location_accuracy_m": observation.get("location_accuracy_m"),
                }
            )
        if not len(hits):
            records.append(
                {
                    "observation_id": observation.observation_id,
                    "source_id": observation.get("source_id"),
                    "road_segment_id": None,
                    "road_name": None,
                    "match_distance_m": None,
                    "match_method": "unmatched",
                    "review_required": True,
                    "review_reason": "outside_aoi"
                    if observation.outside_aoi
                    else "no_candidate_within_radius",
                    "spatial_precision": observation.get("spatial_precision"),
                    "location_accuracy_m": observation.get("location_accuracy_m"),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "observation_id",
            "source_id",
            "road_segment_id",
            "road_name",
            "match_distance_m",
            "match_method",
            "review_required",
            "review_reason",
            "spatial_precision",
            "location_accuracy_m",
        ],
    ).sort_values(["observation_id", "road_segment_id"], ignore_index=True)
