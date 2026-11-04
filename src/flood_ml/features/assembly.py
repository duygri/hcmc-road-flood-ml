"""One-to-one assembly with an explicit model-feature registry."""

import geopandas as gpd
import pandas as pd

from flood_ml.config import PipelineError
from flood_ml.features.raster import FEATURES
from flood_ml.gis.roads import _active

MODEL_FEATURES = frozenset(
    (
        *FEATURES,
        "dist_water_m",
        "water_area_ratio_250m",
        "road_length_m",
        "highway_type",
    )
)
KEY = "road_segment_id"


def assemble_features(
    roads: gpd.GeoDataFrame,
    raster_features: pd.DataFrame,
    water_features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Preserve road order and QC while explicitly identifying safe X columns.

    Only the registry can authorize model input; identifiers, QC and evidence
    columns cannot be promoted by requesting them in feature_columns.
    """
    if not isinstance(feature_columns, list) or len(set(feature_columns)) != len(
        feature_columns
    ):
        raise PipelineError("feature_columns must be a unique list")
    if not set(feature_columns).issubset(MODEL_FEATURES):
        raise PipelineError(
            "Feature allowlist contains identifiers, QC, leakage or unknown fields"
        )
    for table in (roads, raster_features, water_features):
        if KEY not in table or table[KEY].isna().any() or not table[KEY].is_unique:
            raise PipelineError("Every table requires unique nonnull road IDs")
        if set(table[KEY]) != set(roads[KEY]):
            raise PipelineError("Feature tables must contain exactly every road ID")
    base = pd.DataFrame(roads[[KEY]].copy())
    if "length_m" in roads:
        base["road_length_m"] = roads.length_m.to_numpy()
    if "highway_type" in roads:
        base["highway_type"] = roads.highway_type.to_numpy()
    for table in (raster_features, water_features):
        allowed = [
            c
            for c in table
            if c in MODEL_FEATURES
            or c in {"water_context_incomplete", "water_distance_incomplete"}
            or c in {f"{feature}_valid_fraction" for feature in FEATURES}
        ]
        if set(allowed).intersection(base.columns):
            raise PipelineError("Feature columns collide across source tables")
        base = base.merge(
            table[[KEY, *allowed]],
            on=KEY,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    if not set(feature_columns).issubset(base.columns):
        raise PipelineError("Requested features are missing from input tables")
    bridge = roads.get("bridge", pd.Series(False, index=roads.index)).map(_active)
    tunnel = roads.get("tunnel", pd.Series(False, index=roads.index)).map(_active)
    base["score_eligible"] = (~(bridge | tunnel)).to_numpy()
    base["score_exclusion_reason"] = [
        "bridge;tunnel" if b and t else "bridge" if b else "tunnel" if t else ""
        for b, t in zip(bridge, tunnel, strict=True)
    ]
    base.attrs["feature_columns"] = list(feature_columns)
    return base
