"""Exact AOI selection and metric context operations."""

import math
from collections.abc import Mapping

import geopandas as gpd
from pyproj import CRS
from pyproj.exceptions import CRSError

from flood_ml.config import PipelineError


def _crs(value) -> CRS:
    if value is None:
        raise PipelineError("A declared CRS is required")
    try:
        return CRS.from_user_input(value)
    except (CRSError, ValueError, TypeError) as exc:
        raise PipelineError(f"Invalid CRS: {value!r}") from exc


def _require_metric(crs: CRS) -> None:
    axes = crs.axis_info
    if (
        not crs.is_projected
        or len(axes) < 2
        or any(
            axis.unit_name.lower() not in {"metre", "meter"}
            or axis.unit_conversion_factor != 1.0
            for axis in axes[:2]
        )
    ):
        raise PipelineError("Spatial measurements require a projected CRS in metres")


def require_metric_crs(frame: gpd.GeoDataFrame) -> None:
    """Reject missing, geographic, or non-metre CRS before measurements."""
    _require_metric(_crs(frame.crs))


def _require_polygon(frame: gpd.GeoDataFrame) -> None:
    if len(frame) != 1:
        raise PipelineError("AOI must contain exactly one polygon record")
    geometry = frame.geometry.iloc[0]
    if (
        geometry is None
        or geometry.is_empty
        or not geometry.is_valid
        or geometry.geom_type not in {"Polygon", "MultiPolygon"}
    ):
        raise PipelineError(
            "AOI requires valid, nonempty polygon geometry; repair is disabled"
        )


def select_aoi(
    boundaries: gpd.GeoDataFrame, *, field: str, value: str, target_crs: str
) -> gpd.GeoDataFrame:
    """Select exactly one attribute match, then project its unchanged boundary.

    No normalization, row-order fallback, bbox substitution, or implicit repair
    is applied. Source attributes and source index remain available for audit.
    """
    _crs(boundaries.crs)
    target = _crs(target_crs)
    _require_metric(target)
    if (
        not isinstance(field, str)
        or not isinstance(value, str)
        or field not in boundaries
    ):
        raise PipelineError(
            "AOI selector requires an existing field and an exact string value"
        )
    selected = boundaries.loc[boundaries[field].eq(value)].copy()
    _require_polygon(selected)
    try:
        result = selected.to_crs(target)
    except (CRSError, ValueError) as exc:
        raise PipelineError(f"Cannot project selected AOI: {exc}") from exc
    _require_polygon(result)
    return result


def _nonnegative(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise PipelineError(f"{name} must be a finite nonnegative number in metres")
    return float(value)


def make_context(aoi: gpd.GeoDataFrame, radius_m: float) -> gpd.GeoDataFrame:
    """Buffer a valid metric AOI by the caller's full derived context radius.

    This context is for source coverage; downstream outputs still clip to AOI.
    """
    require_metric_crs(aoi)
    _require_polygon(aoi)
    radius_m = _nonnegative(radius_m, "context radius")
    context = aoi.copy()
    context.geometry = aoi.geometry.buffer(radius_m)
    _require_polygon(context)
    return context


def context_radius_m(features: Mapping, *, raster_stencil_m: float) -> float:
    """Largest configured feature reach plus the raster operation's stencil.

    Current feature settings are a flat mapping with *_radius_m/*_buffer_m
    keys. Stencil is the additional support distance (e.g. one cell diagonal
    for a 3x3 slope stencil); raster callers must update it with their grid.
    """
    radii = [
        _nonnegative(value, key)
        for key, value in features.items()
        if key.endswith(("_radius_m", "_buffer_m"))
    ]
    if not radii:
        raise PipelineError("No configured feature radii or buffers")
    return max(radii) + _nonnegative(raster_stencil_m, "raster stencil")
