"""Windowed terrain/land-cover extraction on snapped EPSG:9210 grids.

Only source windows around each road are mosaicked and warped. Source DEM
tiles must share a north-up lattice: a half-pixel mismatch is a data error,
not a seam to conceal with interpolation. Neighbourhoods retain the AOI halo.
"""

import math
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import reproject, transform_bounds
from shapely import transform
from shapely.affinity import affine_transform
from shapely.geometry import box
from shapely.ops import unary_union

from flood_ml.config import PipelineError
from flood_ml.gis.study_area import (
    _require_polygon,
    context_radius_m,
    require_metric_crs,
)

FEATURES = (
    "elevation_median_m",
    "elevation_p10_m",
    "relative_elevation_m",
    "slope_mean_deg",
    "builtup_ratio_250m",
    "vegetation_ratio_250m",
)
TARGET_CRS = "EPSG:9210"
DEM_RESOLUTION = 30
LANDCOVER_RESOLUTION = 10
VEGETATION_CLASSES = (10, 20, 30, 40, 90, 95, 100)


def _grid(bounds, resolution):
    left, bottom, right, top = bounds
    left = math.floor(left / resolution) * resolution
    top = math.ceil(top / resolution) * resolution
    width = math.ceil((right - left) / resolution)
    height = math.ceil((top - bottom) / resolution)
    return from_origin(left, top, resolution, resolution), (height, width)


def _source_metadata(dataset):
    return {
        "path": str(dataset.name),
        "crs": dataset.crs.to_string(),
        "bounds": list(dataset.bounds),
        "resolution": list(dataset.res),
        "shape": [dataset.height, dataset.width],
        "nodata": dataset.nodata,
    }


def _footprint(sources):
    # Form the union in integer source-pixel coordinates first. The lattice
    # validation permits <1e-6 pixel arithmetic noise; independently projecting
    # tile edges can otherwise turn that noise into microscopic seam gaps.
    first = sources[0]
    t = first.transform
    rectangles = []
    for source in sources:
        column = round((source.transform.c - t.c) / t.a)
        row = round((source.transform.f - t.f) / t.e)
        rectangles.append(box(column, row, column + source.width, row + source.height))
    polygon = affine_transform(unary_union(rectangles), [t.a, t.b, t.d, t.e, t.c, t.f])
    left, bottom, right, top = polygon.bounds
    # Densify the outer footprint before projection: geographic edges curve.
    polygon = polygon.segmentize(max(right - left, top - bottom) / 64)
    transformer = Transformer.from_crs(first.crs, TARGET_CRS, always_xy=True)
    return transform(
        polygon,
        lambda coordinates: np.column_stack(
            transformer.transform(coordinates[:, 0], coordinates[:, 1])
        ),
    )


def _validate_sources(sources, context, name):
    for source in sources:
        t = source.transform
        if source.crs is None or source.count != 1:
            raise PipelineError(f"{name}: a single raster band and CRS are required")
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise PipelineError(f"{name}: a north-up source lattice is required")
    if name == "DEM":
        first = sources[0]
        for source in sources[1:]:
            if source.crs != first.crs or not np.allclose(
                source.res, first.res, rtol=1e-9, atol=0
            ):
                raise PipelineError("DEM tiles require the same CRS and resolution")
            offsets = (
                (source.transform.c - first.transform.c) / first.res[0],
                (source.transform.f - first.transform.f) / first.res[1],
            )
            if any(abs(offset - round(offset)) > 1e-6 for offset in offsets):
                raise PipelineError(
                    "DEM tile offset is not aligned to the mosaic lattice"
                )
    footprint = _footprint(sources)
    if not footprint.covers(context):
        raise PipelineError(
            f"{name}: source footprint has insufficient AOI+halo coverage"
        )


def _warp_window(sources, bounds, resolution, resampling):
    """Mosaic before bilinear interpolation, retaining support across tile seams."""
    target_transform, shape = _grid(bounds, resolution)
    target_bounds = array_bounds(*shape, target_transform)
    source = sources[0]
    left, bottom, right, top = transform_bounds(
        TARGET_CRS, source.crs, *target_bounds, densify_pts=21
    )
    xres, yres = source.res
    xorigin, yorigin = source.transform.c, source.transform.f
    # Two source cells retain bilinear kernel support and projection rounding.
    source_bounds = (
        xorigin + (math.floor((left - xorigin) / xres) - 2) * xres,
        yorigin + (math.floor((bottom - yorigin) / yres) - 2) * yres,
        xorigin + (math.ceil((right - xorigin) / xres) + 2) * xres,
        yorigin + (math.ceil((top - yorigin) / yres) + 2) * yres,
    )
    mosaic, source_transform = merge(
        sources,
        bounds=source_bounds,
        dtype="float64",
        nodata=np.nan,
        indexes=[1],
        masked=True,
    )
    values = np.full(shape, np.nan, dtype="float64")
    reproject(
        source=mosaic[0].filled(np.nan),
        destination=values,
        src_transform=source_transform,
        src_crs=source.crs,
        src_nodata=np.nan,
        dst_transform=target_transform,
        dst_crs=TARGET_CRS,
        dst_nodata=np.nan,
        resampling=resampling,
        num_threads=1,
    )
    return values, target_transform


def _slope(dem):
    """Horn's 3x3 derivatives; every stencil cell must have a valid height."""
    windows = np.lib.stride_tricks.sliding_window_view(dem, (3, 3))
    valid = np.isfinite(windows).all(axis=(-2, -1))
    dzdx = (
        windows[..., 0, 2]
        + 2 * windows[..., 1, 2]
        + windows[..., 2, 2]
        - windows[..., 0, 0]
        - 2 * windows[..., 1, 0]
        - windows[..., 2, 0]
    ) / (8 * DEM_RESOLUTION)
    dzdy = (
        windows[..., 2, 0]
        + 2 * windows[..., 2, 1]
        + windows[..., 2, 2]
        - windows[..., 0, 0]
        - 2 * windows[..., 0, 1]
        - windows[..., 0, 2]
    ) / (8 * DEM_RESOLUTION)
    slope = np.full(dem.shape, np.nan)
    interior = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    slope[1:-1, 1:-1] = np.where(valid, interior, np.nan)
    return slope


def _sample(values, transform, geometry):
    selected = geometry_mask(
        [geometry],
        out_shape=values.shape,
        transform=transform,
        all_touched=False,
        invert=True,
    )
    samples = values[selected]
    valid = samples[np.isfinite(samples)]
    return valid, len(valid) / len(samples) if len(samples) else 0.0


def _number(config, name, default, *, allow_zero=False):
    value = config.get(name, default)
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        raise PipelineError(f"{name} requires a finite positive number")
    return value


def compute_raster_features(
    roads: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    dem_paths: list[Path],
    landcover_path: Path,
    config: dict,
) -> pd.DataFrame:
    """Return one row per road in input order, with ``<feature>_valid_fraction``.

    Config is the flat ``Settings.features`` mapping. DEM/land-cover grids are
    fixed to EPSG:9210 at 30/10 m; native DEM tiles must share a lattice.
    Ratios divide by valid pixel count. Fractions divide by all pixel centres
    in each buffer. Relative-height validity is the minimum of the local and
    neighbourhood fractions, and both must meet min_valid_fraction (0.9).
    Below-threshold or empty samples produce NaN, even when threshold is zero.
    Metadata in ``result.attrs['raster_metadata']`` must accompany persistence.
    FABDEM heights are assumed metres in their native vertical reference;
    this function performs no vertical-datum or road-surface correction.
    """
    require_metric_crs(roads)
    require_metric_crs(aoi)
    _require_polygon(aoi)
    if (
        "road_segment_id" not in roads
        or roads.road_segment_id.isna().any()
        or roads.road_segment_id.duplicated().any()
    ):
        raise PipelineError("Roads require non-null unique road_segment_id")
    if (
        roads.geometry.isna().any()
        or roads.geometry.is_empty.any()
        or not roads.geometry.is_valid.all()
        or not roads.geom_type.isin(["LineString", "MultiLineString"]).all()
    ):
        raise PipelineError("Roads require valid nonempty line geometry")
    roads = roads.to_crs(TARGET_CRS) if roads.crs != CRS(TARGET_CRS) else roads
    aoi = aoi.to_crs(TARGET_CRS) if aoi.crs != CRS(TARGET_CRS) else aoi
    polygon = aoi.geometry.iloc[0]
    # Task 4 preserves micrometre-scale boundary roundoff without moving roads.
    if any(geometry.difference(polygon).length > 1e-6 for geometry in roads.geometry):
        raise PipelineError("Road geometry lies outside the AOI")
    if not dem_paths:
        raise PipelineError("At least one DEM source is required")
    buffers = {
        "dem_buffer_m": _number(config, "dem_buffer_m", 30),
        "neighborhood_radius_m": _number(config, "neighborhood_radius_m", 500),
        "landcover_buffer_m": _number(config, "landcover_buffer_m", 250),
    }
    threshold = _number(config, "min_valid_fraction", 0.9, allow_zero=True)
    if threshold > 1:
        raise PipelineError("min_valid_fraction must be in [0, 1]")
    halo = context_radius_m(
        dict(config, **buffers), raster_stencil_m=math.sqrt(2) * DEM_RESOLUTION
    )
    context = polygon.buffer(halo)
    metadata = {
        "halo_m": halo,
        "all_touched": False,
        "min_valid_fraction": threshold,
        "relative_validity": "minimum of road and neighbourhood valid fractions",
        "vertical_datum": config.get("vertical_datum", "source native; unverified"),
        "height_units": "metres assumed; no vertical transformation",
        "slope_method": "Horn 3x3; all nine cells valid; metres",
        "buffers_m": buffers,
    }
    rows, sampled_classes = [], set()
    try:
        with ExitStack() as stack:
            # Bound GDAL's shared block cache; never read an entire source band.
            stack.enter_context(rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024))
            dem_sources = [
                stack.enter_context(rasterio.open(path)) for path in sorted(dem_paths)
            ]
            lc_sources = [stack.enter_context(rasterio.open(landcover_path))]
            for name, sources, resolution, resampling in (
                ("dem", dem_sources, DEM_RESOLUTION, "bilinear"),
                ("landcover", lc_sources, LANDCOVER_RESOLUTION, "nearest"),
            ):
                _validate_sources(sources, context, "DEM" if name == "dem" else name)
                grid_transform, shape = _grid(context.bounds, resolution)
                metadata[name] = {
                    "crs": TARGET_CRS,
                    "resolution_m": resolution,
                    "transform": list(grid_transform)[:6],
                    "shape": list(shape),
                    "resampling": resampling,
                    "sources": [_source_metadata(source) for source in sources],
                }
            for road_id, geometry in zip(roads.road_segment_id, roads.geometry):
                local = geometry.buffer(buffers["dem_buffer_m"])
                neighborhood = geometry.buffer(buffers["neighborhood_radius_m"])
                dem_bounds = (
                    local.union(neighborhood)
                    .buffer(math.sqrt(2) * DEM_RESOLUTION)
                    .bounds
                )
                dem, dt = _warp_window(
                    dem_sources, dem_bounds, DEM_RESOLUTION, Resampling.bilinear
                )
                heights, local_fraction = _sample(dem, dt, local)
                nearby, nearby_fraction = _sample(dem, dt, neighborhood)
                slopes, slope_fraction = _sample(_slope(dem), dt, local)
                cover = geometry.buffer(buffers["landcover_buffer_m"])
                lc, lt = _warp_window(
                    lc_sources, cover.bounds, LANDCOVER_RESOLUTION, Resampling.nearest
                )
                classes, cover_fraction = _sample(lc, lt, cover)
                sampled_classes.update(np.unique(classes).tolist())
                relative_fraction = min(local_fraction, nearby_fraction)
                fractions = (
                    local_fraction,
                    local_fraction,
                    relative_fraction,
                    slope_fraction,
                    cover_fraction,
                    cover_fraction,
                )
                statistics = (
                    lambda values=heights: np.median(values),
                    lambda values=heights: np.percentile(values, 10),
                    lambda values=heights, context_values=nearby: (
                        np.median(values) - np.median(context_values)
                    ),
                    lambda values=slopes: np.mean(values),
                    lambda values=classes: np.mean(values == 50),
                    lambda values=classes: np.mean(np.isin(values, VEGETATION_CLASSES)),
                )
                row = {"road_segment_id": road_id}
                for name, fraction, statistic in zip(FEATURES, fractions, statistics):
                    row[name] = (
                        float(statistic())
                        if fraction > 0 and fraction >= threshold
                        else np.nan
                    )
                    row[f"{name}_valid_fraction"] = fraction
                rows.append(row)
    except (rasterio.errors.RasterioError, OSError) as exc:
        raise PipelineError(f"Cannot extract raster features: {exc}") from exc
    metadata["landcover"]["sampled_classes"] = sorted(sampled_classes)
    columns = ["road_segment_id"] + [
        column for name in FEATURES for column in (name, f"{name}_valid_fraction")
    ]
    result = pd.DataFrame(rows, columns=columns)
    result.attrs["raster_metadata"] = metadata
    return result
