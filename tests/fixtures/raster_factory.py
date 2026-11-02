"""Small synthetic rasters; no licensed source pixels are embedded."""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box


def write_raster(
    path,
    values,
    *,
    resolution=30,
    left=0,
    top=2400,
    crs="EPSG:9210",
    nodata=-9999,
    y_resolution=None,
):
    values = np.asarray(values)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=from_origin(left, top, resolution, y_resolution or resolution),
        nodata=nodata,
    ) as dataset:
        dataset.write(values, 1)
    return path


def raster_case(tmp_path, *, dem=None, landcover=None):
    if dem is None:
        dem = np.full((80, 80), 7, dtype="float32")
    if landcover is None:
        landcover = np.full((240, 240), 10, dtype="uint8")
        landcover[:, 120:] = 50
    return {
        "roads": gpd.GeoDataFrame(
            {"road_segment_id": ["r1"]},
            geometry=[LineString([(1200, 1125), (1200, 1275)])],
            crs=9210,
        ),
        "aoi": gpd.GeoDataFrame(geometry=[box(900, 900, 1500, 1500)], crs=9210),
        "dem_paths": [write_raster(tmp_path / "dem.tif", dem)],
        "landcover_path": write_raster(
            tmp_path / "lc.tif",
            landcover,
            resolution=10,
            nodata=0,
        ),
        "config": {
            "dem_buffer_m": 30,
            "neighborhood_radius_m": 500,
            "landcover_buffer_m": 250,
            "water_buffer_m": 250,
            "min_valid_fraction": 0.9,
        },
    }
