import hashlib
import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from flood_ml.config import PipelineError

factory_spec = importlib.util.spec_from_file_location(
    "raster_factory", Path(__file__).parents[1] / "fixtures/raster_factory.py"
)
factory = importlib.util.module_from_spec(factory_spec)
factory_spec.loader.exec_module(factory)


def compute(**kwargs):
    from flood_ml.features.raster import compute_raster_features

    return compute_raster_features(**kwargs)


@pytest.fixture
def flat_raster_case(tmp_path):
    return factory.raster_case(tmp_path)


def test_flat_dem_has_zero_relative_height(flat_raster_case):
    result = compute(**flat_raster_case)
    assert result["elevation_median_m"].iloc[0] == pytest.approx(7.0)
    assert result["elevation_p10_m"].iloc[0] == pytest.approx(7.0)
    assert result["relative_elevation_m"].iloc[0] == pytest.approx(0.0)
    assert result["slope_mean_deg"].iloc[0] == pytest.approx(0.0)


def test_plane_slope_is_45_degrees_across_mosaic_seam(tmp_path):
    case = factory.raster_case(tmp_path)
    values = np.tile(np.arange(15, 2400, 30, dtype="float32"), (80, 1))
    case["dem_paths"] = [
        factory.write_raster(tmp_path / "west.tif", values[:, :40]),
        factory.write_raster(tmp_path / "east.tif", values[:, 40:], left=1200),
    ]
    result = compute(**case)
    assert result.slope_mean_deg.iloc[0] == pytest.approx(45, abs=1e-5)
    assert result.slope_mean_deg_valid_fraction.iloc[0] == 1
    assert result.elevation_median_m.iloc[0] == pytest.approx(1200)
    assert result.elevation_p10_m.iloc[0] == pytest.approx(1185)


def test_symmetric_half_class50_is_half_builtup(flat_raster_case):
    result = compute(**flat_raster_case)
    assert result.builtup_ratio_250m.iloc[0] == pytest.approx(0.5)
    assert result.vegetation_ratio_250m.iloc[0] == pytest.approx(0.5)
    assert result.builtup_ratio_250m_valid_fraction.iloc[0] == 1


def test_stencil_nodata_invalidates_slope_separately(tmp_path):
    values = np.full((80, 80), 7, dtype="float32")
    # Adjacent to the two sampled columns, outside the 30 m buffer itself.
    values[36:44, 38] = -9999
    case = factory.raster_case(tmp_path, dem=values)
    result = compute(**case)
    assert result.elevation_median_m.iloc[0] == 7
    assert result.elevation_median_m_valid_fraction.iloc[0] == 1
    assert pd.isna(result.slope_mean_deg.iloc[0])
    assert 0 < result.slope_mean_deg_valid_fraction.iloc[0] < 0.9


def test_all_nodata_buffers_are_null_not_zero(tmp_path):
    case = factory.raster_case(
        tmp_path,
        dem=np.full((80, 80), -9999, dtype="float32"),
        landcover=np.zeros((240, 240), dtype="uint8"),
    )
    result = compute(**case)
    for name in (
        "elevation_median_m",
        "elevation_p10_m",
        "relative_elevation_m",
        "slope_mean_deg",
        "builtup_ratio_250m",
        "vegetation_ratio_250m",
    ):
        assert pd.isna(result[name].iloc[0])
        assert result[f"{name}_valid_fraction"].iloc[0] == 0


def test_relative_height_uses_context_outside_aoi(tmp_path):
    values = np.full((80, 80), 2, dtype="float32")
    values[37:43, 39:41] = 7
    case = factory.raster_case(tmp_path, dem=values)
    case["aoi"] = gpd.GeoDataFrame(geometry=[box(1170, 1100, 1230, 1300)], crs=9210)
    result = compute(**case)
    assert result.relative_elevation_m.iloc[0] == pytest.approx(5)
    assert result.relative_elevation_m_valid_fraction.iloc[0] == 1


def test_relative_valid_fraction_is_minimum_of_both_buffers(tmp_path):
    values = np.full((80, 80), -9999, dtype="float32")
    values[35:45, 37:43] = 7
    case = factory.raster_case(tmp_path, dem=values)
    result = compute(**case)
    assert result.elevation_median_m.iloc[0] == 7
    assert pd.isna(result.relative_elevation_m.iloc[0])
    assert 0 < result.relative_elevation_m_valid_fraction.iloc[0] < 0.9


def test_half_pixel_dem_tile_offset_is_rejected(tmp_path):
    case = factory.raster_case(tmp_path)
    values = np.full((80, 40), 7, dtype="float32")
    case["dem_paths"] = [
        factory.write_raster(tmp_path / "west.tif", values),
        factory.write_raster(tmp_path / "east.tif", values, left=1215),
    ]
    with pytest.raises(PipelineError, match="align|lattice|offset"):
        compute(**case)


@pytest.mark.parametrize("source", ["dem", "landcover"])
def test_footprint_must_cover_aoi_plus_full_halo(flat_raster_case, tmp_path, source):
    case = flat_raster_case
    # Covers AOI but not its required 542.426 m halo.
    if source == "dem":
        case["dem_paths"] = [
            factory.write_raster(
                tmp_path / "short.tif",
                np.full((40, 40), 7, dtype="float32"),
                left=600,
                top=1800,
            )
        ]
    else:
        case["landcover_path"] = factory.write_raster(
            tmp_path / "short_lc.tif",
            np.full((120, 120), 50, dtype="uint8"),
            resolution=10,
            left=600,
            top=1800,
            nodata=0,
        )
    with pytest.raises(PipelineError, match="halo|coverage"):
        compute(**case)


def test_nearest_landcover_resampling_preserves_only_source_classes(tmp_path):
    case = factory.raster_case(tmp_path)
    # Offset source forces resampling. Every valid pixel is either built-up or tree.
    classes = np.tile(np.array([10, 50], dtype="uint8"), (240, 120))
    case["landcover_path"] = factory.write_raster(
        tmp_path / "offset_lc.tif",
        classes,
        resolution=10,
        left=5,
        top=2405,
        nodata=0,
    )
    result = compute(**case)
    assert result.builtup_ratio_250m.iloc[0] == pytest.approx(0.5)
    assert result.vegetation_ratio_250m.iloc[0] == pytest.approx(0.5)
    assert result.attrs["raster_metadata"]["landcover"]["sampled_classes"] == [10, 50]


def test_valid_fraction_uses_pixel_centers_and_valid_denominator(tmp_path):
    classes = np.full((240, 240), 50, dtype="uint8")
    classes[:, :120] = 0
    case = factory.raster_case(tmp_path, landcover=classes)
    case["config"]["min_valid_fraction"] = 0.4
    result = compute(**case)
    assert result.builtup_ratio_250m.iloc[0] == 1
    assert result.builtup_ratio_250m_valid_fraction.iloc[0] == pytest.approx(0.5)
    case["config"]["min_valid_fraction"] = 0.9
    assert pd.isna(compute(**case).builtup_ratio_250m.iloc[0])


def test_grid_is_snapped_and_source_bytes_are_unchanged(flat_raster_case):
    paths = [*flat_raster_case["dem_paths"], flat_raster_case["landcover_path"]]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    result = compute(**flat_raster_case)
    metadata = result.attrs["raster_metadata"]
    for name, resolution in (("dem", 30), ("landcover", 10)):
        assert metadata[name]["resolution_m"] == resolution
        transform = metadata[name]["transform"]
        assert transform[2] % resolution == 0
        assert transform[5] % resolution == 0
        assert metadata[name]["crs"] == "EPSG:9210"
    assert metadata["halo_m"] == pytest.approx(542.4264068711929)
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before


def test_geographic_sources_are_warped_to_metric_grid(tmp_path):
    case = factory.raster_case(tmp_path)
    # A ~2.2 km square near central HCMC, and an AOI safely inside it.
    case["dem_paths"] = [
        factory.write_raster(
            tmp_path / "geo_dem.tif",
            np.full((200, 200), 7, dtype="float32"),
            resolution=0.0001,
            left=106.69,
            top=10.79,
            crs=4326,
        )
    ]
    case["landcover_path"] = factory.write_raster(
        tmp_path / "geo_lc.tif",
        np.full((200, 200), 50, dtype="uint8"),
        resolution=0.0001,
        left=106.69,
        top=10.79,
        crs=4326,
        nodata=0,
    )
    case["aoi"] = gpd.GeoDataFrame(
        geometry=[box(106.698, 10.778, 106.702, 10.782)], crs=4326
    ).to_crs(9210)
    case["roads"] = gpd.GeoDataFrame(
        {"road_segment_id": ["geo"]},
        geometry=[LineString([(106.7, 10.7795), (106.7, 10.7805)])],
        crs=4326,
    ).to_crs(9210)
    result = compute(**case)
    assert result.elevation_median_m.iloc[0] == pytest.approx(7)
    assert result.slope_mean_deg.iloc[0] == pytest.approx(0, abs=1e-5)
    assert result.builtup_ratio_250m.iloc[0] == 1


@pytest.mark.parametrize(
    "invalid", ["duplicate", "geographic", "outside", "missing_crs"]
)
def test_rejects_invalid_road_contract(flat_raster_case, invalid):
    roads = flat_raster_case["roads"]
    if invalid == "duplicate":
        flat_raster_case["roads"] = pd.concat([roads, roads], ignore_index=True)
    elif invalid == "geographic":
        flat_raster_case["roads"] = roads.to_crs(4326)
    elif invalid == "outside":
        roads.geometry = [LineString([(0, 0), (100, 100)])]
    else:
        flat_raster_case["roads"] = roads.set_crs(None, allow_override=True)
    with pytest.raises(PipelineError):
        compute(**flat_raster_case)


def test_whole_pixel_gap_between_dem_tiles_is_rejected(tmp_path):
    case = factory.raster_case(tmp_path)
    values = np.full((80, 40), 7, dtype="float32")
    case["dem_paths"] = [
        factory.write_raster(tmp_path / "west.tif", values),
        factory.write_raster(tmp_path / "east.tif", values, left=1230),
    ]
    with pytest.raises(PipelineError, match="halo|coverage"):
        compute(**case)


def test_order_and_values_are_independent_of_callers_chunks(flat_raster_case):
    case = flat_raster_case
    case["roads"] = gpd.GeoDataFrame(
        {"road_segment_id": ["r2", "r1"]},
        geometry=[
            LineString([(1100, 1125), (1100, 1275)]),
            LineString([(1200, 1125), (1200, 1275)]),
        ],
        crs=9210,
    )
    combined = compute(**case)
    chunks = [
        compute(**dict(case, roads=case["roads"].iloc[i : i + 1])) for i in range(2)
    ]
    assert list(combined.road_segment_id) == ["r2", "r1"]
    pd.testing.assert_frame_equal(combined, pd.concat(chunks, ignore_index=True))


def test_empty_roads_return_complete_schema(flat_raster_case):
    flat_raster_case["roads"] = flat_raster_case["roads"].iloc[:0]
    result = compute(**flat_raster_case)
    assert result.empty
    assert len(result.columns) == 13
    assert "slope_mean_deg_valid_fraction" in result


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True])
def test_rejects_invalid_valid_fraction_threshold(flat_raster_case, threshold):
    flat_raster_case["config"]["min_valid_fraction"] = threshold
    with pytest.raises(PipelineError, match="min_valid_fraction"):
        compute(**flat_raster_case)


def test_zero_threshold_still_keeps_empty_buffers_null(tmp_path):
    case = factory.raster_case(tmp_path, dem=np.full((80, 80), -9999, dtype="float32"))
    case["config"]["min_valid_fraction"] = 0
    result = compute(**case)
    assert pd.isna(result.elevation_median_m.iloc[0])
    assert pd.isna(result.slope_mean_deg.iloc[0])


def test_geographic_mosaic_footprints_do_not_create_projected_seam_gaps(tmp_path):
    case = factory.raster_case(tmp_path)
    case["aoi"] = gpd.GeoDataFrame(
        geometry=[box(106.35, 10.95, 106.7, 11.05)], crs=4326
    ).to_crs(9210)
    case["roads"] = case["roads"].iloc[:0]
    case["dem_paths"] = [
        factory.write_raster(
            tmp_path / f"tile_{x}_{y}.tif",
            np.full((80, 80), 7, dtype="float32"),
            resolution=1 / 80,
            y_resolution=(11.00013888888889 - 10.000138888888891) / 80,
            left=x - 0.00013888888888,
            top={10: 11.00013888888889, 11: 12.00013888888889}[y],
            crs=4326,
        )
        for x in (106, 107)
        for y in (10, 11)
    ]
    case["landcover_path"] = factory.write_raster(
        tmp_path / "big_lc.tif",
        np.full((240, 240), 50, dtype="uint8"),
        resolution=1 / 80,
        left=105,
        top=12,
        crs=4326,
        nodata=0,
    )
    result = compute(**case)
    assert result.empty
    assert result.attrs["raster_metadata"]["halo_m"] > 542
