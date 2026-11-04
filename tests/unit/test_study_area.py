"""Synthetic polygons exercise selection, CRS safety, and context coverage."""

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, box

from flood_ml.config import PipelineError
from flood_ml.gis.study_area import (
    context_radius_m,
    make_context,
    require_metric_crs,
    select_aoi,
)


@pytest.fixture
def boundaries():
    return gpd.GeoDataFrame(
        {"GhiChu": ["other", "TP. Ho CHi Minh"]},
        geometry=[box(0, 0, 10, 10), box(20, 20, 40, 40)],
        crs="EPSG:9210",
    )


def select(frame, target_crs="EPSG:9210"):
    return select_aoi(
        frame, field="GhiChu", value="TP. Ho CHi Minh", target_crs=target_crs
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_selection_is_independent_of_row_order(boundaries, reverse):
    frame = boundaries.iloc[::-1] if reverse else boundaries
    result = select(frame)
    assert len(result) == 1
    assert result.geometry.iloc[0].equals(box(20, 20, 40, 40))
    assert result.crs.to_epsg() == 9210
    assert len(frame) == 2  # Selection does not mutate inputs.


@pytest.mark.parametrize(
    "names", [["other", "TP. Ho CHi Minh "], ["TP. Ho CHi Minh"] * 2]
)
def test_absent_or_ambiguous_exact_selector_fails(boundaries, names):
    boundaries["GhiChu"] = names
    with pytest.raises(PipelineError):
        select(boundaries)


def test_missing_selector_field_fails(boundaries):
    with pytest.raises(PipelineError):
        select(boundaries.drop(columns="GhiChu"))


def test_missing_source_crs_fails(boundaries):
    with pytest.raises(PipelineError):
        select(boundaries.set_crs(None, allow_override=True))


@pytest.mark.parametrize(
    "geometry",
    [None, Polygon(), Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]), Point(0, 0)],
)
def test_unusable_selected_geometry_fails(boundaries, geometry):
    boundaries.loc[1, "geometry"] = geometry
    with pytest.raises(PipelineError):
        select(boundaries)


@pytest.fixture
def geographic_frame():
    return gpd.GeoDataFrame(geometry=[box(106.5, 10.5, 106.6, 10.6)], crs="EPSG:4326")


def test_geographic_coordinates_are_rejected(geographic_frame):
    with pytest.raises(PipelineError):
        require_metric_crs(geographic_frame)


@pytest.mark.parametrize("crs", [None, "EPSG:4326", "EPSG:2263"])
def test_metric_guard_rejects_missing_geographic_and_feet_crs(boundaries, crs):
    with pytest.raises(PipelineError):
        require_metric_crs(boundaries.set_crs(crs, allow_override=True))


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:2263", "not a CRS"])
def test_selection_rejects_nonmetric_or_invalid_target(boundaries, crs):
    with pytest.raises(PipelineError):
        select(boundaries, crs)


def test_geographic_source_is_projected_without_changing_boundary(geographic_frame):
    geographic_frame["GhiChu"] = "TP. Ho CHi Minh"
    result = select(geographic_frame)
    require_metric_crs(result)
    assert (
        result.to_crs(4326)
        .geometry.iloc[0]
        .equals_exact(geographic_frame.geometry.iloc[0], tolerance=1e-8)
    )


def test_context_uses_requested_halo_and_preserves_aoi(boundaries):
    aoi = select(boundaries)
    context = make_context(aoi, 530)
    assert context.crs == aoi.crs
    assert context.geometry.iloc[0].covers(box(20, 20, 40, 40))
    assert context.total_bounds.tolist() == pytest.approx([-510, -510, 570, 570])
    assert aoi.total_bounds.tolist() == [20, 20, 40, 40]


@pytest.mark.parametrize("radius", [-1, float("nan"), float("inf"), True, "500"])
def test_context_rejects_invalid_radius(boundaries, radius):
    with pytest.raises(PipelineError):
        make_context(select(boundaries), radius)


def test_context_rejects_geographic_aoi(geographic_frame):
    with pytest.raises(PipelineError):
        make_context(geographic_frame, 530)


def test_context_radius_tracks_largest_feature_plus_stencil():
    features = {"dem_buffer_m": 30, "neighborhood_radius_m": 500, "water_buffer_m": 800}
    assert context_radius_m(features, raster_stencil_m=30) == 830
    features["neighborhood_radius_m"] = 1200
    assert context_radius_m(features, raster_stencil_m=60) == 1260


@pytest.mark.parametrize(
    "features,stencil",
    [({}, 30), ({"dem_buffer_m": -1}, 30), ({"water_buffer_m": 250}, -1)],
)
def test_context_radius_rejects_missing_or_invalid_inputs(features, stencil):
    with pytest.raises(PipelineError):
        context_radius_m(features, raster_stencil_m=stencil)
