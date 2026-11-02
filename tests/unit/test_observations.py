"""Observation evidence must survive normalization and unconfirmed grouping."""

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from flood_ml.config import PipelineError
from flood_ml.labeling.observations import normalize_observations, propose_event_groups


@pytest.fixture
def aoi():
    return gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=9210)


def raw(**overrides):
    values = {
        "Obs_id": ["o1"],
        "Source_id": ["SRC_1"],
        "Date": [None],
        "Year": [2025],
        "Spatial_precision": ["Low "],
        "Geocoding_method": ["Exact_point"],
    }
    values.update(overrides)
    n = len(values["Obs_id"])
    geometry = values.pop("geometry", [Point(10.123456789, 10)] * n)
    return gpd.GeoDataFrame(values, geometry=geometry, crs=9210)


def test_year_precision_does_not_invent_date(aoi):
    result = normalize_observations(raw(), aoi)
    assert result.observed_date.isna().all()
    assert result.date_precision.tolist() == ["year"]
    assert result.observed_year.tolist() == [2025]
    assert result.location_accuracy_m.isna().all()


@pytest.mark.parametrize(
    "code,want", [("Low ", "low"), ("Hgh", "high"), ("Hugh", "high")]
)
def test_precision_correction_preserves_source(code, want, aoi):
    source = raw(Spatial_precision=[code])
    result = normalize_observations(source, aoi)
    assert result.Spatial_precision.tolist() == [code]
    assert result.raw_Spatial_precision.tolist() == [code]
    assert result.spatial_precision.tolist() == [want]
    assert "spatial_precision_normalized" in result.quality_flags.iloc[0]
    assert result.geocoding_method.tolist() == ["exact_point"]
    assert "geocoding_method_normalized" in result.quality_flags.iloc[0]
    assert source.columns.tolist() == raw().columns.tolist()


def test_parseable_nonstandard_iso_has_raw_value_and_audit(aoi):
    result = normalize_observations(
        raw(Date=["2016-10-26T00:00:00.0Z"], Year=[2016]), aoi
    )
    assert result.observed_date.iloc[0] == pd.Timestamp("2016-10-26")
    assert result.raw_Date.iloc[0] == "2016-10-26T00:00:00.0Z"
    assert result.date_precision.iloc[0] == "day"
    assert "date_normalized" in result.quality_flags.iloc[0]


@pytest.mark.parametrize(
    "date,precision,flag",
    [
        (None, "unknown", "missing_date"),
        ("2025", "year", "year_only_date"),
        ("bad", "unknown", "unparseable_date"),
        ("2025-02-30", "unknown", "unparseable_date"),
    ],
)
def test_uncertain_dates_do_not_become_days(date, precision, flag, aoi):
    result = normalize_observations(raw(Date=[date], Year=[None]), aoi)
    assert result.observed_date.isna().all()
    assert result.date_precision.iloc[0] == precision
    assert flag in result.quality_flags.iloc[0]


def test_outside_aoi_is_retained_and_boundary_is_covered(aoi):
    result = normalize_observations(
        raw(
            Obs_id=["a", "b"],
            Source_id=["s", "s"],
            Date=[None] * 2,
            Year=[2025] * 2,
            Spatial_precision=["Low"] * 2,
            Geocoding_method=["Street point"] * 2,
            geometry=[Point(1000, 10), Point(1001, 10)],
        ),
        aoi,
    )
    assert result.outside_aoi.tolist() == [False, True]
    assert "outside_aoi" in result.quality_flags.iloc[1]


def test_duplicate_observation_ids_fail(aoi):
    source = pd.concat([raw(), raw()], ignore_index=True)
    with pytest.raises(PipelineError, match="observation_id"):
        normalize_observations(source, aoi)


def test_explicit_stable_record_keys_preserve_duplicate_native_ids(aoi):
    source = pd.concat([raw(), raw()], ignore_index=True)
    source["observation_id"] = ["ird:fid:1", "ird:fid:2"]
    result = normalize_observations(source, aoi)
    assert result.observation_id.tolist() == ["ird:fid:1", "ird:fid:2"]
    assert result.Obs_id.tolist() == ["o1", "o1"]
    assert result.raw_Obs_id.tolist() == ["o1", "o1"]
    assert all(
        "duplicate_native_observation_id" in flags for flags in result.quality_flags
    )


def observations(aoi):
    return normalize_observations(
        raw(
            Obs_id=["a", "b", "c", "d", "e"],
            Source_id=["s1", "s2", "s3", "s4", "s5"],
            Date=["2025-01-01", "2025-01-02", "2025-02-01", None, "2025-01-01"],
            Year=[2025] * 5,
            Spatial_precision=["High", "Low", "High", "Low", "High"],
            Geocoding_method=["Exact point"] * 5,
            Location_name=["Same street"] * 5,
            geometry=[
                Point(10, 10),
                Point(50, 10),
                Point(10, 10),
                Point(10, 10),
                Point(500, 10),
            ],
        ),
        aoi,
    )


def test_candidate_edges_require_both_space_and_time_preserving_sources(aoi):
    obs = observations(aoi)
    result = propose_event_groups(
        obs, {"candidate_distance_m": 100, "candidate_time_days": 1}
    )
    grouped = result.set_index("observation_id")
    assert (
        grouped.loc["a", "candidate_group_id"] == grouped.loc["b", "candidate_group_id"]
    )
    assert grouped.candidate_group_id.nunique() == 4
    assert set(result.source_id) == {"s1", "s2", "s3", "s4", "s5"}
    assert set(json.loads(grouped.loc["a", "source_ids"])) == {"s1", "s2"}
    assert set(result.spatial_precision) == {"high", "low"}
    assert result.review_required.all()
    assert set(result.candidate_status) == {"unconfirmed"}
    assert "event_group_id" not in result
    assert "missing_day" in grouped.loc["d", "review_reason"]
    assert len(json.loads(grouped.loc["a", "candidate_edges"])) == 1
    rerun = propose_event_groups(
        obs.iloc[::-1], {"candidate_distance_m": 100, "candidate_time_days": 1}
    )
    pd.testing.assert_frame_equal(result, rerun)


def test_config_controls_edges_and_geographic_measurements_fail(aoi):
    obs = observations(aoi)
    assert (
        propose_event_groups(
            obs, {"candidate_distance_m": 10, "candidate_time_days": 1}
        ).candidate_group_id.nunique()
        == 5
    )
    with pytest.raises(PipelineError, match="metres"):
        propose_event_groups(
            obs.to_crs(4326), {"candidate_distance_m": 100, "candidate_time_days": 1}
        )
    for value in [-1, float("nan"), True]:
        with pytest.raises(PipelineError):
            propose_event_groups(
                obs, {"candidate_distance_m": 100, "candidate_time_days": value}
            )


def test_input_crs_and_geometry_contracts(aoi):
    with pytest.raises(PipelineError):
        normalize_observations(raw().set_crs(None, allow_override=True), aoi)
    with pytest.raises(PipelineError):
        normalize_observations(raw(geometry=[None]), aoi)


def test_explicit_accuracy_is_retained_and_year_conflict_flagged(aoi):
    result = normalize_observations(
        raw(Date=["2024-01-01"], location_accuracy_m=[80.0]), aoi
    )
    assert result.location_accuracy_m.tolist() == [80.0]
    assert "date_year_conflict" in result.quality_flags.iloc[0]


def test_empty_inputs_have_stable_schema(aoi):
    obs = normalize_observations(raw().iloc[:0], aoi)
    result = propose_event_groups(
        obs, {"candidate_distance_m": 100, "candidate_time_days": 1}
    )
    assert result.empty
    assert "candidate_group_id" in result
