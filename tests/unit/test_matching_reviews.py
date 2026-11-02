import importlib.util
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from flood_ml.config import PipelineError
from flood_ml.labeling.matching import match_candidates
from flood_ml.labeling.reviews import (
    EVENT_COLUMNS,
    MATCH_COLUMNS,
    apply_reviews,
    read_reviews,
)


@pytest.fixture
def no_review_case():
    roads = gpd.GeoDataFrame(
        {"road_segment_id": ["a", "b"], "road_name": ["A", "B"]},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 20), (100, 20)])],
        crs=9210,
    )
    observations = gpd.GeoDataFrame(
        {
            "observation_id": ["o1", "o2"],
            "source_id": ["s1", "s2"],
            "outside_aoi": [False, True],
            "spatial_precision": ["street", "area"],
            "location_accuracy_m": [None, None],
        },
        geometry=[Point(50, 10), Point(50, 10)],
        crs=9210,
    )
    return {
        "roads": roads,
        "observations": observations,
        "candidates": match_candidates(observations, roads, {}),
        "match_reviews": pd.DataFrame(columns=MATCH_COLUMNS),
        "event_reviews": pd.DataFrame(columns=EVENT_COLUMNS),
    }


def reviewed(case, decision="accepted"):
    case["match_reviews"] = pd.DataFrame(
        [
            [
                "o1",
                "a",
                decision,
                "source page with road extent",
                "human",
                "2026-11-02T12:00:00Z",
            ]
        ],
        columns=MATCH_COLUMNS,
    )
    case["event_reviews"] = pd.DataFrame(
        [
            [
                "o1",
                "e1",
                "l1",
                "verified date and location in source",
                "human",
                "2026-11-02T12:00:00Z",
            ]
        ],
        columns=EVENT_COLUMNS,
    )
    return case


def test_no_review_never_creates_negatives(no_review_case):
    labels, links = apply_reviews(**no_review_case)
    assert set(labels.label_status) == {"unknown"}
    assert labels.label_binary.isna().all()
    assert links.empty


def test_both_roads_remain_candidates_and_outside_is_visible(no_review_case):
    queue = no_review_case["candidates"]
    assert set(queue.loc[queue.observation_id.eq("o1"), "road_segment_id"]) == {
        "a",
        "b",
    }
    assert queue.review_required.all()
    outside = queue.loc[queue.observation_id.eq("o2")]
    assert len(outside) == 1 and outside.road_segment_id.isna().all()
    assert outside.review_reason.tolist() == ["outside_aoi"]
    assert queue.location_accuracy_m.isna().all()


def test_configurable_radius(no_review_case):
    q = match_candidates(
        no_review_case["observations"],
        no_review_case["roads"],
        {"candidate_distance_m": 9},
    )
    assert q.road_segment_id.isna().all()
    assert q.review_reason.tolist() == ["no_candidate_within_radius", "outside_aoi"]


def test_positive_requires_explicit_match_and_event_provenance(no_review_case):
    labels, links = apply_reviews(**reviewed(no_review_case))
    assert labels.set_index("road_segment_id").loc["a", "label_binary"] == 1
    assert labels.set_index("road_segment_id").loc["b", "label_status"] == "unknown"
    assert links[
        ["observation_id", "source_id", "event_group_id", "location_group_id"]
    ].values.tolist() == [["o1", "s1", "e1", "l1"]]
    assert "source page" in labels.iloc[0].label_evidence
    assert links.iloc[0].reviewed_by == "human"


@pytest.mark.parametrize("decision", ["rejected", "needs_info", "no_report"])
def test_nonacceptance_is_unknown_and_history_retained(no_review_case, decision):
    labels, links = apply_reviews(**reviewed(no_review_case, decision))
    assert labels.label_binary.isna().all() and links.empty
    assert decision in labels.iloc[0].decision_history


@pytest.mark.parametrize(
    "table,column,value",
    [
        ("match_reviews", "evidence", ""),
        ("match_reviews", "reviewed_by", None),
        ("match_reviews", "reviewed_at", "bad"),
        ("event_reviews", "evidence", ""),
        ("event_reviews", "reviewed_by", ""),
        ("event_reviews", "reviewed_at", ""),
        ("match_reviews", "road_segment_id", "nonexistent"),
        ("event_reviews", "observation_id", "nonexistent"),
        ("match_reviews", "decision", "verified_negative"),
        ("event_reviews", "event_group_id", ""),
    ],
)
def test_invalid_reviews_fail(no_review_case, table, column, value):
    case = reviewed(no_review_case)
    case[table].loc[0, column] = value
    with pytest.raises(PipelineError):
        apply_reviews(**case)


@pytest.mark.parametrize("table", ["match_reviews", "event_reviews"])
def test_duplicate_or_conflicting_reviews_fail(no_review_case, table):
    case = reviewed(no_review_case)
    case[table] = pd.concat([case[table], case[table]], ignore_index=True)
    with pytest.raises(PipelineError, match="Duplicate"):
        apply_reviews(**case)


def test_missing_event_review_fails(no_review_case):
    case = reviewed(no_review_case)
    case["event_reviews"] = pd.DataFrame(columns=EVENT_COLUMNS)
    with pytest.raises(PipelineError, match="event review"):
        apply_reviews(**case)


def test_nonexistent_candidate_and_event_mismatch_fail(no_review_case):
    case = reviewed(no_review_case)
    case["observations"]["event_group_id"] = ["expected", None]
    with pytest.raises(PipelineError, match="event_group_id"):
        apply_reviews(**case)


def test_csv_preserves_keys_and_requires_schema(tmp_path):
    p = tmp_path / "reviews.csv"
    p.write_text(
        ",".join(MATCH_COLUMNS) + "\n001,a,accepted,page,human,2026-11-02T12:00:00Z\n"
    )
    assert read_reviews(p, MATCH_COLUMNS).observation_id.tolist() == ["001"]
    p.write_text("observation_id\n")
    with pytest.raises(PipelineError, match="columns"):
        read_reviews(p, MATCH_COLUMNS)


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "build_labels", Path(__file__).resolve().parents[2] / "scripts/build_labels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ird_adapter_joins_unique_objectid_not_duplicate_native_ids(tmp_path):
    raw = gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2],
            "Obs_id": ["Obs_239", "Obs_239"],
            "Source_id": ["s1", "s2"],
            "Date": ["2020-01-01", "2021-02-02"],
        },
        geometry=[Point(1, 2), Point(3, 4)],
        crs=9210,
    )
    p = tmp_path / "flood.gpkg"
    raw.to_file(p, layer="Flood_event_locations", driver="GPKG")
    result = load_cli().read_ird(p)
    assert result.observation_id.tolist() == [
        "ird:flood_observations:fid:1",
        "ird:flood_observations:fid:2",
    ]
    assert result.Date.tolist() == ["2020-01-01", "2021-02-02"]
    assert result.geometry.x.tolist() == [1, 3]


def test_ird_adapter_rejects_missing_objectid_geometry_join(tmp_path):
    raw = gpd.GeoDataFrame(
        {"OBJECTID": [10], "Obs_id": ["Obs_239"]}, geometry=[Point(1, 2)], crs=9210
    )
    p = tmp_path / "flood.gpkg"
    raw.to_file(p, layer="Flood_event_locations", driver="GPKG")
    with pytest.raises(PipelineError, match="OBJECTID"):
        load_cli().read_ird(p)


def test_multiple_sources_same_event_preserve_every_link(no_review_case):
    case = reviewed(no_review_case)
    case["observations"].loc[1, "outside_aoi"] = False
    case["candidates"] = match_candidates(case["observations"], case["roads"], {})
    case["match_reviews"] = pd.concat(
        [case["match_reviews"], case["match_reviews"].assign(observation_id="o2")],
        ignore_index=True,
    )
    case["event_reviews"] = pd.concat(
        [case["event_reviews"], case["event_reviews"].assign(observation_id="o2")],
        ignore_index=True,
    )
    labels, links = apply_reviews(**case)
    assert links.source_id.tolist() == ["s1", "s2"]
    assert links.event_group_id.tolist() == ["e1", "e1"]
    assert labels.label_status.tolist() == ["positive", "unknown"]
    case["event_reviews"].loc[1, "location_group_id"] = "different"
    with pytest.raises(PipelineError, match="Inconsistent"):
        apply_reviews(**case)


def test_review_of_nonexistent_pair_fails(no_review_case):
    case = reviewed(no_review_case)
    case["candidates"] = case["candidates"].loc[
        case["candidates"].road_segment_id.ne("a")
    ]
    with pytest.raises(PipelineError, match="candidate pair"):
        apply_reviews(**case)


@pytest.mark.parametrize("radius", [-1, True, float("nan"), "100"])
def test_invalid_radius_fails(no_review_case, radius):
    with pytest.raises(PipelineError):
        match_candidates(
            no_review_case["observations"],
            no_review_case["roads"],
            {"candidate_distance_m": radius},
        )


def test_cli_no_reviews_offline(tmp_path, no_review_case):
    import hashlib
    import json

    import yaml
    from shapely.geometry import box

    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"GhiChu": ["TP. Ho CHi Minh"]}, geometry=[box(-10, -10, 200, 100)], crs=9210
    ).to_file(raw / "boundary.gpkg")
    gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2],
            "Obs_id": ["Obs_239", "Obs_239"],
            "Source_id": ["s1", "s2"],
            "Date": ["2020-01-01", "2020-01-01"],
            "Spatial_precision": ["street", "area"],
        },
        geometry=[Point(50, 10), Point(300, 10)],
        crs=9210,
    ).to_file(raw / "flood.gpkg", layer="Flood_event_locations")
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "configs").mkdir()
    for name in ["sources", "features", "experiments", "labeling", "study_area"]:
        document = yaml.safe_load((root / f"configs/{name}.yaml").read_text())
        if name == "sources":
            document["sources"]["assets"] = [
                {"asset_id": "flood_observations", "path": "flood.gpkg"},
                {"asset_id": "administrative_boundaries", "path": "boundary.gpkg"},
            ]
        (tmp_path / f"configs/{name}.yaml").write_text(yaml.safe_dump(document))
    (tmp_path / "uv.lock").write_bytes(b"fixture lock")
    no_review_case["roads"].to_parquet(tmp_path / "roads.parquet")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    assert (
        load_cli().main(
            [
                "--repo-root",
                str(tmp_path),
                "--roads",
                "roads.parquet",
                "--run-id",
                "fixture",
            ]
        )
        == 0
    )
    output = tmp_path / "data/processed/fixture"
    labels = pd.read_parquet(output / "road_labels.parquet")
    assert labels.label_binary.isna().all()
    assert pd.read_parquet(output / "confirmed_event_segments.parquet").empty
    assert pd.read_parquet(output / "road_flood_matches.parquet").review_required.all()
    assert pd.read_parquet(output / "event_review_history.parquet").empty
    assert (
        json.loads((tmp_path / "reports/runs/fixture/labels_qc.json").read_text())[
            "evaluation_status"
        ]
        == "pending_label_review"
    )
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()
    }


def test_positive_cannot_have_blank_source_provenance(no_review_case):
    case = reviewed(no_review_case)
    case["observations"].loc[0, "source_id"] = ""
    with pytest.raises(PipelineError, match="source_id"):
        apply_reviews(**case)


@pytest.mark.parametrize(
    "table,columns",
    [("match_reviews", MATCH_COLUMNS), ("event_reviews", EVENT_COLUMNS)],
)
def test_csv_rejects_conflicting_duplicate_decision_headers_before_labels(
    tmp_path, no_review_case, table, columns
):
    import csv

    case = reviewed(no_review_case)
    row = case[table].iloc[0].to_dict()
    headers = list(columns)
    values = [row[column] for column in columns]
    if "decision" not in headers:
        headers.append("decision")
        values.append("accepted")
    headers.append("decision")
    values.append("rejected")
    path = tmp_path / f"{table}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerow(values)
    # Parsing must fail before this review table can reach label generation.
    with pytest.raises(PipelineError, match="Duplicate.*CSV.*header"):
        read_reviews(path, columns)


def test_csv_rejects_conflicting_duplicate_event_membership_header(tmp_path):
    path = tmp_path / "event_reviews.csv"
    path.write_text(
        ",".join(EVENT_COLUMNS + ["event_group_id"])
        + "\no1,event1,location1,page,human,2026-11-02T12:00:00Z,event2\n"
    )
    with pytest.raises(PipelineError, match="Duplicate.*CSV.*header"):
        read_reviews(path, EVENT_COLUMNS)


def test_csv_leading_blank_cannot_bypass_original_header_validation(tmp_path):
    path = tmp_path / "reviews.csv"
    path.write_text(
        "\n"
        + ",".join(MATCH_COLUMNS + ["decision"])
        + "\no1,a,accepted,page,human,2026-11-02T12:00:00Z,rejected\n"
    )
    with pytest.raises(PipelineError):
        read_reviews(path, MATCH_COLUMNS)
