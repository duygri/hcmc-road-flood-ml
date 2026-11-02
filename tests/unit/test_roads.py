"""Node topology, metric clipping, provenance and deterministic road IDs."""

import json
import math
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from flood_ml.config import PipelineError
from flood_ml.gis.roads import segment_roads
from flood_ml.ingestion.osm import read_osm_roads


@pytest.fixture
def aoi():
    return gpd.GeoDataFrame(
        {"study_area_id": ["fixture"]}, geometry=[box(0, 0, 1000, 1000)], crs=9210
    )


@pytest.fixture
def ways_case():
    # 10/11: T; 20/21: visual crossing with distinct IDs; 30/31/32:
    # shared node but bridge, surface, and tunnel at different levels.
    cases = [
        (10, [1, 2, 3], [(10, 100), (110, 100), (210, 100)], None, None, None),
        (11, [2, 4], [(110, 100), (110, 180)], None, None, None),
        (20, [5, 6, 7], [(10, 300), (110, 300), (210, 300)], None, None, None),
        (21, [8, 9, 12], [(110, 200), (110, 300), (110, 400)], None, None, None),
        (30, [13, 14, 15], [(10, 500), (110, 500), (210, 500)], "yes", None, "1"),
        (31, [16, 14], [(110, 420), (110, 500)], None, None, "0"),
        (32, [17, 14], [(110, 580), (110, 500)], None, "yes", "-1"),
        (40, [18, 19], [(-100, 700), (1100, 700)], None, None, None),
        (50, [20, 21], [(10, 800), (460, 800)], None, None, None),
        (60, [22, 23], [(10, 900), (50, 900)], None, None, None),
    ]
    rows = [
        {
            "osm_way_id": str(w),
            "node_ids": ids,
            "highway_type": "residential",
            "name": f"Way {w}",
            "bridge": b,
            "tunnel": t,
            "layer": l,
            "tags": {"highway": "residential"},
            "geometry": LineString(coords),
        }
        for w, ids, coords, b, t, l in cases
    ]
    return gpd.GeoDataFrame(rows, crs=9210)


def segment(ways, aoi, **overrides):
    options = {"snapshot_date": "2026-11-02", "target_m": 150.0, "maximum_m": 200.0}
    options.update(overrides)
    return segment_roads(ways, aoi, **options)


def test_segmentation_clips_caps_and_preserves_length(ways_case, aoi):
    result = segment(ways_case, aoi)
    assert result.road_segment_id.is_unique
    assert result.length_m.max() <= 200 + 1e-6
    assert result.geometry.difference(aoi.geometry.union_all()).length.max() <= 1e-6
    assert result.length_m.sum() == pytest.approx(2530, abs=1e-6)
    assert result.geometry.length.sum() == pytest.approx(2530, abs=1e-6)
    assert result.geometry.is_valid.all()
    assert result.geometry_crs.unique().tolist() == ["EPSG:9210"]
    assert result.study_area_id.unique().tolist() == ["fixture"]
    assert result.osm_snapshot_date.unique().tolist() == ["2026-11-02"]


def test_topology_splits_t_but_never_infers_visual_or_different_level_junction(
    ways_case, aoi
):
    result = segment(ways_case, aoi)
    assert sorted(result.loc[result.osm_way_id == "10", "length_m"]) == [100, 100]
    for way in ["20", "21", "30"]:
        assert sorted(result.loc[result.osm_way_id == way, "length_m"]) == [100, 100]
        # A geometric/node split would produce 100 m chains with chainage 0 each;
        # these instead are two equal pieces of the same unbroken 200 m chain.
        assert sorted(result.loc[result.osm_way_id == way, "chainage_start_m"]) == [
            0,
            100,
        ]
    assert result.loc[result.osm_way_id == "10", "chainage_start_m"].tolist() == [0, 0]
    assert result.loc[result.osm_way_id == "30", "bridge"].tolist() == ["yes", "yes"]
    assert result.loc[result.osm_way_id == "32", "tunnel"].tolist() == ["yes"]
    assert result.loc[result.osm_way_id == "50", "length_m"].tolist() == [150] * 3
    assert result.loc[result.osm_way_id == "60", "length_m"].tolist() == [40]
    assert result.loc[result.osm_way_id == "10", "node_ids"].tolist() == [[1, 2, 3]] * 2


def test_ids_ignore_row_and_direction_order_but_include_snapshot_and_config(
    ways_case, aoi
):
    baseline = segment(ways_case, aoi)
    reversed_ways = ways_case.iloc[::-1].copy()
    reversed_ways.geometry = reversed_ways.geometry.map(
        lambda g: LineString(list(g.coords)[::-1])
    )
    reversed_ways["node_ids"] = reversed_ways.node_ids.map(lambda ids: ids[::-1])
    repeated = segment(reversed_ways, aoi)
    assert baseline.road_segment_id.tolist() == repeated.road_segment_id.tolist()
    assert baseline.geometry.to_wkb().tolist() == repeated.geometry.to_wkb().tolist()
    assert set(baseline.road_segment_id).isdisjoint(
        segment(ways_case, aoi, snapshot_date="2026-11-03").road_segment_id
    )
    assert set(baseline.road_segment_id).isdisjoint(
        segment(ways_case, aoi, maximum_m=190).road_segment_id
    )


def test_canonical_hash_uses_micrometre_precision_without_rounding_output(aoi):
    ways = gpd.GeoDataFrame(
        {"osm_way_id": ["1"], "node_ids": [[1, 2]], "highway_type": ["road"]},
        geometry=[LineString([(10, 20), (50, 20)])],
        crs=9210,
    )
    original = segment(ways, aoi)
    ways.geometry = [LineString([(10.0000001, 20), (50.0000001, 20)])]
    perturbed = segment(ways, aoi)
    assert original.road_segment_id.tolist() == perturbed.road_segment_id.tolist()
    assert perturbed.geometry.iloc[0].coords[0][0] == 10.0000001


def test_clip_uses_concave_polygon_and_keeps_disjoint_fragments(ways_case, aoi):
    aoi.geometry = [
        Polygon(
            [
                (0, 0),
                (1000, 0),
                (1000, 1000),
                (600, 1000),
                (600, 600),
                (400, 600),
                (400, 1000),
                (0, 1000),
            ]
        )
    ]
    result = segment(ways_case.loc[ways_case.osm_way_id == "40"], aoi)
    assert result.length_m.sum() == pytest.approx(800, abs=1e-6)
    assert result.geometry.difference(aoi.geometry.iloc[0]).length.max() < 1e-6


def test_boundary_clip_preserves_self_crossing_source_traversal_and_reversed_ids(aoi):
    aoi.geometry = [box(0, 0, 100, 100)]
    ways = gpd.GeoDataFrame(
        {
            "osm_way_id": ["crossing"],
            "node_ids": [[1, 2, 3, 4]],
            "highway_type": ["road"],
        },
        geometry=[LineString([(-10, 10), (90, 90), (10, 90), (90, 10)])],
        crs=9210,
    )
    result = segment(ways, aoi)
    assert result.chain_id.nunique() == 1
    assert len(result) == 3
    expected_length = math.hypot(90, 72) + 80 + math.hypot(80, 80)
    assert result.length_m.sum() == pytest.approx(expected_length, abs=1e-6)
    assert result.length_m.tolist() == pytest.approx(
        [expected_length / 3] * 3, abs=1e-6
    )
    false_junction = Point(410 / 9, 490 / 9)
    for line in result.geometry:
        assert Point(line.coords[0]).distance(false_junction) > 1e-6
        assert Point(line.coords[-1]).distance(false_junction) > 1e-6
    assert result.geometry.difference(aoi.geometry.iloc[0]).length.max() <= 1e-6
    ways.geometry = ways.geometry.map(lambda line: LineString(list(line.coords)[::-1]))
    ways["node_ids"] = [[4, 3, 2, 1]]
    reversed_result = segment(ways, aoi)
    assert reversed_result.road_segment_id.tolist() == result.road_segment_id.tolist()
    assert (
        reversed_result.geometry.to_wkb().tolist() == result.geometry.to_wkb().tolist()
    )


def test_clip_keeps_separate_traversals_after_outside_excursion_to_same_boundary_point(
    aoi,
):
    aoi.geometry = [box(0, 0, 100, 100)]
    ways = gpd.GeoDataFrame(
        {
            "osm_way_id": ["excursion"],
            "node_ids": [[1, 2, 3, 4, 5, 6]],
            "highway_type": ["road"],
        },
        geometry=[
            LineString([(10, 50), (100, 50), (120, 50), (100, 50), (100, 90), (10, 90)])
        ],
        crs=9210,
    )
    result = segment(ways, aoi)
    assert result.chain_id.nunique() == 2
    assert sorted(result.length_m) == pytest.approx([90, 130], abs=1e-6)


@pytest.mark.parametrize("nodes", [[1, 2, 3, 4, 5], [1, 2, 3, 2, 1]])
def test_repeated_clipped_geometry_has_distinct_stable_source_traversal_ids(aoi, nodes):
    aoi.geometry = [box(0, 0, 100, 100)]
    ways = gpd.GeoDataFrame(
        {"osm_way_id": ["repeated"], "node_ids": [nodes], "highway_type": ["road"]},
        geometry=[LineString([(10, 50), (100, 50), (120, 50), (100, 50), (10, 50)])],
        crs=9210,
    )
    result = segment(ways, aoi)
    assert len(result) == 2
    assert result.chain_id.nunique() == 2
    assert result.road_segment_id.is_unique
    assert result.geometry.to_wkb().nunique() == 1
    assert result.length_m.tolist() == pytest.approx([90, 90], abs=1e-6)
    ways.geometry = ways.geometry.map(lambda line: LineString(list(line.coords)[::-1]))
    ways["node_ids"] = [nodes[::-1]]
    reversed_result = segment(ways, aoi)
    assert reversed_result.road_segment_id.tolist() == result.road_segment_id.tolist()
    assert (
        reversed_result.geometry.to_wkb().tolist() == result.geometry.to_wkb().tolist()
    )


@pytest.mark.parametrize(
    "target,maximum", [(0, 200), (201, 200), (150, 0), (float("nan"), 200)]
)
def test_invalid_segmentation_lengths_fail(ways_case, aoi, target, maximum):
    with pytest.raises(PipelineError):
        segment(ways_case, aoi, target_m=target, maximum_m=maximum)


def test_missing_refs_or_nonmetric_crs_fail(ways_case, aoi):
    with pytest.raises(PipelineError):
        segment(ways_case.drop(columns="node_ids"), aoi)
    with pytest.raises(PipelineError):
        segment(ways_case.to_crs(4326), aoi.to_crs(4326))
    ways_case.at[0, "node_ids"] = [1, 2]
    with pytest.raises(PipelineError):
        segment(ways_case, aoi)


def test_empty_allowed_roads_return_schema(ways_case, aoi):
    result = segment(ways_case.iloc[:0], aoi)
    assert result.empty
    assert {"road_segment_id", "node_ids", "length_m", "chainage_start_m"} <= set(
        result
    )
    assert result.crs == aoi.crs


OSM = Path(__file__).parents[1] / "fixtures/osm/intersections.osm"


def test_osmium_reads_real_refs_tags_allowlist_and_projects_before_filtering():
    context = gpd.GeoDataFrame(
        geometry=[box(106.69, 10.69, 106.72, 10.72)], crs=4326
    ).to_crs(9210)
    result = read_osm_roads(OSM, context, ["residential", "primary"])
    assert set(result.osm_way_id) == {
        "10",
        "11",
        "20",
        "21",
        "30",
        "31",
        "32",
        "40",
        "50",
        "60",
    }
    assert result.crs == context.crs
    t = result.set_index("osm_way_id").loc["10"]
    assert t.node_ids == [1, 2, 3]
    assert t.tags["name"] == "T main"
    assert result.set_index("osm_way_id").loc["30", "bridge"] == "yes"
    assert result.set_index("osm_way_id").loc["32", "tunnel"] == "yes"
    assert result.attrs["read_stats"]["ways_seen"] == 12
    # Neither endpoint of crossing way 40 is in the context envelope.
    assert not context.geometry.iloc[0].covers(
        result.set_index("osm_way_id").loc["40"].geometry.boundary
    )


def test_prepare_roads_cli_runs_offline_and_publishes_qc(tmp_path):
    import yaml

    root = Path(__file__).parents[2]
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "roads.osm").write_bytes(OSM.read_bytes())
    boundary = gpd.GeoDataFrame(
        {"GhiChu": ["TP. Ho CHi Minh"]},
        geometry=[box(106.69, 10.69, 106.72, 10.72)],
        crs=4326,
    )
    boundary.to_file(raw / "boundary.gpkg")
    (tmp_path / "configs/local").mkdir(parents=True)
    for name in ["experiments", "study_area", "sources", "features", "labeling"]:
        document = yaml.safe_load((root / f"configs/{name}.yaml").read_text())
        if name == "sources":
            document["sources"]["assets"] = [
                {"dataset_id": "D02", "asset_id": "roads", "path": "roads.osm"},
                {
                    "dataset_id": "D01",
                    "asset_id": "administrative_boundaries",
                    "path": "boundary.gpkg",
                },
            ]
        (tmp_path / f"configs/{name}.yaml").write_text(yaml.safe_dump(document))
    (tmp_path / "configs/local/sources.yaml").write_text(
        yaml.safe_dump({"sources": {"raw_root": str(raw)}})
    )
    (tmp_path / "uv.lock").write_bytes((root / "uv.lock").read_bytes())
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/prepare_roads.py"),
            "--repo-root",
            str(tmp_path),
            "--snapshot-date",
            "2026-11-02",
            "--run-id",
            "fixture_roads",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(
        (tmp_path / "reports/runs/fixture_roads/roads_qc.json").read_text()
    )
    assert report["status"] == "complete"
    assert report["aoi_ways"]["count"] == 10
    assert report["segments"]["count"] > 0
    assert report["segments"]["invalid_geometry"] == 0
    output = gpd.read_file(tmp_path / "data/interim/fixture_roads/road_segments.gpkg")
    assert output.road_segment_id.is_unique
