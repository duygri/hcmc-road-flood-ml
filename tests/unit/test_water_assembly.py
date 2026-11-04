import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from flood_ml.config import PipelineError


def frame(geometries, **columns):
    return gpd.GeoDataFrame(columns, geometry=geometries, crs=9210)


@pytest.fixture
def edge_water_case():
    return {
        "roads": frame([LineString([(0, 0), (0, 10)])], road_segment_id=["a"]),
        "water": frame([box(10, 0, 20, 10)]),
        "coverage_domain": frame([box(-100, -100, 100, 100)]),
        "config": {"water_buffer_m": 250},
    }


def water_features(**kwargs):
    from flood_ml.features.water import compute_water_features

    return compute_water_features(**kwargs)


def test_distance_and_incomplete_buffer(edge_water_case):
    result = water_features(**edge_water_case)
    assert result.dist_water_m.iloc[0] == pytest.approx(10, abs=1e-6)
    assert result.water_context_incomplete.iloc[0]
    assert pd.isna(result.water_area_ratio_250m.iloc[0])


@pytest.mark.parametrize("distance", [100, 110])
def test_nearer_or_tied_domain_boundary_makes_distance_unknown(
    edge_water_case, distance
):
    edge_water_case["water"] = frame([box(distance, 0, distance + 10, 10)])
    edge_water_case["coverage_domain"] = frame([box(-100, -500, 500, 500)])
    result = water_features(**edge_water_case)
    assert pd.isna(result.dist_water_m.iloc[0])
    assert result.water_distance_incomplete.iloc[0]


def test_complete_empty_water_is_zero_not_unknown(edge_water_case):
    edge_water_case["coverage_domain"] = frame([box(-1000, -1000, 1000, 1000)])
    edge_water_case["water"] = frame([])
    result = water_features(**edge_water_case)
    assert result.water_area_ratio_250m.iloc[0] == 0
    assert not result.water_context_incomplete.iloc[0]
    assert pd.isna(result.dist_water_m.iloc[0])


def test_union_prevents_double_counting(edge_water_case):
    edge_water_case["coverage_domain"] = frame([box(-1000, -1000, 1000, 1000)])
    edge_water_case["water"] = frame([box(10, 0, 20, 10)] * 2)
    result = water_features(**edge_water_case)
    area = edge_water_case["roads"].geometry.iloc[0].buffer(250).area
    assert result.water_area_ratio_250m.iloc[0] == pytest.approx(100 / area)


def test_outside_domain_water_cannot_complete_distance(edge_water_case):
    edge_water_case["water"] = frame([box(-105, 0, -101, 10)])
    result = water_features(**edge_water_case)
    assert pd.isna(result.dist_water_m.iloc[0])


@pytest.mark.parametrize("invalid", ["geographic", "duplicate", "bad_geometry"])
def test_invalid_water_inputs_rejected(edge_water_case, invalid):
    if invalid == "geographic":
        edge_water_case["roads"] = edge_water_case["roads"].to_crs(4326)
    elif invalid == "duplicate":
        edge_water_case["roads"] = pd.concat([edge_water_case["roads"]] * 2)
    else:
        edge_water_case["water"] = frame([LineString([(0, 0), (1, 1)])])
    with pytest.raises(PipelineError):
        water_features(**edge_water_case)


def assemble(roads, raster, water, columns):
    from flood_ml.features.assembly import assemble_features

    return assemble_features(roads, raster, water, columns)


@pytest.fixture
def assembly_case():
    roads = frame(
        [LineString([(0, 0), (1, 1)])] * 3,
        road_segment_id=["a", "b", "c"],
        bridge=["yes", "no", None],
        tunnel=[None, "yes", "no"],
        flood_count=[1, 2, 0],
    )
    raster = pd.DataFrame(
        {"road_segment_id": ["c", "a", "b"], "elevation_median_m": [3, 1, 2]}
    )
    water = pd.DataFrame(
        {
            "road_segment_id": ["b", "c", "a"],
            "dist_water_m": [20, 30, 10],
            "water_context_incomplete": [False, False, True],
        }
    )
    return roads, raster, water


def test_assembly_preserves_identity_qc_and_structure_exclusion(assembly_case):
    result = assemble(*assembly_case, ["elevation_median_m", "dist_water_m"])
    assert result.road_segment_id.tolist() == ["a", "b", "c"]
    assert result.dist_water_m.tolist() == [10, 20, 30]
    assert result.score_eligible.tolist() == [False, False, True]
    assert result.score_exclusion_reason.tolist() == ["bridge", "tunnel", ""]
    assert result.water_context_incomplete.tolist() == [True, False, False]
    assert "flood_count" not in result


@pytest.mark.parametrize(
    "column",
    [
        "flood_count",
        "source_count",
        "label_binary",
        "road_segment_id",
        "water_context_incomplete",
        "unknown_feature",
    ],
)
def test_feature_allowlist_rejects_leakage_and_nonfeatures(assembly_case, column):
    with pytest.raises(PipelineError):
        assemble(*assembly_case, [column])


def test_spec_road_covariates_are_available_without_identifier_leakage(assembly_case):
    roads, raster, water = assembly_case
    roads["length_m"] = [10, 20, 30]
    roads["highway_type"] = ["primary", "service", "residential"]
    result = assemble(roads, raster, water, ["road_length_m", "highway_type"])
    assert result.road_length_m.tolist() == [10, 20, 30]
    assert result.highway_type.tolist() == ["primary", "service", "residential"]


@pytest.mark.parametrize("table", [0, 1, 2])
@pytest.mark.parametrize("problem", ["duplicate", "missing", "null"])
def test_one_to_one_joins_reject_invalid_keys(assembly_case, table, problem):
    inputs = list(assembly_case)
    if problem == "duplicate":
        inputs[table] = pd.concat([inputs[table], inputs[table].iloc[:1]])
    elif problem == "missing":
        inputs[table] = inputs[table].iloc[1:]
    else:
        inputs[table] = inputs[table].copy()
        inputs[table].loc[inputs[table].index[0], "road_segment_id"] = None
    with pytest.raises(PipelineError):
        assemble(*inputs, ["dist_water_m"])


def test_build_features_cli_offline(tmp_path, assembly_case):
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    root = Path(__file__).parents[2]
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    frame([box(-1000, -1000, 1000, 1000)], GhiChu=["TP. Ho CHi Minh"]).to_file(
        raw / "boundary.gpkg"
    )
    frame([box(10, 0, 20, 10)]).to_file(raw / "water.gpkg")
    (tmp_path / "configs").mkdir()
    for name in ["sources", "features", "experiments", "labeling", "study_area"]:
        document = yaml.safe_load((root / f"configs/{name}.yaml").read_text())
        if name == "sources":
            document["sources"]["assets"] = [
                {"dataset_id": "D04", "asset_id": "water", "path": "water.gpkg"},
                {
                    "dataset_id": "D01",
                    "asset_id": "administrative_boundaries",
                    "path": "boundary.gpkg",
                },
            ]
        if name == "features":
            document["features"]["feature_columns"] = [
                "dist_water_m",
                "elevation_median_m",
            ]
        (tmp_path / f"configs/{name}.yaml").write_text(yaml.safe_dump(document))
    (tmp_path / "uv.lock").write_bytes(b"fixture lock")
    roads, raster, _ = assembly_case
    roads.to_parquet(tmp_path / "roads.parquet")
    raster.to_parquet(tmp_path / "raster.parquet")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    command = [
        sys.executable,
        str(root / "scripts/build_features.py"),
        "--repo-root",
        str(tmp_path),
        "--roads",
        "roads.parquet",
        "--raster-features",
        "raster.parquet",
        "--run-id",
        "water_fixture",
        "--chunk-size",
        "2",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    output = pd.read_parquet(
        tmp_path / "data/processed/water_fixture/road_features.parquet"
    )
    assert output.road_segment_id.tolist() == ["a", "b", "c"]
    assert output.score_eligible.tolist() == [False, False, True]
    report = json.loads(
        (tmp_path / "reports/runs/water_fixture/features_qc.json").read_text()
    )
    assert report["status"] == "complete"
    assert report["source_bytes_unchanged"]
    assert "not independently verified" in report["coverage_assumption"]
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()
    }
