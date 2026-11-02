"""MCDA contracts use hand-checked synthetic roads, never flood evidence."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from flood_ml.baseline.mcda import compare_weight_sets, fit_normalizer, score_mcda
from flood_ml.config import PipelineError

NAMES = [
    "elevation_median_m",
    "relative_elevation_m",
    "slope_mean_deg",
    "dist_water_m",
    "builtup_ratio_250m",
]
SPEC = dict(zip(NAMES, ["lower_is_higher_risk"] * 4 + ["higher_is_higher_risk"]))
EQUAL = dict.fromkeys(NAMES, 0.2)
SETS = {
    "equal": EQUAL,
    "terrain-heavy": dict(zip(NAMES, [0.15, 0.4, 0.15, 0.15, 0.15])),
    "water-heavy": dict(zip(NAMES, [0.15, 0.15, 0.15, 0.4, 0.15])),
    "builtup-heavy": dict(zip(NAMES, [0.15, 0.15, 0.15, 0.15, 0.4])),
}


def roads(values=(0, 10), ids=None):
    return pd.DataFrame(
        {
            "road_segment_id": ids or [f"r{i}" for i in range(len(values))],
            **{key: list(map(float, values)) for key in NAMES},
            "road_length_m": 100.0,
        }
    )


@pytest.fixture
def normalizer():
    return fit_normalizer(roads(), SPEC)


def test_direction_clip_and_score_endpoints(normalizer):
    f = roads([-20, 30], ["safe", "risk"])
    f.loc[:, NAMES[:4]] = [[30] * 4, [-20] * 4]
    result = score_mcda(f, normalizer, EQUAL)
    assert result.score.tolist() == pytest.approx([0, 100])
    assert result.score_reason.tolist() == ["scored", "scored"]
    assert result[f"risk_{NAMES[0]}"].tolist() == [0, 1]


@pytest.mark.parametrize("feature", NAMES)
@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf, None])
def test_missing_required_value_keeps_null_and_identifies_feature(
    normalizer, feature, missing
):
    f = roads([5])
    f.loc[0, feature] = missing
    result = score_mcda(f, normalizer, EQUAL)
    assert pd.isna(result.score.iloc[0])
    assert pd.isna(result.percentile_rank.iloc[0])
    assert result.score_reason.iloc[0] == "missing_required_feature"
    assert feature in result.missing_required_features.iloc[0]


def test_absent_required_column_is_missing_not_weight_redistributed(normalizer):
    result = score_mcda(roads([5]).drop(columns=NAMES[0]), normalizer, EQUAL)
    assert result.score.isna().all()
    assert result.score_reason.iloc[0] == "missing_required_feature"


@pytest.mark.parametrize("structure", ["bridge", "tunnel"])
def test_structure_flags_and_assembly_exclusion_keep_null(normalizer, structure):
    f = roads([5, 5])
    f[structure] = ["yes", "no"]
    f["score_eligible"] = [True, False]
    f["score_exclusion_reason"] = ["", structure]
    result = score_mcda(f, normalizer, EQUAL)
    assert result.score.isna().all()
    assert result.score_reason.tolist() == [structure, structure]


def test_water_distance_mask_overrides_finite_value_but_area_mask_does_not(normalizer):
    f = roads([5, 5])
    f["water_distance_incomplete"] = [True, False]
    f["water_context_incomplete"] = True
    result = score_mcda(f, normalizer, EQUAL)
    assert pd.isna(result.score.iloc[0])
    assert result.score.iloc[1] == pytest.approx(50)


def test_constant_reference_contribution_neutral_with_flag():
    n = fit_normalizer(roads([4, 4]), SPEC)
    result = score_mcda(roads([-100, 100]), n, EQUAL)
    assert result.score.tolist() == pytest.approx([50, 50])
    assert set(result.constant_reference_features.iloc[0].split(";")) == set(NAMES)
    assert n["features"][NAMES[0]]["constant_reference"]


def test_quantiles_hashes_and_only_complete_eligible_reference_fit():
    f = roads([0, 10, 999, 999, 999])
    f["partition"] = "reference"
    f["score_eligible"] = [True, True, False, True, True]
    f.loc[3, NAMES[0]] = np.nan
    f["water_distance_incomplete"] = [False, False, False, False, True]
    n = fit_normalizer(f, SPEC)
    assert n["reference_row_count"] == 5
    assert n["fit_row_count"] == 2
    assert n["features"][NAMES[0]]["q05"] == pytest.approx(0.5)
    assert n["features"][NAMES[0]]["q95"] == pytest.approx(9.5)
    assert n["feature_spec"] == SPEC
    assert len(n["reference_input_sha256"]) == 64
    assert len(n["feature_spec_sha256"]) == 64
    json.dumps(n, allow_nan=False)


def test_holdout_extreme_cannot_change_reference_normalizer():
    f = roads([0, 10, 20])
    f["partition"] = ["reference", "reference", "holdout"]
    before = fit_normalizer(f.loc[f.partition.eq("reference")], SPEC)
    f.loc[2, NAMES] = 1e12
    assert fit_normalizer(f.loc[f.partition.eq("reference")], SPEC) == before
    with pytest.raises(PipelineError, match="reference"):
        fit_normalizer(f, SPEC)


@pytest.mark.parametrize("case", ["empty", "all_missing", "all_structures"])
def test_empty_fit_population_fails(case):
    f = roads() if case != "empty" else roads().iloc[:0]
    if case == "all_missing":
        f[NAMES[0]] = np.nan
    if case == "all_structures":
        f["bridge"] = "yes"
    with pytest.raises(PipelineError):
        fit_normalizer(f, SPEC)


@pytest.mark.parametrize("bad", [0, -0.2, np.nan, np.inf, True, "0.2"])
def test_invalid_weights_fail(normalizer, bad):
    weights = {**EQUAL, NAMES[0]: bad}
    with pytest.raises(PipelineError):
        score_mcda(roads(), normalizer, weights)


@pytest.mark.parametrize(
    "column",
    ["label_binary", "label_status", "flood_count", "source_count", "flood_observed"],
)
def test_label_columns_and_feature_requests_fail(normalizer, column):
    with pytest.raises(PipelineError):
        fit_normalizer(roads(), {**SPEC, column: "higher_is_higher_risk"})
    with pytest.raises(PipelineError):
        score_mcda(roads().assign(**{column: 1}), normalizer, EQUAL)


def test_diagnostic_elevation_p10_never_changes_scores(normalizer):
    f = roads().assign(elevation_p10_m=[-1e9, 1e9])
    a = score_mcda(f, normalizer, EQUAL)
    f.elevation_p10_m *= -1
    b = score_mcda(f, normalizer, EQUAL)
    pd.testing.assert_series_equal(a.score, b.score)
    with pytest.raises(PipelineError):
        score_mcda(f, normalizer, {**EQUAL, "elevation_p10_m": 0.2})


def test_rank_separate_from_score_and_ties_stable_by_id(normalizer):
    f = roads([5, 5, 5], ["z", "a", "b"])
    r = score_mcda(f, normalizer, EQUAL)
    assert r.score.tolist() == pytest.approx([50, 50, 50])
    assert r.percentile_rank.tolist() == [0, 100, 50]
    assert r["rank"].tolist() == [3, 1, 2]
    reverse = score_mcda(f.iloc[::-1], normalizer, EQUAL)
    pd.testing.assert_frame_equal(
        r.sort_values("road_segment_id"), reverse.sort_values("road_segment_id")
    )


def test_sensitivity_same_eligible_ids_spearman_and_length_budget(normalizer):
    f = roads([5, 5, 5, 5], ["a", "b", "c", "missing"])
    f.loc[:2, "relative_elevation_m"] = [0, 5, 10]
    f.loc[:2, "dist_water_m"] = [10, 5, 0]
    f.loc[3, NAMES[0]] = np.nan
    f["road_length_m"] = [60, 30, 10, 900]
    result = compare_weight_sets(f, normalizer, SETS)
    assert result["eligible_road_segment_ids"] == ["a", "b", "c"]
    assert all(s["eligible_count"] == 3 for s in result["schemes"].values())
    pair = result["comparisons"]["terrain-heavy__water-heavy"]
    assert pair["spearman"] == pytest.approx(-1)
    top = pair["top_k"]["0.1"]
    assert top["length_overlap_fraction"] == 0
    terrain = result["schemes"]["terrain-heavy"]["top_k"]["0.1"]
    assert terrain["selected_road_segment_ids"] == ["a"]
    assert terrain["selected_length_m"] == 60
    assert terrain["target_length_m"] == 10
    assert terrain["overshoot_m"] == 50
    assert terrain["actual_length_fraction"] == 0.6


def test_spearman_ties_are_average_ranks_and_undefined_is_null(normalizer):
    result = compare_weight_sets(roads([5, 5]), normalizer, SETS)
    assert result["comparisons"]["equal__terrain-heavy"]["spearman"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate_id",
        "null_id",
        "bad_length",
        "missing_length",
        "bad_direction",
        "bad_normalizer",
    ],
)
def test_malformed_inputs_fail(normalizer, problem):
    f, n = roads(), copy.deepcopy(normalizer)
    if problem == "duplicate_id":
        f.road_segment_id = "same"
    elif problem == "null_id":
        f.loc[0, "road_segment_id"] = None
    elif problem == "bad_direction":
        with pytest.raises(PipelineError):
            fit_normalizer(f, {**SPEC, NAMES[0]: "high"})
        return
    elif problem == "bad_normalizer":
        n["features"][NAMES[0]]["q95"] = -10
    elif problem == "bad_length":
        f.road_length_m = 0
    else:
        f = f.drop(columns="road_length_m")
    with pytest.raises(PipelineError):
        compare_weight_sets(f, n, SETS)


def test_baseline_cli_offline_reference_join_scope_and_provenance(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString

    root = Path(__file__).parents[2]
    (tmp_path / "configs").mkdir()
    for name in ["sources", "features", "experiments", "labeling", "study_area"]:
        doc = yaml.safe_load((root / f"configs/{name}.yaml").read_text())
        if name == "sources":
            doc["sources"]["assets"] = []
        (tmp_path / f"configs/{name}.yaml").write_text(yaml.safe_dump(doc))
    (tmp_path / "uv.lock").write_bytes(b"fixture lock")
    f = roads([0, 10, 1000]).drop(columns="road_length_m")
    f.to_parquet(tmp_path / "features.parquet")
    pd.DataFrame(
        {
            "road_segment_id": ["r0", "r1", "r2", "uncomputed"],
            "partition": ["reference", "reference", "holdout", "reference"],
        }
    ).to_parquet(tmp_path / "partitions.parquet")
    gpd.GeoDataFrame(
        {"road_segment_id": f.road_segment_id, "bridge": ["no", "no", "yes"]},
        geometry=[LineString([(0, i), (100, i)]) for i in range(3)],
        crs=9210,
    ).to_parquet(tmp_path / "roads.parquet")
    command = [
        sys.executable,
        str(root / "scripts/run_baseline.py"),
        "--repo-root",
        str(tmp_path),
        "--features",
        "features.parquet",
        "--partitions",
        "partitions.parquet",
        "--roads",
        "roads.parquet",
        "--run-id",
        "mcda_fixture",
    ]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    out = pd.read_parquet(
        tmp_path / "data/processed/mcda_fixture/road_susceptibility.parquet"
    )
    assert out.road_segment_id.tolist() == ["r0", "r1", "r2"]
    assert pd.isna(out.score.iloc[2])
    assert out.score_reason.iloc[2] == "bridge"
    report = json.loads(
        (tmp_path / "reports/runs/mcda_fixture/baseline_qc.json").read_text()
    )
    assert report["status"] == "complete"
    assert report["scope"] == "sample_only"
    assert report["partition_population_count"] == 4
    assert report["sample_reference_count"] == 2
    assert report["evaluation_status"] == "pending_label_review"
    n = json.loads((tmp_path / "artifacts/mcda_fixture/normalizer.json").read_text())
    assert n["features"][NAMES[0]]["q95"] == 9.5
    assert len(n["config_hash"]) == 64
    assert set(n["input_hashes"]) == {"features", "partitions", "roads"}
    # Same run ID cannot silently reuse a changed derived input.
    f.loc[2, NAMES[0]] = 1e12
    f.to_parquet(tmp_path / "features.parquet")
    rerun = subprocess.run(command, capture_output=True, text=True, check=False)
    assert rerun.returncode != 0


def test_empty_scoring_retains_schema_and_empty_sensitivity(normalizer):
    f = roads().iloc[:0]
    result = score_mcda(f, normalizer, EQUAL)
    assert result.empty
    assert {"score", "score_reason", "percentile_rank", "rank"}.issubset(result.columns)
    sensitivity = compare_weight_sets(f, normalizer, SETS)
    assert sensitivity["eligible_count"] == 0
    assert sensitivity["eligible_length_m"] == 0
    assert sensitivity["schemes"]["equal"]["top_k"]["0.1"]["selected_length_m"] == 0
