"""Offline reviewed-event evaluation; no source count or unknown label metrics."""

import importlib.util
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from flood_ml.config import PipelineError
from flood_ml.evaluation.capture import evaluate_capture, evaluation_gate
from flood_ml.labeling.reviews import LINK_COLUMNS


@pytest.fixture(autouse=True)
def disable_remote_tracking(monkeypatch):
    """CLI tests must never inherit credentials; SDK tests opt in synthetically."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


def test_evaluation_tests_do_not_inherit_tracking_credentials():
    key_present = bool(os.environ.get("WANDB_API_KEY"))
    assert not key_present


@pytest.fixture
def capture_case():
    ids = ["a", "b", "c", "d", "f", "r"]
    lines = [
        LineString([(0, 0), (10, 0)]),
        LineString([(10, 0), (100, 0)]),
        LineString([(0, 100), (100, 100)]),
        LineString([(0, 200), (100, 200)]),
        LineString([(0, 300), (9700, 300)]),
        LineString([(0, 400), (100, 400)]),
    ]
    roads = gpd.GeoDataFrame({"road_segment_id": ids}, geometry=lines, crs=9210)
    partitions = pd.DataFrame(
        {
            "road_segment_id": ids,
            "block_id": ["H0", "H0", "H1", "H2", "H0", "R0"],
            "partition": ["holdout"] * 5 + ["reference"],
        }
    )
    scores = pd.DataFrame(
        {
            "road_segment_id": ids,
            "score": [100.0, 0, 0, 0, 99, 0],
            "elevation_median_m": [1.0, 0, 2, 3, 4, 0],
            "mcda_eligible": [True] * 6,
        }
    )
    rows = []
    for group, n, segment_ids in [
        ("ref", 20, ["r"]),
        ("h0", 5, ["a", "b"]),
        ("h1", 5, ["c"]),
        ("h2", 10, ["d"]),
    ]:
        for i in range(n):
            for segment_id in segment_ids:
                event = f"{group}-{i}"
                rows.append(
                    {
                        "event_group_id": event,
                        "location_group_id": group,
                        "road_segment_id": segment_id,
                        "observation_id": event,
                        "source_id": event,
                        "evidence": "reviewed road extent",
                        "reviewed_by": "human",
                        "reviewed_at": "2026-11-02T12:00:00Z",
                        "event_evidence": "reviewed place and time",
                        "event_reviewed_by": "human",
                        "event_reviewed_at": "2026-11-02T12:00:00Z",
                        "affected_geometry": roads.set_index("road_segment_id").loc[
                            segment_id, "geometry"
                        ],
                        "affected_geometry_reviewed": True,
                        "affected_geometry_complete": True,
                    }
                )
    return {
        "scores": scores,
        "roads": roads,
        "confirmed_event_segments": pd.DataFrame(rows),
        "partitions": partitions,
        "config": {},
    }


def gate(case):
    return evaluation_gate(case["confirmed_event_segments"], case["partitions"])


def test_empty_review_reports_pending(capture_case):
    empty = pd.DataFrame(columns=LINK_COLUMNS)
    result = evaluation_gate(empty, capture_case["partitions"])
    assert result["status"] == "pending_label_review"
    assert result["missing_reference_events"] == 20
    assert result["missing_holdout_events"] == 20
    assert result["missing_holdout_blocks"] == 3
    assert result["holdout_block_shortfalls"] == {"H0": 5, "H1": 5, "H2": 5}


def test_gate_passes_distinct_reviewed_event_distribution(capture_case):
    result = gate(capture_case)
    assert result["status"] == "ready"
    assert result["reference_events"] == result["holdout_events"] == 20
    assert result["holdout_events_by_block"] == {"H0": 5, "H1": 5, "H2": 10}


@pytest.mark.parametrize(
    "partition, missing",
    [("ref-19", "missing_reference_events"), ("h2-9", "missing_holdout_events")],
)
def test_event_quota_shortfalls(capture_case, partition, missing):
    capture_case["confirmed_event_segments"] = capture_case[
        "confirmed_event_segments"
    ].loc[lambda x: x.event_group_id.ne(partition)]
    result = gate(capture_case)
    assert result["status"] == "pending_label_review"
    assert result[missing] == 1


def test_gate_cannot_reselect_blocks_or_count_sources(capture_case):
    events = capture_case["confirmed_event_segments"]
    events = pd.concat([events.loc[events.event_group_id.ne("h1-4")]] * 5)
    result = evaluation_gate(events, capture_case["partitions"])
    assert result["holdout_events"] == 19
    assert result["holdout_block_shortfalls"]["H1"] == 1
    assert result["missing_holdout_blocks"] == 1
    capture_case["partitions"].loc[lambda x: x.block_id.eq("H1"), "block_id"] = "H0"
    result = gate(capture_case)
    assert result["holdout_events"] == 20
    assert result["missing_holdout_blocks"] == 1
    assert result["status"] == "pending_label_review"


@pytest.mark.parametrize(
    "shared", ["event_group_id", "location_group_id", "observation_id"]
)
def test_cross_partition_groups_are_excluded_and_audited(capture_case, shared):
    events = capture_case["confirmed_event_segments"].copy()
    events.loc[events.event_group_id.eq("h0-0"), shared] = events.loc[
        events.event_group_id.eq("ref-0"), shared
    ].iloc[0]
    result = evaluation_gate(events, capture_case["partitions"])
    assert result["status"] == "pending_label_review"
    assert result["reference_events"] < 20
    assert result["holdout_events"] < 20
    assert "reviewed_group_crosses_partitions" in {
        x["reason"] for x in result["excluded_events"]
    }
    assert capture_case["partitions"].partition.tolist() == ["holdout"] * 5 + [
        "reference"
    ]


def test_missing_review_evidence_does_not_count(capture_case):
    capture_case["confirmed_event_segments"]["event_reviewed_by"] = ""
    result = gate(capture_case)
    assert result["reference_events"] == result["holdout_events"] == 0


def test_pending_returns_without_any_fabricated_metrics(capture_case):
    capture_case["confirmed_event_segments"] = pd.DataFrame(columns=LINK_COLUMNS)
    # Score geometry need not be consulted when review alone already fails.
    capture_case["scores"] = pd.DataFrame()
    result = evaluate_capture(**capture_case)
    assert result["status"] == "pending_label_review"
    assert result["gate"]["missing_reference_events"] == 20
    assert result["evaluation_performed"] is False
    assert not {"methods", "random", "bootstrap", "capture", "lift"}.intersection(
        result
    )
    json.dumps(result, allow_nan=False)


def test_length_capture_10_of_100_is_not_50_percent(capture_case):
    result = evaluate_capture(**capture_case)
    metric = result["methods"]["mcda"]["top_k"]["0.01"]
    assert metric["capture"] == 0
    assert metric["capture_any"] == 0.25
    assert metric["capture_all"] == 0
    assert metric["event_affected_fraction"]["h0-0"] == 0.1
    assert result["eligible_event_count"] == 20
    assert result["methods"]["elevation_only"]["top_k"]["0.01"]["capture"] == 0.25
    capture_case["scores"].loc[lambda x: x.road_segment_id.eq("b"), "score"] = 101
    capture_case["scores"].loc[lambda x: x.road_segment_id.eq("a"), "score"] = 0
    result = evaluate_capture(**capture_case)
    metric = result["methods"]["mcda"]["top_k"]["0.01"]
    assert metric["capture"] == 0.25
    assert metric["capture_all"] == 0
    assert metric["event_affected_fraction"]["h0-0"] == 0.9


def test_duplicates_and_union_geometry_do_not_inflate_denominator(capture_case):
    original = evaluate_capture(**capture_case)
    duplicate = capture_case["confirmed_event_segments"].copy()
    duplicate["source_id"] = "another-news-link"
    duplicate["observation_id"] += "-copy"
    capture_case["confirmed_event_segments"] = pd.concat(
        [duplicate, capture_case["confirmed_event_segments"]], ignore_index=True
    )
    result = evaluate_capture(**capture_case)
    assert result == original
    assert result["event_affected_length_m"]["h0-0"] == 100


def test_ties_and_budget_overshoot_are_deterministic(capture_case):
    capture_case["scores"]["score"] = 5.0
    result = evaluate_capture(**capture_case)
    metric = result["methods"]["mcda"]["top_k"]["0.01"]
    assert metric["selected_road_segment_ids"] == ["a", "b"]
    assert metric["actual_selected_fraction"] == 0.01
    capture_case["scores"] = capture_case["scores"].iloc[::-1]
    assert evaluate_capture(**capture_case) == result
    capture_case["scores"].loc[lambda x: x.road_segment_id.eq("f"), "score"] = 6
    metric = evaluate_capture(**capture_case)["methods"]["mcda"]["top_k"]["0.01"]
    assert metric["selected_length_m"] == 9700
    assert metric["overshoot_m"] == 9600
    assert metric["actual_selected_fraction"] == 0.97


def test_all_methods_share_score_eligible_holdout_length(capture_case):
    capture_case["scores"].loc[lambda x: x.road_segment_id.eq("f"), "mcda_eligible"] = (
        False
    )
    capture_case["roads"] = pd.concat(
        [
            capture_case["roads"],
            gpd.GeoDataFrame(
                {"road_segment_id": ["g"]},
                geometry=[LineString([(0, 500), (100, 500)])],
                crs=9210,
            ),
        ],
        ignore_index=True,
    )
    capture_case["scores"] = pd.concat(
        [
            capture_case["scores"],
            pd.DataFrame(
                [
                    {
                        "road_segment_id": "g",
                        "score": 100,
                        "mcda_eligible": True,
                        "elevation_median_m": np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    capture_case["partitions"] = pd.concat(
        [
            capture_case["partitions"],
            pd.DataFrame(
                [{"road_segment_id": "g", "block_id": "H1", "partition": "holdout"}]
            ),
        ],
        ignore_index=True,
    )
    result = evaluate_capture(**capture_case)
    assert result["eligible_road_segment_ids"] == ["a", "b", "c", "d"]
    assert result["eligible_road_length_m"] == 300
    for name in ["mcda", "elevation_only", "random"]:
        assert result["methods"][name]["eligible_road_length_m"] == 300
        assert set(result["methods"][name]["top_k"]) == {"0.01", "0.05", "0.1", "0.2"}
    random = result["methods"]["random"]
    assert random["seeds"] == list(range(42, 142))
    assert len(random["runs"]) == 100
    assert result["eligible_event_count"] == 20


@pytest.mark.parametrize(
    "column,value,reason",
    [
        ("affected_geometry_complete", False, "incomplete_affected_geometry"),
        ("affected_geometry_reviewed", False, "unreviewed_affected_geometry"),
        ("affected_geometry", None, "missing_affected_geometry"),
    ],
)
def test_incomplete_affected_geometry_excluded(capture_case, column, value, reason):
    links = capture_case["confirmed_event_segments"]
    links.loc[links.event_group_id.eq("h0-0"), column] = value
    result = evaluate_capture(**capture_case)
    assert result["status"] == "pending_label_review"
    assert result["gate"]["reviewed_holdout_events"] == 20
    assert result["gate"]["holdout_events"] == 19
    assert result["gate"]["missing_holdout_events"] == 1
    assert "methods" not in result
    assert result["exclusion_counts"][reason] == 1
    assert result["excluded_affected_length_m"][reason] == (
        None if value is None else 100
    )


def test_accepted_candidate_links_do_not_assert_complete_affected_geometry(
    capture_case,
):
    capture_case["confirmed_event_segments"] = capture_case[
        "confirmed_event_segments"
    ].drop(columns=["affected_geometry_complete"])
    result = evaluate_capture(**capture_case)
    assert result["status"] == "pending_label_review"
    assert result["gate"]["holdout_events"] == 0
    assert result["exclusion_counts"]["incomplete_affected_geometry"] == 20
    assert "methods" not in result


def test_bootstrap_zero_denominators_remain_undefined(capture_case):
    extra = (
        capture_case["confirmed_event_segments"]
        .loc[lambda x: x.event_group_id.str.startswith("h1")]
        .copy()
    )
    extra["event_group_id"] = extra.event_group_id.str.replace("h1", "h3")
    extra["observation_id"] = extra.event_group_id
    extra["location_group_id"] = "h3"
    extra["road_segment_id"] = "f"
    extra["affected_geometry"] = (
        capture_case["roads"].set_index("road_segment_id").loc["f", "geometry"]
    )
    capture_case["partitions"].loc[lambda x: x.road_segment_id.eq("f"), "block_id"] = (
        "H3"
    )
    capture_case["scores"].loc[lambda x: x.road_segment_id.eq("f"), "mcda_eligible"] = (
        False
    )
    capture_case["confirmed_event_segments"] = pd.concat(
        [capture_case["confirmed_event_segments"], extra], ignore_index=True
    )
    result = evaluate_capture(**capture_case)
    ci = result["methods"]["mcda"]["top_k"]["0.01"]["uncertainty"]
    assert ci["replicates"] == 1000
    assert 0 < ci["undefined_replicates"] < 1000
    assert ci["valid_replicates"] + ci["undefined_replicates"] == 1000
    assert ci["capture_ci95"] is not None
    assert ci["undefined_lift_replicates"] >= ci["undefined_replicates"]
    assert result["exploratory"] is True
    assert "source" in result["limitations"].lower()
    json.dumps(result, allow_nan=False)


def test_geometry_outside_confirmed_segment_is_not_evaluated(capture_case):
    links = capture_case["confirmed_event_segments"]
    links.loc[links.event_group_id.eq("h0-0"), "affected_geometry"] = LineString(
        [(0, 0), (200, 0)]
    )
    result = evaluate_capture(**capture_case)
    assert result["exclusion_counts"]["affected_geometry_outside_confirmed_road"] == 1


def test_invalid_partitions_or_duplicate_scores_fail(capture_case):
    bad = capture_case["partitions"].copy()
    bad.loc[0, "partition"] = "train"
    with pytest.raises(PipelineError):
        evaluation_gate(capture_case["confirmed_event_segments"], bad)
    capture_case["scores"] = pd.concat([capture_case["scores"]] * 2)
    with pytest.raises(PipelineError):
        evaluate_capture(**capture_case)


def test_review_output_wkb_hex_geometry_is_supported(capture_case):
    original = evaluate_capture(**capture_case)
    capture_case["confirmed_event_segments"]["affected_geometry"] = capture_case[
        "confirmed_event_segments"
    ].affected_geometry.map(lambda geom: geom.wkb_hex)
    assert evaluate_capture(**capture_case) == original


def test_zero_score_coverage_keeps_metrics_undefined(capture_case):
    capture_case["scores"]["mcda_eligible"] = False
    result = evaluate_capture(**capture_case)
    assert result["eligible_road_length_m"] == 0
    assert result["eligible_event_count"] == 0
    assert result["status"] == "pending_label_review"
    assert result["gate"]["missing_reference_events"] == 20
    assert result["gate"]["missing_holdout_events"] == 20
    assert "methods" not in result


def test_score_coverage_shortfall_cannot_pass_review_quota(capture_case):
    capture_case["scores"].loc[lambda x: x.road_segment_id.eq("c"), "mcda_eligible"] = (
        False
    )
    result = evaluate_capture(**capture_case)
    assert result["status"] == "pending_label_review"
    assert result["review_geometry_gate"]["status"] == "ready"
    assert result["gate"]["holdout_events"] == 15
    assert result["gate"]["missing_holdout_events"] == 5
    assert result["gate"]["missing_holdout_blocks"] == 1
    assert result["exclusion_counts"]["affected_geometry_outside_score_coverage"] == 5
    assert (
        result["excluded_affected_length_m"]["affected_geometry_outside_score_coverage"]
        == 500
    )
    assert "methods" not in result


def test_evaluation_rules_cannot_relax_the_data_gate(capture_case):
    capture_case["config"] = {"evaluation": {"minimum_holdout_events": 1}}
    with pytest.raises(PipelineError, match="predeclared"):
        evaluate_capture(**capture_case)


def track_b_frame(case):
    holdout = case["scores"].loc[
        lambda x: x.road_segment_id.ne("r"), ["road_segment_id", "score"]
    ]
    frames = []
    for ratio, seed in [(10, 42), (10, 43), (5, 42)]:
        frame = holdout.rename(columns={"score": "relative_score"}).copy()
        frame["model_id"] = f"track_b_r{ratio}_s{seed}"
        frame["background_ratio"] = ratio
        frame["seed"] = seed
        frame["partition"] = "holdout"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_capture_without_track_b_keeps_legacy_methods(capture_case):
    original = evaluate_capture(**capture_case)
    assert evaluate_capture(**capture_case, track_b_scores=None) == original
    assert set(original["methods"]) == {"mcda", "elevation_only", "random"}
    assert not {"track_b_stability", "track_b_coverage"}.intersection(original)


def test_capture_reports_each_track_b_draw_on_common_population(capture_case):
    capture_case["scores"].loc[lambda x: x.road_segment_id.eq("f"), "mcda_eligible"] = (
        False
    )
    legacy = evaluate_capture(**capture_case)
    scores = track_b_frame(capture_case)
    result = evaluate_capture(**capture_case, track_b_scores=scores)
    assert result["status"] == "complete"
    assert result["exploratory"] is True
    assert result["track_b_coverage"]["status"] == "ready"
    assert result["eligible_road_segment_ids"] == ["a", "b", "c", "d"]
    for name in ("mcda", "elevation_only", "random"):
        assert result["methods"][name] == legacy["methods"][name]
    assert set(result["methods"]["track_b"]) == set(scores.model_id)
    for model in result["methods"]["track_b"].values():
        assert model["eligible_road_length_m"] == result["eligible_road_length_m"]
        for fraction, metric in model["top_k"].items():
            baseline = legacy["methods"]["mcda"]["top_k"][fraction]
            assert metric["capture"] == baseline["capture"]
            assert metric["capture_any"] == baseline["capture_any"]
            assert metric["capture_all"] == baseline["capture_all"]
            assert metric["lift"] == baseline["lift"]
            assert metric["capture_difference_vs_mcda"] == 0
            assert "lift_vs_elevation_only" in metric
            assert metric["uncertainty"] == baseline["uncertainty"]
    stability = result["track_b_stability"]["by_background_ratio"]["10"]
    assert stability["seeds"] == [42, 43]
    for metric in stability["top_k"].values():
        assert metric["capture"]["iqr"] == 0
        assert metric["mean_pairwise_jaccard"] == 1
    assert result == evaluate_capture(**capture_case, track_b_scores=scores.iloc[::-1])
    json.dumps(result, allow_nan=False)


def test_capture_missing_track_b_score_returns_pending_coverage(capture_case):
    scores = track_b_frame(capture_case)
    # f has no event; coverage must still require every eligible holdout road.
    scores = scores.loc[
        ~(scores.model_id.eq("track_b_r10_s42") & scores.road_segment_id.eq("f"))
    ]
    result = evaluate_capture(**capture_case, track_b_scores=scores)
    assert result["status"] == "pending_track_b_coverage"
    assert result["evaluation_performed"] is False
    assert "methods" not in result
    assert "track_b_stability" not in result
    coverage = result["track_b_coverage"]
    assert coverage["models"]["track_b_r10_s42"]["missing_road_segment_ids"] == ["f"]


def test_capture_empty_track_b_population_is_pending(capture_case):
    result = evaluate_capture(
        **capture_case, track_b_scores=track_b_frame(capture_case).iloc[:0]
    )
    assert result["status"] == "pending_track_b_coverage"
    assert result["track_b_coverage"]["model_count"] == 0
    assert "methods" not in result


def test_track_b_stability_compares_seed_rankings_within_ratio(capture_case):
    scores = track_b_frame(capture_case)
    reverse = scores.model_id.eq("track_b_r10_s43")
    scores.loc[reverse, "relative_score"] *= -1
    result = evaluate_capture(**capture_case, track_b_scores=scores)
    models = result["methods"]["track_b"]
    stability = result["track_b_stability"]["by_background_ratio"]
    for fraction, metric in stability["10"]["top_k"].items():
        draws = [models[f"track_b_r10_s{seed}"]["top_k"][fraction] for seed in (42, 43)]
        captures = [draw["capture"] for draw in draws]
        assert metric["capture"]["median"] == float(np.median(captures))
        assert metric["capture"]["iqr"] == float(
            np.quantile(captures, 0.75) - np.quantile(captures, 0.25)
        )
        first, second = [set(draw["selected_road_segment_ids"]) for draw in draws]
        assert metric["mean_pairwise_jaccard"] == len(first & second) / len(
            first | second
        )
        assert stability["5"]["top_k"][fraction]["mean_pairwise_jaccard"] is None


def test_capture_geometry_gate_withholds_track_b_metrics(capture_case):
    capture_case["confirmed_event_segments"]["affected_geometry_complete"] = False
    result = evaluate_capture(
        **capture_case, track_b_scores=track_b_frame(capture_case)
    )
    assert result["status"] == "pending_label_review"
    assert result["evaluation_performed"] is False
    assert "methods" not in result
    assert "track_b_stability" not in result


@pytest.mark.parametrize(
    "problem", ["duplicate", "nan", "inf", "metadata", "partition"]
)
def test_capture_rejects_duplicate_or_nonfinite_track_b_scores(capture_case, problem):
    scores = track_b_frame(capture_case)
    if problem == "duplicate":
        scores = pd.concat([scores, scores.iloc[[0]]], ignore_index=True)
    elif problem in {"nan", "inf"}:
        scores.loc[0, "relative_score"] = np.nan if problem == "nan" else np.inf
    elif problem == "metadata":
        scores.loc[0, "seed"] = 99
    else:
        scores.loc[0, "partition"] = "reference"
    with pytest.raises(PipelineError, match="Track B"):
        evaluate_capture(**capture_case, track_b_scores=scores)


def test_capture_rejects_nullable_missing_track_b_score_before_metrics(capture_case):
    scores = track_b_frame(capture_case)
    scores["relative_score"] = scores.relative_score.astype("Float64")
    scores.loc[0, "relative_score"] = pd.NA
    with pytest.raises(PipelineError, match="Track B relative scores must be finite"):
        evaluate_capture(**capture_case, track_b_scores=scores)


def test_capture_rejects_two_models_claiming_the_same_ratio_seed(capture_case):
    scores = track_b_frame(capture_case)
    scores.loc[scores.model_id.eq("track_b_r10_s43"), "seed"] = 42
    with pytest.raises(PipelineError, match="Track B"):
        evaluate_capture(**capture_case, track_b_scores=scores)


def test_cli_loads_and_hashes_optional_track_b_input(tmp_path, capture_case, capsys):
    import shutil

    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root / "configs", tmp_path / "configs")
    for name in ("scores", "roads", "confirmed_event_segments", "partitions"):
        frame = capture_case[name].copy()
        if name == "confirmed_event_segments":
            frame["affected_geometry"] = frame.affected_geometry.map(lambda g: g.wkb)
        frame.to_parquet(tmp_path / f"{name}.parquet")
    track_b_frame(capture_case).to_parquet(tmp_path / "track_b.parquet")
    arguments = ["--repo-root", str(tmp_path), "--output", "evaluation.json"]
    for name in ("scores", "roads", "confirmed-event-segments", "partitions"):
        arguments.extend([f"--{name}", f"{name.replace('-', '_')}.parquet"])
    arguments.extend(["--track-b-scores", "track_b.parquet"])
    cli = load_cli()
    assert cli.main(arguments) == 0
    assert (
        json.loads(capsys.readouterr().out)["tracking_status"] == "skipped_no_api_key"
    )
    result = json.loads((tmp_path / "evaluation.json").read_text())
    assert result["status"] == "complete"
    assert result["input_paths"]["track_b_scores"] == "track_b.parquet"
    assert len(result["input_hashes"]["track_b_scores"]) == 64
    assert len(result["methods"]["track_b"]) == 3
    # The new input participates in the existing preflight alias protection.
    before = (tmp_path / "track_b.parquet").read_bytes()
    arguments[arguments.index("--output") + 1] = "track_b.parquet"
    assert cli.main(arguments) == 1
    assert (tmp_path / "track_b.parquet").read_bytes() == before


def load_cli():
    path = Path(__file__).resolve().parents[2] / "scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_track_b_tracking_keeps_evaluation_git_provenance_clean(
    tmp_path, capture_case, monkeypatch
):
    """SDK run files must not dirty the checkout before evaluation snapshots Git."""
    import shutil
    import subprocess
    import sys
    import tempfile
    from types import SimpleNamespace

    from flood_ml.utils.wandb_tracking import log_stage_report

    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / ".gitignore").write_text("reports/\n")
    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root / "configs", repo / "configs")
    for name in ("scores", "roads", "confirmed_event_segments", "partitions"):
        frame = capture_case[name].copy()
        if name == "confirmed_event_segments":
            frame["affected_geometry"] = frame.affected_geometry.map(lambda g: g.wkb)
        frame.to_parquet(repo / f"{name}.parquet")
    for args in (
        ["init"],
        ["add", "."],
        [
            "-c",
            "user.name=Synthetic Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "synthetic fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    monkeypatch.chdir(repo)
    output = repo / "reports" / "evaluation.json"
    arguments = ["--repo-root", str(repo), "--output", "reports/evaluation.json"]
    for name in ("scores", "roads", "confirmed-event-segments", "partitions"):
        arguments.extend([f"--{name}", f"{name.replace('-', '_')}.parquet"])
    cli = load_cli()
    assert cli.main(arguments) == 0
    before = json.loads(output.read_text())["git"]
    assert before["commit"] is not None
    assert before["dirty"] is False

    roots = []

    def initialize(**options):
        run_root = Path(options.get("dir", Path.cwd())).resolve()
        roots.append(run_root)
        # Model the SDK's local output without authentication or a live service.
        run_dir = run_root / "wandb" / options["job_type"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "debug.log").write_text("synthetic SDK output")
        return SimpleNamespace(
            summary={}, log=lambda metrics: None, finish=lambda: None
        )

    monkeypatch.setenv("WANDB_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=initialize, Settings=lambda **kw: kw)
    )
    assert (
        log_stage_report(
            run_id="synthetic", stage="track-b", config={}, summary={}, metrics={}
        )
        == "logged"
    )
    monkeypatch.chdir(repo / "configs")
    assert cli.main(arguments) == 0
    assert json.loads(output.read_text())["git"] == before
    assert len(roots) == 2
    assert roots[0] == roots[1]
    assert roots[0] == Path(tempfile.gettempdir()) / "hcmc-road-flood-ml-wandb"
    assert roots[0].is_absolute()
    assert not roots[0].is_relative_to(repo)
    assert roots[0].is_dir()


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("remote_failure", [False, True])
def test_cli_tracks_only_safe_evaluation_metadata_after_publication(
    tmp_path, capture_case, monkeypatch, capsys, pending, remote_failure
):
    """Catch missing logging, private payload leakage, and writes after logging."""
    import shutil
    import sys
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root / "configs", tmp_path / "configs")
    if pending:
        capture_case["confirmed_event_segments"] = pd.DataFrame(columns=LINK_COLUMNS)
    for name in ("scores", "roads", "confirmed_event_segments", "partitions"):
        frame = capture_case[name].copy()
        if name == "confirmed_event_segments":
            frame["affected_geometry"] = frame.affected_geometry.map(lambda g: g.wkb)
        frame.to_parquet(tmp_path / f"{name}.parquet")
    track_b_frame(capture_case).to_parquet(tmp_path / "track_b.parquet")
    arguments = [
        "--repo-root",
        str(tmp_path),
        "--output",
        "evaluation.json",
        "--run-id",
        "synthetic_evaluation",
    ]
    for name in ("scores", "roads", "confirmed-event-segments", "partitions"):
        arguments.extend([f"--{name}", f"{name.replace('-', '_')}.parquet"])
    arguments.extend(["--track-b-scores", "track_b.parquet"])
    output = tmp_path / "evaluation.json"
    observed = {"logs": [], "finishes": 0}

    def log(metrics):
        observed["logs"].append(metrics)
        if remote_failure:
            raise RuntimeError("synthetic remote failure")

    def finish():
        observed["finishes"] += 1

    summary = {}

    def initialize(**options):
        # The SDK boundary must only be reached after the complete local report.
        observed["before"] = output.read_bytes()
        observed["options"] = options
        if remote_failure and pending:
            raise RuntimeError("synthetic initialization failure")
        return SimpleNamespace(summary=summary, log=log, finish=finish)

    monkeypatch.setenv("WANDB_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=initialize, Settings=lambda **kw: kw)
    )
    assert load_cli().main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    expected_tracking = "failed" if remote_failure else "logged"
    assert printed["tracking_status"] == expected_tracking
    assert output.read_bytes() == observed["before"]
    report = json.loads(output.read_text())
    assert "tracking_status" not in report
    options = observed["options"]
    assert options["name"] == "synthetic_evaluation-evaluation"
    assert options["group"] == "synthetic_evaluation"
    assert options["job_type"] == "evaluation"
    assert set(options["config"]) == {
        "run_id",
        "evaluation",
        "config_hash",
        "source_commit",
        "source_dirty",
    }
    assert options["config"]["evaluation"]["minimum_holdout_events"] == 20
    assert options["config"]["evaluation"]["bootstrap_repetitions"] == 1000
    assert options["config"]["config_hash"] == report["config_hash"]
    assert options["config"]["source_commit"] == report["git"]["commit"]
    assert options["config"]["source_dirty"] == report["git"]["dirty"]
    expected_summary = {
        "status": "pending_label_review" if pending else "complete",
        "evaluation_performed": not pending,
        "reference_events": 0 if pending else 20,
        "holdout_events": 0 if pending else 20,
        "missing_reference_events": 20 if pending else 0,
        "missing_holdout_events": 20 if pending else 0,
        "missing_holdout_blocks": 3 if pending else 0,
        "qualifying_holdout_blocks": 0 if pending else 3,
    }
    assert summary == ({} if remote_failure and pending else expected_summary)
    if pending:
        assert observed["logs"] == []
    else:
        assert len(observed["logs"]) == 1
        metrics = observed["logs"][0]
        prefix = "methods/track_b/track_b_r10_s42/top_k/0.01"
        assert metrics[f"{prefix}/capture_any"] == 0.25
        assert metrics[f"{prefix}/capture_difference_vs_mcda"] == 0
        assert metrics[f"{prefix}/uncertainty/replicates"] == 1000
        assert f"{prefix}/uncertainty/capture_ci95/lower" in metrics
        assert f"{prefix}/uncertainty/capture_ci95/upper" in metrics
        allowed = {
            "capture",
            "capture_any",
            "capture_all",
            "lift",
            "lift_vs_mcda",
            "lift_vs_elevation_only",
            "capture_difference_vs_mcda",
            "capture_difference_vs_elevation_only",
        }
        uncertainty = {
            "replicates",
            "valid_replicates",
            "undefined_replicates",
            "undefined_lift_replicates",
            "capture_ci95/lower",
            "capture_ci95/upper",
            "capture_any_ci95/lower",
            "capture_any_ci95/upper",
            "capture_all_ci95/lower",
            "capture_all_ci95/upper",
            "lift_ci95/lower",
            "lift_ci95/upper",
        }
        assert {key.split("/top_k/")[1].split("/", 1)[1] for key in metrics} <= (
            allowed | {f"uncertainty/{key}" for key in uncertainty}
        )
        assert all(
            np.isfinite(value) and not isinstance(value, bool)
            for value in metrics.values()
        )
    assert observed["finishes"] == (0 if remote_failure and pending else 1)
    payload = json.dumps(
        {"config": options["config"], "summary": summary, "metrics": observed["logs"]}
    )
    for private in (
        "eligible_event_ids",
        "event_affected",
        "selected_road_segment_ids",
        "input_paths",
        "input_hashes",
        "affected_geometry",
        "reviewed_by",
        "reviewed road extent",
        "h0-0",
        "scores.parquet",
        str(tmp_path),
    ):
        assert private not in payload


def test_tracking_metrics_omit_nonfinite_null_boolean_and_private_numeric_leaves():
    report = {
        "evaluation_performed": True,
        "methods": {
            "mcda": {
                "eligible_road_length_m": 123,
                "top_k": {
                    "0.1": {
                        "capture": 0.5,
                        "capture_any": float("inf"),
                        "capture_all": True,
                        "lift": None,
                        "lift_vs_elevation_only": float("nan"),
                        "actual_selected_fraction": 0.11,
                        "event_affected_fraction": {"private-event": 0.8},
                        "uncertainty": {
                            "replicates": 1000,
                            "valid_replicates": 999,
                            "undefined_replicates": 1,
                            "undefined_lift_replicates": 3,
                            "capture_ci95": [0.2, 0.8],
                            "capture_any_ci95": None,
                            "capture_all_ci95": [float("-inf"), 0.9],
                            "lift_ci95": [float("nan"), None],
                            "private_numeric": 456,
                        },
                    }
                },
            },
            "private_method": {"top_k": {"0.1": {"capture": 0.9}}},
        },
    }
    cli = load_cli()
    prefix = "methods/mcda/top_k/0.1"
    assert cli._tracking_metrics(report) == {
        f"{prefix}/capture": 0.5,
        f"{prefix}/uncertainty/replicates": 1000,
        f"{prefix}/uncertainty/valid_replicates": 999,
        f"{prefix}/uncertainty/undefined_replicates": 1,
        f"{prefix}/uncertainty/undefined_lift_replicates": 3,
        f"{prefix}/uncertainty/capture_ci95/lower": 0.2,
        f"{prefix}/uncertainty/capture_ci95/upper": 0.8,
        f"{prefix}/uncertainty/capture_all_ci95/upper": 0.9,
    }
    report["evaluation_performed"] = False
    assert cli._tracking_metrics(report) == {}


def test_cli_atomically_writes_real_empty_review_pending(tmp_path, capture_case):
    root = Path(__file__).resolve().parents[2]
    scores = tmp_path / "scores.parquet"
    roads = tmp_path / "roads.parquet"
    events = tmp_path / "events.parquet"
    partitions = tmp_path / "partitions.parquet"
    output = tmp_path / "evaluation.json"
    capture_case["scores"].to_parquet(scores)
    capture_case["roads"].to_parquet(roads)
    pd.DataFrame(columns=LINK_COLUMNS).to_parquet(events)
    capture_case["partitions"].to_parquet(partitions)
    arguments = [
        "--repo-root",
        str(root),
        "--scores",
        str(scores),
        "--roads",
        str(roads),
        "--confirmed-event-segments",
        str(events),
        "--partitions",
        str(partitions),
        "--output",
        str(output),
    ]
    cli = load_cli()
    # Repo-relative paths are required; use the isolated tmp directory as repo.
    import shutil

    shutil.copytree(root / "configs", tmp_path / "configs")
    arguments[1] = str(tmp_path)
    for i, value in enumerate(arguments):
        if value in map(str, [scores, roads, events, partitions, output]):
            arguments[i] = Path(value).name
    assert cli.main(arguments) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "pending_label_review"
    assert result["gate"]["missing_reference_events"] == 20
    assert result["gate"]["missing_holdout_events"] == 20
    assert result["input_hashes"].keys() == {
        "scores",
        "roads",
        "confirmed_event_segments",
        "partitions",
    }
    assert not {"methods", "bootstrap", "random"}.intersection(result)
    assert list(tmp_path.glob("*.tmp")) == []
    before = scores.read_bytes()
    arguments[-1] = "scores.parquet"
    assert cli.main(arguments) == 1
    assert scores.read_bytes() == before


@pytest.mark.parametrize(
    "target,hardlink,absolute_entry",
    [
        ("selected_yaml", False, False),
        ("selected_yml", False, False),
        ("selected_yml", False, True),
        ("selected_yml", True, False),
        ("sources", False, False),
        ("study_area", False, False),
        ("features", False, False),
        ("labeling", False, False),
        ("features", True, False),
        ("section_entry", False, False),
    ],
)
def test_cli_rejects_actual_loaded_config_alias_before_writes(
    tmp_path, capture_case, capsys, target, hardlink, absolute_entry
):
    import os
    import shutil

    import yaml

    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root / "configs", tmp_path / "configs")
    custom = tmp_path / "custom"
    custom.mkdir()
    experiments = yaml.safe_load((tmp_path / "configs/experiments.yaml").read_text())
    selected = custom / (
        "experiments.yaml" if target == "selected_yaml" else "experiments.yml"
    )
    if target in {"sources", "study_area", "features", "labeling"}:
        destination = custom / f"{target}.bundle"
        shutil.copyfile(tmp_path / f"configs/{target}.yaml", destination)
        experiments["experiments"]["configs"][target] = str(
            destination.relative_to(tmp_path)
        )
    elif target == "section_entry":
        selected = custom / "features.yml"
        shutil.copyfile(tmp_path / "configs/features.yaml", selected)
        destination = selected
    else:
        destination = selected
    if target != "section_entry":
        selected.write_text(yaml.safe_dump(experiments))
    if hardlink:
        alias = custom / "report.json"
        os.link(destination, alias)
        destination = alias
    capture_case["scores"].to_parquet(tmp_path / "scores.parquet")
    capture_case["roads"].to_parquet(tmp_path / "roads.parquet")
    pd.DataFrame(columns=LINK_COLUMNS).to_parquet(tmp_path / "events.parquet")
    capture_case["partitions"].to_parquet(tmp_path / "partitions.parquet")
    entry = str(selected) if absolute_entry else str(selected.relative_to(tmp_path))
    arguments = [
        "--repo-root",
        str(tmp_path),
        "--config",
        entry,
        "--scores",
        "scores.parquet",
        "--roads",
        "roads.parquet",
        "--confirmed-event-segments",
        "events.parquet",
        "--partitions",
        "partitions.parquet",
        "--output",
        str(destination.relative_to(tmp_path)),
    ]

    def snapshot():
        return {
            str(path.relative_to(tmp_path)): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    result = load_cli().main(arguments)
    assert snapshot() == before  # Also proves no report/start/QC marker was added.
    assert result == 1
    assert "aliases publication destination" in capsys.readouterr().err
