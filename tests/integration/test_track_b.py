"""Offline synthetic Track B experiment through the real publication CLIs."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[2]


def synthetic_experiment(tmp_path, *, reviewed):
    root = tmp_path / "repo"
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    (raw / "synthetic-source.txt").write_text("Offline synthetic evidence only.\n")
    (root / "configs").mkdir()
    for name in ("sources", "study_area", "features", "labeling", "experiments"):
        document = yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text())
        if name == "sources":
            document["sources"]["assets"] = []
        (root / f"configs/{name}.yaml").write_text(yaml.safe_dump(document))
    (root / "uv.lock").write_text("synthetic offline lock\n")

    reference_positive = [f"r{i:03}" for i in range(20)]
    background = [f"u{i:03}" for i in range(400)]
    holdout = [f"h{i:03}" for i in range(40)]
    ids = [*reference_positive, *background, *holdout, "guard", "excluded", "negative"]
    values = np.linspace(0.0, 1.0, len(ids))
    features = pd.DataFrame(
        {
            "road_segment_id": ids,
            "elevation_median_m": values * 20,
            "relative_elevation_m": values * 2,
            "slope_mean_deg": values * 5,
            "dist_water_m": values * 1000,
            "builtup_ratio_250m": 1 - values,
            "score_eligible": True,
        }
    )
    positive_ids = set(reference_positive + holdout[:20]) if reviewed else set()
    labels = pd.DataFrame(
        {
            "road_segment_id": ids,
            "label_status": [
                "positive"
                if rid in positive_ids
                else "verified_negative"
                if rid == "negative"
                else "unknown"
                for rid in ids
            ],
            "label_binary": [
                1.0 if rid in positive_ids else 0.0 if rid == "negative" else np.nan
                for rid in ids
            ],
        }
    )
    holdout_blocks = ["H0"] * 5 + ["H1"] * 5 + ["H2"] * 10
    holdout_blocks += [f"H{i % 3}" for i in range(20)]
    partitions = pd.DataFrame(
        {
            "road_segment_id": ids,
            "block_id": [f"R{i % 4}" for i in range(420)]
            + holdout_blocks
            + ["G0", "X0", "R0"],
            "partition": ["reference"] * 420
            + ["holdout"] * 40
            + ["guard", "excluded", "reference"],
        }
    )
    roads = gpd.GeoDataFrame(
        {"road_segment_id": ids},
        geometry=[LineString([(0, i * 10), (100, i * 10)]) for i in range(len(ids))],
        crs=9210,
    )
    geometry = roads.set_index("road_segment_id").geometry
    event_ids = reference_positive + holdout[:20]
    events = pd.DataFrame(
        [
            {
                "road_segment_id": rid,
                "event_group_id": f"event-{rid}",
                "location_group_id": f"location-{rid}",
                "observation_id": f"observation-{rid}",
                "source_id": f"source-{rid}",
                "evidence": "synthetic reviewed road extent",
                "reviewed_by": "synthetic reviewer",
                "reviewed_at": "2026-11-01T00:00:00Z",
                "event_evidence": "synthetic reviewed independent event",
                "event_reviewed_by": "synthetic reviewer",
                "event_reviewed_at": "2026-11-01T00:00:00Z",
                "affected_geometry": geometry.loc[rid].wkb,
                "affected_geometry_reviewed": True,
                "affected_geometry_complete": True,
            }
            for rid in event_ids
        ]
    )
    if not reviewed:
        events = events.iloc[:0]
    for name, frame in {
        "road_features": features,
        "road_labels": labels,
        "confirmed_event_segments": events,
        "partitions": partitions,
        "roads": roads,
    }.items():
        frame.to_parquet(raw / f"{name}.parquet", index=False)
    return root


def run_cli(root, script, run_id, **inputs):
    arguments = ["--repo-root", str(root), "--run-id", run_id]
    for name, value in inputs.items():
        arguments.extend([f"--{name.replace('_', '-')}", value])
    result = subprocess.run(
        [sys.executable, str(ROOT / f"scripts/{script}.py"), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def train_and_baseline(root):
    run_cli(
        root,
        "train",
        "track_b_synthetic",
        features="data/raw/road_features.parquet",
        labels="data/raw/road_labels.parquet",
        confirmed_event_segments="data/raw/confirmed_event_segments.parquet",
        partitions="data/raw/partitions.parquet",
    )
    run_cli(
        root,
        "run_baseline",
        "baseline_synthetic",
        features="data/raw/road_features.parquet",
        partitions="data/raw/partitions.parquet",
        roads="data/raw/roads.parquet",
    )


def capture(root, *, with_track_b):
    inputs = {
        "scores": "data/processed/baseline_synthetic/road_susceptibility.parquet",
        "roads": "data/raw/roads.parquet",
        "confirmed_event_segments": "data/raw/confirmed_event_segments.parquet",
        "partitions": "data/raw/partitions.parquet",
    }
    if with_track_b:
        inputs["track_b_scores"] = (
            "data/processed/track_b_synthetic/track_b_scores.parquet"
        )
    run_cli(root, "evaluate", "capture_synthetic", **inputs)
    return json.loads(
        (root / "reports/runs/capture_synthetic/evaluation.json").read_text()
    )


def input_snapshot(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (root / "data/raw", root / "configs")
        for path in directory.iterdir()
        if path.is_file()
    }


def test_track_b_integration_without_review_is_pending(tmp_path):
    # Removing the review gate would create fitted artifacts and fail this test.
    root = synthetic_experiment(tmp_path, reviewed=False)
    before = input_snapshot(root)
    train_and_baseline(root)
    training = json.loads(
        (root / "reports/runs/track_b_synthetic/track_b.json").read_text()
    )
    assert training["status"] == "pending_positive_review"
    assert training["gate"]["feature_eligible_reference_events"] == 0
    assert not (
        root / "data/processed/track_b_synthetic/track_b_scores.parquet"
    ).exists()
    assert not list((root / "artifacts/track_b_synthetic").rglob("*.joblib"))
    evaluation = capture(root, with_track_b=False)
    assert evaluation["status"] == "pending_label_review"
    assert evaluation["evaluation_performed"] is False
    assert not {"methods", "track_b_stability", "capture", "lift"}.intersection(
        evaluation
    )
    assert input_snapshot(root) == before


def test_track_b_train_then_capture_uses_all_gates(tmp_path):
    # Leaking holdout into fit, dropping draws, changing labels or denominators
    # breaks the observable manifests or comparison assertions below.
    root = synthetic_experiment(tmp_path, reviewed=True)
    before = input_snapshot(root)
    train_and_baseline(root)
    training = json.loads(
        (root / "reports/runs/track_b_synthetic/track_b.json").read_text()
    )
    expected_models = {
        f"track_b_r{ratio}_s{seed}" for ratio in (5, 10, 20) for seed in range(42, 62)
    }
    assert training["status"] == "ready"
    assert training["gate"]["feature_eligible_reference_events"] == 20
    assert training["counts"]["fitted_models"] == 60
    assert training["source_inputs_unchanged"] is True
    assert {row["model_id"] for row in training["draws"]} == expected_models
    for draw in training["draws"]:
        assert draw["status"] == "fitted"
        assert draw["positive_count"] == 20
        assert draw["background_count"] == 20 * draw["background_ratio"]
        assert set(draw["positive_road_ids"]) == {f"r{i:03}" for i in range(20)}
        assert set(draw["background_road_ids"]).issubset(
            {f"u{i:03}" for i in range(400)}
        )
    models = root / "artifacts/track_b_synthetic/track_b_models"
    assert {path.stem for path in models.glob("*.joblib")} == expected_models
    score_path = root / "data/processed/track_b_synthetic/track_b_scores.parquet"
    scores = pd.read_parquet(score_path)
    assert len(scores) == 2400
    assert set(scores.road_segment_id) == {f"h{i:03}" for i in range(40)}
    assert set(scores.model_id) == expected_models
    assert scores.partition.eq("holdout").all()
    assert np.isfinite(scores.relative_score).all()
    assert not scores.duplicated(["road_segment_id", "model_id"]).any()
    assert "label_binary" not in scores
    labels = pd.read_parquet(root / "data/raw/road_labels.parquet")
    unknown = labels.loc[labels.road_segment_id.str.startswith("u")]
    assert unknown.label_status.eq("unknown").all()
    assert unknown.label_binary.isna().all()

    evaluation = capture(root, with_track_b=True)
    assert evaluation["status"] == "complete"
    assert evaluation["evaluation_performed"] is True
    assert evaluation["exploratory"] is True
    assert evaluation["gate"]["reference_events"] == 20
    assert evaluation["gate"]["holdout_events"] == 20
    assert evaluation["gate"]["holdout_events_by_block"] == {
        "H0": 5,
        "H1": 5,
        "H2": 10,
    }
    assert evaluation["track_b_coverage"]["status"] == "ready"
    assert evaluation["eligible_event_count"] == 20
    assert evaluation["eligible_road_length_m"] == 4000.0
    methods = evaluation["methods"]
    assert set(methods["track_b"]) == expected_models
    all_methods = [
        methods["mcda"],
        methods["elevation_only"],
        methods["random"],
        *methods["track_b"].values(),
    ]
    assert all(method["eligible_road_length_m"] == 4000.0 for method in all_methods)
    for method in methods["track_b"].values():
        assert set(method["top_k"]) == {"0.01", "0.05", "0.1", "0.2"}
        for metric in method["top_k"].values():
            assert 0.0 <= metric["capture"] <= 1.0
            assert {
                "capture_any",
                "capture_all",
                "lift",
                "lift_vs_mcda",
            } <= metric.keys()
            assert metric["uncertainty"]["replicates"] == 1000
    for ratio in ("5", "10", "20"):
        stability = evaluation["track_b_stability"]["by_background_ratio"][ratio]
        assert stability["model_count"] == 20
        assert stability["seeds"] == list(range(42, 62))
        assert stability["top_k"]["0.1"]["pair_count"] == 190
        assert 0.0 <= stability["top_k"]["0.1"]["mean_pairwise_jaccard"] <= 1.0
        assert stability["top_k"]["0.1"]["capture"]["defined_draws"] == 20
    assert (
        evaluation["input_hashes"]["track_b_scores"]
        == hashlib.sha256(score_path.read_bytes()).hexdigest()
    )
    assert input_snapshot(root) == before
