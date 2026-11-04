"""Train only synthetic local Track B populations and inspect published artifacts."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline

from flood_ml.config import PipelineError, load_settings

ROOT = Path(__file__).parents[2]
FEATURES = [
    "elevation_median_m",
    "relative_elevation_m",
    "slope_mean_deg",
    "dist_water_m",
    "builtup_ratio_250m",
]
RULES = {
    "feature_columns": FEATURES,
    "primary_background_ratio": 10,
    "sensitivity_background_ratios": [5, 20],
    "seed_start": 42,
    "seed_repetitions": 20,
    "minimum_reference_events": 20,
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
}


def fixture_repo(root, *, reviewed=True, unknowns=400):
    (root / "configs").mkdir()
    for name in ["sources", "features", "experiments", "labeling", "study_area"]:
        doc = yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text())
        if name == "sources":
            doc["sources"]["assets"] = []
        if name == "experiments":
            doc["experiments"]["track_b"] = RULES
        (root / f"configs/{name}.yaml").write_text(yaml.safe_dump(doc))
    (root / "uv.lock").write_bytes(b"synthetic lock\n")
    positives = [f"p{i}" for i in range(20)]
    unknown = [f"u{i}" for i in range(unknowns)]
    holdout = ["h0", "h1", "h_ineligible", "h_missing"]
    ids = positives + unknown + holdout + ["guard", "excluded", "negative"]
    values = np.linspace(0, 1, len(ids))
    features = pd.DataFrame(
        {"road_segment_id": ids, **{name: values for name in FEATURES}}
    )
    features["score_eligible"] = ~features.road_segment_id.eq("h_ineligible")
    features.loc[features.road_segment_id.eq("h_missing"), FEATURES[0]] = np.nan
    labels = pd.DataFrame(
        {
            "road_segment_id": ids,
            "label_status": ["positive"] * 20
            + ["unknown"] * (unknowns + len(holdout) + 2)
            + ["verified_negative"],
        }
    )
    partitions = pd.DataFrame(
        {
            "road_segment_id": ids,
            "block_id": [f"b{i % 4}" for i in range(len(ids))],
            "partition": ["reference"] * (20 + unknowns)
            + ["holdout"] * len(holdout)
            + ["guard", "excluded", "reference"],
        }
    )
    events = pd.DataFrame(
        {
            "road_segment_id": positives,
            "event_group_id": [f"e{i}" for i in range(20)],
            "location_group_id": [f"l{i}" for i in range(20)],
            "evidence": "road review",
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-10-01T00:00:00Z",
            "event_evidence": "event review",
            "event_reviewed_by": "reviewer",
            "event_reviewed_at": "2026-10-01T00:00:00Z",
        }
    )
    if not reviewed:
        events = events.iloc[:0]
        labels["label_status"] = "unknown"
    for name, frame in {
        "features": features,
        "labels": labels,
        "confirmed-event-segments": events,
        "partitions": partitions,
    }.items():
        frame.to_parquet(root / f"{name}.parquet", index=False)
    return root


def arguments(root, run_id="train_fixture", **overrides):
    args = ["--repo-root", str(root), "--run-id", run_id]
    for name in ("features", "labels", "confirmed-event-segments", "partitions"):
        args.extend([f"--{name}", overrides.get(name, f"{name}.parquet")])
    return args


def cli(root, run_id="train_fixture", **overrides):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            *arguments(root, run_id, **overrides),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def report(root, run_id="train_fixture"):
    return json.loads((root / f"reports/runs/{run_id}/track_b.json").read_text())


def assert_no_fitted_outputs(root, run_id="train_fixture"):
    assert not (root / f"data/processed/{run_id}/track_b_scores.parquet").exists()
    assert not list((root / f"artifacts/{run_id}").rglob("*.joblib"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("primary_background_ratio", 0),
        ("primary_background_ratio", 1.5),
        ("primary_background_ratio", True),
        ("sensitivity_background_ratios", [5, 0]),
        ("seed_start", -1),
        ("seed_start", 2**32),
        ("seed_start", "42"),
        ("seed_repetitions", 0),
        ("seed_repetitions", True),
    ],
)
def test_track_b_config_rejects_invalid_ratio_and_seed_values(tmp_path, key, value):
    fixture_repo(tmp_path)
    path = tmp_path / "configs/experiments.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["experiments"]["track_b"][key] = value
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(PipelineError, match=f"track_b.{key}"):
        load_settings(Path("configs/experiments.yaml"), repo_root=tmp_path)


def test_track_b_config_accepts_predeclared_background_count_ratios(tmp_path):
    fixture_repo(tmp_path)
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=tmp_path)
    assert settings.experiments["track_b"]["primary_background_ratio"] == 10


def test_train_cli_without_review_writes_pending_only(tmp_path):
    fixture_repo(tmp_path, reviewed=False)
    features = pd.read_parquet(tmp_path / "features.parquet")
    features.loc[features.road_segment_id.eq("u0"), FEATURES[0]] = np.nan
    features.to_parquet(tmp_path / "features.parquet", index=False)
    result = cli(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert report(tmp_path)["status"] == "pending_positive_review"
    assert report(tmp_path)["gate"]["feature_eligible_reference_events"] == 0
    exclusion = next(
        row
        for row in report(tmp_path)["gate"]["background_exclusions"]
        if row["road_segment_id"] == "u0"
    )
    assert exclusion["reason"] == "incomplete_or_nonfinite_features"
    assert_no_fitted_outputs(tmp_path)


def test_train_cli_rejects_malformed_feature_schema(tmp_path):
    fixture_repo(tmp_path)
    features = pd.read_parquet(tmp_path / "features.parquet")
    features.drop(columns=FEATURES[0]).to_parquet(
        tmp_path / "features.parquet", index=False
    )
    result = cli(tmp_path)
    assert result.returncode != 0
    assert report(tmp_path)["status"] == "failed"
    assert_no_fitted_outputs(tmp_path)


def test_train_cli_primary_background_shortage_publishes_no_partial_outputs(tmp_path):
    fixture_repo(tmp_path, unknowns=100)
    result = cli(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = report(tmp_path)
    assert payload["status"] == "background_pool_insufficient"
    assert payload["failures"]
    assert len(payload["draws"]) == 60
    assert_no_fitted_outputs(tmp_path)


def test_train_cli_ready_writes_scores_models_and_provenance(tmp_path):
    fixture_repo(tmp_path)
    labels_before = (tmp_path / "labels.parquet").read_bytes()
    result = cli(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    scores = pd.read_parquet(
        tmp_path / "data/processed/train_fixture/track_b_scores.parquet"
    )
    assert scores.columns.tolist() == [
        "road_segment_id",
        "model_id",
        "background_ratio",
        "seed",
        "relative_score",
        "partition",
    ]
    assert len(scores) == 120
    assert set(scores.road_segment_id) == {"h0", "h1"}
    assert set(scores.partition) == {"holdout"}
    assert set(scores.background_ratio) == {5, 10, 20}
    assert set(scores.seed) == set(range(42, 62))
    assert not scores.duplicated(["road_segment_id", "model_id"]).any()
    assert np.isfinite(scores.relative_score).all()
    models = list(
        (tmp_path / "artifacts/train_fixture/track_b_models").glob("*.joblib")
    )
    assert len(models) == 60
    assert isinstance(joblib.load(models[0]), Pipeline)
    payload = report(tmp_path)
    assert payload["status"] == "ready"
    assert payload["run_id"] == "train_fixture"
    assert payload["score_semantics"] == "relative_rank_score_not_probability"
    assert payload["source_inputs_unchanged"] is True
    assert len(payload["draws"]) == 60
    assert all(draw["status"] == "fitted" for draw in payload["draws"])
    assert payload["draws"][0]["reference_block_quotas"]
    for name in ("features", "labels", "confirmed_event_segments", "partitions"):
        filename = (
            name.replace("_", "-") if name == "confirmed_event_segments" else name
        )
        assert (
            payload["input_hashes"][name]
            == hashlib.sha256(
                (tmp_path / f"{filename}.parquet").read_bytes()
            ).hexdigest()
        )
    assert payload["partition_sha256"] == payload["input_hashes"]["partitions"]
    assert len(payload["config_hash"]) == 64
    assert len(payload["feature_columns_sha256"]) == 64
    assert payload["lock_sha256"] == hashlib.sha256(b"synthetic lock\n").hexdigest()
    assert "commit" in payload["git"]
    assert {"scikit-learn", "numpy", "pandas", "joblib"} <= payload[
        "model_versions"
    ].keys()
    assert (tmp_path / "labels.parquet").read_bytes() == labels_before


@pytest.mark.parametrize("change", ["contents", "path"])
def test_train_cli_rejects_changed_derived_inputs_without_overwriting_run(
    tmp_path, change
):
    fixture_repo(tmp_path)
    result = cli(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    published = {
        path: path.read_bytes()
        for directory in (
            "reports/runs/train_fixture",
            "artifacts/train_fixture",
            "data/processed/train_fixture",
        )
        for path in (tmp_path / directory).rglob("*")
        if path.is_file()
    }
    overrides = {}
    features = tmp_path / "features.parquet"
    if change == "contents":
        frame = pd.read_parquet(features)
        frame[FEATURES[0]] += 0.1
        frame.to_parquet(features, index=False)
    else:
        (tmp_path / "other-features.parquet").write_bytes(features.read_bytes())
        overrides["features"] = "other-features.parquet"
    result = cli(tmp_path, **overrides)
    assert result.returncode != 0
    assert "Track B inputs changed; choose a new run_id" in result.stderr
    assert {path: path.read_bytes() for path in published} == published


@pytest.mark.parametrize(
    "destination,hardlink",
    [
        ("data/processed/collision/track_b_scores.parquet", False),
        ("artifacts/collision/track_b_models/track_b_r10_s42.joblib", False),
        ("reports/runs/collision/track_b.json", False),
        ("reports/runs/collision/run.json", False),
        ("reports/runs/collision/.start.lock", False),
        ("reports/runs/collision/track_b.json", True),
    ],
)
def test_train_cli_rejects_input_output_alias(tmp_path, destination, hardlink):
    fixture_repo(tmp_path)
    target = tmp_path / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((tmp_path / "features.parquet").read_bytes())
    source = target
    if hardlink:
        source = tmp_path / "alias.parquet"
        os.link(target, source)
    before = {
        str(p.relative_to(tmp_path)): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    result = cli(tmp_path, "collision", features=str(source.relative_to(tmp_path)))
    after = {
        str(p.relative_to(tmp_path)): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert result.returncode != 0
    assert "aliases publication destination" in result.stderr
    assert after == before


def test_train_cli_failed_model_serialization_leaves_no_final_artifacts(
    tmp_path, monkeypatch
):
    fixture_repo(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "track_b_train", ROOT / "scripts/train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fail_serialization(*_args, **_kwargs):
        raise OSError("synthetic serialization failure")

    monkeypatch.setattr(module.joblib, "dump", fail_serialization)
    assert module.main(arguments(tmp_path)) == 1
    assert report(tmp_path)["status"] == "failed"
    assert_no_fitted_outputs(tmp_path)


def test_train_publication_backup_failure_preserves_previous_output(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "track_b_train", ROOT / "scripts/train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    temporary = tmp_path / "staged.tmp"
    destination = tmp_path / "model.joblib"
    temporary.write_bytes(b"new model")
    destination.write_bytes(b"previous model")
    original_replace = os.replace

    def fail_backup(source, target):
        if Path(source) == destination:
            raise OSError("synthetic backup failure")
        return original_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_backup)
    with pytest.raises(OSError, match="synthetic backup failure"):
        module._publish([(temporary, destination)], tmp_path / "report.json", {})
    assert destination.read_bytes() == b"previous model"


def test_train_cli_backup_cleanup_failure_keeps_ready_report_and_outputs(
    tmp_path, monkeypatch
):
    fixture_repo(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "track_b_train", ROOT / "scripts/train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(arguments(tmp_path)) == 0
    original_unlink = Path.unlink

    def fail_backup_cleanup(path, *args, **kwargs):
        if path.exists() and path.name.startswith(".track_b_r10_s42.joblib."):
            raise OSError("synthetic backup cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_backup_cleanup)
        result = module.main(arguments(tmp_path))

    assert result == 0
    payload = report(tmp_path)
    assert payload["status"] == "ready"
    scores = pd.read_parquet(
        tmp_path / "data/processed/train_fixture/track_b_scores.parquet"
    )
    assert len(scores) == 120
    models = list(
        (tmp_path / "artifacts/train_fixture/track_b_models").glob("*.joblib")
    )
    assert len(models) == 60
    assert isinstance(joblib.load(models[0]), Pipeline)
    for path, expected in payload["output_hashes"].items():
        assert hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == expected
