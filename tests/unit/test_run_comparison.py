"""Semantic comparison tests using real Parquet and publication bundles."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from test_run_publisher import RUN_ID, fixture_run

from flood_ml.utils.run_comparison import compare_runs
from flood_ml.utils.run_publisher import publish_run


def update(root, name, **fields):
    path = root / f"reports/runs/{RUN_ID}/{name}.json"
    value = json.loads(path.read_text())
    value.update(fields)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def runs(tmp_path):
    local = fixture_run(tmp_path / "local")
    update(
        local,
        "run",
        manifest_sha256="manifest",
        code_sha256="source",
        config_snapshot={
            "sources": {"raw_root": str(local / "raw")},
            "labeling": {"match_reviews": "reviews/matches.csv"},
            "experiments": {"outputs": {"processed": "data/processed"}, "seed": 42},
        },
    )
    frame = pd.DataFrame(
        {
            "road_segment_id": ["r2", "r1"],
            "feature": [2.0, 1.0],
            "label": [1, 0],
            "partition": ["holdout", "reference"],
        }
    )
    frame.to_parquet(local / f"data/processed/{RUN_ID}/road_features.parquet")
    # Publisher fixture contains an opaque normalizer placeholder; use a real report.
    (local / f"artifacts/{RUN_ID}/normalizer.json").write_text('{"q05": 1.0}')
    published = publish_run(local, tmp_path / "drive", RUN_ID)
    update(
        published,
        "run",
        config_snapshot={
            "sources": {"raw_root": "/content/raw"},
            "labeling": {"match_reviews": "reviews/matches.csv"},
            "experiments": {"outputs": {"processed": "data/processed"}, "seed": 42},
        },
    )
    # Refresh manifest after adapting published runtime provenance to a Colab root.
    refresh(published)
    return published, local


def refresh(root):
    import hashlib

    path = root / "completion_manifest.json"
    manifest = json.loads(path.read_text())
    for relative in manifest["files"]:
        data = (root / relative).read_bytes()
        manifest["files"][relative] = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    path.write_text(json.dumps(manifest))


def alter(local, fn, name="road_features.parquet"):
    path = local / f"data/processed/{RUN_ID}/{name}"
    frame = fn(pd.read_parquet(path))
    frame.to_parquet(path)


def test_compare_runs_accepts_numeric_jitter_within_atol(runs):
    published, local = runs
    alter(local, lambda f: f.assign(feature=f.feature + 0.000009))
    update(local, "evaluation", metrics={"capture": 0.500009}, elapsed_seconds=9)
    update(published, "evaluation", metrics={"capture": 0.5}, elapsed_seconds=1)
    refresh(published)
    assert compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_rejects_numeric_difference_above_atol(runs):
    published, local = runs
    alter(local, lambda f: f.assign(feature=f.feature + 0.00002))
    report = compare_runs(published, local, RUN_ID)
    assert not report["equivalent"]
    assert any(
        d.get("column") == "feature" and d.get("rows") for d in report["differences"]
    )


@pytest.mark.parametrize(
    "mutation", ["keys", "schema", "labels", "order", "nulls", "partition"]
)
def test_compare_runs_requires_exact_keys_schema_labels_order_and_null_masks(
    runs, mutation
):
    published, local = runs

    def change(f):
        if mutation == "keys":
            f.loc[0, "road_segment_id"] = "other"
        if mutation == "schema":
            f["extra"] = 1
        if mutation == "labels":
            f.loc[0, "label"] = 0
        if mutation == "order":
            f = f.iloc[::-1]
        if mutation == "nulls":
            f.loc[0, "feature"] = float("nan")
        if mutation == "partition":
            f.loc[0, "partition"] = "reference"
        return f

    alter(local, change)
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_sorts_and_resets_road_tables_before_numeric_comparison(runs):
    published, local = runs
    alter(local, lambda f: f.set_axis([75, 99]))
    assert compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_uses_composite_track_b_score_key(runs):
    published, local = runs
    scores = pd.DataFrame(
        {
            "road_segment_id": ["r2", "r2", "r1", "r1"],
            "model_id": ["m2", "m1", "m2", "m1"],
            "score": [0.2, 0.3, 0.4, 0.5],
        }
    )
    for root in (published, local):
        scores.to_parquet(root / f"data/processed/{RUN_ID}/track_b_scores.parquet")
    manifest = json.loads((published / "completion_manifest.json").read_text())
    manifest["files"][f"data/processed/{RUN_ID}/track_b_scores.parquet"] = {}
    (published / "completion_manifest.json").write_text(json.dumps(manifest))
    refresh(published)
    assert compare_runs(published, local, RUN_ID)["equivalent"]
    alter(local, lambda f: f.iloc[[1, 0, 2, 3]], "track_b_scores.parquet")
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_rejects_duplicate_row_keys(runs):
    published, local = runs
    alter(local, lambda f: f.assign(road_segment_id="duplicate"))
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("git", {"commit": "other"}),
        ("lock_sha256", "other"),
        ("manifest_sha256", "other"),
        ("input_hashes", {"input": "other"}),
        ("review_hashes", {"review": "other"}),
        ("code_sha256", "other"),
    ],
)
def test_compare_runs_rejects_source_lock_or_input_hash_mismatch(runs, field, value):
    published, local = runs
    update(local, "run", **{field: value})
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_accepts_matching_pending_track_b_without_scores(runs):
    published, local = runs
    report = compare_runs(published, local, RUN_ID)
    assert report["equivalent"]
    assert report["gate_statuses"]["local"]["track_b"] == "pending_positive_review"
    update(local, "track_b", status="ready")
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_reports_ready_model_and_evaluation_statuses(runs):
    published, local = runs
    for root in (published, local):
        update(
            root,
            "track_b",
            status="ready",
            draws=[
                {
                    "status": "fitted",
                    "model_artifact": f"artifacts/{RUN_ID}/model.joblib",
                }
            ],
        )
        update(
            root,
            "evaluation",
            status="ready",
            evaluation_performed=True,
            metrics={"capture": 0.5},
        )
        (root / f"artifacts/{RUN_ID}/model.joblib").write_bytes(b"opaque model")
        pd.DataFrame(
            {"road_segment_id": ["r1"], "model_id": ["m1"], "score": [0.5]}
        ).to_parquet(root / f"data/processed/{RUN_ID}/track_b_scores.parquet")
    manifest = json.loads((published / "completion_manifest.json").read_text())
    for path in published.rglob("*"):
        if path.is_file() and path.name != "completion_manifest.json":
            manifest["files"].setdefault(path.relative_to(published).as_posix(), {})
    manifest["gate_statuses"].update(track_b="ready", evaluation="ready")
    (published / "completion_manifest.json").write_text(json.dumps(manifest))
    refresh(published)
    report = compare_runs(published, local, RUN_ID)
    assert report["equivalent"]
    assert report["gate_statuses"]["local"]["evaluation"] == "ready"
    assert report["gate_statuses"]["local"]["track_b"] == "ready"
    (local / f"artifacts/{RUN_ID}/model.joblib").unlink()
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_cli_writes_report_and_returns_difference_exit_code(runs):
    published, local = runs
    script = Path(__file__).parents[2] / "scripts/verify_colab_run.py"
    args = [
        sys.executable,
        str(script),
        "--published-run",
        str(published),
        "--repo-root",
        str(local),
        "--run-id",
        RUN_ID,
    ]
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = local / f"reports/runs/{RUN_ID}/local_verification.json"
    assert json.loads(report.read_text())["equivalent"]
    alter(local, lambda f: f.assign(feature=9.0))
    assert subprocess.run(args, check=False, capture_output=True).returncode == 1


def test_compare_runs_refuses_overlapping_checkout(runs):
    published, _ = runs
    with pytest.raises(ValueError, match="separate|overlap"):
        compare_runs(published, published, RUN_ID)


def add_table(published, local, name, frame):
    for root in (published, local):
        path = root / f"data/processed/{RUN_ID}/{name}"
        if name.endswith(".gpkg"):
            frame.to_file(path, layer="roads", driver="GPKG")
        else:
            frame.to_parquet(path)
    path = published / "completion_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][f"data/processed/{RUN_ID}/{name}"] = {}
    path.write_text(json.dumps(manifest))
    refresh(published)


def test_compare_runs_accepts_nullable_candidate_discriminator_but_rejects_duplicates(
    runs,
):
    published, local = runs
    frame = pd.DataFrame(
        {
            "observation_id": ["o1", "o1", "o2"],
            "road_segment_id": ["r1", "r2", None],
            "distance": [1.0, 2.0, None],
        }
    )
    add_table(published, local, "road_flood_matches.parquet", frame)
    assert compare_runs(published, local, RUN_ID)["equivalent"]
    alter(local, lambda f: pd.concat([f, f.iloc[[2]]]), "road_flood_matches.parquet")
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_reads_geopackage_layers_and_rejects_geometry_changes(runs):
    import geopandas as gpd
    from shapely.geometry import LineString

    published, local = runs
    frame = gpd.GeoDataFrame(
        {"road_segment_id": ["r1", "r2"], "feature": [1.0, 2.0]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        crs="EPSG:32648",
    )
    add_table(published, local, "roads.gpkg", frame)
    assert compare_runs(published, local, RUN_ID)["equivalent"]
    frame.loc[0, "geometry"] = LineString([(0, 0), (2, 2)])
    frame.to_file(
        local / f"data/processed/{RUN_ID}/roads.gpkg", layer="roads", driver="GPKG"
    )
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


@pytest.mark.parametrize(
    "field", ["config_snapshot", "lock_sha256", "input_hashes", "git"]
)
def test_compare_runs_fails_closed_for_missing_provenance(runs, field):
    published, local = runs
    update(local, "run", **{field: None})
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_rejects_null_road_keys(runs):
    published, local = runs
    alter(local, lambda f: f.assign(road_segment_id=["r2", None]))
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_checks_published_integrity(runs):
    published, local = runs
    update(published, "evaluation", unexpected="corrupt")
    assert any(
        d["check"] == "published_integrity"
        for d in compare_runs(published, local, RUN_ID)["differences"]
    )


def test_compare_runs_ignores_volatile_fingerprints_and_stage_binary_hashes(runs):
    published, local = runs
    update(
        published,
        "pipeline_complete",
        run_fingerprint="colab",
        stage_markers={"features": "colab_hash"},
    )
    update(
        local,
        "pipeline_complete",
        run_fingerprint="local",
        stage_markers={"features": "local_hash"},
    )
    update(published, "pipeline_report", process_peak_working_set_bytes=1000)
    update(local, "pipeline_report", process_peak_working_set_bytes=2000)
    refresh(published)
    assert compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_compares_effective_configuration_exactly(runs):
    published, local = runs
    run = json.loads((local / f"reports/runs/{RUN_ID}/run.json").read_text())
    run["config_snapshot"]["experiments"]["seed"] = 43
    update(local, "run", config_snapshot=run["config_snapshot"])
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_rejects_missing_source_manifest_hash_on_both_runs(runs):
    published, local = runs
    for root in (published, local):
        update(root, "run", manifest_sha256=None)
    refresh(published)
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_uses_reviewed_link_key_for_partition_audit(runs):
    published, local = runs
    frame = pd.DataFrame(
        {
            "observation_id": ["o1", "o2"],
            "road_segment_id": ["r1", "r1"],
            "partition": ["reference", "reference"],
        }
    )
    add_table(published, local, "reviewed_group_partition_audit.parquet", frame)
    assert compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_requires_required_reports_in_manifest(runs):
    published, local = runs
    path = published / "completion_manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files"][f"reports/runs/{RUN_ID}/run.json"]
    path.write_text(json.dumps(manifest))
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_requires_review_hashes_for_configured_review_inputs(runs):
    published, local = runs
    for root in (published, local):
        update(root, "run", review_hashes={})
    manifest_path = published / "completion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["review_hashes"] = {}
    manifest_path.write_text(json.dumps(manifest))
    refresh(published)
    assert not compare_runs(published, local, RUN_ID)["equivalent"]
