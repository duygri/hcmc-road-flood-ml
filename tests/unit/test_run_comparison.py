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
    processed = local / f"data/processed/{RUN_ID}"
    pd.DataFrame(
        {"road_segment_id": ["r1"], "label_status": ["unknown"], "label_binary": [None]}
    ).to_parquet(processed / "road_labels.parquet")
    pd.DataFrame(columns=["observation_id", "road_segment_id"]).to_parquet(
        processed / "confirmed_event_segments.parquet"
    )
    pd.DataFrame(
        {"road_segment_id": ["r1"], "block_id": ["b1"], "partition": ["reference"]}
    ).to_parquet(processed / "road_partitions.parquet")
    pd.DataFrame(columns=["observation_id", "road_segment_id", "partition"]).to_parquet(
        processed / "reviewed_group_partition_audit.parquet"
    )
    pd.DataFrame({"road_segment_id": ["r1"]}).to_parquet(
        processed / "analysis_dataset.parquet"
    )
    pd.DataFrame({"road_segment_id": ["r1"], "score": [0.5]}).to_parquet(
        processed / "road_susceptibility.parquet"
    )
    pd.DataFrame(
        {"road_segment_id": ["r1"], "weight_set": ["equal"], "score": [0.5]}
    ).to_parquet(processed / "sensitivity_scores.parquet")
    (local / f"artifacts/{RUN_ID}/normalizer.json").write_text('{"q05": 1.0}')
    (local / f"artifacts/{RUN_ID}/mcda_rules.json").write_text("{}")
    (local / f"reports/runs/{RUN_ID}/weight_sensitivity.json").write_text("{}")
    refresh_evaluation_inputs(local)
    refresh_stages(local)
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


def refresh_stages(root):
    import hashlib

    completion_path = root / f"reports/runs/{RUN_ID}/pipeline_complete.json"
    completion = json.loads(completion_path.read_text())
    markers = {}
    for stage in completion["stage_markers"]:
        marker_path = root / f"reports/runs/{RUN_ID}/stages/{stage}.json"
        marker = json.loads(marker_path.read_text())
        marker["outputs"] = {
            output: hashlib.sha256(Path(output).read_bytes()).hexdigest()
            for output in marker["outputs"]
        }
        marker_path.write_text(json.dumps(marker))
        markers[stage] = hashlib.sha256(marker_path.read_bytes()).hexdigest()
    completion["stage_markers"] = markers
    completion_path.write_text(json.dumps(completion))


def refresh_evaluation_inputs(root):
    import hashlib

    path = root / f"reports/runs/{RUN_ID}/evaluation.json"
    report = json.loads(path.read_text())
    report["input_hashes"] = {
        name: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for name, relative in report["input_paths"].items()
    }
    path.write_text(json.dumps(report))


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
    assert report["equivalent"], report["differences"]
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


def test_compare_runs_accepts_legacy_empty_no_review_partition_audit(runs):
    published, local = runs
    add_table(
        published, local, "reviewed_group_partition_audit.parquet", pd.DataFrame()
    )
    report = compare_runs(published, local, RUN_ID)
    assert report["equivalent"], report["differences"]


def test_compare_runs_uses_normalizer_threshold_tolerance_and_retains_hashes(runs):
    from flood_ml.baseline.mcda import FEATURE_SPEC, fit_normalizer

    published, local = runs
    base = pd.DataFrame(
        {
            "road_segment_id": [f"r{i:02d}" for i in range(30)],
            "elevation_median_m": [float(i) for i in range(30)],
            "relative_elevation_m": [float(i + 1) for i in range(30)],
            "slope_mean_deg": [float(i + 2) for i in range(30)],
            "dist_water_m": [float(i + 3) for i in range(30)],
            "builtup_ratio_250m": [i / 30 for i in range(30)],
        }
    )
    jittered = base.copy()
    jittered["elevation_median_m"] += 1e-6
    normalizers = [
        fit_normalizer(frame, dict(FEATURE_SPEC)) for frame in (base, jittered)
    ]
    assert (
        normalizers[0]["reference_input_sha256"]
        != normalizers[1]["reference_input_sha256"]
    )
    for root, value in zip((published, local), normalizers, strict=True):
        path = root / f"artifacts/{RUN_ID}/normalizer.json"
        path.write_text(json.dumps(value), encoding="utf-8")
    refresh(published)
    report = compare_runs(published, local, RUN_ID)
    assert report["equivalent"], report["differences"]
    trace = report["derived_artifact_hashes"][f"artifacts/{RUN_ID}/normalizer.json"]
    assert (
        trace["published"]["reference_input_sha256"]
        == normalizers[0]["reference_input_sha256"]
    )
    assert (
        trace["local"]["reference_input_sha256"]
        == normalizers[1]["reference_input_sha256"]
    )


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


@pytest.mark.parametrize("declaration", ["REAL NOT NULL", "DOUBLE"])
def test_compare_runs_rejects_geopackage_declared_schema_changes_with_identical_values(
    runs, declaration
):
    import sqlite3

    import geopandas as gpd
    from shapely.geometry import LineString

    published, local = runs
    frame = gpd.GeoDataFrame(
        {"road_segment_id": ["r1"], "feature": [1.0]},
        geometry=[LineString([(0, 0), (1, 1)])],
        crs="EPSG:32648",
    )
    add_table(published, local, "roads.gpkg", frame)
    assert compare_runs(published, local, RUN_ID)["equivalent"]
    local_path = local / f"data/processed/{RUN_ID}/roads.gpkg"
    with sqlite3.connect(local_path) as connection:
        # Change only the declared constraint; preserve layer registration,
        # row values, geometry and spatial indexes without requiring GIS SQL extensions.
        sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'roads'"
        ).fetchone()[0]
        assert '"feature" REAL' in sql
        connection.execute("PRAGMA writable_schema=ON")
        connection.execute(
            "UPDATE sqlite_schema SET sql = ? WHERE type = 'table' AND name = 'roads'",
            (sql.replace('"feature" REAL', f'"feature" {declaration}'),),
        )
        version = connection.execute("PRAGMA schema_version").fetchone()[0]
        connection.execute(f"PRAGMA schema_version={version + 1}")
        connection.execute("PRAGMA writable_schema=OFF")
    assert gpd.read_file(local_path, layer="roads").equals(frame)
    report = compare_runs(published, local, RUN_ID)
    assert not report["equivalent"]
    difference = next(
        d for d in report["differences"] if d["check"] == "geopackage_schema"
    )
    assert difference["file"].endswith("roads.gpkg:roads")
    assert difference["published"] != difference["local"]


def test_compare_runs_normalizes_real_evaluation_cli_paths_without_hiding_settings(
    tmp_path,
):
    import hashlib
    import shutil
    from dataclasses import asdict

    import geopandas as gpd
    import yaml
    from shapely.geometry import LineString

    from flood_ml.config import load_settings
    from flood_ml.labeling.reviews import LINK_COLUMNS
    from flood_ml.utils.run import _snapshot

    colab = fixture_run(tmp_path / "colab_checkout")
    processed = colab / f"data/processed/{RUN_ID}"
    roads = gpd.GeoDataFrame(
        {"road_segment_id": ["r1"]},
        geometry=[LineString([(0, 0), (1, 1)])],
        crs="EPSG:32648",
    )
    roads.to_parquet(processed / "road_features.parquet")
    pd.DataFrame(
        {
            "road_segment_id": ["r1"],
            "score": [0.5],
            "elevation_median_m": [1.0],
            "mcda_eligible": [True],
        }
    ).to_parquet(processed / "road_susceptibility.parquet")
    pd.DataFrame(
        {"road_segment_id": ["r1"], "block_id": ["b1"], "partition": ["reference"]}
    ).to_parquet(processed / "road_partitions.parquet")
    pd.DataFrame(columns=LINK_COLUMNS).to_parquet(
        processed / "confirmed_event_segments.parquet"
    )
    pd.DataFrame(
        {"road_segment_id": ["r1"], "label_status": ["unknown"], "label_binary": [None]}
    ).to_parquet(processed / "road_labels.parquet")
    pd.DataFrame(columns=["observation_id", "road_segment_id", "partition"]).to_parquet(
        processed / "reviewed_group_partition_audit.parquet"
    )
    pd.DataFrame({"road_segment_id": ["r1"]}).to_parquet(
        processed / "analysis_dataset.parquet"
    )
    pd.DataFrame(
        {"road_segment_id": ["r1"], "weight_set": ["equal"], "score": [0.5]}
    ).to_parquet(processed / "sensitivity_scores.parquet")
    (colab / f"artifacts/{RUN_ID}/normalizer.json").write_text('{"q05": 1.0}')
    (colab / f"artifacts/{RUN_ID}/mcda_rules.json").write_text("{}")
    (colab / f"reports/runs/{RUN_ID}/weight_sensitivity.json").write_text("{}")
    (colab / "uv.lock").write_text("same lock")
    local_configs = colab / "configs/local"
    local_configs.mkdir()
    (local_configs / "review_decisions.csv").write_text("same reviewed snapshot")
    local = tmp_path / "local_checkout"
    shutil.copytree(colab, local)
    script = Path(__file__).parents[2] / "scripts/evaluate.py"
    snapshots = []
    for root in (colab, local):
        (root / "configs/local/sources.yaml").write_text(
            yaml.safe_dump(
                {"sources": {"raw_root": str(tmp_path / f"raw_{root.name}")}}
            )
        )
        settings = load_settings(Path("configs/experiments.yaml"), repo_root=root)
        snapshot = _snapshot(asdict(settings))
        snapshots.append(snapshot)
        update(
            root,
            "run",
            manifest_sha256="manifest",
            code_sha256="source",
            config_snapshot={
                key: value
                for key, value in snapshot.items()
                if key not in {"repo_root", "configuration_paths"}
            },
            review_hashes={
                "match_reviews": hashlib.sha256(b"same reviewed snapshot").hexdigest()
            },
        )
        args = [
            sys.executable,
            str(script),
            "--repo-root",
            str(root),
            "--run-id",
            RUN_ID,
        ]
        for option, filename in (
            ("scores", "road_susceptibility.parquet"),
            ("roads", "road_features.parquet"),
            ("confirmed-event-segments", "confirmed_event_segments.parquet"),
            ("partitions", "road_partitions.parquet"),
        ):
            args.extend([f"--{option}", f"data/processed/{RUN_ID}/{filename}"])
        result = subprocess.run(args, check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    assert snapshots[0]["repo_root"] != snapshots[1]["repo_root"]
    assert snapshots[0]["sources"]["raw_root"] != snapshots[1]["sources"]["raw_root"]
    assert snapshots[0]["configuration_paths"] != snapshots[1]["configuration_paths"]
    refresh_evaluation_inputs(colab)
    refresh_stages(colab)
    published = publish_run(colab, tmp_path / "drive", RUN_ID)
    report = compare_runs(published, local, RUN_ID)
    assert report["equivalent"], report["differences"]
    evaluation_path = local / f"reports/runs/{RUN_ID}/evaluation.json"
    original = json.loads(evaluation_path.read_text())
    changed = json.loads(json.dumps(original))
    changed["config_snapshot"]["experiments"]["evaluation"][
        "primary_capture_fraction"
    ] += 1e-6
    evaluation_path.write_text(json.dumps(changed))
    assert not compare_runs(published, local, RUN_ID)["equivalent"]
    changed = json.loads(json.dumps(original))
    changed["config_snapshot"]["labeling"]["match_reviews"] = (
        "configs/local/other_review.csv"
    )
    evaluation_path.write_text(json.dumps(changed))
    assert not compare_runs(published, local, RUN_ID)["equivalent"]
    changed = json.loads(json.dumps(original))
    changed["config_snapshot"]["sources"]["raw_root"] = str(tmp_path / "unrecorded_raw")
    evaluation_path.write_text(json.dumps(changed))
    assert not compare_runs(published, local, RUN_ID)["equivalent"]
    changed = json.loads(json.dumps(original))
    changed["config_snapshot"]["configuration_paths"][0] = str(
        local / "configs/other.yaml"
    )
    evaluation_path.write_text(json.dumps(changed))
    assert not compare_runs(published, local, RUN_ID)["equivalent"]
    evaluation_path.write_text(json.dumps(original))
    update(local, "run", review_hashes={"match_reviews": "different_review_content"})
    assert not compare_runs(published, local, RUN_ID)["equivalent"]


def test_compare_runs_keeps_derived_report_hashes_as_traceability_only(runs):
    published, local = runs
    for side, root in (("colab", published), ("local", local)):
        update(
            root,
            "evaluation",
            input_hashes={"scores": f"{side}_parquet", "roads": f"{side}_gpkg"},
        )
        update(
            root,
            "track_b",
            input_hashes={
                "features": f"{side}_features",
                "partitions": f"{side}_partitions",
            },
            partition_sha256=f"{side}_partitions",
            output_hashes={"model.joblib": f"{side}_model"},
        )
    refresh(published)
    report = compare_runs(published, local, RUN_ID)
    assert report["equivalent"], report["differences"]
    evaluation = report["derived_artifact_hashes"][
        f"reports/runs/{RUN_ID}/evaluation.json"
    ]
    assert evaluation["published"]["input_hashes"]["roads"] == "colab_gpkg"
    assert evaluation["local"]["input_hashes"]["roads"] == "local_gpkg"
    track_b = report["derived_artifact_hashes"][f"reports/runs/{RUN_ID}/track_b.json"]
    assert track_b["published"]["partition_sha256"] == "colab_partitions"
    assert track_b["local"]["partition_sha256"] == "local_partitions"
    assert track_b["local"]["output_hashes"]["model.joblib"] == "local_model"
    update(local, "evaluation", input_hashes={"scores": "local_parquet"})
    assert not compare_runs(published, local, RUN_ID)["equivalent"]
    update(
        local,
        "evaluation",
        input_hashes={"scores": "local_parquet", "roads": "local_gpkg"},
    )
    alter(local, lambda f: f.assign(feature=9.0))
    assert not compare_runs(published, local, RUN_ID)["equivalent"]
    alter(local, lambda f: f.assign(feature=[2.0, 1.0]))
    update(local, "run", input_hashes={"input": "different_raw_snapshot"})
    assert not compare_runs(published, local, RUN_ID)["equivalent"]
