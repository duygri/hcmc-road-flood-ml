"""Whole offline GIS workflow, publication integrity and honest review gates."""

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from flood_ml.config import PipelineError, load_settings
from flood_ml.pipeline import _write_frame
from flood_ml.utils.run import start_run


@pytest.fixture
def synthetic_pipeline(tmp_path):
    root = tmp_path / "repo"
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "uv.lock").write_bytes(b"offline synthetic lock")
    real_root = Path(__file__).resolve().parents[2]
    for source in (real_root / "configs").glob("*.yaml"):
        (root / "configs" / source.name).write_bytes(source.read_bytes())
    # Four small disconnected AOI parts, in three frozen holdout cells and one
    # reference cell. Raster extent covers all parts and their complete halo.
    cells = [(x, y) for x in range(120, 130) for y in range(240, 250)]
    holdout = [
        c
        for c in cells
        if int(hashlib.sha256(f"42:{c[0]}:{c[1]}".encode()).hexdigest(), 16) / 2**256
        < 0.2
    ]
    reference = next(c for c in cells if c not in holdout)
    cells = [reference, *holdout[:3]]
    centers = [(x * 5000 + 2500, y * 5000 + 2500) for x, y in cells]
    polygons = [box(x - 600, y - 600, x + 600, y + 600) for x, y in centers]
    from shapely import union_all

    gpd.GeoDataFrame(
        {"GhiChu": ["TP. Ho CHi Minh"]}, geometry=[union_all(polygons)], crs=9210
    ).to_file(raw / "aoi.gpkg", driver="GPKG")
    gpd.GeoDataFrame(
        geometry=[box(x - 250, y - 400, x - 100, y + 400) for x, y in centers], crs=9210
    ).to_file(raw / "water.gpkg", driver="GPKG")
    project = Transformer.from_crs(9210, 4326, always_xy=True)
    nodes, ways = [], []
    for i, (x, y) in enumerate(centers):
        for j, dx in enumerate((-60, 60)):
            lon, lat = project.transform(x + dx, y)
            nodes.append(
                f'<node id="{2 * i + j + 1}" lat="{lat:.12f}" lon="{lon:.12f}"/>'
            )
        ways.append(
            f'<way id="{i + 1}"><nd ref="{2 * i + 1}"/><nd ref="{2 * i + 2}"/><tag k="highway" v="residential"/></way>'
        )
    (raw / "roads.osm").write_text(
        '<osm version="0.6">' + "".join(nodes + ways) + "</osm>"
    )
    rows = [
        (i, j, x, y)
        for i, ((x, y), count) in enumerate(zip(centers, (20, 5, 5, 10), strict=True))
        for j in range(count)
    ]
    evidence = gpd.GeoDataFrame(
        {
            "observation_id": [f"o{i}-{j}" for i, j, _, _ in rows],
            "source_id": [f"s{i}-{j}" for i, j, _, _ in rows],
            "observed_date": ["2020-01-01"] * len(rows),
            "spatial_precision": ["high"] * len(rows),
        },
        geometry=[Point(x, y) for _, _, x, y in rows],
        crs=9210,
    )
    evidence.to_file(raw / "evidence.gpkg", driver="GPKG")
    left = min(x for x, _ in centers) - 1500
    top = max(y for _, y in centers) + 1500
    right = max(x for x, _ in centers) + 1500
    bottom = min(y for _, y in centers) - 1500
    for name, res, dtype, value, nodata in (
        ("dem.tif", 30, "float32", 7, -9999),
        ("lc.tif", 10, "uint8", 50, 0),
    ):
        shape = (int(np.ceil((top - bottom) / res)), int(np.ceil((right - left) / res)))
        with rasterio.open(
            raw / name,
            "w",
            driver="GTiff",
            width=shape[1],
            height=shape[0],
            count=1,
            dtype=dtype,
            crs=9210,
            transform=from_origin(left, top, res, res),
            nodata=nodata,
        ) as dst:
            dst.write(np.full(shape, value, dtype=dtype), 1)
    assets = [
        {"dataset_id": d, "asset_id": a, "path": p}
        for d, a, p in (
            ("D01", "administrative_boundaries", "aoi.gpkg"),
            ("D01", "flood_observations", "evidence.gpkg"),
            ("D02", "osm", "roads.osm"),
            ("D03", "dem", "dem.tif"),
            ("D04", "water", "water.gpkg"),
            ("D05", "lc", "lc.tif"),
        )
    ]
    datasets = []
    for dataset_id in dict.fromkeys(a["dataset_id"] for a in assets):
        metadata = {
            k: "synthetic offline fixture"
            for k in (
                "source_url",
                "version",
                "downloaded_on",
                "period",
                "license",
                "attribution",
                "crs",
                "resolution",
                "scope",
                "coverage_assumption",
            )
        }
        metadata.update(
            dataset_id=dataset_id,
            snapshot_date="2026-01-01",
            assets=[
                dict(
                    a,
                    bytes=(raw / a["path"]).stat().st_size,
                    sha256=hashlib.sha256((raw / a["path"]).read_bytes()).hexdigest(),
                )
                for a in assets
                if a["dataset_id"] == dataset_id
            ],
        )
        datasets.append(metadata)
    (raw / "manifest.yaml").write_text(
        yaml.safe_dump({"manifest_version": 2, "datasets": datasets})
    )
    sources = yaml.safe_load((root / "configs/sources.yaml").read_text())
    sources["sources"]["assets"] = assets
    (root / "configs/sources.yaml").write_text(yaml.safe_dump(sources))
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=root)
    return settings, start_run(settings, "synthetic")


def test_full_pipeline_is_honest_without_review(
    synthetic_pipeline, tmp_path, monkeypatch
):
    from flood_ml.pipeline import run_pipeline

    settings, context = synthetic_pipeline
    before = {p: p.read_bytes() for p in settings.sources["raw_root"].iterdir()}
    monkeypatch.chdir(tmp_path)
    summary = run_pipeline(settings, context)
    assert summary["pipeline_status"] == "complete"
    assert summary["evaluation_status"] == "pending_label_review"
    labels = pd.read_parquet(context.processed_dir / "road_labels.parquet")
    assert labels.label_binary.isna().all()
    assert labels.label_status.eq("unknown").all()
    roads = gpd.read_file(context.interim_dir / "road_segments.gpkg")
    features = pd.read_parquet(context.processed_dir / "road_features.parquet")
    assert len(roads) == len(features) == 4
    assert roads.road_segment_id.is_unique
    assert "label_binary" not in features
    scores = pd.read_parquet(context.processed_dir / "road_susceptibility.parquet")
    assert scores.score_reason.notna().all()
    analysis = gpd.read_parquet(context.processed_dir / "analysis_dataset.parquet")
    assert len(analysis) == 4 and analysis.crs.to_epsg() == 9210
    assert (context.interim_dir / "flood_observations.gpkg").exists()
    gate = json.loads((context.reports_dir / "evaluation.json").read_text())["gate"]
    assert (
        gate["missing_reference_events"],
        gate["missing_holdout_events"],
        gate["missing_holdout_blocks"],
    ) == (20, 20, 3)
    assert all(p.read_bytes() == content for p, content in before.items())
    reused = run_pipeline(settings, context)
    assert all(s["reused"] for s in reused["stages"].values())


def test_modified_output_invalidates_downstream(synthetic_pipeline):
    from flood_ml.pipeline import run_pipeline

    settings, context = synthetic_pipeline
    run_pipeline(settings, context)
    path = context.processed_dir / "road_features.parquet"
    data = pd.read_parquet(path)
    data.loc[0, "elevation_median_m"] = 99
    data.to_parquet(path, index=False)
    summary = run_pipeline(settings, context)
    assert summary["stages"]["roads"]["reused"]
    assert not summary["stages"]["features"]["reused"]
    assert not summary["stages"]["mcda"]["reused"]
    assert pd.read_parquet(path).elevation_median_m.eq(7).all()


def test_failure_has_no_stage_completion_marker(synthetic_pipeline):
    from flood_ml.pipeline import run_pipeline

    settings, context = synthetic_pipeline
    changed = replace(settings, features=dict(settings.features, min_valid_fraction=2))
    context = start_run(changed, "failure")
    with pytest.raises(PipelineError):
        run_pipeline(changed, context)
    assert not (context.reports_dir / "stages/features.json").exists()
    assert not (context.reports_dir / "pipeline_complete.json").exists()


def test_rejected_concurrent_run_preserves_completion_marker(synthetic_pipeline):
    from flood_ml.pipeline import run_pipeline

    settings, context = synthetic_pipeline
    run_pipeline(settings, context)
    marker = context.reports_dir / "pipeline_complete.json"
    before = marker.read_bytes()
    (context.reports_dir / ".pipeline.lock").write_text(
        '{"pid": 123456789, "run_id": "synthetic"}', encoding="utf-8"
    )

    with pytest.raises(PipelineError, match="already running or stale lock"):
        run_pipeline(settings, context)

    assert marker.read_bytes() == before


def test_manifest_metadata_and_config_changes_reject_run_reuse(synthetic_pipeline):
    settings, context = synthetic_pipeline
    manifest = settings.repo_root / settings.sources["manifest"]
    manifest.write_text(manifest.read_text() + "\n# metadata-only update\n")
    with pytest.raises(PipelineError, match="fingerprint"):
        start_run(settings, context.run_id)


def test_colab_pipeline_train_evaluate_publish_share_one_run_id(
    synthetic_pipeline, tmp_path
):
    from test_track_b import run_cli

    settings, _ = synthetic_pipeline
    root = settings.repo_root
    # The publisher requires a source commit even for this offline synthetic repo.
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Synthetic Test",
            "-c",
            "user.email=synthetic@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "synthetic fixture",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    run_id = "colab_sequence"
    run_cli(root, "run_pipeline", run_id)
    reports = root / "reports/runs" / run_id
    original_record = (reports / "run.json").read_bytes()
    original = json.loads(original_record)
    processed = f"data/processed/{run_id}"
    inputs = {
        "features": f"{processed}/road_features.parquet",
        "labels": f"{processed}/road_labels.parquet",
        "confirmed_event_segments": f"{processed}/confirmed_event_segments.parquet",
        "partitions": f"{processed}/road_partitions.parquet",
    }
    hashes = {
        name: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for name, path in inputs.items()
    }
    run_cli(root, "train", run_id, **inputs)
    training = json.loads((reports / "track_b.json").read_text())
    assert training["status"] == "pending_positive_review"
    assert training["input_paths"] == inputs
    assert training["input_hashes"] == hashes
    assert training["run_fingerprint"] == original["fingerprint"]
    assert training["config_hash"] == original["config_hash"]
    run_cli(
        root,
        "evaluate",
        run_id,
        scores=f"{processed}/road_susceptibility.parquet",
        roads=f"data/interim/{run_id}/road_segments.gpkg",
        confirmed_event_segments=inputs["confirmed_event_segments"],
        partitions=inputs["partitions"],
    )
    (reports / "runtime.json").write_text(
        json.dumps(
            {
                "python_version": "3.12",
                "uv_version": "uv 0.13.6",
                "platform": "synthetic",
            }
        )
    )
    drive = tmp_path / "drive"
    run_cli(root, "publish_run", run_id, drive_root=str(drive))
    manifest = json.loads(
        (drive / "runs" / run_id / "completion_manifest.json").read_text()
    )
    assert manifest["status"] == "published"
    assert manifest["gate_statuses"]["track_b"] == "pending_positive_review"
    assert (reports / "run.json").read_bytes() == original_record
    assert {
        name: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for name, path in inputs.items()
    } == hashes

    # Replay in an independent checkout with the same commit and raw snapshot.
    # Runtime-sidecar files are part of the published bundle and are recreated
    # locally so the verifier compares equivalent run-scoped file inventories.
    (reports / "colab_summary.md").write_text("synthetic pending run\n")
    (reports / "stdout.log").write_text("synthetic stdout\n")
    (reports / "stderr.log").write_text("synthetic stderr\n")
    # The first publication predates these synthetic sidecars, so publish a
    # fresh bundle after the complete expected inventory exists.
    drive = tmp_path / "drive-replay"
    run_cli(root, "publish_run", run_id, drive_root=str(drive))
    replay = tmp_path / "replay"
    shutil.copytree(
        root,
        replay,
        ignore=shutil.ignore_patterns("reports", "artifacts", "interim", "processed"),
    )
    replay_inputs = {name: path for name, path in inputs.items()}
    run_cli(replay, "run_pipeline", run_id)
    run_cli(replay, "train", run_id, **replay_inputs)
    run_cli(
        replay,
        "evaluate",
        run_id,
        scores=f"{processed}/road_susceptibility.parquet",
        roads=f"data/interim/{run_id}/road_segments.gpkg",
        confirmed_event_segments=inputs["confirmed_event_segments"],
        partitions=inputs["partitions"],
    )
    replay_reports = replay / "reports/runs" / run_id
    for name in ("runtime.json", "colab_summary.md", "stdout.log", "stderr.log"):
        shutil.copyfile(reports / name, replay_reports / name)
    verify = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[2] / "scripts/verify_colab_run.py"),
            "--published-run",
            str(drive / "runs" / run_id),
            "--repo-root",
            str(replay),
            "--run-id",
            run_id,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    verification = json.loads((replay_reports / "local_verification.json").read_text())
    assert verify.returncode == 0, (verify.stderr, verification["differences"])
    assert verification["equivalent"], verification["differences"]

    # The same initialization still rejects source changes before training.
    source = settings.sources["raw_root"] / "evidence.gpkg"
    source.write_bytes(source.read_bytes() + b"changed source snapshot")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[2] / "scripts/train.py"),
            "--repo-root",
            str(root),
            "--run-id",
            run_id,
            *[
                argument
                for name, path in inputs.items()
                for argument in (f"--{name.replace('_', '-')}", path)
            ],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Run fingerprint changed" in result.stderr
    assert (reports / "run.json").read_bytes() == original_record


def test_present_match_review_path_keeps_pipeline_and_train_on_same_run_id(
    synthetic_pipeline,
):
    from test_track_b import run_cli

    from flood_ml.labeling.reviews import MATCH_COLUMNS

    settings, _ = synthetic_pipeline
    root = settings.repo_root
    run_id = "review_path_sequence"
    review_dir = root / "configs/local"
    review_dir.mkdir(parents=True)
    review_path = review_dir / "review_decisions.csv"
    pd.DataFrame(columns=MATCH_COLUMNS).to_csv(review_path, index=False)
    relative_review = review_path.relative_to(root).as_posix()
    run_cli(root, "run_pipeline", run_id, match_reviews=relative_review)
    reports = root / "reports/runs" / run_id
    original = json.loads((reports / "run.json").read_text())
    inputs = {
        "features": f"data/processed/{run_id}/road_features.parquet",
        "labels": f"data/processed/{run_id}/road_labels.parquet",
        "confirmed_event_segments": f"data/processed/{run_id}/confirmed_event_segments.parquet",
        "partitions": f"data/processed/{run_id}/road_partitions.parquet",
    }
    run_cli(root, "train", run_id, **inputs)
    training = json.loads((reports / "track_b.json").read_text())
    assert (
        original["review_hashes"]["match_reviews"]
        == hashlib.sha256(review_path.read_bytes()).hexdigest()
    )
    assert training["run_fingerprint"] == original["fingerprint"]
    assert training["status"] == "pending_positive_review"


def test_reviewed_pipeline_passes_geometry_and_event_quotas(synthetic_pipeline):
    from flood_ml.labeling.reviews import EVENT_TEMPLATE_COLUMNS, MATCH_COLUMNS
    from flood_ml.pipeline import run_pipeline

    settings, context = synthetic_pipeline
    run_pipeline(settings, context, stages=["roads", "observations"])
    roads = gpd.read_file(context.interim_dir / "road_segments.gpkg").set_index(
        "osm_way_id"
    )
    match, events = [], []
    for i, count in enumerate((20, 5, 5, 10)):
        for j in range(count):
            obs = f"o{i}-{j}"
            review = ["synthetic test evidence", "test human", "2026-01-01T00:00:00Z"]
            match.append(
                [obs, roads.loc[str(i + 1), "road_segment_id"], "accepted", *review]
            )
            events.append([obs, f"event-{i}-{j}", f"location-{i}", *review, "true"])
    mr, er = settings.repo_root / "match.csv", settings.repo_root / "event.csv"
    pd.DataFrame(match, columns=MATCH_COLUMNS).to_csv(mr, index=False)
    pd.DataFrame(events, columns=EVENT_TEMPLATE_COLUMNS).to_csv(er, index=False)
    reviewed = replace(
        settings,
        labeling=dict(
            settings.labeling, match_reviews="match.csv", event_reviews="event.csv"
        ),
    )
    reviewed_context = start_run(reviewed, "reviewed")
    result = run_pipeline(reviewed, reviewed_context)
    assert result["evaluation_status"] == "complete"
    report = json.loads((reviewed_context.reports_dir / "evaluation.json").read_text())
    assert report["evaluation_performed"] is True
    assert report["gate"]["reference_events"] == report["gate"]["holdout_events"] == 20


def test_geopackage_output_keeps_native_and_normalized_casefolded_fields(tmp_path):
    frame = gpd.GeoDataFrame(
        {"Source_id": ["native-1"], "source_id": ["normalized-1"]},
        geometry=[Point(1, 2)],
        crs=9210,
    )
    path = tmp_path / "observations.gpkg"
    _write_frame(frame, path, layer="observations")
    restored = gpd.read_file(path, layer="observations")
    assert restored.source_id.tolist() == ["normalized-1"]
    assert restored.native_Source_id.tolist() == ["native-1"]
