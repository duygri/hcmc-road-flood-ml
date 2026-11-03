"""Whole offline GIS workflow, publication integrity and honest review gates."""

import hashlib
import json
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


def test_manifest_metadata_and_config_changes_reject_run_reuse(synthetic_pipeline):
    settings, context = synthetic_pipeline
    manifest = settings.repo_root / settings.sources["manifest"]
    manifest.write_text(manifest.read_text() + "\n# metadata-only update\n")
    with pytest.raises(PipelineError, match="fingerprint"):
        start_run(settings, context.run_id)


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
