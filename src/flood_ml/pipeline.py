"""Dependency-checked orchestration for the offline old-HCMC baseline."""

import ctypes
import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from flood_ml.baseline.mcda import (
    FEATURE_SPEC,
    TOP_K_FRACTIONS,
    compare_weight_sets,
    fit_normalizer,
    score_mcda,
)
from flood_ml.config import PipelineError, Settings, resolve_input, resolve_output
from flood_ml.evaluation.capture import evaluate_capture
from flood_ml.evaluation.splits import assign_spatial_partitions
from flood_ml.features.assembly import assemble_features
from flood_ml.features.raster import compute_raster_features
from flood_ml.features.water import compute_water_features
from flood_ml.gis.roads import _active, segment_roads
from flood_ml.gis.study_area import (
    context_radius_m,
    make_context,
    select_aoi,
)
from flood_ml.ingestion.osm import read_osm_roads
from flood_ml.ingestion.sources import validate_sources
from flood_ml.labeling.ird import read_ird
from flood_ml.labeling.matching import match_candidates
from flood_ml.labeling.observations import normalize_observations, propose_event_groups
from flood_ml.labeling.reviews import (
    EVENT_COLUMNS,
    MATCH_COLUMNS,
    apply_reviews,
    read_reviews,
)
from flood_ml.utils.publication import preflight
from flood_ml.utils.run import _hash_file, _hash_json, write_json_atomic

STAGE_ORDER = (
    "validate",
    "aoi",
    "roads",
    "observations",
    "raster",
    "features",
    "labels",
    "split",
    "mcda",
    "evaluation",
    "report",
)
STAGE_DEPENDENCIES = {
    "validate": (),
    "aoi": ("validate",),
    "roads": ("aoi",),
    "observations": ("aoi",),
    "raster": ("roads", "aoi"),
    "features": ("roads", "raster", "aoi"),
    "labels": ("roads", "observations"),
    "split": ("roads", "labels"),
    "mcda": ("features", "split", "roads"),
    "evaluation": ("mcda", "split", "labels", "roads"),
    "report": (
        "aoi",
        "roads",
        "observations",
        "features",
        "labels",
        "split",
        "mcda",
        "evaluation",
    ),
}


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _peak_working_set_bytes():
    """Return the process peak working set where the host exposes it."""
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        if query(process(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError, ValueError):
        return None


def _write_frame(frame, path: Path, *, layer=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix, delete=False
        ) as stream:
            temp_path = Path(stream.name)
        if path.suffix.lower() == ".gpkg":
            output = frame.copy()
            geometry_column = (
                output.geometry.name if isinstance(output, gpd.GeoDataFrame) else None
            )
            names_by_fold = {}
            for column in output.columns:
                names_by_fold.setdefault(str(column).casefold(), []).append(str(column))
            for folded, duplicates in list(names_by_fold.items()):
                if len(duplicates) < 2:
                    continue
                preferred = next(
                    (name for name in duplicates if name == folded), duplicates[-1]
                )
                for column in duplicates:
                    if column == preferred:
                        continue
                    candidate = f"native_{column}"
                    suffix = 2
                    while candidate.casefold() in names_by_fold:
                        candidate = f"native_{suffix}_{column}"
                        suffix += 1
                    output = output.rename(columns={column: candidate})
                    names_by_fold[candidate.casefold()] = [candidate]
            for column in output.columns:
                if column == geometry_column:
                    continue
                if (
                    output[column].dtype == object
                    and output[column]
                    .map(lambda item: isinstance(item, (dict, list, tuple, set)))
                    .any()
                ):
                    output[column] = output[column].map(
                        lambda item: (
                            json.dumps(item, sort_keys=True)
                            if isinstance(item, (dict, list, tuple, set))
                            else item
                        )
                    )
            output.to_file(
                temp_path,
                driver="GPKG",
                layer=layer or path.stem,
                index=False,
            )
        elif isinstance(frame, gpd.GeoDataFrame):
            frame.to_parquet(temp_path, index=False)
        else:
            frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    except (OSError, ValueError, TypeError) as exc:
        raise PipelineError(f"Cannot atomically publish {path}: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _asset(settings, *, dataset_id=None, asset_id=None):
    selected = [
        asset
        for asset in settings.sources["assets"]
        if (dataset_id is None or asset["dataset_id"] == dataset_id)
        and (asset_id is None or asset["asset_id"] == asset_id)
    ]
    if len(selected) != 1:
        raise PipelineError(
            f"Expected one source asset for dataset_id={dataset_id!r}, asset_id={asset_id!r}; found {len(selected)}"
        )
    return selected[0], resolve_input(
        Path(settings.sources["raw_root"]), selected[0]["path"]
    )


def _manifest_dataset(settings, dataset_id):
    path = resolve_output(settings.repo_root, settings.sources["manifest"])
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PipelineError(f"Cannot read source manifest {path}: {exc}") from exc
    datasets = [
        row
        for row in document.get("datasets", [])
        if row.get("dataset_id") == dataset_id
    ]
    if len(datasets) != 1:
        raise PipelineError(f"Manifest requires one {dataset_id} dataset")
    return datasets[0]


def _read_roads(path):
    return (
        gpd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else gpd.read_file(path)
    )


def _paths(context):
    return {
        "source_audit": context.reports_dir / "source_audit.json",
        "aoi": context.interim_dir / "study_area.gpkg",
        "context": context.interim_dir / "context_area.gpkg",
        "roads": context.interim_dir / "road_segments.gpkg",
        "observations": context.interim_dir / "flood_observations.gpkg",
        "candidate_groups": context.processed_dir
        / "unconfirmed_candidate_groups.parquet",
        "raster_features": context.interim_dir / "raster_features.parquet",
        "water_features": context.interim_dir / "water_features.parquet",
        "road_features": context.processed_dir / "road_features.parquet",
        "matches": context.processed_dir / "road_flood_matches.parquet",
        "review_decisions": context.processed_dir / "review_decision_history.parquet",
        "event_reviews": context.processed_dir / "event_review_history.parquet",
        "labels": context.processed_dir / "road_labels.parquet",
        "confirmed": context.processed_dir / "confirmed_event_segments.parquet",
        "partitions": context.processed_dir / "road_partitions.parquet",
        "group_audit": context.processed_dir / "reviewed_group_partition_audit.parquet",
        "analysis": context.processed_dir / "analysis_dataset.parquet",
        "scores": context.processed_dir / "road_susceptibility.parquet",
        "sensitivity_scores": context.processed_dir / "sensitivity_scores.parquet",
        "normalizer": context.artifacts_dir / "normalizer.json",
        "rules": context.artifacts_dir / "mcda_rules.json",
        "sensitivity": context.reports_dir / "weight_sensitivity.json",
        "evaluation": context.reports_dir / "evaluation.json",
        "report_json": context.reports_dir / "pipeline_report.json",
        "report_html": context.reports_dir / "pipeline_report.html",
        "stage_dir": context.reports_dir / "stages",
        "complete": context.reports_dir / "pipeline_complete.json",
        "lock": context.reports_dir / ".pipeline.lock",
    }


def _all_publications(context):
    paths = _paths(context)
    outputs = [p for key, p in paths.items() if key not in {"stage_dir", "lock"}]
    outputs.extend(paths["stage_dir"] / f"{stage}.json" for stage in STAGE_ORDER)
    outputs.extend(
        context.artifacts_dir / name
        for name in (
            "aoi_roads.png",
            "spatial_partitions.png",
            "baseline_scores.png",
            "feature_distributions.png",
        )
    )
    outputs.append(paths["lock"])
    return outputs


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PipelineError(f"Cannot read JSON record {path}: {exc}") from exc


def _roads_summary(frame):
    return {
        "count": len(frame),
        "unique_ids": bool(frame.road_segment_id.is_unique),
        "invalid_geometry": int((~frame.geometry.is_valid).sum()),
        "outside_aoi": int(
            sum(
                geometry.difference(frame.attrs["aoi_polygon"]).length > 1e-6
                for geometry in frame.geometry
            )
        )
        if "aoi_polygon" in frame.attrs
        else None,
        "total_length_m": float(frame.geometry.length.sum()),
        "bridge_count": int(frame.bridge.map(_active).sum())
        if "bridge" in frame
        else 0,
        "tunnel_count": int(frame.tunnel.map(_active).sum())
        if "tunnel" in frame
        else 0,
    }


def _stage_functions(settings, context, paths):
    def validate():
        audit = validate_sources(settings)
        return [paths["source_audit"]], {
            "assets": len(audit),
            "passed": int(audit.status.eq("PASS").sum()),
            "failed": int(audit.status.ne("PASS").sum()),
        }

    def aoi():
        boundary, boundary_path = _asset(settings, asset_id="administrative_boundaries")
        selector = settings.study_area["boundary_selection"]
        selected = select_aoi(
            gpd.read_file(boundary_path),
            field=selector["field"],
            value=selector["value"],
            target_crs=settings.study_area["target_crs"],
        )
        selected["study_area_id"] = settings.study_area["study_area_id"]
        halo = context_radius_m(
            settings.features,
            raster_stencil_m=settings.study_area["context"]["raster_stencil_m"],
        )
        buffered = make_context(selected, halo)
        _write_frame(selected, paths["aoi"], layer="study_area")
        _write_frame(buffered, paths["context"], layer="context_area")
        return [paths["aoi"], paths["context"]], {
            "study_area_id": settings.study_area["study_area_id"],
            "target_crs": selected.crs.to_string(),
            "area_km2": float(selected.geometry.area.iloc[0] / 1_000_000),
            "context_radius_m": halo,
            "boundary_asset_id": boundary["asset_id"],
        }

    def roads():
        area = gpd.read_file(paths["aoi"])
        buffered = gpd.read_file(paths["context"])
        road_asset, road_path = _asset(settings, dataset_id="D02")
        snapshot = _manifest_dataset(settings, "D02").get("snapshot_date")
        if not snapshot:
            raise PipelineError("D02 manifest requires snapshot_date")
        ways = read_osm_roads(
            road_path, buffered, settings.study_area["roads"]["allowed_highways"]
        )
        segments = segment_roads(
            ways,
            area,
            snapshot_date=snapshot,
            target_m=settings.study_area["segment_target_m"],
            maximum_m=settings.study_area["segment_max_m"],
        )
        stats = _roads_summary(segments)
        if not stats["unique_ids"] or stats["invalid_geometry"]:
            raise PipelineError("Road ID/geometry validation failed")
        _write_frame(segments, paths["roads"], layer="road_segments")
        qc = {
            **stats,
            "osm_snapshot_date": snapshot,
            "segmentation_config_digest": segments.attrs["segmentation_config_digest"],
            "algorithm_version": 4,
            "osm_asset_id": road_asset["asset_id"],
            "read_stats": ways.attrs.get("read_stats", {}),
        }
        return [paths["roads"]], qc

    def observations():
        observation_asset, observation_path = _asset(
            settings, asset_id="flood_observations"
        )
        area = gpd.read_file(paths["aoi"])
        layers = gpd.list_layers(observation_path)
        if "Flood_event_locations" in set(layers.name):
            raw = read_ird(observation_path)
        else:
            raw = gpd.read_file(observation_path, layer=layers.name.iloc[0])
        normalized = normalize_observations(raw, area)
        groups = propose_event_groups(normalized, settings.labeling)
        _write_frame(normalized, paths["observations"], layer="flood_observations")
        _write_frame(groups, paths["candidate_groups"])
        return [paths["observations"], paths["candidate_groups"]], {
            "source_asset_id": observation_asset["asset_id"],
            "observations": len(normalized),
            "outside_aoi": int(normalized.outside_aoi.sum()),
            "unconfirmed_candidate_groups": int(groups.candidate_group_id.nunique()),
            "review_required": int(groups.review_required.sum()),
            "spatial_precision_counts": normalized.spatial_precision.value_counts(
                dropna=False
            ).to_dict(),
        }

    def raster():
        roads_frame = gpd.read_file(paths["roads"])
        area = gpd.read_file(paths["aoi"])
        dem = [
            resolve_input(settings.sources["raw_root"], asset["path"])
            for asset in settings.sources["assets"]
            if asset["dataset_id"] == "D03"
        ]
        if not dem:
            raise PipelineError("At least one D03 DEM asset is required")
        _, landcover = _asset(settings, dataset_id="D05")
        result = compute_raster_features(
            roads_frame, area, dem, landcover, settings.features
        )
        _write_frame(result, paths["raster_features"])
        metadata = result.attrs.get("raster_metadata", {})
        qc = {
            "roads": len(result),
            "feature_columns": list(result.columns[1:]),
            "raster_metadata": metadata,
            "missing": {
                column: int(result[column].isna().sum())
                for column in result.columns
                if column != "road_segment_id"
                and not column.endswith("_valid_fraction")
            },
        }
        return [paths["raster_features"]], qc

    def features():
        roads_frame = gpd.read_file(paths["roads"])
        _, water_path = _asset(settings, asset_id="water")
        coverage = settings.sources.get("water_coverage", {})
        if coverage.get("coverage_domain") != "old_hcmc_aoi" or not coverage.get(
            "coverage_assumption"
        ):
            raise PipelineError(
                "Explicit old_hcmc_aoi water coverage assumption required"
            )
        _, boundary_path = _asset(settings, asset_id=coverage.get("boundary_asset_id"))
        selector = settings.study_area["boundary_selection"]
        domain = select_aoi(
            gpd.read_file(boundary_path),
            field=selector["field"],
            value=selector["value"],
            target_crs=settings.study_area["target_crs"],
        )
        water = gpd.read_file(water_path)
        water_features = compute_water_features(
            roads_frame,
            water,
            domain,
            dict(
                settings.features,
                coverage_assumption=coverage["coverage_assumption"],
            ),
        )
        raster_features = pd.read_parquet(paths["raster_features"])
        result = assemble_features(
            roads_frame,
            raster_features,
            water_features,
            settings.features["feature_columns"],
        )
        _write_frame(water_features, paths["water_features"])
        _write_frame(result, paths["road_features"])
        qc = {
            "roads": len(result),
            "unique_ids": bool(result.road_segment_id.is_unique),
            "feature_columns": settings.features["feature_columns"],
            "missing": {
                column: int(result[column].isna().sum())
                for column in settings.features["feature_columns"]
            },
            "water_context_incomplete": int(result.water_context_incomplete.sum()),
            "water_distance_incomplete": int(result.water_distance_incomplete.sum()),
            "score_excluded_structures": int((~result.score_eligible).sum()),
            "coverage_assumption": coverage["coverage_assumption"],
            "water_metadata": water_features.attrs.get("water_metadata", {}),
        }
        return [paths["water_features"], paths["road_features"]], qc

    def labels():
        roads_frame = gpd.read_file(paths["roads"])
        observations_frame = gpd.read_file(paths["observations"])
        candidates = match_candidates(
            observations_frame, roads_frame, settings.labeling
        )
        match_path = settings.labeling.get("match_reviews")
        event_path = settings.labeling.get("event_reviews")
        match_reviews = (
            read_reviews(resolve_output(settings.repo_root, match_path), MATCH_COLUMNS)
            if match_path
            else pd.DataFrame(columns=MATCH_COLUMNS)
        )
        event_reviews = (
            read_reviews(resolve_output(settings.repo_root, event_path), EVENT_COLUMNS)
            if event_path
            else pd.DataFrame(columns=EVENT_COLUMNS)
        )
        road_labels, confirmed = apply_reviews(
            roads_frame,
            observations_frame,
            candidates,
            match_reviews,
            event_reviews,
        )
        outputs = [
            (candidates, paths["matches"]),
            (match_reviews, paths["review_decisions"]),
            (event_reviews, paths["event_reviews"]),
            (road_labels, paths["labels"]),
            (confirmed, paths["confirmed"]),
        ]
        for frame, path in outputs:
            _write_frame(frame, path)
        qc = {
            "observations": len(observations_frame),
            "outside_aoi": int(observations_frame.outside_aoi.sum()),
            "roads": len(roads_frame),
            "candidate_rows": len(candidates),
            "candidate_links": int(candidates.road_segment_id.notna().sum()),
            "all_candidates_require_review": bool(candidates.review_required.all())
            if len(candidates)
            else True,
            "label_counts": road_labels.label_status.value_counts().to_dict(),
            "confirmed_event_segments": len(confirmed),
            "match_review_count": len(match_reviews),
            "event_review_count": len(event_reviews),
            "affected_geometry_complete_count": int(
                confirmed.affected_geometry_complete.sum()
            )
            if "affected_geometry_complete" in confirmed
            else 0,
            "review_queue_rows": int(candidates.review_required.sum()),
        }
        return [path for _, path in outputs], qc

    def split():
        roads_frame = gpd.read_file(paths["roads"])
        confirmed = pd.read_parquet(paths["confirmed"])
        result = assign_spatial_partitions(
            roads_frame,
            confirmed,
            {"experiments": settings.experiments, "features": settings.features},
        )
        audit = pd.DataFrame(result.attrs.get("confirmed_group_audit", []))
        _write_frame(result, paths["partitions"])
        _write_frame(audit, paths["group_audit"])
        counts = result.partition.value_counts().to_dict()
        qc = {
            "road_count": len(result),
            "partition_counts": counts,
            "block_count": int(result.block_id.nunique()),
            "holdout_blocks": int(
                result.loc[result.partition.eq("holdout"), "block_id"].nunique()
            ),
            "support_rule_m": max(
                float(settings.experiments.get("guard_m", 1000)),
                2
                * max(
                    float(settings.features.get(name, default))
                    for name, default in (
                        ("neighborhood_radius_m", 500),
                        ("dem_buffer_m", 30),
                        ("landcover_buffer_m", 250),
                        ("water_buffer_m", 250),
                    )
                ),
            ),
            "support_scope": "declared target-space feature neighborhoods; raw DEM bilinear source pixels may overlap across partitions",
            "reviewed_group_audit_rows": len(audit),
        }
        return [paths["partitions"], paths["group_audit"]], qc

    def mcda():
        config = settings.experiments.get("mcda")
        expected_sets = {
            "equal": dict.fromkeys(FEATURE_SPEC, 0.2),
            "terrain-heavy": dict(
                zip(FEATURE_SPEC, [0.15, 0.4, 0.15, 0.15, 0.15], strict=True)
            ),
            "water-heavy": dict(
                zip(FEATURE_SPEC, [0.15, 0.15, 0.15, 0.4, 0.15], strict=True)
            ),
            "builtup-heavy": dict(
                zip(FEATURE_SPEC, [0.15, 0.15, 0.15, 0.15, 0.4], strict=True)
            ),
        }
        expected = {
            "feature_spec": [
                {"name": name, "direction": direction}
                for name, direction in FEATURE_SPEC.items()
            ],
            "normalization": "q05_q95_linear_clip",
            "constant_reference_contribution": 0.5,
            "missing_policy": "null_score_no_weight_redistribution",
            "primary_weight_set": "equal",
            "top_k_length_fractions": list(TOP_K_FRACTIONS),
            "weight_sets": expected_sets,
        }
        if config != expected:
            raise PipelineError("MCDA configuration differs from the approved rules")
        roads_frame = gpd.read_file(paths["roads"])
        features = pd.read_parquet(paths["road_features"])
        partitions = pd.read_parquet(paths["partitions"])
        joined = features.merge(
            partitions[["road_segment_id", "partition", "reference_fit_eligible"]],
            on="road_segment_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        lengths = pd.DataFrame(
            {
                "road_segment_id": roads_frame.road_segment_id,
                "road_length_m": roads_frame.geometry.length.to_numpy(),
            }
        )
        for column in ("bridge", "tunnel"):
            if column in roads_frame:
                lengths[column] = roads_frame[column].to_numpy()
        joined = joined.drop(
            columns=["road_length_m", "bridge", "tunnel"], errors="ignore"
        ).merge(lengths, on="road_segment_id", validate="one_to_one", sort=False)
        reference = joined.loc[joined.partition.eq("reference")]
        normalizer = fit_normalizer(reference, dict(FEATURE_SPEC))
        normalizer.update(config_hash=context.config_hash)
        primary = score_mcda(joined, normalizer, expected_sets["equal"])
        sensitivity = compare_weight_sets(joined, normalizer, expected_sets)
        sensitivity.update(
            scope="partition_population",
            ranking_scope="eligible_feature_input_roads_only",
            evaluation_status="pending_label_review",
        )
        sensitivity_scores = pd.concat(
            [
                score_mcda(joined, normalizer, weights).assign(weight_set=name)
                for name, weights in expected_sets.items()
            ],
            ignore_index=True,
        )
        _write_frame(primary, paths["scores"])
        _write_frame(sensitivity_scores, paths["sensitivity_scores"])
        write_json_atomic(paths["sensitivity"], _clean(sensitivity))
        write_json_atomic(paths["normalizer"], _clean(normalizer))
        write_json_atomic(
            paths["rules"],
            _clean({"rules": expected, "config_hash": context.config_hash}),
        )
        # A geometry-bearing analysis dataset is GeoParquet, with labels kept
        # separate in their own authoritative table and joined for inspection.
        analysis = (
            roads_frame[["road_segment_id", "geometry"]]
            .merge(features, on="road_segment_id", validate="one_to_one", sort=False)
            .merge(
                pd.read_parquet(paths["labels"]),
                on="road_segment_id",
                validate="one_to_one",
                sort=False,
            )
        )
        analysis = gpd.GeoDataFrame(analysis, geometry="geometry", crs=roads_frame.crs)
        _write_frame(analysis, paths["analysis"])
        qc = {
            "scope": "partition_population",
            "partition_population_count": len(partitions),
            "feature_input_count": len(features),
            "sample_partition_counts": joined.partition.value_counts().to_dict(),
            "reference_fit_count": int(normalizer["fit_row_count"]),
            "scored_count": int(primary.mcda_eligible.sum()),
            "score_reason_counts": primary.score_reason.value_counts().to_dict(),
            "required_feature_missing_counts": {
                feature: int(
                    primary.missing_required_features.str.split(";")
                    .map(lambda names, feature=feature: feature in names)
                    .sum()
                )
                for feature in FEATURE_SPEC
            },
            "evaluation_status": "pending_label_review",
            "ranking_scope": "eligible_feature_input_roads_only",
        }
        return [
            paths["scores"],
            paths["sensitivity_scores"],
            paths["normalizer"],
            paths["rules"],
            paths["sensitivity"],
            paths["analysis"],
        ], qc

    def evaluation():
        report = evaluate_capture(
            pd.read_parquet(paths["scores"]),
            gpd.read_file(paths["roads"]),
            pd.read_parquet(paths["confirmed"]),
            pd.read_parquet(paths["partitions"]),
            settings.experiments,
        )
        report.update(
            run_id=context.run_id,
            evaluation_performed=bool(report.get("evaluation_performed")),
        )
        write_json_atomic(paths["evaluation"], _clean(report))
        return [paths["evaluation"]], {
            "evaluation_status": report["status"],
            "evaluation_performed": report["evaluation_performed"],
            "gate": report["gate"],
            "metric_count": len(report.get("metrics", [])),
        }

    from flood_ml.reporting import build_report, render_report_artifacts

    def report():
        stage_state = {
            stage: _read_json(paths["stage_dir"] / f"{stage}.json")
            for stage in STAGE_ORDER
            if stage != "report" and (paths["stage_dir"] / f"{stage}.json").exists()
        }
        summary = build_report(context, stage_state)
        artifacts = render_report_artifacts(context, paths)
        summary["artifacts"] = artifacts
        summary["run_id"] = context.run_id
        summary["pipeline_status"] = "complete"
        summary["evaluation_status"] = _read_json(paths["evaluation"])["status"]
        write_json_atomic(paths["report_json"], _clean(summary))
        html = _render_html(summary)
        _atomic_text(paths["report_html"], html)
        return [
            paths["report_json"],
            paths["report_html"],
            *(context.artifacts_dir / name for name in artifacts["files"]),
        ], {
            "evaluation_status": summary["evaluation_status"],
            "report_files": artifacts["files"],
        }

    return {
        "validate": validate,
        "aoi": aoi,
        "roads": roads,
        "observations": observations,
        "raster": raster,
        "features": features,
        "labels": labels,
        "split": split,
        "mcda": mcda,
        "evaluation": evaluation,
        "report": report,
    }


def _atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise PipelineError(f"Cannot atomically publish {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _render_html(report):
    escaped = (
        json.dumps(_clean(report), ensure_ascii=False, indent=2)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Old-HCMC road flood susceptibility run</title>"
        "<style>body{font:15px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#19323a}"
        "h1{font-size:1.7rem}pre{white-space:pre-wrap;background:#f2f6f5;padding:1rem;border-radius:8px;overflow:auto}"
        "img{max-width:100%;height:auto}</style>"
        f"<h1>Old-HCMC road flood susceptibility — {report['run_id']}</h1>"
        f"<p>Pipeline: {report['pipeline_status']} · Evaluation: {report['evaluation_status']}</p>"
        "<p>MCDA scores are relative indices, not flood probabilities. Unknown labels remain unknown.</p>"
        f"<pre>{escaped}</pre></html>\n"
    )


def _dependencies(targets):
    required = set()

    def add(stage):
        if stage not in STAGE_DEPENDENCIES:
            raise PipelineError(f"Unknown pipeline stage: {stage}")
        if stage in required:
            return
        required.add(stage)
        for dependency in STAGE_DEPENDENCIES[stage]:
            add(dependency)

    for target in targets:
        add(target)
    return [stage for stage in STAGE_ORDER if stage in required]


def _stage_outputs(function, stage, result):
    marker = result["marker"]
    if not marker.exists():
        return False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("status") != "complete" or record.get("stage") != stage:
            return False
        if record.get("input_fingerprint") != result["input_fingerprint"]:
            return False
        outputs = record.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            return False
        return all(
            (path := Path(name)).is_file() and _hash_file(path) == digest
            for name, digest in outputs.items()
        )
    except (OSError, ValueError, TypeError, PipelineError):
        return False


def run_pipeline(
    settings: Settings, context, *, stages: list[str] | None = None
) -> dict:
    """Run requested stages and dependencies; checkpoint only verified outputs.

    A full run writes ``pipeline_complete.json`` after evaluation and the offline
    report pass. Partial runs leave completion unclaimed. Stage output hashes and
    dependency checkpoints drive reuse; corrupt outputs invalidate descendants.
    """
    if context.repo_root.resolve() != settings.repo_root.resolve():
        raise PipelineError("RunContext and Settings must have the same repo_root")
    if settings.experiments.get("run_id") != context.run_id:
        settings = replace(
            settings,
            experiments={**settings.experiments, "run_id": context.run_id},
        )
    targets = list(STAGE_ORDER) if stages is None else list(stages)
    if len(targets) != len(set(targets)):
        raise PipelineError("Pipeline stages must be unique")
    plan = _dependencies(targets)
    paths = _paths(context)
    preflight(settings, _all_publications(context))
    run_record = _read_json(context.reports_dir / "run.json")
    lock = paths["lock"]
    paths["stage_dir"].mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "run_id": context.run_id}))
    except FileExistsError as exc:
        raise PipelineError(
            f"Pipeline already running or stale lock exists: {lock}"
        ) from exc
    started = time.perf_counter()
    results = {}
    tainted = set()
    records = {}
    try:
        paths["complete"].unlink(missing_ok=True)
        functions = _stage_functions(settings, context, paths)
        for stage in plan:
            dependencies = STAGE_DEPENDENCIES[stage]
            dependency_outputs = {
                dependency: records[dependency]["outputs"]
                for dependency in dependencies
            }
            input_fingerprint = _hash_json(
                {
                    "run_fingerprint": run_record.get("fingerprint"),
                    "stage": stage,
                    "dependencies": dependency_outputs,
                }
            )
            marker = paths["stage_dir"] / f"{stage}.json"
            result_info = {"marker": marker, "input_fingerprint": input_fingerprint}
            can_reuse = not any(dep in tainted for dep in dependencies)
            reused = can_reuse and _stage_outputs(functions[stage], stage, result_info)
            if reused:
                record = _read_json(marker)
                records[stage] = record
                results[stage] = {
                    "reused": True,
                    "outputs": record["outputs"],
                    "summary": record.get("summary", {}),
                }
                continue

            tainted.add(stage)
            marker.unlink(missing_ok=True)
            began = time.perf_counter()
            output_paths, summary = functions[stage]()
            if not output_paths:
                raise PipelineError(f"Stage {stage} did not declare outputs")
            output_hashes = {}
            for path in output_paths:
                if not Path(path).is_file():
                    raise PipelineError(f"Stage {stage} did not publish {path}")
                output_hashes[str(Path(path).resolve())] = _hash_file(Path(path))
            record = {
                "schema_version": 1,
                "stage": stage,
                "status": "complete",
                "run_id": context.run_id,
                "input_fingerprint": input_fingerprint,
                "outputs": output_hashes,
                "summary": _clean(summary),
                "elapsed_seconds": time.perf_counter() - began,
                "process_peak_working_set_bytes": _peak_working_set_bytes(),
            }
            write_json_atomic(marker, record)
            records[stage] = record
            results[stage] = {
                "reused": False,
                "outputs": output_hashes,
                "summary": record["summary"],
            }

        complete = set(plan) == set(STAGE_ORDER)
        evaluation_status = (
            records["evaluation"]["summary"]["evaluation_status"]
            if "evaluation" in records
            else "not_run"
        )
        summary = {
            "run_id": context.run_id,
            "pipeline_status": "complete" if complete else "partial",
            "evaluation_status": evaluation_status,
            "stages": results,
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_working_set_bytes": _peak_working_set_bytes(),
        }
        if complete:
            write_json_atomic(
                paths["complete"],
                {
                    "schema_version": 1,
                    "status": "complete",
                    "run_id": context.run_id,
                    "run_fingerprint": run_record.get("fingerprint"),
                    "evaluation_status": evaluation_status,
                    "stage_markers": {
                        stage: _hash_file(paths["stage_dir"] / f"{stage}.json")
                        for stage in STAGE_ORDER
                    },
                    "elapsed_seconds": summary["elapsed_seconds"],
                },
            )
        return summary
    except Exception:
        paths["complete"].unlink(missing_ok=True)
        raise
    finally:
        lock.unlink(missing_ok=True)
