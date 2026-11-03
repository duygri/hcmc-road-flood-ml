"""Offline run summary and map artifacts for the old-HCMC baseline."""

import json
import os
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

from flood_ml.config import PipelineError

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PipelineError(f"Cannot read pipeline report input {path}: {exc}") from exc


def _stats(frame, columns):
    result = {}
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values[np.isfinite(values)]
        result[column] = {
            "count": int(valid.count()),
            "missing": int(values.isna().sum()),
            "min": float(valid.min()) if len(valid) else None,
            "p50": float(valid.median()) if len(valid) else None,
            "p95": float(valid.quantile(0.95)) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
        }
    return result


def build_report(context, stage_results: dict) -> dict:
    """Build the report from verified stage summaries and final tables."""
    reports = context.reports_dir
    processed = context.processed_dir
    required = [
        "validate",
        "roads",
        "observations",
        "features",
        "labels",
        "split",
        "mcda",
        "evaluation",
    ]
    if any(name not in stage_results for name in required):
        raise PipelineError("Full pipeline report requires every upstream stage")
    source_audit = _read(reports / "source_audit.json")
    evaluation = _read(reports / "evaluation.json")
    sensitivity = _read(reports / "weight_sensitivity.json")
    features = pd.read_parquet(processed / "road_features.parquet")
    scores = pd.read_parquet(processed / "road_susceptibility.parquet")
    observations = gpd.read_file(context.interim_dir / "flood_observations.gpkg")
    aoi = gpd.read_file(context.interim_dir / "study_area.gpkg")
    labels = pd.read_parquet(processed / "road_labels.parquet")
    feature_columns = [
        "elevation_median_m",
        "elevation_p10_m",
        "relative_elevation_m",
        "slope_mean_deg",
        "builtup_ratio_250m",
        "vegetation_ratio_250m",
        "dist_water_m",
        "water_area_ratio_250m",
    ]
    feature_stats = _stats(features, feature_columns)
    score_reasons = (
        scores.score_reason.value_counts(dropna=False).to_dict()
        if "score_reason" in scores
        else {}
    )
    return {
        "schema_version": 1,
        "run_id": context.run_id,
        "pipeline_status": "complete",
        "evaluation_status": evaluation.get("status", "pending_label_review"),
        "study_area": {
            "study_area_id": str(aoi.study_area_id.iloc[0])
            if "study_area_id" in aoi and len(aoi)
            else "hcmc_pre_2025",
            "crs": aoi.crs.to_string(),
            "area_km2": float(aoi.geometry.area.sum() / 1_000_000),
            "geometry_count": len(aoi),
        },
        "source_coverage": {
            "audit_status": source_audit.get("status"),
            "assets": int(stage_results["validate"]["summary"]["assets"]),
            "passed_assets": int(stage_results["validate"]["summary"]["passed"]),
            "water_coverage_scope": "old-HCMC AOI under the documented D04 completeness assumption; outside the AOI is unknown",
        },
        "roads": stage_results["roads"]["summary"],
        "observations": {
            **stage_results["observations"]["summary"],
            "quality_flags": observations.quality_flags.value_counts().to_dict()
            if "quality_flags" in observations
            else {},
        },
        "review_queue": {
            "candidate_links": stage_results["labels"]["summary"]["candidate_links"],
            "review_queue_rows": stage_results["labels"]["summary"][
                "review_queue_rows"
            ],
            "accepted_links": stage_results["labels"]["summary"][
                "confirmed_event_segments"
            ],
            "label_status_counts": labels.label_status.value_counts().to_dict(),
            "affected_geometry_complete_count": stage_results["labels"]["summary"][
                "affected_geometry_complete_count"
            ],
        },
        "features": {
            **stage_results["features"]["summary"],
            "distributions": feature_stats,
        },
        "splits": stage_results["split"]["summary"],
        "baseline": {
            **stage_results["mcda"]["summary"],
            "score_reason_counts": score_reasons,
            "score_distribution": _stats(scores, ["score"])["score"],
            "sensitivity": sensitivity,
        },
        "evaluation": evaluation,
        "limitations": [
            "The MCDA score is a relative susceptibility index, not a probability or forecast.",
            "Flood observations reflect source reporting and geocoding coverage; unknown roads are not verified negatives.",
            "FABDEM elevations use the native vertical reference without an independently verified vertical transformation.",
            "The D04 water source is assumed complete within the old-HCMC AOI; this has not been independently verified.",
            "Source vintages differ; OSM, FABDEM, WorldCover and flood observations do not represent one synchronized date.",
            "License terms include noncommercial restrictions for some research sources and ODbL obligations for OSM.",
            "Spatial support partitions separate configured target-space neighborhoods; bilinear DEM source pixels may overlap across partition boundaries.",
        ],
    }


def _atomic_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix, delete=False
        ) as stream:
            temporary = Path(stream.name)
        fig.savefig(temporary, dpi=150, bbox_inches="tight", facecolor="white")
        os.replace(temporary, path)
    except OSError as exc:
        raise PipelineError(f"Cannot publish report figure {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def render_report_artifacts(context, paths: dict) -> dict:
    """Render compact, offline maps and feature distributions for the run."""
    roads = gpd.read_file(paths["roads"])
    aoi = gpd.read_file(paths["aoi"])
    partitions = pd.read_parquet(paths["partitions"])
    features = pd.read_parquet(paths["road_features"])
    scores = pd.read_parquet(paths["scores"])
    plot_limit = 100_000

    fig, ax = plt.subplots(figsize=(9, 9))
    aoi.boundary.plot(ax=ax, color="#172b4d", linewidth=1.4)
    plot_roads = roads
    if len(plot_roads) > plot_limit:
        plot_roads = plot_roads.sort_values("road_segment_id", kind="stable").iloc[
            :plot_limit
        ]
    plot_roads.plot(ax=ax, color="#2878a5", linewidth=0.25, alpha=0.62)
    ax.set_title("Old-HCMC road network and study area")
    ax.set_axis_off()
    _atomic_figure(fig, context.artifacts_dir / "aoi_roads.png")
    plt.close(fig)

    split_map = roads.merge(
        partitions[["road_segment_id", "partition"]],
        on="road_segment_id",
        validate="one_to_one",
        sort=False,
    )
    if len(split_map) > plot_limit:
        split_map = split_map.sort_values("road_segment_id", kind="stable").iloc[
            :plot_limit
        ]
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = {
        "reference": "#2878a5",
        "holdout": "#ef7c45",
        "guard": "#8f8f8f",
        "excluded": "#c83e4d",
    }
    for part, group in split_map.groupby("partition", sort=True):
        group.plot(ax=ax, color=colors.get(part, "#8f8f8f"), linewidth=0.3, label=part)
    ax.set_title("Frozen spatial partitions (road sample for display)")
    ax.legend(loc="lower left")
    ax.set_axis_off()
    _atomic_figure(fig, context.artifacts_dir / "spatial_partitions.png")
    plt.close(fig)

    score_map = roads.merge(
        scores[["road_segment_id", "score", "score_reason"]],
        on="road_segment_id",
        validate="one_to_one",
        sort=False,
    )
    score_map = score_map.loc[score_map.score.notna()]
    if len(score_map) > plot_limit:
        score_map = score_map.sort_values("road_segment_id", kind="stable").iloc[
            :plot_limit
        ]
    fig, ax = plt.subplots(figsize=(9, 9))
    if len(score_map):
        score_map.plot(
            ax=ax,
            column="score",
            cmap="viridis",
            linewidth=0.4,
            legend=True,
            legend_kwds={"label": "Relative MCDA score (0–100)"},
        )
    else:
        aoi.boundary.plot(ax=ax, color="#172b4d", linewidth=1.4)
    ax.set_title("Eligible road susceptibility index")
    ax.set_axis_off()
    _atomic_figure(fig, context.artifacts_dir / "baseline_scores.png")
    plt.close(fig)

    histogram_columns = [
        "elevation_median_m",
        "relative_elevation_m",
        "slope_mean_deg",
        "dist_water_m",
        "builtup_ratio_250m",
    ]
    fig, axes = plt.subplots(1, len(histogram_columns), figsize=(16, 3.6))
    for ax, column in zip(axes, histogram_columns, strict=True):
        values = pd.to_numeric(features[column], errors="coerce").dropna()
        ax.hist(values, bins=30, color="#2878a5", alpha=0.88)
        ax.set_title(column.replace("_", " "), fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Observed feature distributions (valid values only)")
    _atomic_figure(fig, context.artifacts_dir / "feature_distributions.png")
    plt.close(fig)

    return {
        "files": [
            "aoi_roads.png",
            "spatial_partitions.png",
            "baseline_scores.png",
            "feature_distributions.png",
        ],
        "road_count": len(roads),
        "map_display_limit": plot_limit,
        "split_roads_displayed": len(split_map),
        "scored_roads_displayed": len(score_map),
        "note": "Large maps display the first stable road IDs up to the stated limit; tables and metrics use the full population.",
    }
