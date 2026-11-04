"""Input identity checks shared by every integration publication."""

from pathlib import Path

from flood_ml.config import PipelineError, resolve_input, resolve_output


def protected_paths(settings):
    paths = [
        resolve_input(Path(settings.sources["raw_root"]), a["path"])
        for a in settings.sources["assets"]
    ]
    if settings.sources.get("manifest"):
        paths.append(resolve_output(settings.repo_root, settings.sources["manifest"]))
    paths.extend(settings.configuration_paths)
    paths.extend(
        resolve_output(settings.repo_root, p)
        for p in settings.experiments.get("configs", {}).values()
    )
    paths.append(settings.repo_root / "uv.lock")
    for key in ("match_reviews", "event_reviews"):
        if settings.labeling.get(key):
            paths.append(resolve_output(settings.repo_root, settings.labeling[key]))
    return paths


def preflight(settings, destinations, *, inputs=()):
    """Reject canonical and hard-link aliases before creating anything."""
    sources = [*protected_paths(settings), *inputs]
    raw = Path(settings.sources["raw_root"]).resolve()
    for target in destinations:
        target = Path(target).resolve()
        if target.is_relative_to(raw):
            raise PipelineError("Publication cannot overwrite raw sources")
        for source in sources:
            source = Path(source).resolve()
            if target == source or (
                target.exists() and source.exists() and target.samefile(source)
            ):
                raise PipelineError(
                    f"Input/config aliases publication destination: {target}"
                )


def run_destinations(settings, run_id):
    """List every known run-scoped file before ``start_run`` writes its record."""
    root = settings.repo_root
    outputs = settings.experiments["outputs"]
    directories = {
        name: resolve_output(root, f"{base}/{run_id}") for name, base in outputs.items()
    }
    interim, processed, reports, artifacts = (
        directories["interim"],
        directories["processed"],
        directories["reports"],
        directories["artifacts"],
    )
    destinations = [
        reports / name
        for name in (
            "run.json",
            ".start.lock",
            ".pipeline.lock",
            "source_audit.json",
            "evaluation.json",
            "pipeline_report.json",
            "pipeline_report.html",
            "pipeline_complete.json",
            "weight_sensitivity.json",
        )
    ]
    destinations.extend(
        reports / "stages" / f"{stage}.json"
        for stage in (
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
    )
    destinations.extend(
        interim / name
        for name in (
            "study_area.gpkg",
            "context_area.gpkg",
            "road_segments.gpkg",
            "flood_observations.gpkg",
            "raster_features.parquet",
            "water_features.parquet",
        )
    )
    destinations.extend(
        processed / name
        for name in (
            "unconfirmed_candidate_groups.parquet",
            "road_features.parquet",
            "road_flood_matches.parquet",
            "review_decision_history.parquet",
            "event_review_history.parquet",
            "road_labels.parquet",
            "confirmed_event_segments.parquet",
            "road_partitions.parquet",
            "reviewed_group_partition_audit.parquet",
            "analysis_dataset.parquet",
            "road_susceptibility.parquet",
            "sensitivity_scores.parquet",
        )
    )
    destinations.extend(
        artifacts / name
        for name in (
            "normalizer.json",
            "mcda_rules.json",
            "aoi_roads.png",
            "spatial_partitions.png",
            "baseline_scores.png",
            "feature_distributions.png",
        )
    )
    return destinations
