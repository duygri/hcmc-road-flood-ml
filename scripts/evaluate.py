"""Write exploratory reviewed-event capture or an explicit pending review report."""

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_ml.config import (
    PipelineError,
    _validate_run_id,
    load_settings,
    resolve_output,
)
from flood_ml.evaluation.capture import RULES, evaluate_capture
from flood_ml.utils.run import (
    _git_state,
    _hash_file,
    _hash_json,
    _snapshot,
    write_json_atomic,
)
from flood_ml.utils.wandb_tracking import log_stage_report


def _tracking_metrics(report: dict) -> dict:
    """Select aggregate capture/lift/bootstrap values; keep row-level data local."""
    metrics = {}
    if not report["evaluation_performed"]:
        return metrics

    def add(name, value):
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        ):
            metrics[name] = value

    methods = report.get("methods", {})
    ranked = [
        (name, methods[name])
        for name in ("mcda", "elevation_only", "random")
        if name in methods
    ]
    ranked.extend(
        (f"track_b/{name}", method)
        for name, method in methods.get("track_b", {}).items()
    )
    for method_name, method in ranked:
        for fraction, values in method["top_k"].items():
            prefix = f"methods/{method_name}/top_k/{fraction}"
            for key in (
                "capture",
                "capture_any",
                "capture_all",
                "lift",
                "lift_vs_mcda",
                "lift_vs_elevation_only",
                "capture_difference_vs_mcda",
                "capture_difference_vs_elevation_only",
            ):
                add(f"{prefix}/{key}", values.get(key))
            uncertainty = values.get("uncertainty", {})
            for key in (
                "replicates",
                "valid_replicates",
                "undefined_replicates",
                "undefined_lift_replicates",
            ):
                add(f"{prefix}/uncertainty/{key}", uncertainty.get(key))
            for key in (
                "capture_ci95",
                "capture_any_ci95",
                "capture_all_ci95",
                "lift_ci95",
            ):
                interval = uncertainty.get(key)
                if isinstance(interval, (list, tuple)) and len(interval) == 2:
                    for endpoint, value in zip(("lower", "upper"), interval):
                        add(f"{prefix}/uncertainty/{key}/{endpoint}", value)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--output", help="Repository-relative report path")
    for name in ("scores", "roads", "confirmed-event-segments", "partitions"):
        parser.add_argument(
            f"--{name}", required=True, help="Repository-relative derived input"
        )
    parser.add_argument(
        "--track-b-scores", help="Optional relative Track B score input"
    )
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config, repo_root=args.repo_root)
        run_id = args.run_id or settings.experiments["run_id"]
        _validate_run_id(run_id)
        inputs = {
            name: getattr(args, name)
            for name in ("scores", "roads", "confirmed_event_segments", "partitions")
        }
        if args.track_b_scores is not None:
            inputs["track_b_scores"] = args.track_b_scores
        paths = {
            name: resolve_output(settings.repo_root, value)
            for name, value in inputs.items()
        }
        output = resolve_output(
            settings.repo_root,
            args.output
            or f"{settings.experiments['outputs']['reports']}/{run_id}/evaluation.json",
        )
        # Match load_settings' validated entry/bundle/link resolution, rather
        # than assuming config files live under configs/ or end in .yaml.
        selected_config = (
            args.config.resolve()
            if args.config.is_absolute()
            else resolve_output(settings.repo_root, str(args.config))
        )
        configuration_paths = {
            selected_config,
            *(
                resolve_output(settings.repo_root, value)
                for value in settings.experiments["configs"].values()
            ),
        }
        if selected_config.stem != "experiments":
            configuration_paths.add(
                resolve_output(settings.repo_root, "configs/experiments.yaml")
            )
        local_sources = resolve_output(settings.repo_root, "configs/local/sources.yaml")
        if local_sources.exists():
            configuration_paths.add(local_sources)
        # All checks precede writes; canonical paths and hard links are guarded.
        protected = [
            *paths.values(),
            *configuration_paths,
        ]
        if output.is_relative_to(settings.sources["raw_root"]):
            raise PipelineError("Evaluation report cannot overwrite raw sources")
        for source in protected:
            if output == source.resolve() or (
                output.exists() and source.exists() and output.samefile(source)
            ):
                raise PipelineError(
                    "Evaluation input/config aliases publication destination"
                )
        hashes = {name: _hash_file(path) for name, path in paths.items()}
        scores = pd.read_parquet(paths["scores"])
        partitions = pd.read_parquet(paths["partitions"])
        links = pd.read_parquet(paths["confirmed_event_segments"])
        roads = (
            gpd.read_parquet(paths["roads"])
            if paths["roads"].suffix.lower() == ".parquet"
            else gpd.read_file(paths["roads"])
        )
        report = evaluate_capture(
            scores,
            roads,
            links,
            partitions,
            settings.experiments,
            track_b_scores=pd.read_parquet(paths["track_b_scores"])
            if "track_b_scores" in paths
            else None,
        )
        if hashes != {name: _hash_file(path) for name, path in paths.items()}:
            raise PipelineError("Derived evaluation inputs changed during evaluation")
        snapshot = _snapshot(asdict(settings))
        report.update(
            run_id=run_id,
            input_paths=inputs,
            input_hashes=hashes,
            source_inputs_unchanged=True,
            score_scope="sample_only"
            if len(scores) < len(partitions)
            else "partition_population",
            partition_population_count=len(partitions),
            score_input_count=len(scores),
            road_input_count=len(roads),
            confirmed_source_link_count=len(links),
            config_snapshot=snapshot,
            config_hash=_hash_json(snapshot),
            git=_git_state(settings.repo_root),
            lock_sha256=_hash_file(settings.repo_root / "uv.lock")
            if (settings.repo_root / "uv.lock").exists()
            else None,
        )
        write_json_atomic(output, report)
        tracking_status = log_stage_report(
            run_id=run_id,
            stage="evaluation",
            config={
                "run_id": run_id,
                "evaluation": dict(RULES),
                "config_hash": report["config_hash"],
                "source_commit": report["git"]["commit"],
                "source_dirty": report["git"]["dirty"],
            },
            summary={
                "status": report["status"],
                "evaluation_performed": report["evaluation_performed"],
                **{
                    key: report["gate"][key]
                    for key in (
                        "reference_events",
                        "holdout_events",
                        "missing_reference_events",
                        "missing_holdout_events",
                        "missing_holdout_blocks",
                        "qualifying_holdout_blocks",
                    )
                },
            },
            metrics=_tracking_metrics(report),
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "evaluation_performed": report["evaluation_performed"],
                    "tracking_status": tracking_status,
                    "output": str(output),
                    "gate": report["gate"],
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
