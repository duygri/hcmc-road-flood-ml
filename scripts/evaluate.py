"""Write exploratory reviewed-event capture or an explicit pending review report."""

import argparse
import json
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
from flood_ml.evaluation.capture import evaluate_capture
from flood_ml.utils.run import (
    _git_state,
    _hash_file,
    _hash_json,
    _snapshot,
    write_json_atomic,
)


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
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "evaluation_performed": report["evaluation_performed"],
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
