"""Build water features and assemble one-to-one with precomputed raster inputs."""

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_ml.config import PipelineError, load_settings, resolve_input, resolve_output
from flood_ml.features.assembly import assemble_features
from flood_ml.features.water import compute_water_features
from flood_ml.gis.study_area import select_aoi
from flood_ml.utils.run import _hash_file, start_run, write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/features.yaml"))
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--roads", required=True, help="Repository-relative roads parquet or GPKG"
    )
    parser.add_argument(
        "--raster-features", required=True, help="Repository-relative Task 6 parquet"
    )
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args(argv)
    run = None
    started = time.perf_counter()
    try:
        if args.chunk_size <= 0:
            raise PipelineError("chunk-size must be positive")
        settings = load_settings(args.config, repo_root=args.repo_root)
        inputs = {"roads": args.roads, "raster_features": args.raster_features}
        paths = {
            key: resolve_output(settings.repo_root, value)
            for key, value in inputs.items()
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise PipelineError(
                f"Missing {', '.join(missing)} prerequisite; run scripts/run_pipeline.py first"
            )
        input_hashes = {key: _hash_file(path) for key, path in paths.items()}
        settings = replace(
            settings,
            features=dict(
                settings.features,
                assembly_inputs=inputs,
                assembly_input_hashes=input_hashes,
                chunk_size=args.chunk_size,
            ),
            experiments=dict(
                settings.experiments,
                run_id=args.run_id or settings.experiments["run_id"],
            ),
        )
        coverage = settings.sources.get("water_coverage", {})
        if coverage.get("coverage_domain") != "old_hcmc_aoi" or not coverage.get(
            "coverage_assumption"
        ):
            raise PipelineError(
                "Explicit old_hcmc_aoi source-domain coverage assumption required"
            )
        assets = settings.sources["assets"]

        def asset_path(asset_id):
            selected = [a for a in assets if a["asset_id"] == asset_id]
            if len(selected) != 1:
                raise PipelineError(f"Exactly one {asset_id} asset required")
            return resolve_input(settings.sources["raw_root"], selected[0]["path"])

        run = start_run(settings, settings.experiments["run_id"])
        qc_path = run.reports_dir / "features_qc.json"
        write_json_atomic(qc_path, {"status": "started"})
        selector = settings.study_area["boundary_selection"]
        domain = select_aoi(
            gpd.read_file(asset_path(coverage["boundary_asset_id"])),
            field=selector["field"],
            value=selector["value"],
            target_crs=settings.study_area["target_crs"],
        )
        water = gpd.read_file(asset_path("water"))
        roads = (
            gpd.read_parquet(paths["roads"])
            if paths["roads"].suffix == ".parquet"
            else gpd.read_file(paths["roads"])
        )
        raster = pd.read_parquet(paths["raster_features"])
        chunks = []
        metadata = None
        for offset in range(0, max(len(roads), 1), args.chunk_size):
            chunk = compute_water_features(
                roads.iloc[offset : offset + args.chunk_size],
                water,
                domain,
                dict(
                    settings.features,
                    coverage_assumption=coverage["coverage_assumption"],
                ),
            )
            metadata = chunk.attrs["water_metadata"]
            chunks.append(chunk)
        water_features = pd.concat(chunks, ignore_index=True)
        result = assemble_features(
            roads, raster, water_features, settings.features["feature_columns"]
        )
        record = json.loads((run.reports_dir / "run.json").read_text())
        after = {
            a["path"]: _hash_file(
                resolve_input(settings.sources["raw_root"], a["path"])
            )
            for a in assets
        }
        if after != record["input_hashes"] or input_hashes != {
            key: _hash_file(path) for key, path in paths.items()
        }:
            raise PipelineError("Source bytes changed during feature assembly")
        with tempfile.TemporaryDirectory(
            prefix=".features-", dir=run.processed_dir
        ) as directory:
            temporary = Path(directory) / "road_features.parquet"
            result.to_parquet(temporary, index=False)
            os.replace(temporary, run.processed_dir / temporary.name)
        qc = {
            "status": "complete",
            "run_id": run.run_id,
            "roads": len(result),
            "unique_ids": bool(result.road_segment_id.is_unique),
            "feature_columns": settings.features["feature_columns"],
            "missing": {
                c: int(result[c].isna().sum())
                for c in settings.features["feature_columns"]
            },
            "water_context_incomplete": int(result.water_context_incomplete.sum()),
            "water_distance_incomplete": int(result.water_distance_incomplete.sum()),
            "score_excluded_structures": int((~result.score_eligible).sum()),
            "coverage_assumption": coverage["coverage_assumption"],
            "water_metadata": metadata,
            "source_hashes_before": record["input_hashes"],
            "source_hashes_after": after,
            "assembly_input_hashes": input_hashes,
            "source_bytes_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json_atomic(qc_path, qc)
        print(json.dumps(qc, indent=2))
        return 0
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        if run:
            write_json_atomic(
                run.reports_dir / "features_qc.json",
                {"status": "failed", "error": str(exc)},
            )
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
