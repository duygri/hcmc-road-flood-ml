"""Build an unconfirmed matching queue and evidence-backed human-reviewed labels."""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_ml.config import PipelineError, load_settings, resolve_input, resolve_output
from flood_ml.gis.study_area import select_aoi
from flood_ml.labeling.ird import read_ird
from flood_ml.labeling.matching import match_candidates
from flood_ml.labeling.observations import normalize_observations, propose_event_groups
from flood_ml.labeling.reviews import (
    EVENT_COLUMNS,
    MATCH_COLUMNS,
    apply_reviews,
    read_reviews,
)
from flood_ml.utils.run import _hash_file, start_run, write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/labeling.yaml"))
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--roads",
        required=True,
        help="Repository-relative current algorithm-version-4 roads GPKG or GeoParquet",
    )
    parser.add_argument(
        "--match-reviews",
        help="Repository-relative human review CSV; omitted means no decisions",
    )
    parser.add_argument(
        "--event-reviews",
        help="Repository-relative event membership CSV; omitted means no confirmations",
    )
    args = parser.parse_args(argv)
    run = None
    try:
        settings = load_settings(args.config, repo_root=args.repo_root)
        inputs = {"roads": args.roads}
        if args.match_reviews:
            inputs["match_reviews"] = args.match_reviews
        if args.event_reviews:
            inputs["event_reviews"] = args.event_reviews
        paths = {k: resolve_output(settings.repo_root, v) for k, v in inputs.items()}
        hashes = {k: _hash_file(v) for k, v in paths.items()}
        labeling = dict(
            settings.labeling,
            review_inputs=inputs,
            review_input_hashes=hashes,
        )
        if args.match_reviews:
            labeling["match_reviews"] = args.match_reviews
        if args.event_reviews:
            labeling["event_reviews"] = args.event_reviews
        settings = replace(
            settings,
            labeling=labeling,
            experiments=dict(
                settings.experiments,
                run_id=args.run_id or settings.experiments["run_id"],
            ),
        )
        run = start_run(settings, settings.experiments["run_id"])
        write_json_atomic(run.reports_dir / "labels_qc.json", {"status": "started"})
        assets = settings.sources["assets"]
        manifest_path = resolve_output(settings.repo_root, settings.sources["manifest"])
        if manifest_path.is_file():
            from flood_ml.pipeline import run_pipeline

            run_pipeline(settings, run, stages=["observations"])
            observations = gpd.read_file(run.interim_dir / "flood_observations.gpkg")
            groups = pd.read_parquet(
                run.processed_dir / "unconfirmed_candidate_groups.parquet"
            )
        else:
            # Legacy offline CLI fixtures can omit the manifest. Production runs
            # with the tracked manifest share the orchestrated AOI/observation stage.
            def asset(asset_id):
                chosen = [a for a in assets if a["asset_id"] == asset_id]
                if len(chosen) != 1:
                    raise PipelineError(f"Exactly one {asset_id} asset required")
                return resolve_input(settings.sources["raw_root"], chosen[0]["path"])

            selector = settings.study_area["boundary_selection"]
            aoi = select_aoi(
                gpd.read_file(asset("administrative_boundaries")),
                field=selector["field"],
                value=selector["value"],
                target_crs=settings.study_area["target_crs"],
            )
            observations = normalize_observations(
                read_ird(asset("flood_observations")), aoi
            )
            groups = propose_event_groups(observations, settings.labeling)
        roads = (
            gpd.read_parquet(paths["roads"])
            if paths["roads"].suffix == ".parquet"
            else gpd.read_file(paths["roads"])
        )
        candidates = match_candidates(observations, roads, settings.labeling)
        mr = (
            read_reviews(paths["match_reviews"], MATCH_COLUMNS)
            if "match_reviews" in paths
            else pd.DataFrame(columns=MATCH_COLUMNS)
        )
        er = (
            read_reviews(paths["event_reviews"], EVENT_COLUMNS)
            if "event_reviews" in paths
            else pd.DataFrame(columns=EVENT_COLUMNS)
        )
        labels, links = apply_reviews(roads, observations, candidates, mr, er)
        record = json.loads((run.reports_dir / "run.json").read_text())
        after = {
            a["path"]: _hash_file(
                resolve_input(settings.sources["raw_root"], a["path"])
            )
            for a in assets
        }
        if after != record["input_hashes"] or hashes != {
            k: _hash_file(p) for k, p in paths.items()
        }:
            raise PipelineError("Input bytes changed during labeling")
        outputs = {
            "flood_observations": observations,
            "unconfirmed_candidate_groups": groups,
            "road_flood_matches": candidates,
            "review_decision_history": mr,
            "event_review_history": er,
            "road_labels": labels,
            "confirmed_event_segments": links,
        }
        with tempfile.TemporaryDirectory(
            prefix=".labels-", dir=run.processed_dir
        ) as directory:
            for name, frame in outputs.items():
                frame.to_parquet(Path(directory) / f"{name}.parquet", index=False)
            for name in outputs:
                os.replace(
                    Path(directory) / f"{name}.parquet",
                    run.processed_dir / f"{name}.parquet",
                )
        qc = {
            "status": "complete",
            "evaluation_status": "pending_label_review",
            "run_id": run.run_id,
            "observations": len(observations),
            "outside_aoi": int(observations.outside_aoi.sum()),
            "roads": len(roads),
            "candidates": int(candidates.road_segment_id.notna().sum()),
            "queue_rows": len(candidates),
            "all_candidates_require_review": bool(candidates.review_required.all()),
            "label_counts": labels.label_status.value_counts().to_dict(),
            "confirmed_event_segments": len(links),
            "match_review_count": len(mr),
            "event_review_count": len(er),
            "candidate_distance_m": settings.labeling["candidate_distance_m"],
            "numeric_location_accuracy_count": int(
                observations.location_accuracy_m.notna().sum()
            ),
            "source_hashes_before": record["input_hashes"],
            "source_hashes_after": after,
            "input_hashes": hashes,
            "source_bytes_unchanged": True,
        }
        write_json_atomic(run.reports_dir / "labels_qc.json", qc)
        print(json.dumps(qc, indent=2))
        return 0
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        if run:
            write_json_atomic(
                run.reports_dir / "labels_qc.json",
                {"status": "failed", "error": str(exc)},
            )
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
