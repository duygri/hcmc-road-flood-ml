"""Prepare topology-based road segments and per-run QC without altering raw."""

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import yaml
from shapely import covers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_ml.config import PipelineError, load_settings, resolve_input, resolve_output
from flood_ml.gis.roads import _active, segment_roads
from flood_ml.gis.study_area import context_radius_m, make_context, select_aoi
from flood_ml.ingestion.osm import read_osm_roads
from flood_ml.utils.run import _hash_file, start_run, write_json_atomic


def _publish_gpkg(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".roads-", dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        output = frame.copy()
        for column in ("node_ids", "tags"):
            if column in output:
                output[column] = output[column].map(
                    lambda value: json.dumps(value, sort_keys=True)
                )
        output.to_file(temporary, driver="GPKG", layer=path.stem, index=False)
        os.replace(temporary, path)


def _summary(frame):
    return {
        "count": len(frame),
        "by_highway": {
            str(k): int(v) for k, v in frame.highway_type.value_counts().items()
        },
        "bridge": int(frame.bridge.map(_active).sum()),
        "tunnel": int(frame.tunnel.map(_active).sum()),
        "invalid_geometry": int((~frame.geometry.is_valid).sum()),
    }


def _peak_memory_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in [
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                ]
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if query(process(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/study_area.yaml"))
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--snapshot-date", help="ISO OSM snapshot date; defaults to source manifest"
    )
    args = parser.parse_args(argv)
    run = None
    started = time.perf_counter()
    try:
        settings = load_settings(args.config, repo_root=args.repo_root)
        if args.run_id:
            settings = replace(
                settings, experiments=dict(settings.experiments, run_id=args.run_id)
            )
        raw_root = settings.sources["raw_root"]
        assets = settings.sources["assets"]
        roads = [asset for asset in assets if asset["dataset_id"] == "D02"]
        boundaries = [
            asset
            for asset in assets
            if asset["asset_id"] == "administrative_boundaries"
        ]
        if len(roads) != 1 or len(boundaries) != 1:
            raise PipelineError(
                "Exactly one D02 OSM asset and administrative boundary required"
            )
        snapshot = args.snapshot_date
        if snapshot is None:
            manifest_path = resolve_output(
                settings.repo_root, settings.sources["manifest"]
            )
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            datasets = [
                dataset
                for dataset in manifest["datasets"]
                if dataset["dataset_id"] == "D02"
            ]
            if len(datasets) != 1 or not datasets[0].get("snapshot_date"):
                raise PipelineError(
                    "OSM snapshot date must be explicit or declared in manifest"
                )
            snapshot = datasets[0]["snapshot_date"]
        # CLI overrides belong to provenance/fingerprint, not just the output.
        study = dict(settings.study_area, osm_snapshot_date=snapshot)
        settings = replace(settings, study_area=study)
        run = start_run(settings, settings.experiments["run_id"])
        qc_path = run.reports_dir / "roads_qc.json"
        write_json_atomic(qc_path, {"status": "started", "run_id": run.run_id})
        selector = study["boundary_selection"]
        aoi = select_aoi(
            gpd.read_file(resolve_input(raw_root, boundaries[0]["path"])),
            field=selector["field"],
            value=selector["value"],
            target_crs=study["target_crs"],
        )
        aoi["study_area_id"] = study["study_area_id"]
        radius = context_radius_m(
            settings.features, raster_stencil_m=study["context"]["raster_stencil_m"]
        )
        context = make_context(aoi, radius)
        ways = read_osm_roads(
            resolve_input(raw_root, roads[0]["path"]),
            context,
            study["roads"]["allowed_highways"],
        )
        segments = segment_roads(
            ways,
            aoi,
            snapshot_date=snapshot,
            target_m=study["segment_target_m"],
            maximum_m=study["segment_max_m"],
        )
        stats = _summary(segments)
        # segment_roads prepared this exact polygon. The vast majority of
        # segments are fully covered; only boundary roundoff candidates need
        # the expensive overlay to apply the 1 micrometre outside tolerance.
        uncovered = segments.loc[~covers(aoi.geometry.iloc[0], segments.geometry.array)]
        stats.update(
            shorter_than_100m=int((segments.length_m < 100).sum()),
            shorter_than_target=int(
                (segments.length_m < study["segment_target_m"]).sum()
            ),
            min_length_m=float(segments.length_m.min()) if len(segments) else None,
            max_length_m=float(segments.length_m.max()) if len(segments) else None,
            total_length_m=float(segments.length_m.sum()),
            unique_ids=bool(segments.road_segment_id.is_unique),
            outside_aoi=int(
                (
                    uncovered.geometry.difference(aoi.geometry.iloc[0]).length > 1e-6
                ).sum()
            ),
        )
        if stats["invalid_geometry"] or stats["outside_aoi"] or not stats["unique_ids"]:
            raise PipelineError("Road geometry/identity QC failed")
        run_record = json.loads(
            (run.reports_dir / "run.json").read_text(encoding="utf-8")
        )
        after = {
            asset["path"]: _hash_file(resolve_input(raw_root, asset["path"]))
            for asset in assets
        }
        if after != run_record["input_hashes"]:
            raise PipelineError("Source bytes changed during road preparation")
        _publish_gpkg(aoi, run.interim_dir / "study_area.gpkg")
        _publish_gpkg(segments, run.interim_dir / "road_segments.gpkg")
        qc = {
            "status": "complete",
            "run_id": run.run_id,
            "snapshot_date": snapshot,
            "geometry_crs": aoi.crs.to_string(),
            "context_radius_m": radius,
            "read_stats": ways.attrs["read_stats"],
            "context_ways": _summary(ways),
            "aoi_ways": _summary(segments.drop_duplicates("osm_way_id")),
            "segments": stats,
            "segmentation_config_digest": segments.attrs["segmentation_config_digest"],
            "source_hashes_before": run_record["input_hashes"],
            "source_hashes_after": after,
            "source_bytes_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_memory_bytes(),
            "count_note": "Final segments clipped to actual AOI; context-way and historic bbox inspection counts are different populations.",
        }
        write_json_atomic(qc_path, qc)
        print(json.dumps(qc, indent=2))
        return 0
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        if run is not None:
            write_json_atomic(
                run.reports_dir / "roads_qc.json",
                {"status": "failed", "error": str(exc)},
            )
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
