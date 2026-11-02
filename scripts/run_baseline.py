"""Score explicit precomputed features using a frozen spatial reference partition."""

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

from flood_ml.baseline.mcda import (
    FEATURE_SPEC,
    TOP_K_FRACTIONS,
    compare_weight_sets,
    fit_normalizer,
    score_mcda,
)
from flood_ml.config import PipelineError, load_settings, resolve_output
from flood_ml.gis.study_area import require_metric_crs
from flood_ml.labeling.matching import require_ids
from flood_ml.utils.run import (
    _hash_file,
    _hash_json,
    start_run,
    write_json_atomic,
)


def _rules(config):
    if not isinstance(config, dict):
        raise PipelineError("experiments.mcda rules are required")
    expected_sets = {
        "equal": dict.fromkeys(FEATURE_SPEC, 0.2),
        "terrain-heavy": dict(zip(FEATURE_SPEC, [0.15, 0.4, 0.15, 0.15, 0.15])),
        "water-heavy": dict(zip(FEATURE_SPEC, [0.15, 0.15, 0.15, 0.4, 0.15])),
        "builtup-heavy": dict(zip(FEATURE_SPEC, [0.15, 0.15, 0.15, 0.15, 0.4])),
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
        raise PipelineError(
            "MCDA configuration must match the predeclared rules/weights"
        )
    return {**config, "feature_spec": dict(FEATURE_SPEC)}


def _parquet_atomic(frame, path):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".parquet", delete=False
        ) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--features", required=True, help="Repository-relative feature parquet"
    )
    parser.add_argument(
        "--partitions",
        required=True,
        help="Repository-relative frozen Task 9 partition parquet",
    )
    parser.add_argument(
        "--roads",
        required=True,
        help="Repository-relative metric road GeoParquet/GPKG for lengths",
    )
    args = parser.parse_args(argv)
    run = None
    try:
        settings = load_settings(args.config, repo_root=args.repo_root)
        rules = _rules(settings.experiments.get("mcda"))
        inputs = {
            name: getattr(args, name) for name in ("features", "partitions", "roads")
        }
        paths = {
            name: resolve_output(settings.repo_root, value)
            for name, value in inputs.items()
        }
        hashes = {name: _hash_file(path) for name, path in paths.items()}
        settings = replace(
            settings,
            experiments={
                **settings.experiments,
                "run_id": args.run_id or settings.experiments["run_id"],
                "baseline_inputs": inputs,
                "baseline_input_hashes": hashes,
            },
        )
        run = start_run(settings, settings.experiments["run_id"])
        qc_path = run.reports_dir / "baseline_qc.json"
        write_json_atomic(qc_path, {"status": "started", "run_id": run.run_id})
        features = pd.read_parquet(paths["features"])
        partitions = pd.read_parquet(paths["partitions"])
        require_ids(features, "road_segment_id")
        require_ids(partitions, "road_segment_id")
        if (
            "partition" not in partitions
            or not partitions.partition.isin(
                ["reference", "holdout", "guard", "excluded"]
            ).all()
        ):
            raise PipelineError(
                "Frozen partition table requires valid partition values"
            )
        if not set(features.road_segment_id).issubset(set(partitions.road_segment_id)):
            raise PipelineError("Feature IDs absent from frozen partition population")
        if "partition" in features:
            raise PipelineError("Features must be separate from partition input")
        columns = ["road_segment_id", "partition"]
        if "reference_fit_eligible" in partitions:
            columns.append("reference_fit_eligible")
        joined = features.merge(
            partitions[columns],
            on="road_segment_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        roads = (
            gpd.read_parquet(paths["roads"])
            if paths["roads"].suffix.lower() == ".parquet"
            else gpd.read_file(paths["roads"])
        )
        require_ids(roads, "road_segment_id")
        require_metric_crs(roads)
        if (
            roads.geometry.isna().any()
            or roads.geometry.is_empty.any()
            or not roads.geometry.is_valid.all()
            or not roads.geom_type.eq("LineString").all()
        ):
            raise PipelineError("Road lengths require valid nonempty LineStrings")
        length_table = pd.DataFrame(
            {
                "road_segment_id": roads.road_segment_id,
                "road_length_m": roads.geometry.length.to_numpy(),
            }
        )
        structures = [column for column in ("bridge", "tunnel") if column in roads]
        for column in structures:
            length_table[column] = roads[column].to_numpy()
        # Use geometry lengths consistently even if upstream carries a covariate.
        joined = joined.drop(
            columns=["road_length_m", *structures], errors="ignore"
        ).merge(
            length_table,
            on="road_segment_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        reference = joined.loc[joined.partition.eq("reference")]
        normalizer = fit_normalizer(reference, rules["feature_spec"])
        normalizer.update(
            config_hash=run.config_hash,
            rules_sha256=_hash_json(rules),
            input_hashes=hashes,
        )
        primary = score_mcda(
            joined, normalizer, rules["weight_sets"][rules["primary_weight_set"]]
        )
        sensitivity = compare_weight_sets(joined, normalizer, rules["weight_sets"])
        scope = (
            "sample_only" if len(features) < len(partitions) else "partition_population"
        )
        sensitivity.update(
            scope=scope,
            ranking_scope="eligible_feature_input_roads_only",
            evaluation_status="pending_label_review",
        )
        all_scores = pd.concat(
            [
                score_mcda(joined, normalizer, weights).assign(weight_set=name)
                for name, weights in rules["weight_sets"].items()
            ],
            ignore_index=True,
        )
        outputs = [
            run.processed_dir / "road_susceptibility.parquet",
            run.processed_dir / "sensitivity_scores.parquet",
            run.artifacts_dir / "normalizer.json",
            run.artifacts_dir / "mcda_rules.json",
            run.reports_dir / "weight_sensitivity.json",
        ]
        _parquet_atomic(primary, outputs[0])
        _parquet_atomic(all_scores, outputs[1])
        write_json_atomic(outputs[2], normalizer)
        write_json_atomic(
            outputs[3],
            {
                "rules": rules,
                "rules_sha256": _hash_json(rules),
                "config_hash": run.config_hash,
            },
        )
        write_json_atomic(outputs[4], sensitivity)
        if {name: _hash_file(path) for name, path in paths.items()} != hashes:
            raise PipelineError("Derived source inputs changed during baseline run")
        qc = {
            "status": "complete",
            "run_id": run.run_id,
            "scope": scope,
            "limitation": "Precomputed feature input only; no population feature completeness claim or label evaluation.",
            "evaluation_status": "pending_label_review",
            "label_evaluation_performed": False,
            "partition_population_count": len(partitions),
            "feature_input_count": len(features),
            "sample_partition_counts": joined.partition.value_counts().to_dict(),
            "sample_reference_count": len(reference),
            "normalizer_fit_count": normalizer["fit_row_count"],
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
            "constant_reference_features": [
                f
                for f, rule in normalizer["features"].items()
                if rule["constant_reference"]
            ],
            "ranking_scope": "eligible_feature_input_roads_only",
            "input_hashes": hashes,
            "config_hash": run.config_hash,
            "rules_sha256": _hash_json(rules),
            "source_inputs_unchanged": True,
            "output_hashes": {
                str(path.relative_to(settings.repo_root)): _hash_file(path)
                for path in outputs
            },
        }
        write_json_atomic(qc_path, qc)
        print(json.dumps(qc, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if run is not None:
            write_json_atomic(
                run.reports_dir / "baseline_qc.json",
                {"status": "failed", "error": str(exc)},
            )
        print(f"Baseline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
