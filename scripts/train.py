"""Train the reviewed-reference Track B pilot and score eligible holdout roads."""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_ml.baseline.mcda import _values_and_reasons
from flood_ml.config import (
    PipelineError,
    _validate_run_id,
    load_settings,
    resolve_output,
)
from flood_ml.training.presence_background import fit_track_b, score_track_b
from flood_ml.training.sampling import FEATURE_COLUMNS, build_background_draws
from flood_ml.utils.publication import preflight
from flood_ml.utils.run import _hash_file, _hash_json, start_run, write_json_atomic


def _draw_settings(rules):
    return [
        (f"track_b_r{ratio}_s{seed}", ratio, seed)
        for ratio in (
            rules["primary_background_ratio"],
            *rules["sensitivity_background_ratios"],
        )
        for seed in range(
            rules["seed_start"], rules["seed_start"] + rules["seed_repetitions"]
        )
    ]


def _preflight_publications(settings, run_id, inputs, rules):
    """Protect derived inputs, configs, and provenance before start_run writes."""
    _validate_run_id(run_id)
    directories = {
        name: resolve_output(settings.repo_root, f"{base}/{run_id}")
        for name, base in settings.experiments["outputs"].items()
    }
    models_dir = directories["artifacts"] / "track_b_models"
    publications = {
        "scores": directories["processed"] / "track_b_scores.parquet",
        "report": directories["reports"] / "track_b.json",
        "run": directories["reports"] / "run.json",
        "lock": directories["reports"] / ".start.lock",
        **{
            model_id: models_dir / f"{model_id}.joblib"
            for model_id, _, _ in _draw_settings(rules)
        },
    }
    publications = {
        name: resolve_output(
            settings.repo_root, str(path.relative_to(settings.repo_root))
        )
        for name, path in publications.items()
    }
    preflight(
        settings,
        [*publications.values(), *directories.values(), models_dir],
        inputs=inputs.values(),
    )
    return publications


def _temporary_for(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        return Path(stream.name)


def _publish(staged, report_path, payload):
    """Publish completed files, restoring previous files if publication fails."""
    backups, published = {}, []
    try:
        for temporary, destination in staged:
            if destination.exists():
                backup = _temporary_for(destination)
                try:
                    os.replace(destination, backup)
                except OSError:
                    backup.unlink(missing_ok=True)
                    raise
                backups[destination] = backup
            os.replace(temporary, destination)
            published.append(destination)
        write_json_atomic(report_path, payload)
    except (OSError, ValueError, TypeError):
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    else:
        for backup in backups.values():
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                # Publication is committed; leftover backups cannot undo it.
                pass


def _draw_reports(sampling, rules):
    available = {draw.model_id: draw for draw in sampling.draws}
    failures = {failure["model_id"]: failure for failure in sampling.failures}
    rows = []
    for model_id, ratio, seed in _draw_settings(rules):
        row = {"model_id": model_id, "background_ratio": ratio, "seed": seed}
        if model_id in failures:
            row.update(failures[model_id], status="background_pool_insufficient")
        elif model_id in available:
            draw = available[model_id]
            row.update(
                draw.manifest,
                status="sampled",
                positive_count=int((draw.y == 1).sum()),
                background_count=int((draw.y == 0).sum()),
            )
        else:
            row["status"] = f"not_fitted_{sampling.status}"
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-id")
    for name in ("features", "labels", "confirmed-event-segments", "partitions"):
        parser.add_argument(
            f"--{name}", required=True, help="Repository-relative Parquet input"
        )
    args = parser.parse_args(argv)
    run, staged = None, []
    try:
        settings = load_settings(args.config, repo_root=args.repo_root)
        rules = settings.experiments.get("track_b")
        if not isinstance(rules, dict):
            raise PipelineError("experiments.track_b configuration is required")
        inputs = {
            name: getattr(args, name)
            for name in ("features", "labels", "confirmed_event_segments", "partitions")
        }
        paths = {
            name: resolve_output(settings.repo_root, value)
            for name, value in inputs.items()
        }
        run_id = args.run_id or settings.experiments["run_id"]
        publications = _preflight_publications(settings, run_id, paths, rules)
        hashes = {name: _hash_file(path) for name, path in paths.items()}
        settings = replace(
            settings,
            experiments={
                **settings.experiments,
                "run_id": run_id,
                "track_b_inputs": inputs,
                "track_b_input_hashes": hashes,
            },
        )
        run = start_run(settings, run_id)
        write_json_atomic(
            publications["report"], {"status": "started", "run_id": run_id}
        )
        features, labels, events, partitions = (
            pd.read_parquet(paths[name])
            for name in ("features", "labels", "confirmed_event_segments", "partitions")
        )
        if "partition" in features:
            raise PipelineError(
                "Features must be separate from the frozen partition input"
            )
        if not set(FEATURE_COLUMNS).issubset(features.columns):
            raise PipelineError("Feature table is missing approved Track B features")
        if (
            "label_status" not in labels
            or not labels.label_status.isin(
                ["positive", "unknown", "verified_negative"]
            ).all()
        ):
            raise PipelineError("Labels require valid label_status values")
        # Apply the canonical structural/context policy before building any draw.
        values, reasons = _values_and_reasons(features)
        prepared = features.copy()
        prepared[FEATURE_COLUMNS] = values
        prepared["score_eligible"] = reasons.eq("")
        sampling = build_background_draws(prepared, labels, events, partitions, rules)
        provenance = json.loads(publications["run"].read_text(encoding="utf-8"))
        payload = {
            "schema_version": 1,
            "status": sampling.status,
            "run_id": run_id,
            "method": "sampled_presence_background_logistic_regression",
            "score_semantics": "relative_rank_score_not_probability",
            "evaluation_status": "exploratory_pending_capture_evaluation",
            "ranking_scope": "eligible_holdout_feature_input_roads_only",
            "config": rules,
            "config_hash": run.config_hash,
            "feature_columns_sha256": _hash_json(FEATURE_COLUMNS),
            "input_paths": inputs,
            "input_hashes": hashes,
            "partition_sha256": hashes["partitions"],
            "git": provenance["git"],
            "lock_sha256": provenance["lock_sha256"],
            "code_sha256": provenance["code_sha256"],
            "run_fingerprint": provenance["fingerprint"],
            "model_versions": {
                package: version(package)
                for package in ("scikit-learn", "numpy", "pandas", "joblib")
            },
            "counts": {
                "feature_input": len(features),
                "labels": labels.label_status.value_counts().to_dict(),
                "partitions": partitions.partition.value_counts().to_dict(),
            },
            "gate": sampling.gate,
            "failures": list(sampling.failures),
            "draws": _draw_reports(sampling, rules),
            "output_hashes": {},
        }
        if sampling.status == "ready":
            holdout = prepared.merge(
                partitions[["road_segment_id", "partition"]],
                on="road_segment_id",
                validate="one_to_one",
                sort=False,
            )
            holdout = holdout.loc[holdout.partition.eq("holdout")]
            fitted = [
                fit_track_b(draw, **rules["logistic_regression"])
                for draw in sampling.draws
            ]
            scores = score_track_b(fitted, holdout)
            score_temp = _temporary_for(publications["scores"])
            staged.append((score_temp, publications["scores"]))
            scores.to_parquet(score_temp, index=False)
            fitted_by_id = {model.model_id: model for model in fitted}
            for model in fitted:
                destination = publications[model.model_id]
                temporary = _temporary_for(destination)
                staged.append((temporary, destination))
                joblib.dump(model.estimator, temporary)
            for row in payload["draws"]:
                if row["model_id"] in fitted_by_id:
                    model = fitted_by_id[row["model_id"]]
                    row.update(model.manifest, status="fitted")
                    row["model_artifact"] = str(
                        publications[model.model_id].relative_to(settings.repo_root)
                    )
            payload["counts"].update(
                fitted_models=len(fitted),
                scored_rows=len(scores),
                scored_holdout_roads=int(scores.road_segment_id.nunique()),
            )
            payload["output_hashes"] = {
                str(destination.relative_to(settings.repo_root)): _hash_file(temporary)
                for temporary, destination in staged
            }
        if {name: _hash_file(path) for name, path in paths.items()} != hashes:
            raise PipelineError("Derived source inputs changed during Track B run")
        payload["source_inputs_unchanged"] = True
        _publish(staged, publications["report"], payload)
        print(json.dumps({"status": payload["status"], "run_id": run_id}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if run is not None:
            try:
                write_json_atomic(
                    publications["report"],
                    {"status": "failed", "run_id": run.run_id, "error": str(exc)},
                )
            except (OSError, ValueError, TypeError):
                pass
        print(f"Track B training failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
