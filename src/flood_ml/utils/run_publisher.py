"""Safely copy completed run outputs to a mounted Drive directory."""

import hashlib
import json
import shutil
from pathlib import Path

from flood_ml.config import (
    PipelineError,
    _validate_run_id,
    load_settings,
    resolve_output,
)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read required run report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"Expected a JSON object in {path}")
    return value


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _transient(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".tmp", ".temp", ".lock", ".swp", "~")) or name.startswith(
        (".start.lock", ".pipeline.lock")
    )


def _manifest_required_files(root: Path, track_b: dict) -> set[Path]:
    required = set()
    if track_b.get("status") == "ready":
        output_hashes = track_b.get("output_hashes") or {}
        paths = set(output_hashes)
        for draw in track_b.get("draws", []):
            if draw.get("status") == "fitted" and draw.get("model_artifact"):
                paths.add(draw["model_artifact"])
        score_paths = [p for p in paths if "track_b_scores" in p]
        if not score_paths or not any(p for p in paths if p.endswith(".joblib")):
            raise PipelineError(
                "Ready Track B report must declare its score and model artifacts"
            )
        for value in paths:
            required.add(resolve_output(root, value))
    elif track_b.get("status") not in {
        "pending_positive_review",
        "background_pool_insufficient",
    }:
        raise PipelineError(
            f"Track B status is not publishable: {track_b.get('status')!r}"
        )
    return required


def publish_run(
    repo_root: Path,
    drive_root: Path,
    run_id: str,
    *,
    config: Path = Path("configs/experiments.yaml"),
    publish_interim: bool = False,
) -> Path:
    """Publish one complete local run and write its manifest only after verification."""
    _validate_run_id(run_id)
    repo_root = Path(repo_root).resolve()
    drive_root = Path(drive_root).resolve()
    try:
        settings = load_settings(config, repo_root=repo_root)
    except (OSError, ValueError) as exc:
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(f"Cannot load run publication config: {exc}") from exc
    outputs = settings.experiments["outputs"]
    source_roots = {
        name: resolve_output(repo_root, f"{base}/{run_id}")
        for name, base in outputs.items()
        if name in {"processed", "artifacts", "reports", "interim"}
        and (name != "interim" or publish_interim)
    }
    report_root = source_roots["reports"]
    required_reports = {
        "run": report_root / "run.json",
        "completion": report_root / "pipeline_complete.json",
        "pipeline": report_root / "pipeline_report.json",
        "evaluation": report_root / "evaluation.json",
        "track_b": report_root / "track_b.json",
        "runtime": report_root / "runtime.json",
    }
    reports = {key: _read_json(path) for key, path in required_reports.items()}
    run_record = reports["run"]
    completion = reports["completion"]
    pipeline, evaluation, track_b, runtime = (
        reports["pipeline"],
        reports["evaluation"],
        reports["track_b"],
        reports["runtime"],
    )
    python_version = runtime.get("python_version", runtime.get("python"))
    uv_version = runtime.get("uv_version", runtime.get("uv"))
    runtime_platform = runtime.get("platform")
    if not all(
        isinstance(value, str) and value
        for value in (python_version, uv_version, runtime_platform)
    ):
        raise PipelineError("Runtime report must declare Python, uv, and platform")
    if completion.get("status") != "complete" or completion.get("run_id") != run_id:
        raise PipelineError(
            "Pipeline completion marker is missing, incomplete, or for another run"
        )
    if run_record.get("run_id") != run_id:
        raise PipelineError("Run provenance is for another run")
    if pipeline.get("pipeline_status") != "complete":
        raise PipelineError("Pipeline report does not declare a complete run")
    if not evaluation.get("status") or not pipeline.get("evaluation_status"):
        raise PipelineError("Evaluation gate status is missing")
    extra_required = _manifest_required_files(repo_root, track_b)
    files = []
    for source_root in source_roots.values():
        if not source_root.is_dir():
            raise PipelineError(
                f"Required run output directory is missing: {source_root}"
            )
        for path in sorted(source_root.rglob("*")):
            if path.is_file() and not _transient(path):
                if path.is_symlink() or not path.resolve().is_relative_to(source_root):
                    raise PipelineError(f"Run output contains a linked file: {path}")
                files.append(path)
    absent = [
        path
        for path in [*required_reports.values(), *extra_required]
        if not path.is_file()
    ]
    if absent:
        raise PipelineError(f"Required run output is missing: {absent[0]}")
    for path in extra_required:
        if path not in files:
            files.append(path)
    selected_size = sum(path.stat().st_size for path in files)
    try:
        local_free = shutil.disk_usage(repo_root).free
        if selected_size > local_free:
            raise PipelineError(
                "Insufficient local free space to stage selected run outputs"
            )
    except OSError:
        pass
    target = drive_root / "runs" / run_id
    if target.exists():
        raise PipelineError(f"Drive run target already exists: {target}")
    try:
        drive_root.mkdir(parents=True, exist_ok=True)
        drive_free = shutil.disk_usage(drive_root).free
        if selected_size > drive_free:
            raise PipelineError(
                "Drive reports insufficient free space for selected run outputs"
            )
    except OSError:
        # Mounted Drive quota reporting is not consistently available or authoritative.
        pass
    try:
        target.mkdir(parents=True, exist_ok=False)
        copied = {}
        for source in files:
            relative = source.relative_to(repo_root)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected = {"size": source.stat().st_size, "sha256": _hash_file(source)}
            shutil.copy2(source, destination)
            copied[relative.as_posix()] = expected
        for relative, record in copied.items():
            destination = target / Path(relative)
            if (
                destination.stat().st_size != record["size"]
                or _hash_file(destination) != record["sha256"]
            ):
                raise PipelineError(f"Copied file hash verification failed: {relative}")
        manifest = {
            "schema_version": 1,
            "status": "published",
            "run_id": run_id,
            "source_commit": run_record.get("git", {}).get("commit"),
            "source_commit_sha": run_record.get("git", {}).get("commit"),
            "runtime": {
                "python": python_version,
                "uv": uv_version,
                "platform": runtime_platform,
            },
            "lock_sha256": run_record.get("lock_sha256"),
            "config_sha256": run_record.get("config_hash"),
            "configuration_hashes": run_record.get("configuration_hashes", {}),
            "input_hashes": run_record.get("input_hashes", {}),
            "review_hashes": run_record.get("review_hashes", {}),
            "gate_statuses": {
                "pipeline": pipeline.get("pipeline_status"),
                "evaluation": evaluation.get("status"),
                "track_b": track_b.get("status"),
            },
            "publish_interim": publish_interim,
            "files": copied,
        }
        manifest_path = target / "completion_manifest.json"
        temporary_manifest = target / ".completion_manifest.json.tmp"
        with temporary_manifest.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
        temporary_manifest.replace(manifest_path)
    except PipelineError:
        raise
    except (OSError, shutil.Error) as exc:
        raise PipelineError(f"Drive publication failed: {exc}") from exc
    return target
