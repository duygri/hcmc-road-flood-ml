"""Atomic JSON and per-run reproducibility records (not stage completion)."""

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from flood_ml.config import (
    PipelineError,
    Settings,
    _validate_run_id,
    resolve_input,
    resolve_output,
)


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    run_id: str
    seed: int
    config_hash: str
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    artifacts_dir: Path


def write_json_atomic(path: Path, payload: dict) -> None:
    """Publish a complete JSON document by same-directory atomic replacement."""
    path = Path(path)
    temporary = None
    try:
        # Serialize first so invalid JSON cannot alter even the directory tree.
        content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        raise PipelineError(f"Cannot atomically write {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _hash_file(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise PipelineError(f"Cannot hash input {path}: {exc}") from exc


def _code_hash(repo_root: Path) -> str:
    """Fingerprint runnable source so old stage outputs cannot mask code edits."""
    digest = hashlib.sha256()
    code_files = [
        *Path(repo_root, "src/flood_ml").rglob("*.py"),
        *Path(repo_root, "scripts").rglob("*.py"),
    ]
    code_files.extend(
        path
        for path in (
            Path(repo_root, ".python-version"),
            Path(repo_root, "pyproject.toml"),
        )
        if path.is_file()
    )
    for path in sorted(set(code_files)):
        if not path.is_file():
            continue
        try:
            digest.update(path.relative_to(repo_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        except OSError as exc:
            raise PipelineError(f"Cannot fingerprint code file {path}: {exc}") from exc
    return digest.hexdigest()


def _snapshot(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    return value


def _hash_json(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _git_state(repo_root: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def start_run(settings: Settings, run_id: str) -> RunContext:
    """Create provenance or reuse an identical fingerprint; never mark completion.

    Hashes are calculated from the configured source assets, not directory scans.
    Task 2 will populate assets when the downloaded inventory is standardized.
    Concurrent writers to the same run_id are rejected by an exclusive lock.
    """
    _validate_run_id(run_id)
    root = settings.repo_root.resolve()
    snapshot = _snapshot(
        {
            name: getattr(settings, name)
            for name in (
                "sources",
                "study_area",
                "features",
                "labeling",
                "experiments",
                "seed",
            )
        }
    )
    config_hash = _hash_json(snapshot)
    raw_root = Path(settings.sources["raw_root"]).resolve()
    inputs = {
        asset["path"]: _hash_file(resolve_input(raw_root, asset["path"]))
        for asset in settings.sources["assets"]
    }
    manifest_path = (
        resolve_output(root, settings.sources["manifest"])
        if settings.sources.get("manifest")
        else None
    )
    # Standalone legacy stage CLIs can be exercised with isolated raw fixtures
    # that intentionally omit a manifest. A full pipeline still requires and
    # validates it before processing; when present, its metadata bytes are part
    # of the run fingerprint.
    manifest_hash = (
        _hash_file(manifest_path)
        if manifest_path is not None and manifest_path.is_file()
        else None
    )
    configuration_hashes = {str(p): _hash_file(p) for p in settings.configuration_paths}
    review_hashes = {
        key: _hash_file(resolve_output(root, settings.labeling[key]))
        for key in ("match_reviews", "event_reviews")
        if settings.labeling.get(key)
    }
    lock_hash = _hash_file(resolve_output(root, "uv.lock"))
    code_hash = _code_hash(root)
    fingerprint = _hash_json(
        {
            "config_hash": config_hash,
            "lock_sha256": lock_hash,
            "input_hashes": inputs,
            "manifest_sha256": manifest_hash,
            "configuration_hashes": configuration_hashes,
            "review_hashes": review_hashes,
            "code_sha256": code_hash,
        }
    )
    dirs = {}
    for name, base in settings.experiments["outputs"].items():
        directory = resolve_output(root, f"{base}/{run_id}")
        if directory.is_relative_to(raw_root) or raw_root.is_relative_to(directory):
            raise PipelineError("Run output must not overlap raw_root")
        dirs[f"{name}_dir"] = directory
    context = RunContext(
        repo_root=root,
        run_id=run_id,
        seed=settings.seed,
        config_hash=config_hash,
        **dirs,
    )
    manifest_path = context.reports_dir / "run.json"
    lock_path = context.reports_dir / ".start.lock"
    from flood_ml.utils.publication import preflight, run_destinations

    preflight(settings, [manifest_path, lock_path, *run_destinations(settings, run_id)])
    owns_lock = False
    try:
        context.reports_dir.mkdir(parents=True, exist_ok=True)
        with lock_path.open("x", encoding="utf-8"):
            owns_lock = True
            if manifest_path.exists():
                try:
                    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise PipelineError(
                        f"Invalid existing run record: {manifest_path}"
                    ) from exc
                if (
                    not isinstance(previous, dict)
                    or previous.get("fingerprint") != fingerprint
                ):
                    raise PipelineError("Run fingerprint changed; choose a new run_id")
            else:
                for directory in dirs.values():
                    directory.mkdir(parents=True, exist_ok=True)
                write_json_atomic(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "status": "started",
                        "seed": settings.seed,
                        "config_hash": config_hash,
                        "fingerprint": fingerprint,
                        "config_snapshot": snapshot,
                        "started_at_utc": datetime.now(UTC).isoformat(),
                        "git": _git_state(root),
                        "lock_sha256": lock_hash,
                        "input_hashes": inputs,
                        "manifest_sha256": manifest_hash,
                        "configuration_hashes": configuration_hashes,
                        "review_hashes": review_hashes,
                        "code_sha256": code_hash,
                    },
                )
    except FileExistsError as exc:
        raise PipelineError(
            f"Run is being initialized (or stale lock requires review): {lock_path}"
        ) from exc
    except OSError as exc:
        raise PipelineError(f"Cannot initialize run {run_id}: {exc}") from exc
    finally:
        # Do not remove somebody else's lock when exclusive creation failed.
        if owns_lock:
            lock_path.unlink(missing_ok=True)
    return context
