"""Read-only streaming integrity checks with explicit, persisted failure audits.

Config identifies the assets to consume; the versioned manifest holds expected
hashes and provenance. Building a candidate manifest never publishes it.
"""

import copy
import hashlib
import re
from pathlib import Path

import pandas as pd
import yaml

from flood_ml.config import (
    PipelineError,
    Settings,
    _validate_run_id,
    resolve_input,
    resolve_output,
)
from flood_ml.utils.run import write_json_atomic

COLUMNS = [
    "dataset_id",
    "asset_id",
    "path",
    "bytes",
    "expected_sha256",
    "actual_sha256",
    "status",
]
METADATA = (
    "source_url",
    "version",
    "downloaded_on",
    "period",
    "license",
    "attribution",
    "crs",
    "resolution",
    "scope",
    "coverage_assumption",
)


def _inventory(settings: Settings):
    assets = settings.sources.get("assets")
    if not isinstance(assets, list):
        raise PipelineError("Source inventory requires assets list")
    if not assets:
        raise PipelineError("Source inventory is empty")
    ids, paths = set(), set()
    for asset in assets:
        if not isinstance(asset, dict) or any(
            not isinstance(asset.get(key), str) or not asset[key].strip()
            for key in ("dataset_id", "asset_id", "path")
        ):
            raise PipelineError("Each asset requires dataset_id, asset_id and path")
        path = resolve_input(Path(settings.sources["raw_root"]), asset["path"])
        if asset["asset_id"] in ids or path in paths:
            raise PipelineError("Duplicate source asset ID or path")
        ids.add(asset["asset_id"])
        paths.add(path)
    return assets


def _manifest(settings: Settings) -> dict:
    path = resolve_output(settings.repo_root, settings.sources.get("manifest"))
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PipelineError(f"Cannot read source manifest {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("manifest_version") != 2:
        raise PipelineError("Source manifest requires manifest_version 2")
    datasets = document.get("datasets")
    if not isinstance(datasets, list):
        raise PipelineError("Manifest datasets must be a list")
    ids, asset_ids, paths = set(), set(), set()
    for dataset in datasets:
        if not isinstance(dataset, dict) or not isinstance(
            dataset.get("dataset_id"), str
        ):
            raise PipelineError("Manifest dataset requires dataset_id")
        if dataset["dataset_id"] in ids:
            raise PipelineError("Duplicate manifest dataset ID")
        ids.add(dataset["dataset_id"])
        for key in METADATA:
            if not isinstance(dataset.get(key), str) or not dataset[key].strip():
                raise PipelineError(
                    f"Dataset {dataset['dataset_id']} requires {key}; record unknown with reason"
                )
            if dataset[key].strip().lower() == "unknown":
                raise PipelineError(f"Unknown {key} requires a reason")
        if not isinstance(dataset.get("assets"), list):
            raise PipelineError("Dataset requires assets list")
        for asset in dataset["assets"]:
            if not isinstance(asset, dict) or any(
                not isinstance(asset.get(key), str) or not asset[key].strip()
                for key in ("asset_id", "path")
            ):
                raise PipelineError("Manifest asset requires asset_id and path")
            path = resolve_input(Path(settings.sources["raw_root"]), asset["path"])
            if asset["asset_id"] in asset_ids or path in paths:
                raise PipelineError("Duplicate manifest asset ID or path")
            asset_ids.add(asset["asset_id"])
            paths.add(path)
            if type(asset.get("bytes")) is not int or asset["bytes"] < 0:
                raise PipelineError("Manifest asset requires nonnegative bytes")
            if not isinstance(asset.get("sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", asset["sha256"]
            ):
                raise PipelineError("Manifest asset requires lowercase SHA-256")
    return document


def _measure(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _audit_path(settings: Settings) -> Path:
    run_id = settings.experiments["run_id"]
    _validate_run_id(run_id)
    path = resolve_output(
        settings.repo_root,
        f"{settings.experiments['outputs']['reports']}/{run_id}/source_audit.json",
    )
    raw = Path(settings.sources["raw_root"]).resolve()
    if path.is_relative_to(raw) or raw.is_relative_to(path.parent):
        raise PipelineError("Source audit must not overlap raw_root")
    from flood_ml.utils.publication import preflight

    preflight(settings, [path])
    return path


def _audit_sources(settings: Settings) -> pd.DataFrame:
    """Persist all integrity results before returning, including failed rows.

    Audit bytes are observed bytes, with null size/hash for missing/unreadable
    files. Size mismatches take precedence over SHA mismatches. Structural errors
    persist a FAIL audit with an error field, without pretending to audit assets.
    """
    audit_path = _audit_path(settings)
    rows = []
    try:
        assets = _inventory(settings)
        document = _manifest(settings)
        expected = {
            asset["asset_id"]: dict(asset, dataset_id=dataset["dataset_id"])
            for dataset in document["datasets"]
            for asset in dataset["assets"]
        }
        actual_inventory = {(a["dataset_id"], a["asset_id"], a["path"]) for a in assets}
        manifest_inventory = {
            (a["dataset_id"], a["asset_id"], a["path"]) for a in expected.values()
        }
        if actual_inventory != manifest_inventory:
            raise PipelineError("Config and manifest inventory differ")
        for asset in assets:
            prior = expected[asset["asset_id"]]
            row = {key: asset[key] for key in ("dataset_id", "asset_id", "path")}
            row.update(
                bytes=None,
                expected_sha256=prior["sha256"],
                actual_sha256=None,
                status="PASS",
            )
            try:
                size, digest = _measure(
                    resolve_input(Path(settings.sources["raw_root"]), asset["path"])
                )
                row.update(bytes=size, actual_sha256=digest)
                if size != prior["bytes"]:
                    row["status"] = "FAIL_BYTES"
                elif digest != prior["sha256"]:
                    row["status"] = "FAIL_SHA256"
            except FileNotFoundError:
                row["status"] = "FAIL_MISSING"
            except OSError:
                row["status"] = "FAIL_UNREADABLE"
            rows.append(row)
    except PipelineError as exc:
        write_json_atomic(
            audit_path, {"status": "FAIL", "error": str(exc), "assets": rows}
        )
        raise
    failed = any(row["status"] != "PASS" for row in rows)
    write_json_atomic(
        audit_path, {"status": "FAIL" if failed else "PASS", "assets": rows}
    )
    return pd.DataFrame(rows, columns=COLUMNS)


def validate_sources(settings: Settings) -> pd.DataFrame:
    """Persist every result, then reject any failed source integrity check."""
    audit = _audit_sources(settings)
    if (audit["status"] != "PASS").any():
        raise PipelineError(f"Source integrity failed; inspect {_audit_path(settings)}")
    return audit


def build_manifest(settings: Settings) -> dict:
    """Return refreshed bytes/hash inventory; never change expected metadata on disk."""
    assets = _inventory(settings)
    document = copy.deepcopy(_manifest(settings))
    datasets = {d["dataset_id"]: d for d in document["datasets"]}
    prior = {a["asset_id"]: a for d in document["datasets"] for a in d["assets"]}
    for dataset in datasets.values():
        dataset["assets"] = []
    for asset in assets:
        if asset["dataset_id"] not in datasets:
            raise PipelineError(f"Missing dataset metadata: {asset['dataset_id']}")
        refreshed = dict(prior.get(asset["asset_id"], {}), **asset)
        refreshed.pop("dataset_id")
        try:
            refreshed["bytes"], refreshed["sha256"] = _measure(
                resolve_input(Path(settings.sources["raw_root"]), asset["path"])
            )
        except OSError as exc:
            raise PipelineError(
                f"Cannot refresh asset {asset['asset_id']}: {exc}"
            ) from exc
        datasets[asset["dataset_id"]]["assets"].append(refreshed)
    return document
