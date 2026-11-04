"""Read-only semantic verification of a published run against a local replay.

Binary hashes provide traceability and published-bundle integrity, never numeric
proof. The caller must replay in a separate checkout before invoking this module.
"""

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio

from flood_ml.config import (
    PipelineError,
    _validate_run_id,
    load_settings,
    resolve_output,
)
from flood_ml.utils.run_publisher import _hash_file, _read_json

_VOLATILE = {
    "fingerprint",
    "run_fingerprint",
    "stage_markers",
    "outputs",
    "reused",
    "process_peak_working_set_bytes",
    "config_hash",
    "config_sha256",
    "configuration_hashes",
    "output_hashes",
    "input_fingerprint",
    "stage_fingerprint",
    "runtime",
    "elapsed_seconds",
    "elapsed_s",
    "duration_seconds",
    "peak_memory_mb",
    "peak_memory_bytes",
    "machine",
    "platform",
    "python_version",
    "uv_version",
}


def _canonical(value, snapshot):
    """Normalize only configured runtime roots; retain all effective settings."""
    sources = snapshot.get("sources", {})
    labeling = snapshot.get("labeling", {})
    roots = []
    if sources.get("raw_root"):
        roots.append(
            (str(sources["raw_root"]).replace("\\", "/").rstrip("/"), "<RAW_ROOT>")
        )
    for key in ("match_reviews", "event_reviews"):
        path = labeling.get(key)
        if path:
            # Normalize the directory, retaining the review filename and content identity.
            parent = str(path).replace("\\", "/").rsplit("/", 1)
            if len(parent) == 2:
                roots.append((parent[0], "<REVIEW_ROOT>"))
    outputs = snapshot.get("experiments", {}).get("outputs", {})
    for key, path in outputs.items():
        roots.append((str(path).replace("\\", "/").rstrip("/"), f"<RUN_ROOT>/{key}"))
    roots.sort(key=lambda pair: len(pair[0]), reverse=True)

    def walk(item):
        if isinstance(item, dict):
            return {key: walk(val) for key, val in item.items()}
        if isinstance(item, list):
            return [walk(val) for val in item]
        if isinstance(item, str):
            normalized = item.replace("\\", "/")
            for root, replacement in roots:
                if normalized == root or normalized.startswith(root + "/"):
                    return replacement + normalized[len(root) :]
            return normalized
        return item

    return walk(value)


def _semantic_report(value):
    if isinstance(value, dict):
        return {
            key: _semantic_report(item)
            for key, item in value.items()
            if key not in _VOLATILE and not key.endswith(("_at_utc", "_timestamp"))
        }
    if isinstance(value, list):
        return [_semantic_report(item) for item in value]
    return value


def _compare_json(left, right, path, differences, atol, *, tolerant=True):
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            differences.append(
                {
                    "file": path,
                    "check": "json_keys",
                    "published": sorted(left),
                    "local": sorted(right),
                }
            )
        for key in left.keys() & right.keys():
            _compare_json(
                left[key],
                right[key],
                f"{path}.{key}",
                differences,
                atol,
                tolerant=tolerant,
            )
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append({"file": path, "check": "json_length"})
        for i, (a, b) in enumerate(zip(left, right)):
            _compare_json(a, b, f"{path}[{i}]", differences, atol, tolerant=tolerant)
    elif tolerant and type(left) in (int, float) and type(right) in (int, float):
        if not np.allclose(left, right, atol=atol, rtol=0):
            differences.append(
                {"file": path, "check": "numeric", "published": left, "local": right}
            )
    elif type(left) is not type(right) or left != right:
        differences.append(
            {"file": path, "check": "exact", "published": left, "local": right}
        )


def _geopackage_schema(path: Path, layer: str) -> list[dict]:
    """Read ordered declared layer columns, excluding SQLite/GPKG support tables.

    Include the layer's declared FID/geometry columns consistently with its user
    fields, but do not compare spatial indexes, SQLite internals, or row IDs.
    Bound layer names as values so quoted/non-ASCII identifiers remain safe.
    """
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        columns = connection.execute(
            'SELECT name, type, "notnull", pk FROM pragma_table_info(?) ORDER BY cid',
            (layer,),
        ).fetchall()
    return [
        {
            "name": name,
            "type": declared_type,
            "not_null": bool(not_null),
            "primary_key": primary_key,
        }
        for name, declared_type, not_null, primary_key in columns
    ]


def _keys(frame, name):
    if "track_b_scores" in name:
        return ["road_segment_id", "model_id"]
    if name in {
        "road_flood_matches.parquet",
        "review_decision_history.parquet",
        "confirmed_event_segments.parquet",
        "reviewed_group_partition_audit.parquet",
    }:
        return ["observation_id", "road_segment_id"]
    if "road_segment_id" in frame:
        if "weight_set" in frame:
            return ["road_segment_id", "weight_set"]
        return ["road_segment_id"]
    for key in (
        "observation_id",
        "event_group_id",
        "candidate_group_id",
        "block_id",
        "source_id",
    ):
        if key in frame:
            return [key]
    # Singleton AOI/context layers carry one geometry and may have no explicit ID.
    if len(frame) <= 1:
        return []
    raise PipelineError(f"No declared row key for table {name}")


def _compare_frame(left, right, name, differences, atol):
    def difference(check, **details):
        differences.append({"file": name, "check": check, **details})

    if list(left.columns) != list(right.columns) or list(map(str, left.dtypes)) != list(
        map(str, right.dtypes)
    ):
        difference(
            "schema",
            published={str(k): str(v) for k, v in left.dtypes.items()},
            local={str(k): str(v) for k, v in right.dtypes.items()},
        )
        return
    if getattr(left, "crs", None) != getattr(right, "crs", None):
        difference("crs")
    keys = _keys(left, Path(name.split(":")[0]).name)
    for side, frame in (("published", left), ("local", right)):
        if any(key not in frame for key in keys):
            difference("missing_row_key", side=side, keys=keys)
            return
        key_frame = frame[keys].copy()
        if (
            Path(name).name == "road_flood_matches.parquet"
            and "road_segment_id" in key_frame
        ):
            # An unmatched observation is one explicit candidate state, not a road.
            key_frame["road_segment_id"] = key_frame.road_segment_id.map(
                lambda value: ("missing",) if pd.isna(value) else ("road", value)
            )
        if keys and (key_frame.isna().any().any() or key_frame.duplicated().any()):
            difference("invalid_row_keys", side=side, keys=keys)
            return
    if len(left) != len(right):
        difference("row_count", published=len(left), local=len(right))
        return
    # Check submission order first; sorting below cannot erase this exact check.
    if keys and not left[keys].reset_index(drop=True).equals(
        right[keys].reset_index(drop=True)
    ):
        difference("row_key_order", keys=keys)
    if keys:
        left = left.sort_values(keys, kind="stable").reset_index(drop=True)
        right = right.sort_values(keys, kind="stable").reset_index(drop=True)
    else:
        left, right = left.reset_index(drop=True), right.reset_index(drop=True)
    for column in left.columns:
        a, b = left[column], right[column]
        if not a.isna().equals(b.isna()):
            difference("null_mask", column=column)
            continue
        present = a.notna()
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            # Labels, IDs and partition membership always remain exact.
            exact = column in keys or any(
                token in column.lower() for token in ("label", "partition", "eligible")
            )
            av, bv = a[present].to_numpy(dtype=float), b[present].to_numpy(dtype=float)
            matches = av == bv if exact else np.isclose(av, bv, atol=atol, rtol=0)
            if not matches.all():
                positions = np.flatnonzero(present.to_numpy())[~matches][:5]
                difference(
                    "exact" if exact else "numeric",
                    column=column,
                    rows=[
                        {
                            "row": int(i),
                            "key": {k: str(left.iloc[i][k]) for k in keys},
                            "published": str(a.iloc[i]),
                            "local": str(b.iloc[i]),
                        }
                        for i in positions
                    ],
                )
        elif str(a.dtype) == "geometry":
            if (
                not a[present]
                .to_wkb()
                .reset_index(drop=True)
                .equals(b[present].to_wkb().reset_index(drop=True))
            ):
                difference("geometry", column=column)
        else:
            # Arrow list-valued cells need structural equality rather than array truth.
            av = (
                a[present]
                .map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                .tolist()
            )
            bv = (
                b[present]
                .map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                .tolist()
            )
            if av != bv:
                difference("exact", column=column)


def compare_runs(
    published_run: Path,
    local_repo_root: Path,
    run_id: str,
    *,
    config: Path = Path("configs/experiments.yaml"),
    atol: float = 1e-5,
) -> dict:
    """Compare existing outputs without modifying either run or launching replay."""
    _validate_run_id(run_id)
    if not math.isfinite(atol) or atol < 0:
        raise PipelineError("atol must be finite and nonnegative")
    published_run, local_repo_root = (
        Path(published_run).resolve(),
        Path(local_repo_root).resolve(),
    )
    if published_run.is_relative_to(local_repo_root) or local_repo_root.is_relative_to(
        published_run
    ):
        raise PipelineError(
            "Published bundle and local replay require separate, non-overlapping checkouts"
        )
    settings = load_settings(config, repo_root=local_repo_root)
    report_relative = f"{settings.experiments['outputs']['reports']}/{run_id}"
    manifest = _read_json(published_run / "completion_manifest.json")
    differences, hashes = [], {}
    records = {
        side: {
            name: _read_json(root / report_relative / f"{name}.json")
            for name in (
                "run",
                "pipeline_complete",
                "pipeline_report",
                "evaluation",
                "track_b",
            )
        }
        for side, root in (("published", published_run), ("local", local_repo_root))
    }
    provenance = {}
    gates = {}
    for side, reports in records.items():
        run = reports["run"]
        snapshot = run.get("config_snapshot")
        required = ("lock_sha256", "input_hashes", "code_sha256", "manifest_sha256")
        if (
            not isinstance(snapshot, dict)
            or not snapshot
            or any(not run.get(key) for key in required)
            or not (run.get("git") or {}).get("commit")
        ):
            differences.append(
                {"file": "run.json", "check": "required_provenance", "side": side}
            )
        configured_reviews = (snapshot or {}).get("labeling", {})
        if any(
            configured_reviews.get(key) for key in ("match_reviews", "event_reviews")
        ) and not run.get("review_hashes"):
            differences.append(
                {"file": "run.json", "check": "missing_review_hashes", "side": side}
            )
        provenance[side] = {
            "run_id": run.get("run_id"),
            "source_commit": (run.get("git") or {}).get("commit"),
            **{
                key: run.get(key)
                for key in (
                    "lock_sha256",
                    "manifest_sha256",
                    "input_hashes",
                    "review_hashes",
                    "code_sha256",
                )
            },
            "effective_config": _canonical(snapshot or {}, snapshot or {}),
        }
        gates[side] = {
            "pipeline": reports["pipeline_report"].get("pipeline_status"),
            "evaluation": reports["evaluation"].get("status"),
            "track_b": reports["track_b"].get("status"),
        }
        if (
            run.get("run_id") != run_id
            or reports["pipeline_complete"].get("run_id") != run_id
            or reports["pipeline_complete"].get("status") != "complete"
            or gates[side]["pipeline"] != "complete"
        ):
            differences.append(
                {"file": "run.json", "check": "incomplete_or_wrong_run", "side": side}
            )
        if not all(gates[side].values()):
            differences.append(
                {"file": "reports", "check": "missing_gate_status", "side": side}
            )
    _compare_json(
        provenance["published"],
        provenance["local"],
        "provenance",
        differences,
        atol,
        tolerant=False,
    )
    _compare_json(
        gates["published"],
        gates["local"],
        "gate_statuses",
        differences,
        atol,
        tolerant=False,
    )
    if manifest.get("status") != "published" or manifest.get("run_id") != run_id:
        differences.append(
            {"file": "completion_manifest.json", "check": "invalid_manifest"}
        )
    for key, value in {
        "source_commit": provenance["published"]["source_commit"],
        "lock_sha256": provenance["published"]["lock_sha256"],
        "input_hashes": provenance["published"]["input_hashes"],
        "review_hashes": provenance["published"]["review_hashes"],
        "gate_statuses": gates["published"],
    }.items():
        if manifest.get(key) != value:
            differences.append(
                {
                    "file": "completion_manifest.json",
                    "check": "manifest_provenance",
                    "field": key,
                }
            )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise PipelineError(
            "Published completion manifest requires a nonempty file map"
        )
    for name in (*records["published"], "runtime"):
        relative = f"{report_relative}/{name}.json"
        if relative not in files:
            differences.append({"file": relative, "check": "missing_manifest_entry"})
    allowed_roots = [
        f"{base}/{run_id}/" for base in settings.experiments["outputs"].values()
    ]
    for relative, expected in files.items():
        if not any(relative.startswith(prefix) for prefix in allowed_roots):
            raise PipelineError(f"Manifest file is outside selected run: {relative}")
        published, local = (
            resolve_output(published_run, relative),
            resolve_output(local_repo_root, relative),
        )
        if not published.is_file() or not local.is_file():
            differences.append(
                {
                    "file": relative,
                    "check": "missing_file",
                    "published": published.is_file(),
                    "local": local.is_file(),
                }
            )
            continue
        hashes[relative] = {
            "published": _hash_file(published),
            "local": _hash_file(local),
        }
        if (
            expected.get("sha256") != hashes[relative]["published"]
            or expected.get("size") != published.stat().st_size
        ):
            differences.append({"file": relative, "check": "published_integrity"})
        try:
            if published.suffix == ".parquet":
                schema_a, schema_b = pq.read_schema(published), pq.read_schema(local)

                def logical_schema(schema):
                    metadata = json.loads((schema.metadata or {}).get(b"pandas", b"{}"))
                    indexes = {
                        item
                        for item in metadata.get("index_columns", [])
                        if isinstance(item, str)
                    }
                    return [
                        (field.name, str(field.type), field.nullable)
                        for field in schema
                        if field.name not in indexes
                    ]

                if logical_schema(schema_a) != logical_schema(schema_b):
                    differences.append({"file": relative, "check": "arrow_schema"})
                geo = b"geo" in (schema_a.metadata or {})
                reader = gpd.read_parquet if geo else pd.read_parquet
                _compare_frame(
                    reader(published), reader(local), relative, differences, atol
                )
            elif published.suffix == ".gpkg":
                layers_a, layers_b = (
                    pyogrio.list_layers(published),
                    pyogrio.list_layers(local),
                )
                if layers_a.tolist() != layers_b.tolist():
                    differences.append({"file": relative, "check": "layers"})
                for layer in set(layers_a[:, 0]) & set(layers_b[:, 0]):
                    schema_a = _geopackage_schema(published, layer)
                    schema_b = _geopackage_schema(local, layer)
                    if schema_a != schema_b:
                        differences.append(
                            {
                                "file": f"{relative}:{layer}",
                                "check": "geopackage_schema",
                                "published": schema_a,
                                "local": schema_b,
                            }
                        )
                    _compare_frame(
                        gpd.read_file(published, layer=layer, engine="pyogrio"),
                        gpd.read_file(local, layer=layer, engine="pyogrio"),
                        f"{relative}:{layer}",
                        differences,
                        atol,
                    )
            elif published.suffix == ".json" and published.name not in {
                "run.json",
                "runtime.json",
                "local_verification.json",
            }:
                _compare_json(
                    _semantic_report(_read_json(published)),
                    _semantic_report(_read_json(local)),
                    relative,
                    differences,
                    atol,
                )
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
            differences.append(
                {
                    "file": relative,
                    "check": "unreadable_table_or_report",
                    "error": str(exc),
                }
            )
    # Detect additional local semantic artifacts, including scores absent in a pending bundle.
    for prefix in allowed_roots:
        if prefix.startswith(
            settings.experiments["outputs"]["interim"] + "/"
        ) and not manifest.get("publish_interim"):
            continue
        directory = local_repo_root / prefix
        for path in directory.rglob("*"):
            relative = path.relative_to(local_repo_root).as_posix()
            if (
                path.suffix in {".parquet", ".gpkg", ".joblib"}
                and relative not in files
            ):
                differences.append({"file": relative, "check": "extra_local_artifact"})
    for side, reports in records.items():
        track = reports["track_b"]
        if track.get("status") == "ready":
            root = published_run if side == "published" else local_repo_root
            models = [
                draw.get("model_artifact")
                for draw in track.get("draws", [])
                if draw.get("status") == "fitted"
            ]
            if (
                not models
                or any(
                    not model or not resolve_output(root, model).is_file()
                    for model in models
                )
                or not any(
                    "track_b_scores" in name and (root / name).is_file()
                    for name in files
                )
            ):
                differences.append(
                    {
                        "file": "track_b.json",
                        "check": "missing_ready_artifacts",
                        "side": side,
                    }
                )
        elif track.get("status") not in {
            "pending_positive_review",
            "background_pool_insufficient",
        }:
            differences.append(
                {"file": "track_b.json", "check": "invalid_fit_status", "side": side}
            )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "equivalent": not differences,
        "atol": atol,
        "rtol": 0,
        "gate_statuses": gates,
        "provenance": provenance,
        "file_hashes": hashes,
        "differences": differences,
    }
