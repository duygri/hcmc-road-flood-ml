"""Validate configured assets; refresh expected checksums only on explicit request."""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import yaml

from flood_ml.config import (
    PipelineError,
    _validate_run_id,
    load_settings,
    resolve_input,
    resolve_output,
)
from flood_ml.ingestion.sources import build_manifest, validate_sources
from flood_ml.utils.run import start_run


def _refresh(settings):
    document = build_manifest(settings)
    path = resolve_output(settings.repo_root, settings.sources["manifest"])
    if path in {
        resolve_input(Path(settings.sources["raw_root"]), asset["path"])
        for asset in settings.sources["assets"]
    }:
        raise PipelineError("Manifest cannot replace a source asset")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise PipelineError(f"Cannot refresh manifest: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sources.yaml")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Explicitly replace expected bytes and SHA-256 from local assets",
    )
    args = parser.parse_args(argv)
    settings = None
    validation_started = False
    try:
        settings = load_settings(
            Path(args.config), repo_root=Path(__file__).resolve().parents[1]
        )
        if args.run_id:
            settings = replace(
                settings, experiments=dict(settings.experiments, run_id=args.run_id)
            )
        _validate_run_id(settings.experiments["run_id"])
        # Validation writes the audit even when files are missing; start_run hashes
        # inputs and therefore must happen after validation, not before it.
        if args.refresh_manifest:
            _refresh(settings)
        validation_started = True
        audit = validate_sources(settings)
        for row in audit.to_dict("records"):
            print(
                f"{row['status']} {row['dataset_id']}/{row['asset_id']} {row['path']}"
            )
        context = start_run(settings, settings.experiments["run_id"])
        print(f"Audit: {context.reports_dir / 'source_audit.json'}")
        return 0
    except PipelineError as exc:
        if settings is not None and validation_started:
            # Only print this invocation's persisted validation result.
            from flood_ml.ingestion.sources import _audit_path

            try:
                document = json.loads(_audit_path(settings).read_text(encoding="utf-8"))
                for row in document["assets"]:
                    print(
                        f"{row['status']} {row['dataset_id']}/{row['asset_id']} {row['path']}"
                    )
            except (PipelineError, OSError, ValueError, KeyError):
                pass
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
