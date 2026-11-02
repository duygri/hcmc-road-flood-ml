"""Source validation uses tiny files with independently known checksums."""

import copy
import json
from dataclasses import replace

import pytest
import yaml

from flood_ml.config import PipelineError, Settings
from flood_ml.ingestion.sources import build_manifest, validate_sources

ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.fixture
def configured_sources(tmp_path):
    raw = tmp_path / "inputs"
    raw.mkdir()
    (raw / "abc.bin").write_bytes(b"abc")
    asset = {"dataset_id": "D03", "asset_id": "tile_1", "path": "abc.bin"}
    dataset = {
        key: "unknown: synthetic fixture"
        for key in (
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
    }
    dataset.update(dataset_id="D03", assets=[dict(asset, bytes=3, sha256=ABC)])
    manifest = {"manifest_version": 2, "datasets": [dataset]}
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return Settings(
        tmp_path,
        {"raw_root": raw, "manifest": "manifest.yaml", "assets": [asset]},
        {},
        {},
        {},
        {"run_id": "audit_001", "outputs": {"reports": "reports"}},
        42,
    )


def audit_path(settings):
    return settings.repo_root / "reports/audit_001/source_audit.json"


def test_published_checksum_is_validated(configured_sources):
    audit = validate_sources(configured_sources)
    assert audit["status"].tolist() == ["PASS"]
    assert audit["actual_sha256"].tolist() == [ABC]
    assert audit["expected_sha256"].tolist() == [ABC]
    assert audit["bytes"].tolist() == [3]
    assert (
        json.loads(audit_path(configured_sources).read_text())["assets"][0]["status"]
        == "PASS"
    )


@pytest.mark.parametrize(
    "mutation,status",
    [("missing", "FAIL_MISSING"), ("hash", "FAIL_SHA256"), ("size", "FAIL_BYTES")],
)
def test_failure_is_audited_before_raising(configured_sources, mutation, status):
    source = configured_sources.sources["raw_root"] / "abc.bin"
    if mutation == "missing":
        source.unlink()
    elif mutation == "hash":
        source.write_bytes(b"abd")
    else:
        source.write_bytes(b"abcd")
    manifest = configured_sources.repo_root / "manifest.yaml"
    expected_before = manifest.read_bytes()
    with pytest.raises(PipelineError):
        validate_sources(configured_sources)
    saved = json.loads(audit_path(configured_sources).read_text())
    assert saved["assets"][0]["status"] == status
    assert manifest.read_bytes() == expected_before


@pytest.mark.parametrize("duplicate", ["id", "path"])
def test_duplicate_assets_are_rejected(configured_sources, duplicate):
    sources = copy.deepcopy(configured_sources.sources)
    second = dict(sources["assets"][0])
    if duplicate == "id":
        second["path"] = "other.bin"
    else:
        second["asset_id"] = "other"
    sources["assets"].append(second)
    with pytest.raises(PipelineError, match="Duplicate"):
        validate_sources(replace(configured_sources, sources=sources))
    assert json.loads(audit_path(configured_sources).read_text())["status"] == "FAIL"


def test_d03_multiple_tiles_are_preserved_and_hashed(configured_sources):
    sources = copy.deepcopy(configured_sources.sources)
    (sources["raw_root"] / "tile2.tif").write_bytes(b"abc")
    sources["assets"].append(
        {"dataset_id": "D03", "asset_id": "tile_2", "path": "tile2.tif"}
    )
    settings = replace(configured_sources, sources=sources)
    original = (settings.repo_root / "manifest.yaml").read_bytes()
    manifest = build_manifest(settings)
    assert [a["asset_id"] for a in manifest["datasets"][0]["assets"]] == [
        "tile_1",
        "tile_2",
    ]
    assert [a["sha256"] for a in manifest["datasets"][0]["assets"]] == [ABC, ABC]
    assert (settings.repo_root / "manifest.yaml").read_bytes() == original
    (settings.repo_root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )
    assert validate_sources(settings)["status"].tolist() == ["PASS", "PASS"]


def test_manifest_and_config_must_describe_same_assets(configured_sources):
    sources = copy.deepcopy(configured_sources.sources)
    sources["assets"][0]["path"] = "unlisted.bin"
    with pytest.raises(PipelineError, match="inventory"):
        validate_sources(replace(configured_sources, sources=sources))


def test_empty_inventory_cannot_report_success(configured_sources):
    manifest = configured_sources.repo_root / "manifest.yaml"
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    document["datasets"][0]["assets"] = []
    manifest.write_text(yaml.safe_dump(document), encoding="utf-8")
    settings = replace(
        configured_sources, sources=dict(configured_sources.sources, assets=[])
    )
    with pytest.raises(PipelineError, match="empty"):
        validate_sources(settings)
    assert json.loads(audit_path(settings).read_text())["status"] == "FAIL"


def test_cli_validates_and_only_refreshes_on_explicit_flag(
    configured_sources, monkeypatch, capsys
):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/validate_sources.py"
    spec = importlib.util.spec_from_file_location("validate_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = configured_sources.repo_root
    monkeypatch.setattr(module, "__file__", str(root / "scripts/validate_sources.py"))
    (root / "configs").mkdir()
    sections = {
        "sources": dict(configured_sources.sources, raw_root="inputs"),
        "study_area": {"target_crs": "EPSG:9210"},
        "features": {},
        "labeling": {"positive_requires_review": True},
        "experiments": {
            "seed": 42,
            "run_id": "audit_001",
            "outputs": {
                "reports": "reports",
                "interim": "interim",
                "processed": "processed",
                "artifacts": "artifacts",
            },
            "configs": {
                name: f"configs/{name}.yaml"
                for name in ("sources", "study_area", "features", "labeling")
            },
        },
    }
    for name, section in sections.items():
        (root / f"configs/{name}.yaml").write_text(
            yaml.safe_dump({name: section}), encoding="utf-8"
        )
    (root / "uv.lock").write_bytes(b"test lock")
    assert module.main(["--config", "configs/sources.yaml"]) == 0
    assert "PASS D03/tile_1" in capsys.readouterr().out
    (root / "inputs/abc.bin").write_bytes(b"abd")
    original = (root / "manifest.yaml").read_bytes()
    assert module.main(["--config", "configs/sources.yaml", "--run-id", "changed"]) == 1
    assert "FAIL_SHA256 D03/tile_1" in capsys.readouterr().out
    assert (root / "manifest.yaml").read_bytes() == original
    assert (
        module.main(
            [
                "--config",
                "configs/sources.yaml",
                "--run-id",
                "refreshed",
                "--refresh-manifest",
            ]
        )
        == 0
    )
    assert (root / "inputs/abc.bin").read_bytes() == b"abd"
    assert (root / "manifest.yaml").read_bytes() != original
    before_invalid = (root / "manifest.yaml").read_bytes()
    (root / "inputs/abc.bin").write_bytes(b"bad")
    assert module.main(["--run-id", "../invalid", "--refresh-manifest"]) == 1
    assert (root / "manifest.yaml").read_bytes() == before_invalid
