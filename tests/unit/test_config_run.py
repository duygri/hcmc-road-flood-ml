"""Behavioral tests use only synthetic local files, never research sources."""

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from flood_ml.config import PipelineError, load_settings, resolve_input, resolve_output
from flood_ml.utils import run
from flood_ml.utils.run import start_run, write_json_atomic


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "uv.lock").write_bytes(b"synthetic lock\n")
    sections = {
        "sources": {"raw_root": "data/raw", "assets": []},
        "study_area": {"target_crs": "EPSG:9210"},
        "features": {"neighborhood_radius_m": 500},
        "labeling": {"candidate_distance_m": 100, "positive_requires_review": True},
        "experiments": {
            "seed": 42,
            "run_id": "fixture_001",
            "configs": {
                name: f"configs/{name}.yaml"
                for name in ("sources", "study_area", "features", "labeling")
            },
            "outputs": {
                "interim": "data/interim",
                "processed": "data/processed",
                "reports": "reports/runs",
                "artifacts": "artifacts",
            },
        },
    }
    for name, value in sections.items():
        (root / "configs" / f"{name}.yaml").write_text(
            yaml.safe_dump({name: value}), encoding="utf-8"
        )
    return root


def edit_section(repo, section, edit):
    path = repo / "configs" / f"{section}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data[section])
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_config_paths_are_independent_of_cwd(repo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(Path("configs/features.yaml"), repo_root=repo)
    assert settings.sources["raw_root"] == repo / "data/raw"
    assert settings.seed == 42
    assert (
        start_run(settings, "fixture_001").processed_dir
        == repo / "data/processed/fixture_001"
    )


@pytest.mark.parametrize("value", [None, True, "42", -1, 2**32])
def test_invalid_seed_rejected(repo, value):
    edit_section(repo, "experiments", lambda s: s.update(seed=value))
    with pytest.raises(PipelineError):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


def test_missing_seed_rejected(repo):
    edit_section(repo, "experiments", lambda s: s.pop("seed"))
    with pytest.raises(PipelineError):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


def test_missing_section_rejected(repo):
    (repo / "configs/features.yaml").write_text("{}", encoding="utf-8")
    with pytest.raises(PipelineError):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


def test_positive_review_requirement_accepts_true(repo):
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    assert settings.labeling["positive_requires_review"] is True


@pytest.mark.parametrize("value", [False, "false", 0, 1, None, "true"])
def test_positive_review_requirement_cannot_be_disabled_or_malformed(repo, value):
    edit_section(repo, "labeling", lambda s: s.update(positive_requires_review=value))
    with pytest.raises(PipelineError, match="positive_requires_review"):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


def test_positive_review_requirement_must_be_explicit(repo):
    edit_section(repo, "labeling", lambda s: s.pop("positive_requires_review"))
    with pytest.raises(PipelineError, match="positive_requires_review"):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


@pytest.mark.parametrize("value", [-5, True, "500", float("nan")])
def test_invalid_radius_rejected(repo, value):
    edit_section(repo, "features", lambda s: s.update(neighborhood_radius_m=value))
    with pytest.raises(PipelineError):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


def test_section_run_id_must_match_shared_run(repo):
    edit_section(repo, "features", lambda s: s.update(run_id="different_run"))
    with pytest.raises(PipelineError):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


def test_absolute_raw_root_requires_local_override(repo, tmp_path):
    edit_section(repo, "sources", lambda s: s.update(raw_root=str(tmp_path)))
    with pytest.raises(PipelineError):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


def test_unsafe_yaml_rejected(repo):
    (repo / "configs/features.yaml").write_text(
        "features: !!python/object/apply:builtins.eval ['1 + 1']", encoding="utf-8"
    )
    with pytest.raises(PipelineError):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


def test_missing_input_does_not_create_run(repo):
    edit_section(repo, "sources", lambda s: s.update(assets=[{"path": "missing.bin"}]))
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    with pytest.raises(PipelineError):
        start_run(settings, "fixture_001")
    assert not (repo / "reports/runs/fixture_001").exists()


def directory_link(link, target):
    # Junctions need no Windows symlink privilege and exercise the same boundary.
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def test_input_symlink_escape_rejected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "tiny.bin").write_bytes(b"abc")
    directory_link(raw / "link", outside)
    with pytest.raises(PipelineError):
        resolve_input(raw, "link/tiny.bin")


def test_output_symlink_escape_rejected(tmp_path):
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    directory_link(root / "link", outside)
    with pytest.raises(PipelineError):
        resolve_output(root, "link/result.json")


def test_output_cannot_escape_workspace(tmp_path):
    with pytest.raises(PipelineError):
        resolve_output(tmp_path, "../outside")


@pytest.mark.parametrize(
    "path",
    [
        "/outside",
        "C:/outside",
        "C:outside",
        "\\\\host\\share",
        "data/../../outside",
        "data/x:stream",
    ],
)
def test_absolute_or_unsafe_output_rejected(tmp_path, path):
    with pytest.raises(PipelineError):
        resolve_output(tmp_path, path)


def test_output_resolves_under_repo(tmp_path):
    assert resolve_output(tmp_path, "data/x") == tmp_path / "data/x"


def test_config_output_traversal_rejected(repo):
    edit_section(
        repo, "experiments", lambda s: s["outputs"].update(reports="../outside")
    )
    with pytest.raises(PipelineError):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


def test_input_traversal_rejected(tmp_path):
    with pytest.raises(PipelineError):
        resolve_input(tmp_path, "../outside")


def test_external_raw_root_is_explicit_and_read_only(repo, tmp_path):
    raw = tmp_path / "external"
    raw.mkdir()
    asset = raw / "tiny.bin"
    asset.write_bytes(b"abc")
    local = repo / "configs/local"
    local.mkdir()
    (local / "sources.yaml").write_text(
        yaml.safe_dump({"sources": {"raw_root": str(raw)}})
    )
    edit_section(repo, "sources", lambda s: s.update(assets=[{"path": "tiny.bin"}]))
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    assert resolve_input(settings.sources["raw_root"], "tiny.bin") == asset
    context = start_run(settings, "fixture_001")
    manifest = json.loads((context.reports_dir / "run.json").read_text())
    assert manifest["input_hashes"] == {
        "tiny.bin": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    }
    assert asset.read_bytes() == b"abc"
    assert list(raw.iterdir()) == [asset]


def test_outputs_cannot_overlap_raw_root(repo):
    edit_section(repo, "sources", lambda s: s.update(raw_root="data"))
    with pytest.raises(PipelineError):
        load_settings(Path("configs/experiments.yaml"), repo_root=repo)


def test_run_provenance_and_reuse(repo):
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    context = start_run(settings, "fixture_001")
    path = context.reports_dir / "run.json"
    content = path.read_bytes()
    manifest = json.loads(content)
    assert manifest["status"] == "started"
    assert manifest["seed"] == 42
    assert manifest["config_snapshot"]["study_area"]["target_crs"] == "EPSG:9210"
    assert manifest["started_at_utc"].endswith("+00:00")
    assert manifest["lock_sha256"] == hashlib.sha256(b"synthetic lock\n").hexdigest()
    assert manifest["git"] == {"commit": None, "dirty": None}
    assert start_run(settings, "fixture_001") == context
    assert path.read_bytes() == content
    assert not (context.reports_dir / "complete.json").exists()


def test_run_id_config_change_rejected(repo):
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    start_run(settings, "fixture_001")
    with pytest.raises(PipelineError):
        start_run(replace(settings, seed=7), "fixture_001")


def test_run_id_lock_change_rejected(repo):
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    start_run(settings, "fixture_001")
    (repo / "uv.lock").write_text("changed", encoding="utf-8")
    with pytest.raises(PipelineError):
        start_run(settings, "fixture_001")


def test_run_id_source_code_change_rejected(repo):
    source = repo / "src/flood_ml"
    source.mkdir(parents=True)
    module = source / "stage.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    start_run(settings, "fixture_001")
    module.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(PipelineError, match="fingerprint"):
        start_run(settings, "fixture_001")


def test_run_id_input_change_rejected(repo):
    raw = repo / "data/raw"
    raw.mkdir(parents=True)
    (raw / "tiny.bin").write_bytes(b"abc")
    edit_section(repo, "sources", lambda s: s.update(assets=[{"path": "tiny.bin"}]))
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    start_run(settings, "fixture_001")
    (raw / "tiny.bin").write_bytes(b"changed")
    with pytest.raises(PipelineError):
        start_run(settings, "fixture_001")


@pytest.mark.parametrize("run_id", ["../evil", "a/b", "", "CON", "abc."])
def test_unsafe_run_id_rejected(repo, run_id):
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    with pytest.raises(PipelineError):
        start_run(settings, run_id)


def test_atomic_write_failure_leaves_no_final_output(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(run.os, "replace", fail)
    target = tmp_path / "result.json"
    with pytest.raises(PipelineError):
        write_json_atomic(target, {"score": 12})
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_failure_preserves_previous_output(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text('{"old": true}', encoding="utf-8")

    def fail(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(run.os, "replace", fail)
    with pytest.raises(PipelineError):
        write_json_atomic(target, {"new": True})
    assert json.loads(target.read_text()) == {"old": True}


def test_start_failure_does_not_publish_run_or_completion(repo, monkeypatch):
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)

    def fail(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(run.os, "replace", fail)
    with pytest.raises(PipelineError):
        start_run(settings, "fixture_001")
    assert not (repo / "reports/runs/fixture_001/run.json").exists()
    assert not (repo / "reports/runs/fixture_001/complete.json").exists()


def test_failed_start_releases_its_lock_for_retry(repo, monkeypatch):
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    original = run.os.replace

    def fail(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(run.os, "replace", fail)
    with pytest.raises(PipelineError):
        start_run(settings, "fixture_001")
    monkeypatch.setattr(run.os, "replace", original)
    context = start_run(settings, "fixture_001")
    assert (context.reports_dir / "run.json").is_file()


def test_existing_initialization_lock_is_preserved(repo):
    reports = repo / "reports/runs/fixture_001"
    reports.mkdir(parents=True)
    (reports / ".start.lock").write_text("owned by another process")
    settings = load_settings(Path("configs/experiments.yaml"), repo_root=repo)
    with pytest.raises(PipelineError):
        start_run(settings, "fixture_001")
    assert (reports / ".start.lock").read_text() == "owned by another process"


def test_atomic_success_and_nonfinite_json_rejected(tmp_path):
    target = tmp_path / "result.json"
    write_json_atomic(target, {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}
    with pytest.raises(PipelineError):
        write_json_atomic(target, {"score": float("nan")})
    assert json.loads(target.read_text()) == {"ok": True}
