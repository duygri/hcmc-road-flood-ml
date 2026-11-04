"""Offline tests for safe publication of completed run bundles."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from flood_ml.config import PipelineError
from flood_ml.utils.run_publisher import publish_run

RUN_ID = "publisher_fixture"


def fixture_run(root: Path, *, track_b_status="pending_positive_review") -> Path:
    config = root / "configs/experiments.yaml"
    config.parent.mkdir(parents=True)
    source_config = Path(__file__).parents[2] / "configs/experiments.yaml"
    for source in source_config.parent.glob("*.yaml"):
        (config.parent / source.name).write_bytes(source.read_bytes())
    for directory, name, content in [
        (
            "reports/runs",
            "run.json",
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "status": "started",
                "git": {"commit": "abc123"},
                "lock_sha256": "lockhash",
                "config_hash": "confighash",
                "input_hashes": {"input": "inputhash"},
                "review_hashes": {"review": "reviewhash"},
            },
        ),
        (
            "reports/runs",
            "pipeline_complete.json",
            {
                "status": "complete",
                "run_id": RUN_ID,
                "evaluation_status": "pending_label_review",
            },
        ),
        (
            "reports/runs",
            "pipeline_report.json",
            {
                "pipeline_status": "complete",
                "evaluation_status": "pending_label_review",
            },
        ),
        (
            "reports/runs",
            "evaluation.json",
            {"status": "pending_label_review", "evaluation_performed": False},
        ),
        (
            "reports/runs",
            "track_b.json",
            {
                "status": track_b_status,
                "gate": {"status": "pending"},
                "counts": {"primary_background_pool": 0},
            },
        ),
        (
            "reports/runs",
            "runtime.json",
            {"python": "3.12.1", "uv": "0.5.0", "platform": "linux-x86_64"},
        ),
    ]:
        path = root / directory / RUN_ID / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")
    for relative, content in [
        (f"data/processed/{RUN_ID}/road_features.parquet", b"processed"),
        (f"data/interim/{RUN_ID}/roads.gpkg", b"interim"),
        (f"artifacts/{RUN_ID}/normalizer.json", b"artifact"),
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def test_publisher_copies_required_run_outputs_and_records_hashes(tmp_path):
    repo = fixture_run(tmp_path / "repo")
    published = publish_run(repo, tmp_path / "drive", RUN_ID)
    manifest = json.loads((published / "completion_manifest.json").read_text())
    assert published == tmp_path / "drive/runs" / RUN_ID
    assert (
        published / f"data/processed/{RUN_ID}/road_features.parquet"
    ).read_bytes() == b"processed"
    assert manifest["status"] == "published"
    assert manifest["run_id"] == RUN_ID
    assert manifest["source_commit"] == "abc123"
    assert manifest["runtime"] == {
        "python": "3.12.1",
        "uv": "0.5.0",
        "platform": "linux-x86_64",
    }
    assert manifest["lock_sha256"] == "lockhash"
    assert manifest["config_sha256"] == "confighash"
    assert manifest["input_hashes"] == {"input": "inputhash"}
    assert manifest["review_hashes"] == {"review": "reviewhash"}
    record = manifest["files"][f"data/processed/{RUN_ID}/road_features.parquet"]
    assert record == {"size": 9, "sha256": hashlib.sha256(b"processed").hexdigest()}
    assert manifest["gate_statuses"]["evaluation"] == "pending_label_review"


def test_publisher_skips_interim_by_default(tmp_path):
    repo = fixture_run(tmp_path / "repo")
    published = publish_run(repo, tmp_path / "drive", RUN_ID)
    assert not (published / f"data/interim/{RUN_ID}").exists()


def test_publisher_includes_interim_when_enabled(tmp_path):
    repo = fixture_run(tmp_path / "repo")
    published = publish_run(repo, tmp_path / "drive", RUN_ID, publish_interim=True)
    assert (published / f"data/interim/{RUN_ID}/roads.gpkg").read_bytes() == b"interim"
    assert (
        json.loads((published / "completion_manifest.json").read_text())[
            "publish_interim"
        ]
        is True
    )


def test_publisher_accepts_pending_track_b_without_model_or_track_b_scores(tmp_path):
    repo = fixture_run(tmp_path / "repo")
    published = publish_run(repo, tmp_path / "drive", RUN_ID)
    assert (published / "completion_manifest.json").is_file()


def test_publisher_requires_ready_track_b_artifacts(tmp_path):
    repo = fixture_run(tmp_path / "repo", track_b_status="ready")
    with pytest.raises(PipelineError, match="Track B.*artifact|track_b.*artifact"):
        publish_run(repo, tmp_path / "drive", RUN_ID)
    assert not (tmp_path / "drive/runs" / RUN_ID / "completion_manifest.json").exists()


@pytest.mark.parametrize("failure", ["copy", "hash"])
def test_publisher_does_not_write_completion_manifest_after_copy_or_hash_failure(
    tmp_path, monkeypatch, failure
):
    repo = fixture_run(tmp_path / "repo")
    if failure == "copy":
        original = shutil.copy2

        def fail_copy(source, destination, *args, **kwargs):
            if Path(source).name == "road_features.parquet":
                raise OSError("simulated Drive quota exceeded")
            return original(source, destination, *args, **kwargs)

        monkeypatch.setattr(shutil, "copy2", fail_copy)
    else:
        import flood_ml.utils.run_publisher as publisher

        original = publisher._hash_file

        def fail_hash(path):
            result = original(path)
            if Path(path).name == "road_features.parquet" and "drive" in str(path):
                return "0" * 64
            return result

        monkeypatch.setattr(publisher, "_hash_file", fail_hash)
    with pytest.raises(PipelineError):
        publish_run(repo, tmp_path / "drive", RUN_ID)
    assert not (tmp_path / "drive/runs" / RUN_ID / "completion_manifest.json").exists()


def test_publisher_refuses_existing_run_without_overwriting_it(tmp_path):
    repo = fixture_run(tmp_path / "repo")
    target = tmp_path / "drive/runs" / RUN_ID
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_text("existing")
    with pytest.raises(PipelineError, match="already exists"):
        publish_run(repo, tmp_path / "drive", RUN_ID)
    assert sentinel.read_text() == "existing"
