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
        ("reports/runs", "pipeline_complete.json", {}),
        (
            "reports/runs",
            "pipeline_report.json",
            {
                "pipeline_status": "complete",
                "evaluation_status": "pending_label_review",
            },
        ),
        ("reports/runs", "evaluation.json", {}),
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
    output_bytes = {
        f"data/processed/{RUN_ID}/road_features.parquet": b"processed",
        f"data/processed/{RUN_ID}/road_labels.parquet": b"labels",
        f"data/processed/{RUN_ID}/confirmed_event_segments.parquet": b"events",
        f"data/processed/{RUN_ID}/road_partitions.parquet": b"partitions",
        f"data/processed/{RUN_ID}/reviewed_group_partition_audit.parquet": b"audit",
        f"data/processed/{RUN_ID}/analysis_dataset.parquet": b"analysis",
        f"data/processed/{RUN_ID}/road_susceptibility.parquet": b"scores",
        f"data/processed/{RUN_ID}/sensitivity_scores.parquet": b"sensitivity-scores",
        f"data/interim/{RUN_ID}/roads.gpkg": b"interim",
        f"artifacts/{RUN_ID}/normalizer.json": b"artifact",
        f"artifacts/{RUN_ID}/mcda_rules.json": b"rules",
        f"reports/runs/{RUN_ID}/weight_sensitivity.json": b"sensitivity",
        f"reports/runs/{RUN_ID}/pipeline_report.html": b"report-html",
    }
    for relative, content in output_bytes.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    processed = f"data/processed/{RUN_ID}"
    evaluation_inputs = {
        "scores": f"{processed}/road_susceptibility.parquet",
        "roads": f"data/interim/{RUN_ID}/roads.gpkg",
        "confirmed_event_segments": f"{processed}/confirmed_event_segments.parquet",
        "partitions": f"{processed}/road_partitions.parquet",
    }
    (root / f"reports/runs/{RUN_ID}/evaluation.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "status": "pending_label_review",
                "evaluation_performed": False,
                "input_paths": evaluation_inputs,
                "input_hashes": {
                    name: hashlib.sha256((root / path).read_bytes()).hexdigest()
                    for name, path in evaluation_inputs.items()
                },
                "source_inputs_unchanged": True,
            }
        ),
        encoding="utf-8",
    )
    stage_outputs = {
        "features": [f"data/processed/{RUN_ID}/road_features.parquet"],
        "labels": [f"data/processed/{RUN_ID}/road_labels.parquet"],
        "split": [
            f"data/processed/{RUN_ID}/confirmed_event_segments.parquet",
            f"data/processed/{RUN_ID}/road_partitions.parquet",
            f"data/processed/{RUN_ID}/reviewed_group_partition_audit.parquet",
        ],
        "mcda": [
            f"data/processed/{RUN_ID}/analysis_dataset.parquet",
            f"data/processed/{RUN_ID}/road_susceptibility.parquet",
            f"data/processed/{RUN_ID}/sensitivity_scores.parquet",
            f"artifacts/{RUN_ID}/normalizer.json",
            f"artifacts/{RUN_ID}/mcda_rules.json",
            f"reports/runs/{RUN_ID}/weight_sensitivity.json",
        ],
        "evaluation": [f"reports/runs/{RUN_ID}/evaluation.json"],
        "report": [
            f"reports/runs/{RUN_ID}/pipeline_report.json",
            f"reports/runs/{RUN_ID}/pipeline_report.html",
        ],
    }
    stage_markers = {}
    for stage in ("validate", "aoi", "roads", "observations", "raster", *stage_outputs):
        outputs = {
            str((root / relative).resolve()): hashlib.sha256(
                (root / relative).read_bytes()
            ).hexdigest()
            for relative in stage_outputs.get(stage, [])
        }
        marker = root / f"reports/runs/{RUN_ID}/stages/{stage}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "status": "complete",
                    "run_id": RUN_ID,
                    "outputs": outputs,
                }
            ),
            encoding="utf-8",
        )
        stage_markers[stage] = hashlib.sha256(marker.read_bytes()).hexdigest()
    completion = {
        "status": "complete",
        "run_id": RUN_ID,
        "evaluation_status": "pending_label_review",
        "stage_markers": stage_markers,
    }
    (root / f"reports/runs/{RUN_ID}/pipeline_complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    return root


def ready_track_b(repo: Path, *, extra_output: str | None = None) -> None:
    score = repo / f"data/processed/{RUN_ID}/track_b_scores.parquet"
    model = repo / f"artifacts/{RUN_ID}/model.joblib"
    score.write_bytes(b"scores")
    model.write_bytes(b"model")
    report_path = repo / f"reports/runs/{RUN_ID}/track_b.json"
    report = json.loads(report_path.read_text())
    report.update(
        status="ready",
        output_hashes={
            f"data/processed/{RUN_ID}/track_b_scores.parquet": hashlib.sha256(
                score.read_bytes()
            ).hexdigest(),
            f"artifacts/{RUN_ID}/model.joblib": hashlib.sha256(
                model.read_bytes()
            ).hexdigest(),
        },
        draws=[
            {"status": "fitted", "model_artifact": f"artifacts/{RUN_ID}/model.joblib"}
        ],
    )
    if extra_output:
        extra_path = repo / extra_output
        report["output_hashes"][extra_output] = (
            hashlib.sha256(extra_path.read_bytes()).hexdigest()
            if extra_path.is_file()
            else "0" * 64
        )
    report_path.write_text(json.dumps(report), encoding="utf-8")


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


@pytest.mark.parametrize(
    "relative",
    [
        f"data/processed/{RUN_ID}/analysis_dataset.parquet",
        f"data/processed/{RUN_ID}/road_susceptibility.parquet",
    ],
)
def test_publisher_rejects_missing_required_pending_stage_output(tmp_path, relative):
    repo = fixture_run(tmp_path / "repo")
    (repo / relative).unlink()
    with pytest.raises(PipelineError, match="missing|stage output"):
        publish_run(repo, tmp_path / "drive", RUN_ID)
    assert not (tmp_path / "drive/runs" / RUN_ID).exists()


def test_publisher_rejects_missing_declared_stage_artifact(tmp_path):
    repo = fixture_run(tmp_path / "repo")
    (repo / f"data/processed/{RUN_ID}/sensitivity_scores.parquet").unlink()
    with pytest.raises(PipelineError, match="missing|stage output"):
        publish_run(repo, tmp_path / "drive", RUN_ID)
    assert not (tmp_path / "drive/runs" / RUN_ID).exists()


def test_publisher_rejects_present_ready_model_with_wrong_recorded_digest(tmp_path):
    repo = fixture_run(tmp_path / "repo")
    ready_track_b(repo)
    report_path = repo / f"reports/runs/{RUN_ID}/track_b.json"
    report = json.loads(report_path.read_text())
    report["output_hashes"][f"artifacts/{RUN_ID}/model.joblib"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(PipelineError, match="Track B.*hash|digest"):
        publish_run(repo, tmp_path / "drive", RUN_ID)
    assert not (tmp_path / "drive/runs" / RUN_ID).exists()


def test_publisher_publishes_ready_track_b_with_real_hashes(tmp_path):
    repo = fixture_run(tmp_path / "repo")
    ready_track_b(repo)
    published = publish_run(repo, tmp_path / "drive", RUN_ID)
    manifest = json.loads((published / "completion_manifest.json").read_text())
    for relative in (
        f"data/processed/{RUN_ID}/track_b_scores.parquet",
        f"artifacts/{RUN_ID}/model.joblib",
    ):
        record = manifest["files"][relative]
        assert (
            record["sha256"]
            == hashlib.sha256((published / relative).read_bytes()).hexdigest()
        )


def test_publisher_requires_ready_track_b_artifacts(tmp_path):
    repo = fixture_run(tmp_path / "repo", track_b_status="ready")
    with pytest.raises(PipelineError, match="Track B.*artifact|track_b.*artifact"):
        publish_run(repo, tmp_path / "drive", RUN_ID)
    assert not (tmp_path / "drive/runs" / RUN_ID / "completion_manifest.json").exists()


@pytest.mark.parametrize(
    "extra_output",
    [
        f"data/interim/{RUN_ID}/roads.gpkg",
        "artifacts/another_run/model.joblib",
    ],
)
def test_publisher_rejects_ready_track_b_artifacts_outside_selected_run_roots(
    tmp_path, extra_output
):
    repo = fixture_run(tmp_path / "repo")
    if "another_run" in extra_output:
        other_run_artifact = repo / extra_output
        other_run_artifact.parent.mkdir(parents=True)
        other_run_artifact.write_bytes(b"other run")
    ready_track_b(repo, extra_output=extra_output)
    with pytest.raises(PipelineError, match="outside selected run outputs"):
        publish_run(repo, tmp_path / "drive", RUN_ID)
    assert not (tmp_path / "drive/runs" / RUN_ID).exists()


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("git", "commit"), None),
        (("lock_sha256",), None),
        (("config_hash",), None),
        (("input_hashes",), {}),
    ],
)
def test_publisher_rejects_missing_required_provenance_before_creating_target(
    tmp_path, field_path, replacement
):
    repo = fixture_run(tmp_path / "repo")
    path = repo / f"reports/runs/{RUN_ID}/run.json"
    record = json.loads(path.read_text())
    section, *keys = field_path
    if keys:
        record[section][keys[0]] = replacement
    else:
        record[section] = replacement
    path.write_text(json.dumps(record), encoding="utf-8")
    target = tmp_path / "drive/runs" / RUN_ID
    with pytest.raises(PipelineError, match="provenance"):
        publish_run(repo, tmp_path / "drive", RUN_ID)
    assert not target.exists()


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
