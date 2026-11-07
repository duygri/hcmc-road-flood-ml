"""Keep the Colab orchestrator valid, stripped, and aligned with PU-Learning."""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "colab_first_workflow.ipynb"
SECRET_RE = re.compile(
    r"(WANDB_API_KEY|API_KEY|TOKEN|PASSWORD)\s*=\s*['\"][^'\"]+['\"]"
    r"|ghp_[A-Za-z0-9]{20,}"
)
ALLOWED_CELL_METADATA = {"tags"}


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_sources(doc):
    return [
        "".join(cell.get("source", []))
        for cell in doc["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_colab_notebook_is_parseable_stripped_and_free_of_secrets():
    doc = _notebook()
    assert doc["nbformat"] == 4
    assert doc["cells"], "Expected at least one cell"
    for cell in doc["cells"]:
        assert set(cell.get("metadata", {})) <= ALLOWED_CELL_METADATA
        source = "".join(cell.get("source", []))
        assert not SECRET_RE.search(source)
        if cell.get("cell_type") == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None
            ast.parse(source)


def test_colab_notebook_delegates_core_stages_in_order():
    source = "\n".join(_code_sources(_notebook()))
    stage_positions = [
        source.index("scripts/run_pipeline.py"),
        source.index("scripts/train.py"),
        source.index("scripts/evaluate.py"),
        source.index("scripts/publish_run.py"),
    ]
    assert stage_positions == sorted(stage_positions)


def test_colab_notebook_enforces_mounted_drive_and_space_preflight():
    source = "\n".join(_code_sources(_notebook()))
    assert 'Path("/content/drive/MyDrive")' in source
    assert "drive.mount" in source
    assert "shutil.disk_usage" in source
    assert "OUTPUT_RESERVE_GIB" in source
    assert "manifest.yaml" in source


def test_colab_notebook_redacts_secrets_and_captures_stage_logs():
    sources = _code_sources(_notebook())
    source = "\n".join(sources)
    assert 'ENV.pop("WANDB_API_KEY", None)' in source
    assert "redact_tracking_secret" in source
    assert '(REPORTS / f"{name}.stdout.log").write_text' in source
    assert '(REPORTS / f"{name}.stderr.log").write_text' in source
    assert '(REPORTS / "publish.stdout.log").write_text' in source
    assert '(REPORTS / "publish.stderr.log").write_text' in source
    assert source.count(".check_returncode()") >= 2


def test_colab_notebook_writes_runtime_and_summary_before_publishing():
    publish_cell = next(
        c for c in _code_sources(_notebook()) if "scripts/publish_run.py" in c
    )
    runtime_pos = publish_cell.index('(REPORTS / "runtime.json").write_text')
    summary_pos = publish_cell.index('(REPORTS / "colab_summary.md").write_text')
    publish_pos = publish_cell.index("scripts/publish_run.py")
    assert runtime_pos < publish_pos
    assert summary_pos < publish_pos


def test_colab_notebook_prioritizes_pu_capture_lift_and_ablation():
    source = "\n".join(_code_sources(_notebook()))
    assert "Capture@10%" in source
    assert "Lift@10%" in source
    assert "LogisticRegression" in source
    assert "full_study_area_road_risk_scores.parquet" in source
    assert "from getpass import getpass" in source


def test_colab_notebook_bootstraps_missing_or_corrupt_drive_source():
    doc = _notebook()
    cell = next(
        c for c in doc["cells"] if c.get("id") == "8dd0d8092fe74a7c96281538738b07e2"
    )
    source = "".join(cell.get("source", []))
    assert "_is_valid_source_dir" in source
    assert "tomllib.loads" in source
    assert "archive/refs/heads/main.zip" in source
    assert "importlib.invalidate_caches()" in source
