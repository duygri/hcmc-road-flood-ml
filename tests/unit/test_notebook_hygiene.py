"""Check the distributable notebook without fetching code or running raw data."""

import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[2] / "notebooks/colab_first_workflow.ipynb"


def notebook():
    assert NOTEBOOK.is_file(), "Colab workflow notebook is missing"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code():
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_template_is_valid_json_and_python_cells_parse():
    document = notebook()
    assert document["nbformat"] == 4
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_notebook_template_has_empty_outputs_and_null_execution_counts():
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_notebook_template_has_no_widget_or_transient_colab_metadata():
    document = notebook()
    assert set(document["metadata"]) <= {"kernelspec", "language_info"}
    assert all(cell["metadata"] == {} for cell in document["cells"])


def test_notebook_contains_no_credentials_or_user_data_paths():
    source = "\n".join("".join(cell["source"]) for cell in notebook()["cells"])
    assert not re.search(r"gh[pousr]_[A-Za-z0-9]+|AIza[A-Za-z0-9_-]+", source)
    assert not re.search(r"[A-Za-z]:\\|/Users/|/home/[^/]+/", source)
    imports = [
        node for node in ast.walk(ast.parse(code())) if isinstance(node, ast.Import)
    ]
    assert not {alias.name for node in imports for alias in node.names} & {
        "pandas",
        "geopandas",
        "yaml",
        "flood_ml",
    }


def test_notebook_runs_pipeline_train_evaluate_publish_cells_in_order():
    source = code()
    stages = [
        source.index(f'"scripts/{name}.py"')
        for name in ("run_pipeline", "train", "evaluate", "publish_run")
    ]
    assert stages == sorted(stages)
    assert source.index("tests/integration/test_track_b.py") < stages[0]
    for required in (
        "hashlib.file_digest",
        "disk_usage",
        '"bytes"',
        "OUTPUT_RESERVE_GIB",
        "runtime.json",
        "colab_summary.md",
        "stdout",
        "stderr",
    ):
        assert required in source


def test_notebook_pins_source_and_uses_uv_project_environment():
    source = code()
    for required in (
        "https://github.com/duygri/hcmc-road-flood-ml.git",
        "https://astral.sh/uv/0.13.6/install.sh",
        "uv 0.13.6",
        "UV_PROJECT_ENVIRONMENT",
        "/content/.venv",
        '"--locked"',
        '"--group", "dev"',
        '"--project"',
        '"3.12"',
        '"checkout", "--detach", SOURCE_COMMIT',
        '"rev-parse", "HEAD"',
    ):
        assert required in source
    assert 'subprocess.run(["sh"], input=' in source
    assert "[0-9a-fA-F]{40}" in source
    assert "pip install" not in source


@pytest.mark.parametrize(
    "status,ready",
    [
        ("ready", True),
        ("pending_positive_review", False),
        ("background_pool_insufficient", False),
    ],
)
def test_notebook_keeps_missing_reviews_pending_and_only_passes_ready_track_b_scores(
    tmp_path, status, ready
):
    document = notebook()
    source = next(
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
        and '"scripts/evaluate.py"' in "".join(cell["source"])
    )
    report = tmp_path / "track_b.json"
    report.write_text(json.dumps({"status": status}))
    commands = []
    exec(  # noqa: S102 - execute only a trusted notebook cell with captured CLI calls
        compile(source, "<evaluation-cell>", "exec"),
        {
            "json": json,
            "REPORTS": tmp_path,
            "RUN_ID": "test",
            "COMMON": [],
            "run_project": lambda args, name: commands.append(args),
        },
    )
    assert ("--track-b-scores" in commands[0]) is ready
    all_code = code()
    assert "if review.is_file():" in all_code
    assert "pending_positive_review" in all_code


def test_notebook_documents_same_snapshot_replay_in_separate_checkout():
    text = "\n".join("".join(cell["source"]) for cell in notebook()["cells"])
    for required in (
        "separate checkout",
        "same raw/review snapshot",
        "SOURCE_COMMIT",
        "verify_colab_run.py",
        "--published-run",
        "--repo-root",
        "--run-id",
        "--atol",
        "1e-5",
    ):
        assert required in text
    local_script = text.split("```python\n")[-1].split("```")[0]
    ast.parse(local_script)
    # The verifier requires local counterparts even for runtime/summary/log files.
    for required in (
        "runtime.json",
        "colab_summary.md",
        "stdout.log",
        "stderr.log",
        "preflight",
    ):
        assert required in local_script
