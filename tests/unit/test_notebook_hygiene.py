"""Check the distributable notebook without fetching code or running raw data."""

import ast
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
        'expected="0.13.6"',
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


def test_notebook_bootstrap_and_windows_replay_parse_real_uv_release_output():
    document = notebook()
    bootstrap = next(
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
        and '"https://astral.sh/uv/0.13.6/install.sh"' in "".join(cell["source"])
    )
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "markdown"
    )
    replay = markdown.split("```python\n")[-1].split("```")[0]
    assert "import re" in replay
    actual = "uv 0.13.6 (329bf9448 2026-10-26 x86_64-pc-windows-msvc)"
    for source in (bootstrap, replay):
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "require_uv_version"
        )
        parser_module = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.Import)
                and any(alias.name == "re" for alias in node.names)
            ]
            + [function],
            type_ignores=[],
        )
        namespace = {"RuntimeError": RuntimeError, "re": re}
        exec(  # noqa: S102 - execute only the notebook's isolated version parser
            compile(parser_module, "<uv-parser>", "exec"),
            namespace,
        )
        assert namespace["require_uv_version"](actual) == actual
        with pytest.raises(RuntimeError):
            namespace["require_uv_version"]("uv 0.13.5 (build metadata)")


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


@pytest.mark.parametrize(
    "secret_status", ["available", "missing", "denied", "empty", "outside_colab"]
)
def test_notebook_colab_secret_is_private_to_project_children(
    monkeypatch, tmp_path, capsys, secret_status
):
    # Catch missing Secret handling, global environment writes, and leaked child logs.
    sources = [
        "".join(cell["source"])
        for cell in notebook()["cells"]
        if cell["cell_type"] == "code"
    ]
    secret_cell = next(
        (source for source in sources if "from google.colab import userdata" in source),
        None,
    )
    assert secret_cell is not None, "Notebook must read the Colab Secret"
    sentinel = "synthetic-secret-for-notebook-test"
    requested = []

    def get_secret(name):
        requested.append(name)
        if secret_status in {"missing", "denied"}:
            raise RuntimeError(sentinel)  # Exception text must not be displayed.
        return sentinel if secret_status == "available" else ""

    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.userdata = SimpleNamespace(get=get_secret)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(
        sys.modules, "google.colab", None if secret_status == "outside_colab" else colab
    )
    monkeypatch.setenv("WANDB_API_KEY", "inherited-value-must-not-be-used")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    before = dict(os.environ)
    namespace = {"ENV": dict(os.environ), "os": os}
    exec(compile(secret_cell, "<colab-secrets>", "exec"), namespace)  # noqa: S102
    assert requested == ([] if secret_status == "outside_colab" else ["WANDB_API_KEY"])
    assert dict(os.environ) == before
    env = namespace["ENV"]
    assert env["WANDB_ENTITY"] == "duygri-saigon-university-sgu-"
    assert env["WANDB_PROJECT"] == "hcmc-road-flood-ml"
    if secret_status == "available":
        assert env["WANDB_API_KEY"] == sentinel
    else:
        assert not env.get("WANDB_API_KEY")
        assert "skip" in capsys.readouterr().out.lower()
    # Run actual notebook log-handling functions, substituting only the subprocess.
    bootstrap = next(source for source in sources if "def run_project(" in source)
    functions = ast.Module(
        body=[
            node
            for node in ast.parse(bootstrap).body
            if isinstance(node, ast.FunctionDef)
        ],
        type_ignores=[],
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            stdout="child output "
            + (sentinel if secret_status == "available" else "ok"),
            stderr="child error "
            + (sentinel if secret_status == "available" else "ok"),
            check_returncode=lambda: None,
        )

    namespace.update(
        UV="uv",
        CHECKOUT=tmp_path,
        REPORTS=tmp_path,
        subprocess=SimpleNamespace(run=run),
    )
    exec(compile(functions, "<project-commands>", "exec"), namespace)  # noqa: S102
    result = namespace["run_project"](["python", "scripts/train.py"], "train")
    assert calls[0][0][:2] == ["uv", "run"]
    assert calls[0][1]["env"] is env
    assert sentinel not in result.stdout + result.stderr
    assert sentinel not in capsys.readouterr().out
    publication = next(
        source for source in sources if "publication = subprocess.run(" in source
    )
    statements = ast.parse(publication).body
    start = next(
        index
        for index, node in enumerate(statements)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "publication"
            for target in node.targets
        )
    )
    namespace.update(
        publish_args=["python", "scripts/publish_run.py"],
        summary="summary",
        DRIVE_ROOT=tmp_path,
        RUN_ID="test",
    )
    exec(  # noqa: S102 - only the publication log handler; subprocess is captured
        compile(
            ast.Module(body=statements[start:], type_ignores=[]),
            "<publication>",
            "exec",
        ),
        namespace,
    )
    assert calls[1][0][:2] == ["uv", "run"]
    assert calls[1][1]["env"] is env
    assert sentinel not in capsys.readouterr().out
    assert all(sentinel not in path.read_text() for path in tmp_path.iterdir())


def test_notebook_has_no_literal_wandb_credential():
    tree = ast.parse(code())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "WANDB_API_KEY"
                ):
                    assert not isinstance(node.value, ast.Constant)
    assert not re.search(r"(?<![A-Za-z0-9])[a-fA-F0-9]{40}(?![A-Za-z0-9])", code())


def test_notebook_bootstrap_does_not_forward_inherited_wandb_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-inherited-key")
    bootstrap = next(
        "".join(cell["source"])
        for cell in notebook()["cells"]
        if cell["cell_type"] == "code" and "UV_BIN_DIR =" in "".join(cell["source"])
    )
    statements = ast.parse(bootstrap).body
    boundary = next(
        index for index, node in enumerate(statements) if isinstance(node, ast.With)
    )
    namespace = {"os": os, "Path": Path}
    exec(  # noqa: S102 - only environment construction, before any installer/network call
        compile(
            ast.Module(body=statements[:boundary], type_ignores=[]),
            "<bootstrap-env>",
            "exec",
        ),
        namespace,
    )
    assert "WANDB_API_KEY" not in namespace["ENV"]
    assert os.environ["WANDB_API_KEY"] == "synthetic-inherited-key"
