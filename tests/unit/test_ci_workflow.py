from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"


def test_ci_matrix_contains_windows_and_linux_and_uses_locked_commands():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"].values()
    matrix_jobs = [
        job
        for job in jobs
        if isinstance(job.get("strategy", {}).get("matrix", {}).get("os"), list)
    ]

    assert matrix_jobs, "CI must define an operating-system matrix"
    assert {
        runner for job in matrix_jobs for runner in job["strategy"]["matrix"]["os"]
    } >= {"windows-latest", "ubuntu-latest"}

    required_commands = {
        "uv lock --check",
        "uv sync --locked --group dev",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run pytest -q",
    }
    for job in matrix_jobs:
        assert job["runs-on"] == "${{ matrix.os }}"
        setup_uv = [
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("astral-sh/setup-uv@")
        ]
        assert len(setup_uv) == 1
        assert setup_uv[0]["with"]["version"] == "0.13.6"
        assert setup_uv[0]["with"]["python-version"] == "3.12"
        commands = {step.get("run") for step in job["steps"]}
        assert required_commands <= commands
