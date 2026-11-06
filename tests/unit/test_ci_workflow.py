"""Keep the GitHub Actions quality gate aligned with repository tooling."""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"


def test_ci_workflow_runs_lint_and_unit_checks():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True, {}))
    assert "push" in triggers
    assert "pull_request" in triggers
    assert workflow["permissions"] == {"contents": "read"}

    steps = workflow["jobs"]["check"]["steps"]
    uses = [step.get("uses") for step in steps if "uses" in step]
    assert "actions/checkout@v4" in uses
    assert any(u and u.startswith("astral-sh/setup-uv@") for u in uses)
    assert "actions/setup-python@v5" in uses

    setup_python = [
        step for step in steps if step.get("uses") == "actions/setup-python@v5"
    ]
    assert setup_python[0]["with"]["python-version"] == "3.11"

    commands = [step.get("run") for step in steps if "run" in step]
    assert commands == [
        'uv pip install --system -e . "pytest>=8.3" "ruff>=0.9" "scipy>=1.13"',
        "ruff check .",
        "pytest -q",
    ]
