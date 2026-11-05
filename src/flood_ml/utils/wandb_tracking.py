"""Optional reporting of caller-sanitized stage metadata to Weights & Biases."""

import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


def _numeric_metrics(metrics: Mapping[str, object], prefix: str = "") -> dict:
    """Flatten mapping leaves only; lists may contain private row-level data."""
    result = {}
    for key, value in metrics.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            result.update(_numeric_metrics(value, name))
        elif not isinstance(value, bool) and (
            isinstance(value, int)
            or (isinstance(value, float) and math.isfinite(value))
        ):
            result[name] = value
    return result


def log_stage_report(
    *,
    run_id: str,
    stage: str,
    config: Mapping[str, object],
    summary: Mapping[str, object],
    metrics: Mapping[str, object],
) -> str:
    """Report safe metadata without making remote failures fatal to local work.

    Callers must select config, summary, and metric names explicitly: never pass
    raw reports, input paths, review evidence, datasets, or artifact objects.
    Numeric metric mappings are flattened with slash-separated names; other
    leaves and lists are ignored. No SDK import occurs without an API key.
    """
    if stage not in {"track-b", "evaluation"}:
        return "failed"
    if not os.environ.get("WANDB_API_KEY", "").strip():
        return "skipped_no_api_key"

    try:
        import wandb

        # Both stages share a portable run root outside the working checkout;
        # SDK files must not affect the next stage's Git provenance snapshot.
        run_root = Path(tempfile.gettempdir()) / "hcmc-road-flood-ml-wandb"
        run_root.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            dir=str(run_root),
            entity=os.environ.get("WANDB_ENTITY", "duygri-saigon-university-sgu-"),
            project=os.environ.get("WANDB_PROJECT", "hcmc-road-flood-ml"),
            name=f"{run_id}-{stage}",
            group=run_id,
            job_type=stage,
            config=dict(config),
            settings=wandb.Settings(
                disable_code=True,
                save_code=False,
                disable_git=True,
                disable_job_creation=True,
                console="off",
                x_disable_meta=True,
                x_disable_stats=True,
                finish_timeout=5,
                finish_timeout_raises=True,
            ),
        )
        try:
            run.summary.update(dict(summary))
            numeric_metrics = _numeric_metrics(metrics)
            if numeric_metrics:
                run.log(numeric_metrics)
        finally:
            run.finish()
    except Exception:  # noqa: BLE001 -- optional remote reporting must not block local work
        # Do not expose SDK errors: they can contain credentials or local paths.
        return "failed"
    return "logged"
