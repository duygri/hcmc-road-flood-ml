"""Prepare the AOI and normalized IRD observation stage without labels."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_ml.config import PipelineError, load_settings
from flood_ml.pipeline import run_pipeline
from flood_ml.utils.run import start_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/labeling.yaml"))
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config, repo_root=args.repo_root)
        if args.run_id:
            settings = replace(
                settings,
                experiments={**settings.experiments, "run_id": args.run_id},
            )
        context = start_run(settings, settings.experiments["run_id"])
        summary = run_pipeline(settings, context, stages=["observations"])
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Observation preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
