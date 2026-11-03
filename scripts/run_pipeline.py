"""Run the reproducible offline old-HCMC road susceptibility workflow."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_ml.config import PipelineError, load_settings, resolve_output
from flood_ml.pipeline import STAGE_ORDER, run_pipeline
from flood_ml.utils.run import start_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGE_ORDER,
        help="Run selected stages and their prerequisites; repeat to select more than one",
    )
    parser.add_argument(
        "--match-reviews", help="Repository-relative reviewed match CSV"
    )
    parser.add_argument("--event-reviews", help="Repository-relative event review CSV")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config, repo_root=args.repo_root)
        labeling = dict(settings.labeling)
        for key, value in (
            ("match_reviews", args.match_reviews),
            ("event_reviews", args.event_reviews),
        ):
            if value:
                resolved = resolve_output(settings.repo_root, value)
                labeling[key] = str(resolved.relative_to(settings.repo_root))
        settings = replace(
            settings,
            labeling=labeling,
            experiments={
                **settings.experiments,
                "run_id": args.run_id or settings.experiments["run_id"],
            },
        )
        context = start_run(settings, settings.experiments["run_id"])
        summary = run_pipeline(settings, context, stages=args.stage)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
