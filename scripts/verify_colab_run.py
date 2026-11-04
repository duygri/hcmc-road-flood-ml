"""Verify an existing separate-checkout replay against a published Colab run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_ml.config import PipelineError, load_settings, resolve_output
from flood_ml.utils.run import write_json_atomic
from flood_ml.utils.run_comparison import compare_runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-run", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/experiments.yaml"), type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--atol", default=1e-5, type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        root, published = args.repo_root.resolve(), args.published_run.resolve()
        settings = load_settings(args.config, repo_root=root)
        output = args.output or Path(
            f"{settings.experiments['outputs']['reports']}/{args.run_id}/local_verification.json"
        )
        output = (
            output.resolve()
            if output.is_absolute()
            else resolve_output(root, str(output))
        )
        if not output.is_relative_to(root) or output.is_relative_to(published):
            raise PipelineError(
                "Verification report must be inside the separate local checkout"
            )
        if output.exists() and output.name != "local_verification.json":
            raise PipelineError("Refusing to overwrite an existing run artifact")
        report = compare_runs(
            published, root, args.run_id, config=args.config, atol=args.atol
        )
        write_json_atomic(output, report)
        print(json.dumps({"equivalent": report["equivalent"], "report": str(output)}))
        return 0 if report["equivalent"] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Run verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
