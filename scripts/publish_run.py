"""Publish a completed run bundle to a mounted Google Drive directory."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_ml.config import PipelineError
from flood_ml.utils.run_publisher import publish_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--publish-interim", action="store_true")
    args = parser.parse_args(argv)
    try:
        published = publish_run(
            args.repo_root,
            args.drive_root,
            args.run_id,
            config=args.config,
            publish_interim=args.publish_interim,
        )
    except (PipelineError, OSError, ValueError) as exc:
        print(f"Run publication failed: {exc}", file=sys.stderr)
        return 1
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
