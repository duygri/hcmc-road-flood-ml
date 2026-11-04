"""Tracked hash inputs must keep identical bytes across Git checkouts."""

import hashlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *map(str, args)],
        cwd=cwd,
        check=True,
        capture_output=True,
    ).stdout


@pytest.mark.parametrize("autocrlf", ["true", "false"])
def test_hash_critical_text_bytes_match_git_blobs_in_real_checkouts(tmp_path, autocrlf):
    source = tmp_path / "source"
    source.mkdir()
    relative_sources = {
        "src/flood_ml/config.py": ROOT / "src/flood_ml/config.py",
        "uv.lock": ROOT / "uv.lock",
        "data/raw/manifest.yaml": ROOT / "data/raw/manifest.yaml",
        "configs/review-template.csv": ROOT / "tests/fixtures/review-template.csv",
    }
    for relative, original in relative_sources.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
    (source / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    git("init", source)
    git("-C", source, "config", "user.name", "EOL test")
    git("-C", source, "config", "user.email", "eol@example.invalid")
    git("-C", source, "add", ".")
    git("-C", source, "commit", "-m", "tracked text fixtures")

    checkout = tmp_path / f"checkout-{autocrlf}"
    subprocess.run(
        ["git", "-c", f"core.autocrlf={autocrlf}", "clone", str(source), str(checkout)],
        check=True,
        capture_output=True,
    )
    for relative in relative_sources:
        expected_blob = git("-C", checkout, "show", f"HEAD:{relative}")
        checked_out = (checkout / relative).read_bytes()
        assert checked_out == expected_blob, relative
        assert (
            hashlib.sha256(checked_out).digest()
            == hashlib.sha256(expected_blob).digest()
        )
