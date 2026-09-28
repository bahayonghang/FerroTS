"""Fetch official upstream repositories at pinned commits into tools/reference/.cache/.

Usage: python tools/reference/fetch_upstream.py

Development-time only. Production builds do not use Python or these checkouts.
"""

import subprocess
import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parent / ".cache"

# Pinned 2026-09-28 (main HEAD on that date).
UPSTREAMS = {
    "Time-Series-Library": (
        "https://github.com/thuml/Time-Series-Library.git",
        "4e938a1767106324dd753b2a44832bf870a0252e",
    ),
    "iTransformer": (
        "https://github.com/thuml/iTransformer.git",
        "c2426e68ca13f74aaec08045c5c724d8ad328124",
    ),
}


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch(name: str, url: str, sha: str) -> None:
    dest = CACHE / name
    if not dest.exists():
        dest.mkdir(parents=True)
        git("init", "-q", cwd=dest)
        git("remote", "add", "origin", url, cwd=dest)
    git("fetch", "-q", "--depth", "1", "origin", sha, cwd=dest)
    git("-c", "advice.detachedHead=false", "checkout", "-q", "--force", sha, cwd=dest)
    head = git("rev-parse", "HEAD", cwd=dest)
    if head != sha:
        raise SystemExit(f"{name}: HEAD {head} != pinned {sha}")
    print(f"{name}: {sha} -> {dest}")


def main() -> int:
    for name, (url, sha) in UPSTREAMS.items():
        fetch(name, url, sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
