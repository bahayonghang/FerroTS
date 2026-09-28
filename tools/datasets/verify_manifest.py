"""Verify dataset files against docs/datasets/manifest.toml.

Usage: python tools/datasets/verify_manifest.py docs/datasets/manifest.toml

Paths in the manifest are relative to the repository root. Exit code is 0
when every file exists with the listed size and SHA256, otherwise 1.
"""

import hashlib
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    manifest = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    failed = 0
    for entry in manifest["file"]:
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            print(f"MISSING  {entry['path']}")
            failed += 1
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            print(f"SIZE     {entry['path']}: {size} != {entry['bytes']}")
            failed += 1
            continue
        actual = sha256_of(path)
        if actual != entry["sha256"]:
            print(f"SHA256   {entry['path']}: {actual} != {entry['sha256']}")
            failed += 1
            continue
        print(f"OK       {entry['path']}")
    total = len(manifest["file"])
    print(f"{total - failed}/{total} files OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
