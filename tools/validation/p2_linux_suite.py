"""Run one fresh Linux P2 CUDA suite. Completed gates and entries are verified, not repeated."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "tools/reference/.venv/bin/python"
ACCEPTANCE = ROOT / "tools/validation/p2_acceptance.py"
QUALITY = ROOT / "tools/validation/p2_quality.py"
BINARY = "target/linux-cuda/debug/ferrots"
STAGES = ("data", "numerics", "resume", "ett-reference", "ett-rust", "d4", "cement", "recompute", "final")


def run(command: list[str]) -> None:
    print("START " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def passed(path: Path) -> bool:
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("passed") is True


def main() -> None:
    suite = sys.argv[1]
    directory = ROOT / suite
    if not passed(directory / "preflight.json"):
        run([str(PYTHON), str(ACCEPTANCE), "preflight", "--suite", suite, "--cuda-bin", BINARY])
    if not passed(directory / "quality.json"):
        run([str(PYTHON), str(QUALITY), suite])
    for stage in STAGES:
        run([str(PYTHON), str(ACCEPTANCE), stage, "--suite", suite, "--cuda-bin", BINARY])
    print(f"PASS fresh Linux suite {suite}", flush=True)


if __name__ == "__main__":
    if os.environ.get("RUSTUP_AUTO_INSTALL") != "0" or os.environ.get("RUSTUP_TOOLCHAIN") != "1.98.0":
        raise SystemExit("fixed Rust 1.98.0 environment required")
    main()
