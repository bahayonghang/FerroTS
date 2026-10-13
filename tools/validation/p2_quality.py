"""Record the Linux CUDA quality gate. Windows keeps tools/validation/p2_quality.ps1."""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from p1_acceptance import sha256, write_json
from p2_acceptance import QUALITY_COMMANDS


def require_environment() -> str:
    if os.environ.get("RUSTUP_AUTO_INSTALL") != "0" or os.environ.get("RUSTUP_TOOLCHAIN") != "1.98.0":
        raise SystemExit("fixed Rust 1.98.0 environment required")
    if not os.environ.get("CUDA_PATH") or not os.environ.get("CARGO_TARGET_DIR"):
        raise SystemExit("Linux CUDA cargo environment required")
    if "1.98.0" not in os.environ.get("RUSTC", "") or "1.98.0" not in os.environ.get("RUSTDOC", ""):
        raise SystemExit("RUSTC and RUSTDOC must be pinned to 1.98.0")
    return subprocess.check_output(["rustup", "which", "cargo", "--toolchain", "1.98.0"], text=True).strip()


def input_hashes() -> dict:
    files = [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
    for folder in ("crates", "configs", "tests/fixtures", "tools/validation"):
        files.extend(path for path in (ROOT / folder).rglob("*") if path.is_file() and path.suffix in {".rs", ".toml", ".json", ".safetensors", ".py", ".ps1"})
    files.extend(path for path in (ROOT / "tools/reference").glob("*") if path.is_file() and path.suffix in {".py", ".toml", ".lock"})
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(set(files))}


def run_command(name: str, executable: str, expected_failure: bool, directory: Path, records: list, inputs: dict) -> None:
    arguments = QUALITY_COMMANDS[name]
    log = directory / f"{name}.log"
    started = time.perf_counter()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"START quality {name}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run([executable, *arguments], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=os.environ)
    text = log.read_text(encoding="utf-8", errors="replace")
    passed = result.returncode != 0 if expected_failure else result.returncode == 0
    if expected_failure:
        passed = passed and "requested_device: cuda:9" in text and "device initialization" in text
    records.append(dict(
        key=name, executable=executable, arguments=arguments, started_at=started_at,
        ended_at=dt.datetime.now(dt.timezone.utc).isoformat(), seconds=time.perf_counter() - started,
        exit_code=result.returncode, expected_failure=expected_failure, passed=passed,
        log=str(log.resolve()), log_sha256=sha256(log),
        rustup_auto_install=os.environ["RUSTUP_AUTO_INSTALL"], rustup_toolchain=os.environ["RUSTUP_TOOLCHAIN"],
        rustc=os.environ["RUSTC"], rustdoc=os.environ["RUSTDOC"], target="x86_64-unknown-linux-gnu",
        cuda_path=os.environ["CUDA_PATH"], cl=None, rustflags=None, nvcc_append_flags=None,
    ))
    write_json(directory / "quality.json", dict(passed=all(row["passed"] for row in records), script_sha256=sha256(Path(__file__)), records=records, inputs=inputs))
    print(f"END quality {name} exit={result.returncode} passed={passed}", flush=True)
    if not passed:
        raise SystemExit(f"Quality gate failed: {name}")


def main() -> None:
    cargo = require_environment()
    suite = (ROOT / sys.argv[1]).resolve()
    if not (suite / "suite.json").is_file():
        raise SystemExit("preflight suite.json is missing")
    final = suite / "quality.json"
    if final.is_file():
        import json
        existing = json.loads(final.read_text(encoding="utf-8"))
        if existing.get("passed"):
            print("VERIFIED existing quality gate", flush=True)
            return
    directory = suite / "quality" / dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    recorded = input_hashes()
    records: list = []
    plan = [
        ("fmt", cargo, False), ("cuda-build", cargo, False), ("cuda-clippy", cargo, False),
        ("cuda-test", cargo, False), ("cuda-doctor", cargo, False), ("cuda-invalid-device", cargo, True),
        ("python-tests", "uv", False), ("task-validate", "uv", False),
        ("private-data", "bash", False), ("diff-check", "git", False),
    ]
    for name, executable, expected_failure in plan:
        run_command(name, executable, expected_failure, directory, records, recorded)
    write_json(final, dict(passed=True, script_sha256=sha256(Path(__file__)), records=records, inputs=recorded))
    print(f"PASS full quality gates; records: {directory}", flush=True)


if __name__ == "__main__":
    main()
