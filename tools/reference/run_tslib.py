"""Run the official Time-Series-Library run.py with a given seed on local data.

Usage:
    uv run --project tools/reference python tools/reference/run_tslib.py \
        --seed 2021 -- <run.py arguments from an official script>

- Give `--root_path` relative to the repository root, for example `data/ETT-small/`.
  The data file must exist. The wrapper stops before TSLib can download data from
  Hugging Face.
- TSLib sets `fix_seed = 2021` in run.py. The wrapper replaces that value with `--seed`.
- Output goes to runs/reference/<time>_<model_id>_seed<seed>/: stdout.log, metrics.json,
  and checkpoints/. TSLib also writes results/ and test_results/ in its own directory
  under tools/reference/.cache/.

Development-time only. Production builds do not use Python or this script.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from fetch_upstream import CACHE, UPSTREAMS, git

REPO_ROOT = Path(__file__).resolve().parents[2]
TSLIB = CACHE / "Time-Series-Library"
RUN_PY = TSLIB / "run.py"
SEED_LINE = "fix_seed = 2021"
METRIC_RE = re.compile(r"^mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", re.M)

# Runs run.py as __main__ in a child process after the seed line is replaced.
BOOTSTRAP = (
    "import sys\n"
    "path, seed = sys.argv[1], sys.argv[2]\n"
    "src = open(path, encoding='utf-8').read()\n"
    f"src = src.replace({SEED_LINE!r}, 'fix_seed = ' + seed)\n"
    "sys.argv = [path] + sys.argv[3:]\n"
    "exec(compile(src, path, 'exec'), {'__name__': '__main__', '__file__': path})\n"
)


def main() -> int:
    # TSLib prints emoji; a GBK console cannot encode them.
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("run_args", nargs=argparse.REMAINDER, help="arguments after --")
    args = parser.parse_args()
    run_args = args.run_args[1:] if args.run_args[:1] == ["--"] else args.run_args

    paths = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    paths.add_argument("--root_path", required=True)
    paths.add_argument("--data_path", required=True)
    paths.add_argument("--model_id", required=True)
    data_args, _ = paths.parse_known_args(run_args)

    sha = UPSTREAMS["Time-Series-Library"][1]
    if not RUN_PY.exists() or git("rev-parse", "HEAD", cwd=TSLIB) != sha:
        raise SystemExit(
            f"{TSLIB} is not at {sha}. Run tools/reference/fetch_upstream.py."
        )
    if RUN_PY.read_text(encoding="utf-8").count(SEED_LINE) != 1:
        raise SystemExit(f"{RUN_PY}: expected exactly one line '{SEED_LINE}'")

    root = (REPO_ROOT / data_args.root_path).resolve()
    data_file = root / data_args.data_path
    if not data_file.is_file():
        raise SystemExit(f"data file not found: {data_file}")

    run_dir = (
        REPO_ROOT
        / "runs"
        / "reference"
        / (f"{time.strftime('%Y%m%d-%H%M%S')}_{data_args.model_id}_seed{args.seed}")
    )
    run_dir.mkdir(parents=True)
    run_args = [
        str(root) + os.sep if a == data_args.root_path else a for a in run_args
    ] + ["--checkpoints", str(run_dir / "checkpoints") + os.sep]

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-c", BOOTSTRAP, str(RUN_PY), str(args.seed), *run_args]
    start = time.perf_counter()
    with open(run_dir / "stdout.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=TSLIB,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    elapsed = time.perf_counter() - start

    text = (run_dir / "stdout.log").read_text(encoding="utf-8")
    tests = [{"mse": float(m[0]), "mae": float(m[1])} for m in METRIC_RE.findall(text)]
    metrics = {
        "upstream": "Time-Series-Library",
        "upstream_sha": sha,
        "seed": args.seed,
        "run_args": run_args,
        "exit_code": code,
        "elapsed_s": round(elapsed, 1),
        "tests": tests,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"run dir: {run_dir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
