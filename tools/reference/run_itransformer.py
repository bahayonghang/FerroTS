"""Run the pinned official iTransformer on local data and retain a receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fetch_upstream import CACHE, UPSTREAMS, git

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = CACHE / "iTransformer"
SEED_LINE = "fix_seed = 2023"
BOOTSTRAP = '''import argparse, json, sys, os
from pathlib import Path
path, seed = sys.argv[1:3]
sys.path.insert(0, str(Path(path).parent))
import torch
sys.path.insert(0, os.environ["FERROTS_REFERENCE_TOOLS"])
from run_itransformer import install_numpy_compatibility
install_numpy_compatibility()
from p2_observer import install
install()
parse = argparse.ArgumentParser.parse_args
def record_args(self, *a, **kw):
    args = parse(self, *a, **kw)
    if args.use_gpu and (not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count()):
        raise RuntimeError("requested reference CUDA device is unavailable")
    device = torch.cuda.get_device_name(args.gpu) if args.use_gpu else "cpu"
    Path("resolved-args.json").write_text(json.dumps(vars(args), sort_keys=True, indent=2), encoding="utf-8")
    import numpy as np
    Path("environment.json").write_text(json.dumps(dict(python=sys.version, torch=torch.__version__, torch_sha=torch.version.git_version, cuda=torch.version.cuda, device=device, numpy=np.__version__, compatibility="np.Inf=np.inf (legacy constant alias)"), sort_keys=True, indent=2), encoding="utf-8")
    return args
argparse.ArgumentParser.parse_args = record_args
src = Path(path).read_text(encoding="utf-8")
if src.count("fix_seed = 2023") != 1:
    raise RuntimeError("official seed assignment changed")
src = src.replace("fix_seed = 2023", "fix_seed = " + seed)
sys.argv = [path] + sys.argv[3:]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
'''


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_numpy_compatibility():
    """Restore the removed spelling used by the pinned EarlyStopping constructor."""
    import numpy as np
    if not hasattr(np, "Inf"):
        np.Inf = np.inf


def prepare(run_args: list[str]):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ("root_path", "data_path", "model_id", "model"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--checkpoints")
    args, _ = parser.parse_known_args(run_args)
    if args.model != "iTransformer":
        raise ValueError("this reference runner requires model iTransformer")
    if args.checkpoints is not None:
        raise ValueError("the wrapper owns the checkpoint directory")
    for name in ("root_path", "data_path", "model_id", "model"):
        if sum(a == f"--{name}" or a.startswith(f"--{name}=") for a in run_args) != 1:
            raise ValueError(f"duplicate --{name}")
    data_root = (ROOT / args.root_path).resolve()
    data_file = (data_root / args.data_path).resolve()
    if not data_file.is_file():
        raise ValueError(f"local data file is missing: {data_file}")
    if not data_file.is_relative_to(data_root):
        raise ValueError("data_path must be inside root_path")
    if not args.model_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.model_id):
        raise ValueError("model_id must contain only ASCII letters, digits, underscores or hyphens")
    normalized = []
    index = 0
    while index < len(run_args):
        value = run_args[index]
        if value == "--root_path":
            normalized.extend([value, str(data_root) + os.sep])
            index += 2
        elif value.startswith("--root_path="):
            normalized.append("--root_path=" + str(data_root) + os.sep)
            index += 1
        else:
            normalized.append(value)
            index += 1
    return args, normalized, data_file


def run(seed: int, run_args: list[str], destination: Path | None = None):
    sha = UPSTREAMS["iTransformer"][1]
    if git("rev-parse", "HEAD", cwd=UPSTREAM) != sha:
        raise ValueError("iTransformer checkout does not match the pinned commit")
    if git("status", "--porcelain", "--untracked-files=all", cwd=UPSTREAM):
        raise ValueError("iTransformer checkout is dirty")
    entry = UPSTREAM / "run.py"
    if entry.read_text(encoding="utf-8").count(SEED_LINE) != 1:
        raise ValueError("official seed assignment changed")
    args, normalized, data_file = prepare(run_args)
    destination = (destination or ROOT / "runs/reference" /
                   f"{time.time_ns()}_{args.model_id}_seed{seed}").resolve()
    if not destination.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("reference output must be inside the ignored runs directory")
    destination.mkdir(parents=True, exist_ok=False)
    normalized.extend(["--checkpoints", str(destination / "checkpoints") + os.sep])
    command = [sys.executable, "-c", BOOTSTRAP, str(entry), str(seed), *normalized]
    receipt = dict(format_version=1, seed=seed, upstream_sha=sha,
                   wrapper_sha256=digest(__file__), entry_sha256=digest(entry),
                   data_sha256=digest(data_file), arguments=normalized,
                   source_hashes={str(p.relative_to(UPSTREAM)): digest(p) for p in sorted(UPSTREAM.rglob("*.py")) if "__pycache__" not in p.parts},
                   status="running", exit_code=None)
    receipt_path = destination / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    begin = time.perf_counter()
    process = None
    try:
        with (destination / "stdout.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=destination,
                                       env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", FERROTS_REFERENCE_TOOLS=str(ROOT / "tools/reference")),
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            receipt["exit_code"] = process.wait()
        receipt["status"] = "completed" if receipt["exit_code"] == 0 else "failed"
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            receipt["exit_code"] = process.wait()
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - begin
        receipt["artifacts"] = {str(p.relative_to(destination)): dict(bytes=p.stat().st_size, sha256=digest(p))
                                  for p in sorted(destination.rglob("*")) if p.is_file() and p != receipt_path}
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt["exit_code"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("run_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.run_args[1:] if args.run_args[:1] == ["--"] else args.run_args
    sys.exit(run(args.seed, arguments, args.output))
