"""Run pinned TSLib Informer in an isolated ignored run directory with a receipt."""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fetch_upstream import CACHE, UPSTREAMS, git
from run_itransformer import digest

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = CACHE / "Time-Series-Library"

BOOTSTRAP = r'''
import argparse, inspect, json, os, sys, textwrap
from pathlib import Path
entry, seed = sys.argv[1:3]
sys.path.insert(0, str(Path(entry).parent))
sys.path.insert(0, os.environ["FERROTS_REFERENCE_TOOLS"])
import torch
from run_itransformer import install_numpy_compatibility
install_numpy_compatibility()
from run_informer import install_model_source_path, begin_observation
install_model_source_path()
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import p2_observer as observer
source = observer.instrument(inspect.getsource(Exp_Long_Term_Forecast.train))
source = source.replace("_p2_begin(self, epoch, len(train_loader))", "_p2_begin(self, epoch, train_loader)")
def begin(model, epoch, loader):
    begin_observation(observer, model, epoch, loader)
namespace = dict(Exp_Long_Term_Forecast.train.__globals__, _p2_begin=begin,
    _p2_first_batch=observer.first_batch, _p2_end_train=observer.end_train,
    _p2_validation=observer.validation, _p2_checkpoint=observer.checkpoint)
exec(compile(textwrap.dedent(source), "<informer-performance-observer>", "exec"), namespace)
Exp_Long_Term_Forecast.train = namespace["train"]
parse = argparse.ArgumentParser.parse_args
def record_args(self, *a, **kw):
    args = parse(self, *a, **kw)
    if not args.use_gpu or args.gpu_type != "cuda" or not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
        raise RuntimeError("Informer reference requires the requested CUDA device")
    Path("resolved-args.json").write_text(json.dumps(vars(args), sort_keys=True, indent=2))
    import numpy as np
    Path("environment.json").write_text(json.dumps(dict(python=sys.version,
        torch=torch.__version__, torch_sha=torch.version.git_version,
        cuda=torch.version.cuda, device=torch.cuda.get_device_name(args.gpu),
        numpy=np.__version__, compatibility="np.Inf=np.inf; absolute model scanner path; timing observer"),sort_keys=True,indent=2))
    return args
argparse.ArgumentParser.parse_args = record_args
source = Path(entry).read_text(encoding="utf-8")
if source.count("fix_seed = 2021") != 1:
    raise RuntimeError("official seed assignment changed")
source = source.replace("fix_seed = 2021", "fix_seed = " + seed)
sys.argv = [entry] + sys.argv[3:]
exec(compile(source, entry, "exec"), {"__name__":"__main__", "__file__":entry})
'''


def begin_observation(observer, model, epoch, loader) -> None:
    """Count the actual retained tail without changing the frozen P2 observer."""
    observer.begin(model, epoch, len(loader))
    windows = len(loader) * loader.batch_size if loader.drop_last else len(loader.dataset)
    model._p2_observation.update(training_windows=windows,
        steady_windows=max(0, windows - loader.batch_size))


def install_model_source_path() -> None:
    """Resolve scanner filesystem access without changing Python module names."""
    import textwrap
    from exp.exp_basic import Exp_Basic
    method = Exp_Basic._scan_models_directory
    source = inspect.getsource(method)
    for call in ("os.path.exists", "os.listdir"):
        anchor = call + "(models_dir)"
        if source.count(anchor) != 1:
            raise RuntimeError("official model scanner anchor changed: " + call)
        source = source.replace(anchor, call + "(" + repr(str(UPSTREAM / "models")) + ")")
    namespace = dict(method.__globals__)
    exec(compile(textwrap.dedent(source), "<informer-model-source-path>", "exec"), namespace)
    Exp_Basic._scan_models_directory = namespace["_scan_models_directory"]


def prepare(arguments: list[str]) -> tuple[list[str], Path]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ("root_path", "data_path", "model", "model_id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--checkpoints")
    args, _ = parser.parse_known_args(arguments)
    if args.model != "Informer" or args.checkpoints is not None:
        raise ValueError("runner owns Informer checkpoints and requires model Informer")
    for name in ("root_path", "data_path", "model", "model_id"):
        if sum(value == f"--{name}" or value.startswith(f"--{name}=") for value in arguments) != 1:
            raise ValueError(f"duplicate --{name}")
    data_root = (ROOT / args.root_path).resolve()
    data = (data_root / args.data_path).resolve()
    if not data.is_file() or not data.is_relative_to(data_root):
        raise ValueError("reference data must already exist inside root_path; no remote fallback")
    if not args.model_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.model_id):
        raise ValueError("model_id must be a simple ASCII identifier")
    normalized = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--root_path":
            normalized.extend([value, str(data_root) + os.sep])
            index += 2
        elif value.startswith("--root_path="):
            normalized.append("--root_path=" + str(data_root) + os.sep)
            index += 1
        else:
            normalized.append(value)
            index += 1
    return normalized, data


def run(seed: int, arguments: list[str], destination: Path) -> int:
    sha = UPSTREAMS["Time-Series-Library"][1]
    if git("rev-parse", "HEAD", cwd=UPSTREAM) != sha or git("status", "--porcelain", "--untracked-files=all", cwd=UPSTREAM):
        raise ValueError("reference requires the clean pinned TSLib checkout")
    normalized, data = prepare(arguments)
    destination = destination.resolve()
    if not destination.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("reference output must be inside ignored runs/")
    destination.mkdir(parents=True, exist_ok=False)
    normalized.extend(["--checkpoints", str(destination / "checkpoints") + os.sep])
    command = [sys.executable, "-c", BOOTSTRAP, str(UPSTREAM / "run.py"), str(seed), *normalized]
    receipt = dict(format_version=1, model="Informer", seed=seed, upstream_sha=sha,
        wrapper_sha256=digest(__file__), observer_sha256=digest(ROOT / "tools/reference/p2_observer.py"),
        data_sha256=digest(data), arguments=normalized, command=command,
        source_hashes={str(p.relative_to(UPSTREAM)): digest(p) for p in sorted(UPSTREAM.rglob("*.py"))},
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status="running", exit_code=None)
    receipt_path = destination / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    begin = time.perf_counter()
    process = None
    try:
        with (destination / "stdout.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=destination,
                env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", FERROTS_REFERENCE_TOOLS=str(ROOT / "tools/reference")),
                stdout=log, stderr=subprocess.STDOUT)
            receipt["exit_code"] = process.wait()
        receipt["status"] = "completed" if receipt["exit_code"] == 0 else "failed"
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            receipt["exit_code"] = process.wait()
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - begin
        receipt["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        receipt["artifacts"] = {str(p.relative_to(destination)): dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted(destination.rglob("*")) if p.is_file() and p != receipt_path}
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(dict(output=str(destination), status=receipt["status"], exit_code=receipt["exit_code"])))
    return receipt["exit_code"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("run_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.run_args[1:] if args.run_args[:1] == ["--"] else args.run_args
    sys.exit(run(args.seed, arguments, args.output))
