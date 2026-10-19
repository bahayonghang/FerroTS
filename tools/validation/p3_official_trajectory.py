"""Bounded, paired official Informer CUDA update trace. Development diagnosis only.

The pinned checkout and production reference runner remain untouched. Two independent
official processes advance one optimizer update at a time under a parent barrier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/reference"))
sys.path.insert(0, str(ROOT / "tools/validation"))
import run_informer
from p1_acceptance import sha256, source_snapshot, write_json
from p3_smoke import official_arguments

SOURCE_SHA = "8a39414aa10f879daa0dd692590a26fd17460b6f362c170965a465ae04a06a47"
DATA_SHA = "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"
MAX_STEPS = 100
MAX_SECONDS = 900
MAX_BYTES = 2_000_000_000

HOOK = r'''
import hashlib, json, os, time
import numpy as np
from pathlib import Path
from torch import nn

_diag_dir = Path(os.environ['FERROTS_TRAJECTORY_DIR'])
_diag_dir.mkdir(parents=True, exist_ok=True)
_diag_step = None
_diag_items = []
_diag_arrays = {}
_diag_tensors = {}

def _diag_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())

def _diag_rng():
    return dict(cpu=_diag_hash(torch.get_rng_state()),
                cuda=_diag_hash(torch.cuda.get_rng_state(0)))

def _diag_state(model):
    return {name: _diag_hash(value) for name, value in sorted(model.state_dict().items())}

def _diag_optimizer(optimizer, model):
    values = {}
    for name, parameter in sorted(model.named_parameters()):
        state = optimizer.state.get(parameter, {})
        values[name] = {key: _diag_hash(value) if isinstance(value, torch.Tensor) else value
                        for key, value in sorted(state.items())}
    return values

_diag_original_init = Exp_Long_Term_Forecast.__init__
def _diag_init(self, args):
    before = dict(cpu=_diag_hash(torch.get_rng_state()),
                  numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest())
    _diag_original_init(self, args)
    after = _diag_rng()
    row = dict(before=before, after=after, state=_diag_state(self.model),
               precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                              cudnn_tf32=torch.backends.cudnn.allow_tf32,
                              cudnn_benchmark=torch.backends.cudnn.benchmark,
                              cudnn_deterministic=torch.backends.cudnn.deterministic,
                              amp=args.use_amp), device=str(self.device))
    (_diag_dir / 'initial.json').write_text(json.dumps(row, sort_keys=True))
    torch.save(self.model.state_dict(), _diag_dir / 'initial-state.pth')
Exp_Long_Term_Forecast.__init__ = _diag_init

_diag_original_randint = torch.randint
def _diag_randint(*args, **kwargs):
    result = _diag_original_randint(*args, **kwargs)
    if _diag_step is not None:
        name = 'sample.%02d' % len([key for key in _diag_arrays if key.startswith('sample.')])
        array = result.detach().cpu().contiguous().numpy()
        _diag_arrays[name] = array
        _diag_items.append(dict(name=name, value=_diag_hash(result)))
    return result
torch.randint = _diag_randint

_diag_original_dropout = nn.Dropout.forward
def _diag_dropout(self, input):
    if _diag_step is None or not self.training or self.p == 0:
        return _diag_original_dropout(self, input)
    state = torch.cuda.get_rng_state(input.device) if input.is_cuda else torch.get_rng_state()
    try:
        mask = _diag_original_dropout(self, torch.ones_like(input)) != 0
    finally:
        if input.is_cuda:
            torch.cuda.set_rng_state(state, input.device)
        else:
            torch.set_rng_state(state)
    result = _diag_original_dropout(self, input)
    name = 'dropout.%02d' % len([key for key in _diag_arrays if key.startswith('dropout.')])
    array = mask.detach().cpu().contiguous().numpy()
    _diag_arrays[name] = np.packbits(array.reshape(-1))
    _diag_items.append(dict(name=name, shape=list(array.shape), value=_diag_hash(mask)))
    return result
nn.Dropout.forward = _diag_dropout

def _diag_module(name):
    def hook(_module, _inputs, output):
        if _diag_step is None:
            return
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            label = 'op.%03d.%s' % (len([x for x in _diag_items if x['name'].startswith('op.')]), name)
            _diag_items.append(dict(name=label, value=_diag_hash(tensor)))
            _diag_tensors[label] = tensor.detach().cpu()
    return hook

def _diag_prepare(model, optimizer):
    from layers.SelfAttention_Family import ProbAttention
    for name, module in model.model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.BatchNorm1d, nn.Linear, nn.LayerNorm, ProbAttention)):
            module.register_forward_hook(_diag_module(name))

def _diag_begin(model, epoch, index, batch_x, batch_y, batch_x_mark, batch_y_mark, optimizer):
    global _diag_step, _diag_items, _diag_arrays, _diag_tensors, _diag_started
    _diag_step = epoch * len(model._diag_loader) + index + 1
    if _diag_step > 100:
        raise RuntimeError('bounded probe exceeded 100 updates')
    _diag_started = time.perf_counter()
    _diag_items, _diag_arrays, _diag_tensors = [], {}, {}
    model._diag_row = dict(step=_diag_step, epoch=epoch + 1, batch=index + 1,
        rng_before=_diag_rng(), state_before=_diag_state(model.model),
        optimizer_before=_diag_optimizer(optimizer, model.model),
        input={name: _diag_hash(value) for name, value in
               [('x', batch_x), ('y', batch_y), ('x_mark', batch_x_mark), ('y_mark', batch_y_mark)]})
    _diag_tensors['input.x'] = batch_x.detach().cpu()
    _diag_tensors['input.y'] = batch_y.detach().cpu()

def _diag_forward(model, output):
    model._diag_row['forward'] = _diag_hash(output)
    model._diag_row['rng_after_forward'] = _diag_rng()
    model._diag_row['state_after_forward'] = _diag_state(model.model)
    _diag_tensors['forward'] = output.detach().cpu()

def _diag_backward(model, loss):
    model._diag_row['loss'] = _diag_hash(loss)
    model._diag_row['loss_value'] = float(loss.detach().item())
    model._diag_row['gradients'] = {
        name: _diag_hash(parameter.grad) if parameter.grad is not None else None
        for name, parameter in sorted(model.model.named_parameters())}
    _diag_tensors.update({'grad.' + name: parameter.grad.detach().cpu()
                          for name, parameter in model.model.named_parameters()
                          if parameter.grad is not None})

def _diag_snapshot(model, optimizer, step):
    torch.save(dict(model=model.model.state_dict(),
                    gradients={name: p.grad for name, p in model.model.named_parameters() if p.grad is not None},
                    optimizer=optimizer.state_dict(), tensors=_diag_tensors),
               _diag_dir / ('step-%03d.pth' % step))

def _diag_wait(model, optimizer, step):
    while True:
        if (_diag_dir / 'stop').exists():
            raise SystemExit(0)
        capture = _diag_dir / ('capture-%03d' % step)
        if capture.exists():
            _diag_snapshot(model, optimizer, step)
            capture.unlink()
            (_diag_dir / ('captured-%03d' % step)).touch()
        gate = _diag_dir / ('go-%03d' % (step + 1))
        if gate.exists():
            gate.unlink()
            return
        time.sleep(0.02)

def _diag_after_update(model, optimizer):
    global _diag_step
    step = _diag_step
    row = model._diag_row
    row.update(items=_diag_items, rng_after_update=_diag_rng(),
               state_after_update=_diag_state(model.model),
               optimizer_after=_diag_optimizer(optimizer, model.model),
               elapsed_seconds=time.perf_counter() - _diag_started)
    np.savez_compressed(_diag_dir / ('sampling-%03d.npz' % step), **_diag_arrays)
    if step in (1, 2, 5):
        _diag_snapshot(model, optimizer, step)
    pending = _diag_dir / ('step-%03d.json.pending' % step)
    pending.write_text(json.dumps(row, sort_keys=True, allow_nan=False))
    pending.rename(_diag_dir / ('step-%03d.json' % step))
    _diag_step = None
    _diag_wait(model, optimizer, step)
'''


def bootstrap() -> str:
    anchor = "namespace = dict(Exp_Long_Term_Forecast.train.__globals__, _p2_begin=begin,"
    if run_informer.BOOTSTRAP.count(anchor) != 1:
        raise ValueError("official bootstrap anchor changed")
    body = run_informer.BOOTSTRAP.replace(anchor, HOOK + "\n" + anchor, 1)
    anchor = "source = source.replace(\"_p2_begin(self, epoch, len(train_loader))\", \"_p2_begin(self, epoch, train_loader)\")"
    if body.count(anchor) != 1:
        raise ValueError("official train source anchor changed")
    injection = r'''
def _diag_replace(old, new, count=1):
    global source
    if source.count(old) != count:
        raise RuntimeError('official train anchor changed: ' + old)
    source = source.replace(old, new)
_diag_replace('model_optim = self._select_optimizer()',
              'model_optim = self._select_optimizer()\n        self._diag_loader = train_loader\n        _diag_prepare(self, model_optim)')
_diag_replace('for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):',
              'for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):\n                _diag_begin(self, epoch, i, batch_x, batch_y, batch_x_mark, batch_y_mark, model_optim)')
_diag_replace('else:\n                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)',
              'else:\n                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)\n                    _diag_forward(self, outputs)')
_diag_replace('loss.backward()\n                    model_optim.step()',
              'loss.backward()\n                    _diag_backward(self, loss)\n                    model_optim.step()\n                    _diag_after_update(self, model_optim)')
'''
    body = body.replace(anchor, anchor + "\n" + injection, 1)
    old = "_p2_validation=observer.validation, _p2_checkpoint=observer.checkpoint)"
    if body.count(old) != 1:
        raise ValueError("official callback anchor changed")
    return body.replace(old, old[:-1] + ", _diag_prepare=_diag_prepare, _diag_begin=_diag_begin, "
                        "_diag_forward=_diag_forward, _diag_backward=_diag_backward, "
                        "_diag_after_update=_diag_after_update)", 1)


def worker(attempt: Path) -> None:
    run_informer.BOOTSTRAP = bootstrap()
    arguments = official_arguments("ETTh2", 96, "trajectory_ETTh2_h96_s2023")
    rc = run_informer.run(2023, arguments, attempt / "official")
    raise SystemExit(rc)


def wait_file(path: Path, processes: list[subprocess.Popen], started: float) -> None:
    while not path.exists():
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError(f"bounded CUDA diagnosis exceeded {MAX_SECONDS}s")
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"official process exited {process.returncode} before {path}")
        time.sleep(0.05)


def first_difference(a: object, b: object, path: str = "") -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}: missing"
            result = first_difference(a[key], b[key], f"{path}.{key}")
            if result:
                return result
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (left, right) in enumerate(zip(a, b)):
            result = first_difference(left, right, f"{path}[{index}]")
            if result:
                return result
        return None
    return None if a == b else path


def compare_step(left: dict, right: dict) -> str | None:
    for field in ("step", "epoch", "batch", "rng_before", "state_before",
                  "optimizer_before", "input", "items", "forward", "rng_after_forward",
                  "state_after_forward", "loss", "loss_value", "gradients",
                  "optimizer_after", "state_after_update", "rng_after_update"):
        difference = first_difference(left[field], right[field], field)
        if difference:
            return difference
    return None


def attempt_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run(output: Path) -> None:
    output = output.resolve()
    if not output.is_relative_to((ROOT / "runs/p3-development").resolve()):
        raise ValueError("output must be in runs/p3-development")
    if output.exists():
        raise FileExistsError(output)
    source = source_snapshot()
    data = ROOT / "data/ETT-small/ETTh2.csv"
    if source["source_sha256"] != SOURCE_SHA or sha256(data) != DATA_SHA:
        raise ValueError("runtime source or ETTh2 data differs from reviewed identity")
    upstream = run_informer.UPSTREAM
    if run_informer.git("rev-parse", "HEAD", cwd=upstream) != run_informer.UPSTREAMS["Time-Series-Library"][1] or run_informer.git("status", "--porcelain", "--untracked-files=all", cwd=upstream):
        raise ValueError("pinned official checkout is not clean")
    output.mkdir(parents=True)
    write_json(output / "identity.json", dict(task=".trellis/tasks/09-28-p3-informer",
        classification="bounded official-to-official CUDA diagnosis; zero formal D4 entries",
        source=source, data_sha256=sha256(data), seed=2023,
        profile=official_arguments("ETTh2", 96, "trajectory_ETTh2_h96_s2023"),
        script_sha256=sha256(Path(__file__)), wrapper_sha256=sha256(ROOT / "tools/reference/run_informer.py"),
        observer_sha256=sha256(ROOT / "tools/reference/p2_observer.py"),
        bootstrap_sha256=hashlib.sha256(bootstrap().encode()).hexdigest(),
        limits=dict(processes=2, updates_per_process=MAX_STEPS, seconds=MAX_SECONDS,
                    bytes_per_attempt=MAX_BYTES)))
    processes = []
    started = time.monotonic()
    try:
        for name in ("attempt-001", "attempt-002"):
            attempt = output / name
            attempt.mkdir()
            trace = attempt / "trace"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(attempt)]
            with (attempt / "launcher.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                    env=dict(os.environ, FERROTS_TRAJECTORY_DIR=str(trace)), stdout=log,
                    stderr=subprocess.STDOUT)
            processes.append(process)
            wait_file(trace / "step-001.json", processes, started)
            if name == "attempt-001":
                first_seconds = json.loads((trace / "step-001.json").read_text())["elapsed_seconds"]
                first_bytes = attempt_bytes(attempt)
                write_json(output / "cost-probe.json", dict(first_step_seconds=first_seconds,
                    first_attempt_bytes=first_bytes, projected_seconds=first_seconds * 200,
                    snapshot_upper_bound_bytes=first_bytes * 6))
                if first_seconds * 200 > MAX_SECONDS or first_bytes * 6 > MAX_BYTES:
                    raise RuntimeError("measured one-step cost exceeds bounded diagnosis limits")
        for step in range(1, MAX_STEPS + 1):
            a = output / "attempt-001/trace" / f"step-{step:03d}.json"
            b = output / "attempt-002/trace" / f"step-{step:03d}.json"
            if step > 1:
                for index in range(2):
                    trace = output / f"attempt-{index+1:03d}/trace"
                    (trace / f"go-{step:03d}").touch()
                    wait_file(trace / f"step-{step:03d}.json", processes, started)
            if step == 1:
                initial_a = json.loads((output / "attempt-001/trace/initial.json").read_text())
                initial_b = json.loads((output / "attempt-002/trace/initial.json").read_text())
                initial_difference = first_difference(initial_a, initial_b, "initial")
                if initial_difference:
                    write_json(output / "comparison.json", dict(first_difference_step=0,
                        first_difference_path=initial_difference, last_equal_step=None,
                        scope="two new independent official processes; original full-run trajectory unavailable"))
                    break
            left, right = json.loads(a.read_text()), json.loads(b.read_text())
            difference = compare_step(left, right)
            if difference:
                for index in range(2):
                    trace = output / f"attempt-{index+1:03d}/trace"
                    if step not in (1, 2, 5):
                        (trace / f"capture-{step:03d}").touch()
                        wait_file(trace / f"captured-{step:03d}", processes, started)
                write_json(output / "comparison.json", dict(first_difference_step=step,
                    first_difference_path=difference, last_equal_step=step - 1,
                    scope="two new independent official processes; original full-run trajectory unavailable"))
                break
            if any(attempt_bytes(output / f"attempt-{index+1:03d}") > MAX_BYTES for index in range(2)):
                raise RuntimeError("bounded diagnosis disk limit reached")
            if step == 5:
                write_json(output / "first-five.json", dict(equal=True, steps=list(range(1, 6))))
            if step == MAX_STEPS:
                write_json(output / "comparison.json", dict(first_difference_step=None,
                    last_equal_step=MAX_STEPS, scope="two new independent official processes only"))
        for index in range(2):
            (output / f"attempt-{index+1:03d}/trace/stop").touch()
        for process in processes:
            process.wait(timeout=30)
        write_json(output / "summary.json", dict(comparison=json.loads((output / "comparison.json").read_text()),
            attempt_bytes={f"attempt-{i+1:03d}": attempt_bytes(output / f"attempt-{i+1:03d}") for i in range(2)},
            elapsed_seconds=time.monotonic()-started, exit_codes=[p.returncode for p in processes]))
    except BaseException as error:
        write_json(output / "controller-failure.json", dict(error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started, processes_started=len(processes)))
        raise
    finally:
        for index, process in enumerate(processes):
            if process.poll() is None:
                trace = output / f"attempt-{index+1:03d}/trace"
                trace.mkdir(parents=True, exist_ok=True)
                (trace / "stop").touch()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.worker):
        parser.error("select exactly one of --output or --worker")
    if args.worker:
        worker(args.worker)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
