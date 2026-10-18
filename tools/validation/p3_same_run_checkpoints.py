"""Retain two epochs from one pinned official ETTh2/H96 training attempt.

Development diagnosis only. The pinned checkout, reference runner, training
profile, early stopping, and formal suite are not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/reference"))
sys.path.insert(0, str(ROOT / "tools/validation"))
import run_informer
from p1_acceptance import sha256, source_snapshot, write_json
from p3_smoke import official_arguments


HOOK = r'''
import hashlib

def _diag_rng(model):
    cpu = torch.get_rng_state().cpu().numpy().tobytes()
    cuda = torch.cuda.get_rng_state(model.device).cpu().numpy().tobytes()
    return hashlib.sha256(cpu + cuda).hexdigest()

def _diag_validation(model, name, data, loader, criterion):
    epoch = model._p2_observation['epoch']
    if epoch not in (1, 2):
        return observer.validation(model, name, data, loader, criterion)
    before = _diag_rng(model)
    samples = hashlib.sha256()
    batches = hashlib.sha256()
    calls = 0
    windows = 0
    tail = 0
    actual_randint = torch.randint
    def tapped_randint(*args, **kwargs):
        nonlocal calls
        value = actual_randint(*args, **kwargs)
        array = value.detach().cpu().contiguous().numpy()
        samples.update(str(array.shape).encode())
        samples.update(array.tobytes())
        calls += 1
        return value
    class TapLoader:
        def __iter__(self):
            nonlocal windows, tail
            for batch in loader:
                windows += len(batch[0])
                tail = len(batch[0])
                for tensor in batch:
                    array = tensor.detach().cpu().contiguous().numpy()
                    batches.update(str(array.shape).encode())
                    batches.update(array.tobytes())
                yield batch
    torch.randint = tapped_randint
    try:
        loss = observer.validation(model, name, data, TapLoader(), criterion)
    finally:
        torch.randint = actual_randint
    row = dict(epoch=epoch, split=name, loss=float(loss), windows=windows,
               tail_windows=tail, shuffle=type(loader.sampler).__name__,
               sample_calls=calls, sample_sha256=samples.hexdigest(),
               batches_sha256=batches.hexdigest(), rng_before_sha256=before,
               rng_after_sha256=_diag_rng(model))
    with Path('validation-sampling.jsonl').open('a') as output:
        output.write(json.dumps(row, sort_keys=True) + '\n')
    return loss

def _diag_checkpoint(model, early, loss, path, training_loss, optimizer):
    epoch = model._p2_observation['epoch']
    before = _diag_rng(model)
    observer.checkpoint(model, early, loss, path, training_loss, optimizer)
    if epoch not in (1, 2):
        return
    checkpoint = Path(path) / f'epoch-{epoch}.pth'
    torch.save(model.model.state_dict(), checkpoint)
    after = _diag_rng(model)
    if before != after:
        raise RuntimeError('checkpoint capture changed CPU/CUDA RNG state')
    state = model.model.state_dict()
    bn = {key: value for key, value in state.items()
          if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key}
    digest = hashlib.sha256()
    for key, value in sorted(bn.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    file_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    row = dict(epoch=epoch, validation_loss=float(loss), training_loss=float(training_loss),
               selected_best=bool(early.best_score == -float(loss) and early.counter == 0),
               best_validation_loss=float(early.val_loss_min),
               early_stop_counter=early.counter, checkpoint=str(checkpoint),
               checkpoint_sha256=file_hash, state_keys=len(state),
               bn_buffer_keys=sorted(bn), bn_sha256=digest.hexdigest(),
               rng_before_sha256=before, rng_after_sha256=after,
               precision=dict(cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                              cudnn_tf32=torch.backends.cudnn.allow_tf32,
                              cudnn_benchmark=torch.backends.cudnn.benchmark))
    with Path('epoch-checkpoints.jsonl').open('a') as output:
        output.write(json.dumps(row, sort_keys=True) + '\n')
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    if not attempt.is_relative_to((ROOT / "runs/p3-development").resolve()):
        raise ValueError("attempt must be under runs/p3-development")
    source = source_snapshot()
    data = ROOT / "data/ETT-small/ETTh2.csv"
    if source["source_sha256"] != "8a39414aa10f879daa0dd692590a26fd17460b6f362c170965a465ae04a06a47":
        raise ValueError("runtime source differs from the reviewed P3 diagnosis")
    if sha256(data) != "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b":
        raise ValueError("ETTh2 data differs from the reviewed diagnosis")
    anchor = "namespace = dict(Exp_Long_Term_Forecast.train.__globals__, _p2_begin=begin,"
    if run_informer.BOOTSTRAP.count(anchor) != 1:
        raise ValueError("official observer anchor changed")
    bootstrap = run_informer.BOOTSTRAP.replace(anchor, HOOK + "\n" + anchor, 1)
    old = "_p2_validation=observer.validation, _p2_checkpoint=observer.checkpoint)"
    if bootstrap.count(old) != 1:
        raise ValueError("official observer callbacks changed")
    run_informer.BOOTSTRAP = bootstrap.replace(
        old, "_p2_validation=_diag_validation, _p2_checkpoint=_diag_checkpoint)", 1
    )
    profile = official_arguments("ETTh2", 96, "same_run_ETTh2_h96_s2023")
    attempt.mkdir(parents=True, exist_ok=False)
    write_json(attempt / "identity.json", dict(
        classification="bounded development diagnosis; zero formal D4 entries",
        task=".trellis/tasks/09-28-p3-informer", seed=2023,
        source=source, data_sha256=sha256(data),
        rust_config_sha256=sha256(ROOT / "configs/reference/informer/ett-h2-h96.toml"),
        official_profile=profile, script_sha256=sha256(Path(__file__)),
        wrapper_sha256=sha256(ROOT / "tools/reference/run_informer.py"),
        bootstrap_sha256=hashlib.sha256(run_informer.BOOTSTRAP.encode()).hexdigest(),
    ))
    rc = run_informer.run(2023, profile, attempt / "official")
    if rc:
        raise SystemExit(rc)
    rows = [json.loads(line) for line in (attempt / "official/epoch-checkpoints.jsonl").read_text().splitlines()]
    if [row["epoch"] for row in rows] != [1, 2]:
        raise ValueError("same-run epoch 1/2 checkpoints missing")
    for row in rows:
        if sha256(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError("captured checkpoint hash mismatch")
    write_json(attempt / "capture-summary.json", dict(attempt=str(attempt), epochs=rows,
        receipt_sha256=sha256(attempt / "official/receipt.json")))


if __name__ == "__main__":
    main()
