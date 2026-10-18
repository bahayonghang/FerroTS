"""Paired CUDA replay of retained same-run Informer epoch checkpoints.

The test split is reported only. It never selects a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/reference"))
sys.path.insert(0, str(ROOT / "tools/validation"))
from p1_acceptance import sha256, write_json
from run_informer import UPSTREAM, install_model_source_path
from run_itransformer import install_numpy_compatibility

install_numpy_compatibility()
sys.path.insert(0, str(UPSTREAM))
install_model_source_path()
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

RUST = ROOT / "runs/p3-development/20261018-single-case/ETTh2-h96/rust/diag-ETTh2-h96-s2023"
DRAW_SEEDS = tuple(range(2023, 2031))


def rust_bundle() -> Path:
    pointers = list((RUST / "best").glob("*.json"))
    if len(pointers) != 1:
        raise ValueError("expected one Rust best pointer")
    generation = json.loads(pointers[0].read_text())["generation"]
    return RUST / "checkpoints" / generation


def buffer_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if not any(part in key for part in ("running_mean", "running_var", "num_batches_tracked")):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    if not attempt.is_relative_to((ROOT / "runs/p3-development").resolve()):
        raise ValueError("attempt must be under runs/p3-development")
    official = attempt / "official"
    rows = [json.loads(line) for line in (official / "epoch-checkpoints.jsonl").read_text().splitlines()]
    if [row["epoch"] for row in rows] != [1, 2]:
        raise ValueError("same-run epoch checkpoints are incomplete")
    bundle = rust_bundle()
    states = {
        f"official_epoch{row['epoch']}": torch.load(
            row["checkpoint"], map_location="cpu", weights_only=True
        ) for row in rows
    }
    rust_state = dict(load_file(bundle / "model.safetensors"))
    for key, value in load_file(bundle / "buffers.safetensors").items():
        rust_state[key] = value.to(torch.int64) if key.endswith("num_batches_tracked") else value
    states["rust_epoch1"] = rust_state
    keys = set(states["official_epoch1"])
    if any(set(state) != keys for state in states.values()):
        raise ValueError("candidate state keys differ")
    candidates = {
        f"official_epoch{row['epoch']}": dict(
            path=row["checkpoint"], checkpoint_sha256=sha256(Path(row["checkpoint"])),
            buffer_sha256=buffer_digest(states[f"official_epoch{row['epoch']}"]),
            num_batches_tracked=int(states[f"official_epoch{row['epoch']}"][
                "encoder.conv_layers.0.norm.num_batches_tracked"].item()),
        ) for row in rows
    }
    candidates["rust_epoch1"] = dict(
        path=str(bundle), model_sha256=sha256(bundle / "model.safetensors"),
        buffers_file_sha256=sha256(bundle / "buffers.safetensors"),
        buffer_sha256=buffer_digest(rust_state),
        num_batches_tracked=int(rust_state[
            "encoder.conv_layers.0.norm.num_batches_tracked"].item()),
    )
    for row in rows:
        name = f"official_epoch{row['epoch']}"
        if candidates[name]["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError(f"{name} checkpoint hash changed")
        if candidates[name]["buffer_sha256"] != row["bn_sha256"]:
            raise ValueError(f"{name} BatchNorm buffer hash changed")
    if (candidates["official_epoch1"]["num_batches_tracked"],
        candidates["official_epoch2"]["num_batches_tracked"],
        candidates["rust_epoch1"]["num_batches_tracked"]) != (265, 530, 265):
        raise ValueError("candidate BatchNorm batch counters differ from epoch identity")

    model_args = Namespace(**json.loads((official / "resolved-args.json").read_text()))
    exp = Exp_Long_Term_Forecast(model_args)
    data, _ = exp._get_data(args.split)
    loader = DataLoader(data, batch_size=model_args.batch_size,
                        shuffle=False, num_workers=0, drop_last=False)
    if len(data) != 2785 or len(loader) != 88:
        raise ValueError("unexpected ETTh2 replay coverage")
    criterion = exp._select_criterion()
    original_randint = torch.randint
    result_rows = []

    def score(name: str, draw_seed: int) -> dict:
        exp.model.load_state_dict(states[name], strict=True)
        exp.model.to("cuda:0")
        exp.model.eval()
        torch.manual_seed(draw_seed)
        random.seed(draw_seed)
        np.random.seed(draw_seed)
        cpu_before = torch.get_rng_state().cpu().numpy().tobytes()
        cuda_before = torch.cuda.get_rng_state("cuda:0").cpu().numpy().tobytes()
        sample_digest = hashlib.sha256()
        sample_calls = 0

        def tapped_randint(*a, **kw):
            nonlocal sample_calls
            values = original_randint(*a, **kw)
            array = values.detach().cpu().contiguous().numpy()
            sample_digest.update(str(array.shape).encode())
            sample_digest.update(array.tobytes())
            sample_calls += 1
            return values

        torch.randint = tapped_randint
        mse_batches = []
        mae_batches = []
        sum_sq = sum_abs = 0.0
        labels = 0
        try:
            with torch.no_grad():
                for bx, by, bx_mark, by_mark in loader:
                    bx = bx.float().to("cuda:0")
                    by = by.float()
                    bx_mark = bx_mark.float().to("cuda:0")
                    by_mark = by_mark.float().to("cuda:0")
                    decoder = torch.cat((by[:, :model_args.label_len, :],
                                         torch.zeros_like(by[:, -model_args.pred_len:, :])),
                                        dim=1).float().to("cuda:0")
                    pred = exp.model(bx, bx_mark, decoder, by_mark)[:, -model_args.pred_len:, :]
                    target = by[:, -model_args.pred_len:, :].to("cuda:0")
                    delta = (pred - target).float()
                    mse_batches.append(float(criterion(pred, target).item()))
                    mae_batches.append(float(delta.abs().mean().item()))
                    sum_sq += float(delta.square().sum().item())
                    sum_abs += float(delta.abs().sum().item())
                    labels += delta.numel()
        finally:
            torch.randint = original_randint
        cpu_after = torch.get_rng_state().cpu().numpy().tobytes()
        cuda_after = torch.cuda.get_rng_state("cuda:0").cpu().numpy().tobytes()
        if buffer_digest(dict(exp.model.state_dict())) != candidates[name]["buffer_sha256"]:
            raise ValueError(f"{name} BatchNorm buffers changed during evaluation")
        return dict(candidate=name, draw_seed=draw_seed, windows=len(data),
                    batches=len(loader), labels=labels,
                    mse_batch_equal=float(np.mean(mse_batches)),
                    mae_batch_equal=float(np.mean(mae_batches)),
                    mse_label_weighted=sum_sq / labels,
                    mae_label_weighted=sum_abs / labels,
                    tail_batch_windows=len(data) % model_args.batch_size,
                    tail_mse=mse_batches[-1], tail_mae=mae_batches[-1],
                    sample_calls=sample_calls, sample_sha256=sample_digest.hexdigest(),
                    rng_before_sha256=hashlib.sha256(cpu_before + cuda_before).hexdigest(),
                    rng_after_sha256=hashlib.sha256(cpu_after + cuda_after).hexdigest())

    for draw_seed in DRAW_SEEDS:
        paired = [score(name, draw_seed) for name in states]
        if len({row["sample_calls"] for row in paired}) != 1 or \
           len({row["sample_sha256"] for row in paired}) != 1 or \
           len({row["rng_before_sha256"] for row in paired}) != 1:
            raise ValueError(f"paired sparse sampling differs for draw {draw_seed}")
        result_rows.extend(paired)
        print(f"{args.split} draw {draw_seed}: " + ", ".join(
            f"{row['candidate']}={row['mse_label_weighted']:.9f}" for row in paired
        ), flush=True)

    report = dict(classification="paired development replay; no D4 or test selection",
                  split=args.split, draw_seeds=DRAW_SEEDS,
                  replay_script_sha256=sha256(Path(__file__)),
                  capture_script_sha256=sha256(ROOT / "tools/validation/p3_same_run_checkpoints.py"),
                  attempt_identity_sha256=sha256(attempt / "identity.json"),
                  official_receipt_sha256=sha256(official / "receipt.json"),
                  data_sha256=sha256(ROOT / "data/ETT-small/ETTh2.csv"),
                  evaluator="pinned Exp_Long_Term_Forecast; ordered full windows, paired torch.randint draws",
                  precision=dict(cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                 cudnn_tf32=torch.backends.cudnn.allow_tf32),
                  candidates=candidates, rows=result_rows)
    output = attempt / "replay" / f"report-{args.split}.json"
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise ValueError(f"replay report exists: {output}")
    write_json(output, report)
    print(output, flush=True)


if __name__ == "__main__":
    main()
