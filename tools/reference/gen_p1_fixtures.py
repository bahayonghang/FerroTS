"""Generate public ETT and synthetic DLinear fixtures from the pinned TSLib."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save

from fetch_upstream import CACHE, UPSTREAMS, git

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "tests/fixtures/p1"
UPSTREAM = CACHE / "Time-Series-Library"
SHA = UPSTREAMS["Time-Series-Library"][1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def emit(name: str, data: bytes, check: bool) -> None:
    path = DEST / name
    if check:
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"fixture differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(f"{'verified' if check else 'generated'} {path.relative_to(ROOT)} ({len(data)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    check = parser.parse_args().check
    if git("rev-parse", "HEAD", cwd=UPSTREAM) != SHA:
        raise ValueError("TSLib SHA does not match the protocol")
    sys.path.insert(0, str(UPSTREAM))
    from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
    from models.DLinear import Model
    from utils.timefeatures import time_features

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tensors = {}
    cases = []
    for case_id, individual, batch, length, horizon, channels, kernel in [
        ("shared", False, 2, 5, 3, 3, 3),
        ("individual", True, 2, 5, 3, 3, 3),
        ("length_one", False, 1, 1, 2, 2, 25),
    ]:
        torch.manual_seed(137)
        model = Model(SimpleNamespace(task_name="long_term_forecast", seq_len=length, pred_len=horizon, moving_avg=kernel, enc_in=channels), individual=individual)
        x = torch.randn(batch, length, channels, requires_grad=True)
        target = torch.randn(batch, horizon, channels)
        prefix = case_id + "."
        tensors[prefix + "input"] = x.detach().clone()
        tensors[prefix + "target"] = target.clone()
        for name, parameter in model.named_parameters():
            tensors[prefix + "initial." + name] = parameter.detach().clone()
        seasonal, trend = model.decompsition(x)
        prediction = model(x, None, None, None)
        loss = (prediction - target).square().mean()
        tensors[prefix + "seasonal"] = seasonal.detach().contiguous()
        tensors[prefix + "trend"] = trend.detach().contiguous()
        tensors[prefix + "prediction"] = prediction.detach().contiguous()
        tensors[prefix + "loss"] = loss.detach().reshape(1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False, fused=False)
        loss.backward()
        tensors[prefix + "input_grad"] = x.grad.contiguous()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            tensors[prefix + "grad." + name] = parameter.grad.contiguous().clone()
        optimizer.step()
        for name, parameter in model.named_parameters():
            tensors[prefix + "updated." + name] = parameter.detach().contiguous().clone()
        cases.append(dict(id=case_id, individual=individual, batch=batch, lookback=length, horizon=horizon, channels=channels, moving_avg=kernel))
    metadata = dict(version=1, source="synthetic only", upstream_sha=SHA, torch=torch.__version__, seed=137, cases=cases, learning_rate=0.001, atol=1e-5, rtol=1e-4)
    emit("dlinear.safetensors", save(tensors, metadata={"fixture": json.dumps(metadata, sort_keys=True)}), check)
    emit("dlinear.json", (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(), check)

    datasets = []
    for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        path = ROOT / "data/ETT-small" / (name + ".csv")
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        cls = Dataset_ETT_hour if name.startswith("ETTh") else Dataset_ETT_minute
        counts = {}
        scaler = None
        for horizon in (96, 192, 336, 720):
            counts[str(horizon)] = {}
            for split in ("train", "val", "test"):
                ds = cls(SimpleNamespace(augmentation_ratio=0), str(path.parent), flag=split, size=[96, 48, horizon], features="M", data_path=path.name, timeenc=1, freq="h")
                counts[str(horizon)][split] = len(ds)
                scaler = ds.scaler
        indices = sorted(set([0, 1, 23, 24, 95, 96, len(frame) // 2, len(frame) - 1]))
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame.date.iloc[indices]))
        time_rows = [dict(row=int(row), timestamp=str(stamp), features=features.tolist()) for row, stamp, features in zip(indices, timestamps, time_features(timestamps, freq="h").T)]
        datasets.append(dict(name=name, rows=len(frame), columns=list(frame.columns[1:]), sha256=digest(path), counts=counts, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), time_rows=time_rows))
    stamps = pd.DatetimeIndex(["2024-02-29 23:59:00", "2024-12-31 00:00:00", "2025-01-01 01:15:00"])
    calendar = {freq: [dict(timestamp=str(stamp), features=values.tolist()) for stamp, values in zip(stamps, time_features(stamps, freq=freq).T)] for freq in ("h", "t")}
    public = dict(version=1, source="public ETT and synthetic calendar timestamps", upstream_sha=SHA, datasets=datasets, calendar=calendar)
    emit("ett.json", (json.dumps(public, indent=2, sort_keys=True) + "\n").encode(), check)


if __name__ == "__main__":
    main()
