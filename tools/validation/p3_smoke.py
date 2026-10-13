"""Measure Informer CUDA diagnostics and prepare a reviewable formal-run budget.

These runs are diagnostics only. No entry is added to the 105-run formal matrix.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from gpu_observer import DeviceMemoryObserver
from p1_acceptance import artifacts, bundle_at, read_json, sha256, source_snapshot, write_json

ETT = {"ETTh1": "ett-h1", "ETTh2": "ett-h2", "ETTm1": "ett-m1", "ETTm2": "ett-m2"}
SEEDS = (2021, 2022, 2023)
HORIZONS = (96, 192, 336, 720)


def official_arguments(name: str, horizon: int, model_id: str, epochs: int = 10) -> list[str]:
    return ["--task_name", "long_term_forecast", "--is_training", "1",
        "--root_path", "data/ETT-small", "--data_path", name + ".csv",
        "--model_id", model_id, "--model", "Informer", "--data", name,
        "--features", "M", "--seq_len", "96", "--label_len", "48",
        "--pred_len", str(horizon), "--e_layers", "2", "--d_layers", "1",
        "--factor", "3", "--enc_in", "7", "--dec_in", "7", "--c_out", "7",
        "--d_model", "512", "--n_heads", "8", "--d_ff", "2048",
        "--batch_size", "32", "--learning_rate", "0.0001", "--train_epochs", str(epochs),
        "--patience", "3", "--num_workers", "0", "--gpu", "0", "--gpu_type", "cuda",
        "--des", "Exp", "--itr", "1"]


def planned_matrix(binary: Path, output: Path) -> list[dict]:
    rows = []
    for name, short in ETT.items():
        for horizon in HORIZONS:
            config = f"configs/reference/informer/{short}-h{horizon}.toml"
            for seed in SEEDS:
                key = f"{name}-h{horizon}-s{seed}"
                rows.append(dict(engine="official", dataset=name, horizon=horizon, seed=seed,
                    command=[sys.executable, str(ROOT / "tools/reference/run_informer.py"), "--seed", str(seed),
                        "--output", str(output / "official" / key), "--", *official_arguments(name, horizon, key)]))
                rows.append(dict(engine="rust", dataset=name, horizon=horizon, seed=seed,
                    config=config, config_sha256=sha256(ROOT / config), command=[str(binary), "train", "--config", config,
                        "--device", "cuda:0", "--seed", str(seed), "--output-root", str(output / "rust"), "--run-id", key]))
    for horizon in (15, 30, 60):
        config = f"configs/engineering/informer-cement-h{horizon}.toml"
        for seed in SEEDS:
            rows.append(dict(engine="rust", dataset="cement", horizon=horizon, seed=seed,
                config=config, config_sha256=sha256(ROOT / config), command=[str(binary), "train", "--config", config,
                    "--device", "cuda:0", "--seed", str(seed), "--output-root", str(output / "rust"),
                    "--run-id", f"cement-h{horizon}-s{seed}"]))
    assert len(rows) == 105
    return rows


def execute(directory: Path, command: list[str], *, observe: bool = False) -> tuple[dict, dict]:
    directory.mkdir(parents=True, exist_ok=False)
    receipt = dict(command=command, cwd=str(ROOT), started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   status="running", exit_code=None)
    write_json(directory / "receipt.json", receipt)
    print("START " + directory.name, flush=True)
    begin = time.perf_counter()
    try:
        with (directory / "stdout.json").open("w") as out, (directory / "stderr.log").open("w") as err:
            if observe:
                with DeviceMemoryObserver(directory / "memory", device=0):
                    result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
            else:
                result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
    except BaseException as error:
        receipt.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}",
                       elapsed_seconds=time.perf_counter()-begin,
                       ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        receipt["artifacts"] = artifacts([p for p in directory.rglob("*") if p.is_file() and p.name != "receipt.json"])
        write_json(directory / "receipt.json", receipt)
        raise
    receipt.update(exit_code=result.returncode, status="completed" if result.returncode == 0 else "failed",
                   elapsed_seconds=time.perf_counter() - begin, ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    receipt["artifacts"] = artifacts([p for p in directory.rglob("*") if p.is_file() and p.name != "receipt.json"])
    write_json(directory / "receipt.json", receipt)
    if result.returncode:
        raise RuntimeError(f"diagnostic failed: {directory}; see retained stdout and stderr")
    print("END " + directory.name, flush=True)
    return read_json(directory / "stdout.json"), receipt


def rust_case(output: Path, binary: Path, name: str, config: str, steps: int | None) -> dict:
    report, _ = execute(output / f"{name}-data", [str(binary), "data", "inspect", "--config", config,
        "--cache-dir", str(output / "cache")])
    command = [str(binary), "train", "--config", config, "--device", "cuda:0", "--run-id", name,
        "--output-root", str(output / "rust"), "--cache-dir", str(output / "cache")]
    if steps is not None:
        command.extend(["--stop-after-steps", str(steps)])
    summary, receipt = execute(output / name, command, observe=True)
    run = Path(summary["run_dir"])
    checkpoint = bundle_at(run, "last")
    state = read_json(checkpoint / "trainer-state.json")
    if steps is not None:
        if summary.get("status") != "paused" or summary["optimizer_steps"] != steps:
            raise ValueError("bounded CUDA diagnostic did not stop at the requested boundary")
        throughput = state["steady_windows"] / state["steady_seconds"]
    else:
        throughput = sum(e["steady_windows"] for e in summary["epochs"]) / sum(e["steady_seconds"] for e in summary["epochs"])
        if state["phase"] != "complete":
            raise ValueError("synthetic smoke did not complete")
        execute(output / f"{name}-evaluate", [str(binary), "evaluate", "--checkpoint", str(run / "best")])
        execute(output / f"{name}-predict", [str(binary), "predict", "--checkpoint", str(run / "best"),
            "--future", "--output", str(output / "synthetic-future.csv")])
    if not math.isfinite(throughput) or throughput <= 0:
        raise ValueError("invalid measured throughput")
    return dict(name=name, config=config, config_sha256=sha256(ROOT / config), summary=summary,
        checkpoint=str(checkpoint), state=state, window_counts=report["window_counts"],
        steady_windows_per_second=throughput, memory=read_json(output / name / "memory/device-memory.json"),
        receipt=receipt, classification="diagnostic only; zero formal matrix entries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-bin", type=Path, default=ROOT / "target/linux-cuda/debug/ferrots")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    output, binary = args.output.resolve(), args.cuda_bin.resolve()
    if not output.is_relative_to((ROOT / "runs/p3-development").resolve()) or args.steps < 2:
        raise ValueError("diagnostics require runs/p3-development and at least two CUDA steps")
    output.mkdir(parents=True, exist_ok=False)
    source = source_snapshot()
    doctor = subprocess.run([str(binary), "doctor", "--device", "cuda:0"], cwd=ROOT, text=True, capture_output=True, check=True)
    if source["source_sha256"] not in doctor.stdout:
        raise ValueError("CUDA binary does not match the current runtime source")
    write_json(output / "identity.json", dict(source=source, binary=dict(path=str(binary), sha256=sha256(binary)),
        tools=artifacts([Path(__file__), ROOT / "tools/reference/run_informer.py", ROOT / "tools/reference/gen_p3_fixtures.py"])))
    rows = []
    for name, config, steps in [
        ("synthetic", "configs/reference/informer/synthetic.toml", None),
        ("ETTh1-h96", "configs/reference/informer/ett-h1-h96.toml", args.steps),
        ("ETTh1-h720", "configs/reference/informer/ett-h1-h720.toml", args.steps),
        ("cement-h60", "configs/engineering/informer-cement-h60.toml", args.steps),
    ]:
        rows.append(rust_case(output, binary, name, config, steps))
        write_json(output / "rust-smoke.json", dict(passed=True, runs=rows, formal_entries=0))
    finish_budget(output, binary, source, rows)


def finish_budget(output: Path, binary: Path, source: dict, rows: list[dict]) -> None:
    nonce = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    official_output = output / ("official-ETTh1-h96-" + nonce)
    result, receipt = execute(output / ("official-command-" + nonce), [sys.executable, str(ROOT / "tools/reference/run_informer.py"),
        "--seed", "2021", "--output", str(official_output), "--", *official_arguments("ETTh1", 96, "smoke_ETTh1_96", 1)], observe=True)
    epochs = [json.loads(line) for line in (official_output / "performance.jsonl").read_text().splitlines()]
    if len(epochs) != 1 or result["status"] != "completed":
        raise ValueError("official one-epoch diagnostic did not complete normally")
    torch_throughput = epochs[0]["steady_samples_per_second"]
    speed = {row["name"]: row["steady_windows_per_second"] for row in rows}
    # Estimates are deliberately separate from measured data. Validation,
    # checkpointing, evaluation, exports and cold parsing add overhead.
    budget = dict(passed=True, formal_entries=0, scope="diagnostics and budget only", runs=rows,
        official=dict(result=result, receipt=receipt, epochs=epochs),
        estimates=dict(max_epochs=10, rust_ETT_training_only_hours_estimate=None, rust_cement_training_only_hours_estimate=None,
                       excludes="validation/checkpoint/evaluation/export/parse overhead; early stopping unknown"))
    ett_seconds = 0.
    for name, short in ETT.items():
        for horizon in HORIZONS:
            cfg = tomllib.loads((ROOT / f"configs/reference/informer/{short}-h{horizon}.toml").read_text())
            count = cfg["data"]["split"]["train_end"] - 96 - horizon + 1
            # Endpoint interpolation is an estimate, not an observed interior-horizon rate.
            seconds_per_window = 1/speed["ETTh1-h96"] + (horizon-96)/(720-96)*(1/speed["ETTh1-h720"]-1/speed["ETTh1-h96"])
            ett_seconds += count*seconds_per_window*10*3
    cement_counts = rows[-1]["window_counts"][0]
    budget["estimates"]["rust_ETT_training_only_hours_estimate"] = ett_seconds/3600
    budget["estimates"]["rust_cement_training_only_hours_estimate"] = cement_counts/speed["cement-h60"]*10*9/3600
    budget["estimates"]["official_ETT_hours_h96_rate_only"] = sum(
        (tomllib.loads((ROOT / f"configs/reference/informer/{short}-h{horizon}.toml").read_text())["data"]["split"]["train_end"]-96-horizon+1)
        /torch_throughput*10*3 for short in ETT.values() for horizon in HORIZONS)/3600
    budget["estimates"]["assumptions"] = "ETT Rust interpolates H96/H720 endpoint cost; cement uses measured H60 cost at all H (H15/H30 counts slightly larger); official uses H96 rate at all H and is a rough estimate. One experiment per GPU; figures are GPU hours, not two-GPU elapsed time."
    write_json(output / "budget.json", budget)
    formal = ROOT / "runs/p3-acceptance/proposed-cuda-suite"
    write_json(output / "planned-matrix.json", dict(formal_runs=105, executed=0, source=source, runs=planned_matrix(binary, formal)))
    print(json.dumps(dict(budget=str(output / "budget.json"), matrix=str(output / "planned-matrix.json"), formal_entries=0),indent=2))


if __name__ == "__main__":
    main()
