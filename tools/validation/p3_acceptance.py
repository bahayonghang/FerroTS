"""Freeze and execute the approved Informer CUDA matrix with immutable attempts."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import json
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

from p1_acceptance import (ROOT, PRIVATE, PRIVATE_SHA, TSLIB_SHA, CEMENT_COUNTS,
    artifacts, bundle_at, read_json, require, sha256, source_snapshot, verify_artifacts, write_json)
from p2_acceptance import execute, validate_epochs, verify_receipt
from p2_quality import require_environment
from p3_smoke import ETT, HORIZONS, SEEDS, official_arguments

UPSTREAM = ROOT / "tools/reference/.cache/Time-Series-Library"
# 2026-10-13 required 186 GiB free for the first suite. The 2026-10-14 approval
# keeps that suite and starts a second one. Measured free space was about 140 GiB,
# and the finished ETT portion was 51G. Freeze now requires 100 GiB free. Each
# later cell still stops below 5 GiB free and does not delete retained artifacts.
RERUN_FREE_BYTES = 100 * 2**30
RERUN_APPROVAL = dict(
    date="2026-10-14",
    decision="用户批准",
    budget="second suite beside retained 20261013-linux-cuda; prior attempt measured 51G and about 11h47m for 96/105; cement unmeasured; freeze floor 100 GiB; rust parent 95ea51d",
)


def config_path(name, horizon):
    return f"configs/engineering/informer-cement-h{horizon}.toml" if name == "cement" else f"configs/reference/informer/{ETT[name]}-h{horizon}.toml"


def key_for(engine, name, horizon, seed):
    return f"{engine}-{name}-h{horizon}-s{seed}"


def matrix(cement=False):
    if cement:
        return [("rust", "cement", h, s) for h in CEMENT_COUNTS for s in SEEDS]
    return [(engine, name, h, s) for name in ETT for h in HORIZONS for s in SEEDS for engine in ("official", "rust")]


def context(suite):
    suite = (ROOT / suite).resolve()
    require(suite.is_relative_to(ROOT / "runs/p3-acceptance") and suite != ROOT / "runs/p3-acceptance", "P3 suite must be inside runs/p3-acceptance")
    info = read_json(suite / "suite.json")
    require(info["upstream_sha"] == TSLIB_SHA and info["formal_runs"] == 105 and info["ddof"] == 0, "suite protocol changed")
    require(source_snapshot()["source_sha256"] == info["source"]["source_sha256"], "runtime changed; use a new suite")
    verify_artifacts(info["inputs"])
    verify_artifacts(info["binary_files"])
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip() == TSLIB_SHA, "upstream changed")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=UPSTREAM, text=True).strip(), "upstream dirty")
    return suite, info


def gate(suite, name, value):
    value["suite_sha256"] = sha256(suite / "suite.json")
    write_json(suite / (name + ".json"), value)
    return value


def checked_gate(suite, info, name):
    value = read_json(suite / (name + ".json"))
    require(value["passed"] and value["suite_sha256"] == sha256(suite / "suite.json"), "missing or stale gate: " + name)
    for receipt in value.get("receipts", []):
        verify_receipt(receipt, info, suite)
    if value.get("artifacts"):
        verify_artifacts(value["artifacts"])
    return value


def freeze(suite, binary):
    import torch
    suite = (ROOT / suite).resolve()
    require(suite.is_relative_to(ROOT / "runs/p3-acceptance") and suite != ROOT / "runs/p3-acceptance", "invalid P3 suite path")
    require_environment()
    require(torch.__version__ == "2.14.0+cu130" and torch.version.git_version == "08187d9e0fba026dc8217405802ab5381dc88d90", "torch pin differs")
    require(torch.cuda.device_count() >= 2, "approved two-GPU execution requires both CUDA devices")
    binary = (ROOT / binary).resolve()
    source = source_snapshot()
    require(source["source_sha256"] in subprocess.check_output([str(binary), "doctor", "--device", "cuda:0"], cwd=ROOT, text=True), "binary source differs")
    manifest = tomllib.loads((ROOT / "docs/datasets/manifest.toml").read_text())
    data = {}
    for item in manifest["file"]:
        path = ROOT / item["path"]
        require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], "public fingerprint differs")
        data[item["path"]] = item["sha256"]
    require(PRIVATE.stat().st_size == 1068005224 and sha256(PRIVATE) == PRIVATE_SHA, "private fingerprint differs")
    data[str(PRIVATE.relative_to(ROOT))] = PRIVATE_SHA
    paths = [ROOT / config_path(n, h) for n in ETT for h in HORIZONS]
    paths += [ROOT / config_path("cement", h) for h in CEMENT_COUNTS]
    for folder in ("tools/validation", "tools/reference"):
        paths += list((ROOT / folder).glob("*.py"))
    paths += list((ROOT / "tools/validation/tests").glob("test_*.py"))
    paths += [p for p in (ROOT / "crates").rglob("*.rs") if "tests" in p.relative_to(ROOT).parts]
    paths += [p for p in (ROOT / "tests/fixtures").rglob("*") if p.is_file()]
    paths += [ROOT / "docs/results/p1-cement-baselines.md", ROOT / "docs/results/p2-itransformer-cement.md",
              ROOT / "tools/reference/uv.lock", ROOT / "tools/reference/pyproject.toml"]
    require(shutil.disk_usage(ROOT).free > RERUN_FREE_BYTES, "approved disk allowance is unavailable")
    suite.mkdir(parents=True, exist_ok=False)
    # Preserve a suite-owned binary so subsequent builds cannot silently replace it.
    frozen = suite / "bin/ferrots"
    frozen.parent.mkdir()
    shutil.copy2(binary, frozen)
    info = dict(format_version=1, model="informer", upstream_sha=TSLIB_SHA, source=source,
        formal_runs=105, ddof=0, seeds=list(SEEDS), horizons=list(HORIZONS), devices=["cuda:0", "cuda:1"],
        approval=RERUN_APPROVAL,
        inputs=artifacts(paths), data=data, binary_files=artifacts([frozen]), binary=str(frozen),
        environment=dict(os=platform.platform(), python=sys.version, torch=torch.__version__, torch_sha=torch.version.git_version,
            cuda=torch.version.cuda, gpus=[torch.cuda.get_device_name(i) for i in range(2)]),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), free_bytes=shutil.disk_usage(ROOT).free)
    write_json(suite / "suite.json", info)
    context(suite)
    gate(suite, "preflight", dict(passed=True, data_fingerprints=data))
    print(json.dumps(dict(suite=str(suite), source=source["source_sha256"], status="frozen", formal_entries=0)), flush=True)


def quality(suite, info, name):
    if (suite / (name + ".json")).exists():
        return checked_gate(suite, info, name)
    cargo = require_environment()
    receipts = []
    commands = [
        ("fmt", [cargo, "fmt", "--all", "--check"]),
        ("clippy", [cargo, "clippy", "--workspace", "--all-targets", "--features", "cuda", "--locked", "--", "-D", "warnings"]),
        ("workspace", [cargo, "test", "--workspace", "--features", "cuda", "--locked"]),
        ("doctor", [info["binary"], "doctor", "--device", "cuda:0", "--training-smoke-test"]),
        ("python", [sys.executable, "-m", "unittest", "discover", "-s", "tools/validation/tests", "-p", "test_*.py"]),
        ("task", [sys.executable, ".trellis/scripts/task.py", "validate", "09-28-p3-informer"]),
        ("privacy", ["bash", "tools/check_private_data.sh"]), ("diff", ["git", "diff", "--check"])]
    counts = {}
    for key, command in commands:
        receipt, output = execute(suite, info, name + "-" + key, command)
        receipts.append(receipt)
        if key == "workspace":
            results = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", output)
            require(results and all(int(f) == int(i) == 0 for _, f, i in results), "failed/ignored CUDA workspace tests")
            counts["rust"] = sum(int(p) for p, _, _ in results)
        if key == "python":
            log = Path(receipt["directory"]) / "stderr.log"
            text = output + log.read_text()
            match = re.search(r"Ran (\d+) tests", text)
            require(match and re.search(r"^OK$", text, re.M) and "skipped=" not in text, "Python test count/result missing")
            counts["python"] = int(match[1])
    invalid = subprocess.run([info["binary"], "doctor", "--device", "cuda:9"], cwd=ROOT, capture_output=True, text=True)
    text = invalid.stdout + invalid.stderr
    require(invalid.returncode != 0 and "requested_device: cuda:9" in text and "device initialization" in text, "invalid CUDA device did not fail explicitly")
    rejection = suite / (name + "-invalid-device.json")
    write_json(rejection, dict(command=[info["binary"], "doctor", "--device", "cuda:9"], exit_code=invalid.returncode, output=text, expected_failure=True))
    return gate(suite, name, dict(passed=True, counts=counts, receipts=receipts, artifacts=artifacts([rejection])))


def data_gate(suite, info):
    if (suite / "data-gate.json").exists():
        return checked_gate(suite, info, "data-gate")
    reports, receipts = {}, []
    expected = {r["name"]: r for r in read_json(ROOT / "tests/fixtures/p1/ett.json")["datasets"]}
    for name, horizons in [*[(n, HORIZONS) for n in ETT], ("cement", CEMENT_COUNTS)]:
        for horizon in horizons:
            config = config_path(name, horizon)
            receipt, output = execute(suite, info, f"data-{name}-{horizon}", [info["binary"], "data", "inspect", "--config", config, "--cache-dir", str(suite / "cache")])
            report = json.loads(output)
            audit = report["window_audit"]
            require(audit["all_same_segment"] and audit["all_target_split"] and audit["checked"] == sum(report["window_counts"]), "window leakage")
            cfg = tomllib.loads((ROOT / config).read_text())
            require(report["cache"]["source_sha256"] == info["data"][cfg["data"]["path"]], "data identity differs")
            counts = CEMENT_COUNTS[horizon] if name == "cement" else [expected[name]["counts"][str(horizon)][k] for k in ("train", "val", "test")]
            require(report["window_counts"] == counts, "window coverage differs")
            if name != "cement":
                for col, mean, scale in zip(report["scaler"]["columns"], expected[name]["mean"], expected[name]["scale"]):
                    require(abs(col["mean"] - mean) < 1e-9 and abs(col["scale"] - scale) < 1e-9, "train scaler differs")
            else:
                require(len(report["scaler"]["columns"]) == 18 and all(c["count"] == 303009 for c in report["scaler"]["columns"]), "private scaler coverage differs")
                if horizon != 15:
                    require(report["scaler"] == reports["cement-15"]["scaler"], "private scaler changes with horizon")
            reports[f"{name}-{horizon}"] = report
            receipts.append(receipt)
    return gate(suite, "data-gate", dict(passed=True, reports=reports, cache_dir=str(suite / "cache"), receipts=receipts))


def validate_config(actual, name, horizon, seed, device):
    expected = tomllib.loads((ROOT / config_path(name, horizon)).read_text())
    require(actual["model"] == expected["model"] and actual["window"] == expected["window"], "model/window profile differs")
    for key, value in expected["data"].items():
        if key == "missing":
            require(all(actual["data"][key].get(k) == v for k, v in value.items()), "missing policy differs")
        else:
            require(actual["data"].get(key) == value, "data profile differs: " + key)
    for key, value in dict(expected["train"], seed=seed, device=device).items():
        require(actual["train"].get(key, False if key == "drop_last" else None) == value, "training profile differs: " + key)
    require(not actual["data"].get("future_known_columns"), "unexpected future numeric values")


def verify_entry(entry, suite, info):
    identity = entry["identity"]
    engine, name, horizon, seed = (identity[k] for k in ("engine", "dataset", "horizon", "seed"))
    require((engine, name, horizon, seed) in matrix(name == "cement"), "invalid matrix cell")
    require(identity["model"] == "informer" and identity["device"] in info["devices"] and identity["source_sha256"] == info["source"]["source_sha256"], "entry identity differs")
    require(identity["config_sha256"] == info["inputs"][str(ROOT / config_path(name, horizon))]["sha256"], "entry config identity differs")
    verify_artifacts(entry["artifacts"])
    for receipt in entry["receipts"]:
        verify_receipt(receipt, info, suite)
    run = Path(entry["run"])
    require(run.is_relative_to(suite / engine), "run outside its suite")
    counts = read_json(suite / "data-gate.json")["reports"][f"{name}-{horizon}"]["window_counts"]
    cfg = tomllib.loads((ROOT / config_path(name, horizon)).read_text())
    channels = len(cfg["data"]["target_columns"])
    epochs = [json.loads(line) for line in (run / "performance.jsonl").read_text().splitlines()]
    if engine == "rust":
        require(identity["binary_sha256"] == info["binary_files"][info["binary"]]["sha256"], "entry binary identity differs")
        expected_command = [info["binary"], "train", "--config", config_path(name, horizon), "--seed", str(seed), "--device", identity["device"],
            "--cache-dir", str(suite / "cache"), "--output-root", str(run.parent), "--run-id", run.name]
        require(entry["receipts"][0]["command"] == expected_command, "Rust executed command differs")
        require(entry["receipts"][1]["command"] == [info["binary"], "evaluate", "--checkpoint", str(run / "best"), "--device", identity["device"], "--output", str(run / "evaluation.json")], "evaluation command differs")
        actual = tomllib.loads((run / "resolved-config.toml").read_text())
        validate_config(actual, name, horizon, seed, identity["device"])
        summary, evaluation = entry["summary"], entry["evaluation"]
        require(summary == read_json(run / "train-summary.json") and evaluation == read_json(run / "evaluation.json"), "Rust output changed")
        best = validate_epochs(summary["epochs"], counts[0], horizon, channels)
        require(summary["model"] == evaluation["model"] == "informer" and summary["best_epoch"] == best["epoch"], "selection/model differs")
        require(summary["access"]["windows"] == [counts[0] * len(epochs), counts[1] * len(epochs), 0], "training accesses test or misses windows")
        require(summary["source_sha256"] == identity["source_sha256"] and summary["data_sha256"] == evaluation["data_sha256"] == info["data"][cfg["data"]["path"]], "Rust provenance differs")
        require(evaluation["seed"] == seed and evaluation["windows"] == counts[2] and evaluation["metrics"]["standardized"]["overall"]["count"] == counts[2] * horizon * channels, "Rust test coverage differs")
        require(evaluation["sampling_policy"] == "evaluation-split-batch-v1" and evaluation["batch_size"] == 32 and evaluation["split"] == 2, "evaluation sampling identity differs")
        best_dir = bundle_at(run)
        manifest = read_json(best_dir / "manifest.json")
        require(manifest["format_version"] == 2 and manifest["upstream_sha"] == TSLIB_SHA and manifest["epoch"] == best["epoch"]
            and manifest["provenance"]["source_sha256"] == identity["source_sha256"] and evaluation["checkpoint_sha256"] == sha256(best_dir / "manifest.json"), "best checkpoint differs")
        require((best_dir / "buffers.safetensors").stat().st_size > 0 and read_json(bundle_at(run, "last") / "trainer-state.json")["phase"] == "complete", "buffers/end state missing")
        require(len(epochs) == len(summary["epochs"]), "performance coverage differs")
        for row, epoch in zip(epochs, summary["epochs"]):
            require(all(row[k] == epoch[k] for k in ("epoch", "training_windows", "train_seconds", "validation_seconds", "steady_windows", "steady_seconds")), "performance differs from epoch")
    else:
        import numpy as np
        result = read_json(run / "receipt.json")
        require(result == entry["result"] and result["status"] == "completed" and result["exit_code"] == 0 and result["seed"] == seed and result["upstream_sha"] == TSLIB_SHA, "official exit/source/seed differs")
        require(result["wrapper_sha256"] == info["inputs"][str(ROOT / "tools/reference/run_informer.py")]["sha256"] and result["data_sha256"] == info["data"][cfg["data"]["path"]], "official wrapper/data differs")
        verify_artifacts({str(UPSTREAM / p): {"sha256": h, "bytes": (UPSTREAM / p).stat().st_size} for p, h in result["source_hashes"].items()})
        verify_artifacts({str(run / p): value for p, value in result["artifacts"].items()})
        arguments = official_arguments(name, horizon, key_for(engine, name, horizon, seed))
        arguments[arguments.index("--gpu") + 1] = identity["device"].split(":")[1]
        arguments[arguments.index("--root_path") + 1] = str(ROOT / "data/ETT-small") + os.sep
        arguments += ["--checkpoints", str(run / "checkpoints") + os.sep]
        require(result["arguments"] == arguments, "official arguments differ")
        resolved = read_json(run / "resolved-args.json")
        require(resolved["model"] == "Informer" and resolved["use_gpu"] and resolved["gpu_type"] == "cuda" and resolved["gpu"] == int(identity["device"][-1]) and not resolved["use_amp"] and not resolved["use_multi_gpu"], "official device/precision differs")
        environment = read_json(run / "environment.json")
        require(environment["torch"] == info["environment"]["torch"] and environment["torch_sha"] == info["environment"]["torch_sha"] and environment["cuda"] == info["environment"]["cuda"] and environment["device"] == info["environment"]["gpus"][int(identity["device"][-1])], "official environment differs")
        log = (run / "stdout.log").read_text()
        for flag, count in zip(("train", "val", "test"), counts):
            require(re.search(rf"^{flag} {count}$", log, re.M), "official split differs")
        settings = re.findall(r">+testing : (.+?)<+", log)
        require(len(settings) == 1, "official result identity ambiguous")
        result_dir = run / "results" / settings[0]
        metrics = np.load(result_dir / "metrics.npy", allow_pickle=False)
        require(entry["metrics"] == dict(mae=float(metrics[0]), mse=float(metrics[1]), rmse=float(metrics[2])), "official metrics changed")
        for filename in ("pred.npy", "true.npy"):
            array = np.load(result_dir / filename, mmap_mode="r", allow_pickle=False)
            require(array.shape == (counts[2], horizon, channels) and array.dtype == np.float32, "official full test coverage differs")
            require(all(np.isfinite(array[i:i+256]).all() for i in range(0, len(array), 256)), "nonfinite official output")
        require((run / "checkpoints" / settings[0] / "checkpoint.pth").is_file(), "official best checkpoint missing")
        validate_epochs(epochs, counts[0], horizon, channels, official=True)
    require(all(math.isfinite(row["checkpoint_seconds"]) and row["checkpoint_seconds"] >= 0 for row in epochs), "checkpoint performance invalid")
    metrics = entry["metrics"] if engine == "official" else entry["evaluation"]["metrics"]["standardized"]["overall"]
    require(all(math.isfinite(metrics[k]) and metrics[k] >= 0 for k in ("mse", "mae", "rmse")), "nonfinite metrics")
    memory = read_json(Path(entry["receipts"][0]["directory"]) / "memory/device-memory.json")
    require(memory["device"] == int(identity["device"][-1]), "memory observed wrong GPU")


def run_cell(suite, info, cell, device):
    engine, name, horizon, seed = cell
    key = key_for(*cell)
    destination = suite / "entries" / (key + ".json")
    if destination.exists():
        entry = read_json(destination)
        verify_entry(entry, suite, info)
        return entry
    context(suite)
    require(shutil.disk_usage(ROOT).free > 5 * 2**30, "insufficient space for another experiment; retained artifacts are not deleted")
    run_id = key + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    run = suite / engine / run_id
    identity = dict(engine=engine, dataset=name, horizon=horizon, seed=seed, model="informer", device=device, source_sha256=info["source"]["source_sha256"],
        config_sha256=info["inputs"][str(ROOT / config_path(name, horizon))]["sha256"])
    if engine == "rust":
        identity["binary_sha256"] = info["binary_files"][info["binary"]]["sha256"]
        command = [info["binary"], "train", "--config", config_path(name, horizon), "--seed", str(seed), "--device", device,
            "--cache-dir", str(suite / "cache"), "--output-root", str(run.parent), "--run-id", run_id]
        receipt, output = execute(suite, info, key + "-train", command, observe=True, gpu=int(device[-1]))
        second, result = execute(suite, info, key + "-evaluate", [info["binary"], "evaluate", "--checkpoint", str(run / "best"), "--device", device, "--output", str(run / "evaluation.json")])
        entry = dict(identity=identity, run=str(run), summary=json.loads(output), evaluation=json.loads(result), receipts=[receipt, second])
    else:
        import numpy as np
        arguments = official_arguments(name, horizon, key)
        arguments[arguments.index("--gpu") + 1] = device[-1]
        receipt, _ = execute(suite, info, key, [sys.executable, str(ROOT / "tools/reference/run_informer.py"), "--seed", str(seed), "--output", str(run), "--", *arguments], observe=True, gpu=int(device[-1]))
        log = (run / "stdout.log").read_text()
        settings = re.findall(r">+testing : (.+?)<+", log)
        require(len(settings) == 1, "official output setting missing")
        values = np.load(run / "results" / settings[0] / "metrics.npy", allow_pickle=False)
        entry = dict(identity=identity, run=str(run), result=read_json(run / "receipt.json"),
            metrics=dict(mae=float(values[0]), mse=float(values[1]), rmse=float(values[2])), receipts=[receipt])
    entry["artifacts"] = artifacts([p for p in run.rglob("*") if p.is_file()])
    # Retain a candidate even if validation fails, without counting it as an accepted cell.
    write_json(run / "candidate-entry.json", entry)
    verify_entry(entry, suite, info)
    write_json(destination, entry)
    print("ACCEPTED " + key + " " + device, flush=True)
    return entry


def run_matrix(suite, info, cells):
    jobs = queue.Queue()
    stopped = threading.Event()
    for cell in cells:
        jobs.put(cell)
    def worker(device):
        while not stopped.is_set():
            try:
                cell = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                run_cell(suite, info, cell, device)
            except BaseException:
                stopped.set()
                raise
    # Every worker owns exactly one GPU; one sequential run/evaluation at a time.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, device) for device in info["devices"]]
        for future in futures:
            future.result()


def entries(suite, info):
    rows = [read_json(path) for path in sorted((suite / "entries").glob("*.json"))]
    for row in rows:
        verify_entry(row, suite, info)
    return rows


def recompute(suite, info, rows):
    if (suite / "recompute.json").exists():
        return checked_gate(suite, info, "recompute")
    entry = next(r for r in rows if r["identity"]["dataset"] == "cement" and r["identity"]["horizon"] == 60 and r["identity"]["seed"] == 2021)
    directory = suite / "recompute" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=False)
    prediction = directory / "predictions.csv.gz"
    columns = ["窑喂料反馈", "窑头煤反馈", "分解炉内温度1反馈"]
    receipt, output = execute(suite, info, "private-three-variable-export", [info["binary"], "predict", "--checkpoint", str(Path(entry["run"]) / "best"), "--device", entry["identity"]["device"], "--columns", ",".join(columns), "--output", str(prediction)])
    result = json.loads(output)
    require(result["metrics"] == entry["evaluation"]["metrics"] and result["checkpoint_sha256"] == entry["evaluation"]["checkpoint_sha256"] and result["data_sha256"] == PRIVATE_SHA and result["prediction_rows"] == 18574560, "export provenance or coverage differs")
    second, _ = execute(suite, info, "private-independent-recompute", [sys.executable, str(ROOT / "tools/validation/recompute_predictions.py"), "--predictions", str(prediction), "--evaluation", str(Path(entry["run"]) / "evaluation.json"), "--columns", *columns, "--horizon", "60", "--output", str(directory / "independent.json")])
    independent = read_json(directory / "independent.json")
    require(independent["passed"] and independent["rows"] == 18574560 and len(independent["checks"]) == 6, "independent recomputation failed")
    return gate(suite, "recompute", dict(passed=True, result=independent, receipts=[receipt, second], artifacts=artifacts(list(directory.iterdir()))))


def run(suite):
    from p3_reports import write_reports
    suite, info = context(suite)
    owner_lock = (suite / "executor.lock").open("a")
    fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    owner = dict(pid=os.getpid(), started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        process_start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().split()[21], status="running", suite_sha256=sha256(suite / "suite.json"))
    write_json(suite / "executor.json", owner)
    try:
        quality(suite, info, "initial-quality")
        data_gate(suite, info)
        run_matrix(suite, info, matrix())
        rows = entries(suite, info)
        reports = write_reports(suite, info, rows)
        require(reports["d4"]["passed"], "D4 failed; original bounds and evidence retained; cement not started")
        run_matrix(suite, info, matrix(cement=True))
        rows = entries(suite, info)
        reports = write_reports(suite, info, rows)
        require(reports["passed"], "formal reports incomplete or failed")
        recompute(suite, info, rows)
        quality(suite, info, "final-quality")
        context(suite)
        gate(suite, "final", dict(passed=True, formal_runs=len(rows), d4_passed=32, cement_passed=9,
            recompute_checks=6, model_status="preview pending final independent evidence review", elapsed_seconds=time.time() - started))
        owner["status"] = "execution_passed_pending_independent_review"
    except BaseException as error:
        owner.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        owner["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(suite / "executor.json", owner)
        fcntl.flock(owner_lock, fcntl.LOCK_UN)
        owner_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "verify"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--cuda-bin", default="target/linux-cuda/debug/ferrots")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.suite, args.cuda_bin)
    elif args.stage == "run":
        run(args.suite)
    else:
        suite, info = context(args.suite)
        rows = entries(suite, info)
        print(json.dumps(dict(accepted=len(rows), expected=105, suite=str(suite))))


if __name__ == "__main__":
    main()
