"""Run the complete, local P1 acceptance suite. Production training remains Rust."""
from __future__ import annotations

import argparse
import csv
import gzip
import datetime as dt
import hashlib
import json
import os
import math
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TSLIB_SHA = "4e938a1767106324dd753b2a44832bf870a0252e"
PRIVATE = ROOT / "data/阳泉回转窑联合训练孪生样本/data_in.csv"
PRIVATE_SHA = "52145f5a4ef4a5d69f10ab851274efe0f324ca8dc1e8a577746d1e1a2eac2b80"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def source_snapshot() -> dict:
    names = set(git("ls-files").splitlines()) | set(git("ls-files", "--others", "--exclude-standard").splitlines())
    included = {}
    for name in sorted(names):
        path = ROOT / name
        if path.is_file() and (name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml") or (name.startswith("crates/") and "tests" not in path.parts and (path.suffix == ".rs" or path.name == "Cargo.toml"))):
            included[name] = sha256(path)
    payload = json.dumps(included, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return dict(code_sha=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")), source_sha256=hashlib.sha256(payload).hexdigest(), files=included)


def cpu_environment() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("CUDA", "CUDNN"))}
    env["PATH"] = os.pathsep.join(part for part in env.get("PATH", "").split(os.pathsep) if "NVIDIA GPU Computing Toolkit" not in part)
    return env


def preflight(args: argparse.Namespace) -> None:
    if os.environ.get("RUSTUP_AUTO_INSTALL") != "0" or os.environ.get("RUSTUP_TOOLCHAIN") != "1.98.0":
        raise ValueError("the fixed P1 environment is required")
    bins = {key: (ROOT / value).resolve() for key, value in {"cpu": args.cpu_bin, "cuda": args.cuda_bin}.items()}
    for path in bins.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source = source_snapshot()
    official = ROOT / "tools/reference/.cache/Time-Series-Library"
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=official, text=True).strip()
    if actual != TSLIB_SHA:
        raise ValueError("official source SHA mismatch")
    manifest = tomllib.loads((ROOT / "docs/datasets/manifest.toml").read_text(encoding="utf-8"))
    fingerprints = {}
    for item in manifest["file"]:
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"public fingerprint mismatch: {item['path']}")
        fingerprints[item["path"]] = item["sha256"]
    if PRIVATE.stat().st_size != 1068005224 or sha256(PRIVATE) != PRIVATE_SHA:
        raise ValueError("private source fingerprint mismatch")
    fingerprints[PRIVATE.relative_to(ROOT).as_posix()] = PRIVATE_SHA
    import torch
    if not torch.cuda.is_available():
        raise ValueError("official CUDA device is unavailable")
    suite = (ROOT / args.suite).resolve()
    if ROOT / "runs" not in suite.parents:
        raise ValueError("the suite must be under ignored runs/")
    suite.mkdir(parents=True, exist_ok=False)
    for mode, binary in bins.items():
        env = cpu_environment() if mode == "cpu" else dict(os.environ)
        device = "cpu" if mode == "cpu" else "cuda:0"
        result = subprocess.run([str(binary), "doctor", "--device", device, "--training-smoke-test"], cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True)
        (suite / f"preflight-{mode}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{mode} doctor failed; see the suite log")
        if source["source_sha256"] not in result.stdout:
            raise ValueError(f"{mode} binary does not match the runtime source snapshot")
    record = dict(format_version=1, created_at=dt.datetime.now(dt.timezone.utc).isoformat(), status="preflight_passed", source=source, upstream_sha=TSLIB_SHA, data=fingerprints, binaries={k: dict(path=str(v), sha256=sha256(v)) for k, v in bins.items()}, environment=dict(os=platform.platform(), python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), free_bytes=shutil.disk_usage(ROOT).free), seeds=[2021, 2022, 2023], horizons=[96, 192, 336, 720], ddof=0)
    record["configs"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / "configs").rglob("*.toml"))}
    record["reference_files"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / "tools/reference").glob("*.py"))}
    write_json(suite / "suite.json", record)
    print(json.dumps({"suite": str(suite), "status": record["status"], "public_files": len(manifest["file"]), "free_bytes": record["environment"]["free_bytes"]}))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def suite_context(args: argparse.Namespace) -> tuple[Path, dict]:
    suite = (ROOT / args.suite).resolve()
    require(ROOT / "runs" in suite.parents, "suite must be under runs/")
    info = read_json(suite / "suite.json")
    require(info["source"]["source_sha256"] == source_snapshot()["source_sha256"], "runtime sources changed; rebuild and rerun affected experiments")
    for group in ("binaries", "configs", "reference_files"):
        for name, item in info[group].items():
            path, expected = (Path(item["path"]), item["sha256"]) if group == "binaries" else (ROOT / name, item)
            require(sha256(path) == expected, f"{group} fingerprint changed: {name}")
    for name, expected in info["data"].items():
        require(sha256(ROOT / name) == expected, f"data fingerprint changed: {name}")
    return suite, info


def execute(suite: Path, info: dict, key: str, command: list[str], mode: str = "cuda") -> tuple[dict, str]:
    """Retain every attempt. An exit-zero receipt alone does not prove acceptance."""
    directory = suite / "commands" / key / dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    tick = time.perf_counter()
    print(f"START {key}", flush=True)
    env = cpu_environment() if mode == "cpu" else dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", RUSTUP_AUTO_INSTALL="0", RUSTUP_TOOLCHAIN="1.98.0")
    with (directory / "stdout.log").open("w", encoding="utf-8") as out, (directory / "stderr.log").open("w", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    record = dict(key=key, command=command, started_at=started, ended_at=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_s=time.perf_counter()-tick, exit_code=result.returncode, source=info["source"]["source_sha256"], data=info["data"], configs=info["configs"], binaries=info["binaries"], upstream_sha=info["upstream_sha"], environment=info["environment"], validation_tool_sha256=sha256(Path(__file__)), directory=str(directory))
    record["logs"] = {p.name: sha256(p) for p in directory.glob("*.log")}
    write_json(directory / "receipt.json", record)
    print(f"END {key} exit={result.returncode} seconds={record['elapsed_s']:.3f}", flush=True)
    require(result.returncode == 0, f"command failed; see {directory}")
    return record, (directory / "stdout.log").read_text(encoding="utf-8")


ETT = {"ETTh1": "ett-h1", "ETTh2": "ett-h2", "ETTm1": "ett-m1", "ETTm2": "ett-m2"}
HORIZONS = [96, 192, 336, 720]
SEEDS = [2021, 2022, 2023]
CEMENT_COUNTS = {15: [299819, 57806, 103417], 30: [299384, 57746, 103342], 60: [298514, 57626, 103192]}


def inspect_data(suite: Path, info: dict, name: str, horizon: int, config: str, suffix: str = "") -> tuple[dict, dict]:
    key = f"data-{name}-h{horizon}{suffix}"
    receipt, stdout = execute(suite, info, key, [info["binaries"]["cpu"]["path"], "data", "inspect", "--config", config, "--horizon", str(horizon), "--cache-dir", str(suite / "cache")], "cpu")
    report = json.loads(stdout)
    require(report["window_audit"]["all_same_segment"] and report["window_audit"]["all_target_split"], f"window audit failed: {name}")
    require(report["window_audit"]["checked"] == sum(report["window_counts"]), "window audit coverage differs")
    require(report["cache"]["source_sha256"] == info["data"][tomllib.loads((ROOT / config).read_text(encoding="utf-8"))["data"]["path"]], "inspected data hash differs")
    return report, receipt


def data_gate(args: argparse.Namespace) -> None:
    suite, info = suite_context(args)
    fixture = read_json(ROOT / "tests/fixtures/p1/ett.json")
    reports, receipts = {}, []
    for expected in fixture["datasets"]:
        name = expected["name"]
        for horizon in HORIZONS:
            actual, receipt = inspect_data(suite, info, name, horizon, f"configs/reference/{ETT[name]}.toml")
            receipts.append(receipt)
            reports[f"{name}-{horizon}"] = actual
            require(actual["raw"]["rows"] == expected["rows"], f"ETT rows: {name}")
            counts = expected["counts"][str(horizon)]
            require(actual["window_counts"] == [counts[k] for k in ["train", "val", "test"]], f"ETT window counts: {name}/{horizon}")
            for i, col in enumerate(actual["scaler"]["columns"]):
                require(col["name"] == expected["columns"][i], "ETT scaler columns differ")
                require(abs(col["mean"] - expected["mean"][i]) < 1e-9 and abs(col["scale"] - expected["scale"][i]) < 1e-9, "ETT scaler differs")
    exchange, receipt = inspect_data(suite, info, "exchange", 96, "configs/reference/exchange.toml")
    receipts.append(receipt)
    reports["exchange"] = exchange
    require(exchange["raw"]["rows"] == 7588 and exchange["raw"]["start"] == "1990-01-01 00:00:00.000000" and exchange["raw"]["end"] == "2010-10-10 00:00:00.000000", "Exchange rows/times differ")
    require(exchange["prepared"]["usable_rows"] == [5311, 760, 1517], "Exchange split differs")
    means = [558.919,16.2511,805.0574,-58.1947,887.5173,905.4728,1369.4377,1119.7764,1114.0591,637.2028,225.0093,934.7145,872.8465,11.2186,9.551,9.7854,6.6468,13.6605]
    scales = [163.8968,1.5547,83.4736,42.5745,27.6103,50.4096,242.2032,137.0094,70.9388,538.4648,25.0956,71.3482,23.8773,2.1341,1.2228,1.257,0.8263,2.3094]
    for horizon in CEMENT_COUNTS:
        actual, receipt = inspect_data(suite, info, "cement", horizon, "configs/engineering/cement-twin.toml")
        receipts.append(receipt)
        reports[f"cement-{horizon}"] = actual
        raw, prepared = actual["raw"], actual["prepared"]
        require([raw[k] for k in ["rows", "non_increasing", "duplicate_groups", "duplicate_rows", "duplicate_removed"]] == [6914135,7,7,14,7], "cement raw counters differ")
        require(raw["start"] == "2023-01-01 00:00:05.000000" and raw["end"] == "2025-05-21 09:42:02.000000", "cement raw time range differs")
        require([c["missing"] for c in raw["columns"]] == [620602,0,0,1,0,0,1,0,0,0,0,0,895122,0,0,0,0,0], "cement raw missing counts differ")
        require(all(c["non_numeric"] == 0 for c in raw["columns"]), "unexpected nonnumeric data")
        require([prepared[k] for k in ["rows", "running_segments", "segment_rows"]] == [1254823,48,529531], "cement prepared counters differ")
        require(prepared["split_segments"] == [37,4,7] and prepared["usable_rows"] == [303009,58246,104051], "cement split differs")
        require(prepared["imputed_per_column"] == [13]*18, "cement fill counters differ")
        require(actual["window_counts"] == CEMENT_COUNTS[horizon], "cement window counts differ")
        for i, col in enumerate(actual["scaler"]["columns"]):
            require(col["count"] == 303009 and math.isclose(col["mean"], means[i], rel_tol=1e-3) and math.isclose(col["scale"], scales[i], rel_tol=1e-3), "cement scaler differs")
    first = reports["cement-15"]["cache"]
    second, receipt = inspect_data(suite, info, "cement", 15, "configs/engineering/cement-twin.toml", "-repeat")
    receipts.append(receipt)
    reports["cement-repeat"] = second
    require(not first["hit"] and first["parse_count"] == 1, "first cement load must use a fresh cache namespace")
    require(second["cache"]["hit"] and second["cache"]["parse_count"] == 0, "second cement load parsed CSV")
    write_json(suite / "data-gate.json", dict(passed=True, reports=reports, receipts=receipts))
    print("PASS data: ETT 16, Exchange 1, cement 3 horizons and fresh/hit cache", flush=True)


def artifacts(paths: list[Path]) -> dict:
    return {str(p.resolve()): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(set(paths))}


def verify_artifacts(files: dict) -> None:
    require(bool(files), "missing artifact inventory")
    for name, expected in files.items():
        p = Path(name)
        require(p.stat().st_size == expected["bytes"] and sha256(p) == expected["sha256"], f"artifact changed: {name}")


def bundle_at(run: Path, selector: str = "best") -> Path:
    refs = sorted((run / selector).glob("*.json"), key=lambda p: int(p.stem))
    require(bool(refs), "checkpoint selector is empty")
    ref = read_json(refs[-1])
    path = run / "checkpoints" / ref["generation"]
    require(sha256(path / "manifest.json") == ref["manifest_sha256"], "checkpoint manifest hash mismatch")
    for name, record in read_json(path / "manifest.json")["files"].items():
        verify_artifacts({str(path / name): record})
    return path


def validate_config(actual: dict, config: str, horizon: int, seed: int, mode: str, model: str) -> None:
    expected = tomllib.loads((ROOT / config).read_text(encoding="utf-8"))
    if 'missing' in expected['data']:
        expected['data']['missing'].setdefault('tokens', ['', 'NaN', 'nan'])
    expected["window"]["horizon"] = horizon
    expected["train"].update(seed=seed, device="cpu" if mode == "cpu" else "cuda:0")
    expected["model"] = {"name": model.replace("-", "_")}
    if model == "dlinear":
        expected["model"].update(individual=False, moving_avg=25)
    elif model == "seasonal-naive":
        expected["model"]["period"] = 60
    for section in ("data", "window", "model", "train"):
        for key, value in expected[section].items():
            require(actual[section].get(key) == value, f"resolved config differs: {section}.{key}")
    for key, value in dict(accumulation_steps=1,beta1=0.9,beta2=0.999,eps=1e-8,weight_decay=0.0,dtype="f32",max_epochs=10,patience=3,batch_size=32,learning_rate=0.0001,lr_schedule="type1").items():
        require(actual["train"][key] == value, f"training contract differs: {key}")


def rust_run(suite: Path, info: dict, name: str, horizon: int, seed: int, mode: str, model: str = "dlinear", predict: bool = False) -> dict:
    key = f"rust-{name}-{model}-{mode}-h{horizon}-s{seed}"
    entry_path = suite / "entries" / f"{key}.json"
    config = "configs/engineering/cement-twin.toml" if name == "cement" else f"configs/reference/{ETT[name]}.toml"
    identity = dict(dataset=name, horizon=horizon, seed=seed, mode=mode, model=model, source_sha256=info["source"]["source_sha256"], config_sha256=sha256(ROOT / config), binary_sha256=info["binaries"][mode]["sha256"], predict=predict)
    if entry_path.exists():
        entry = read_json(entry_path)
        require(entry["identity"] == identity, "completed entry identity differs")
        verify_artifacts(entry["artifacts"])
        print(f"VERIFIED existing {key}", flush=True)
        return entry
    run_id = f"{key}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    run = suite / "rust" / run_id
    binary = info["binaries"][mode]["path"]
    device = "cpu" if mode == "cpu" else "cuda:0"
    command = [binary, "train", "--config", config, "--device", device, "--seed", str(seed), "--horizon", str(horizon), "--model", model, "--cache-dir", str(suite / "cache"), "--output-root", str(run.parent), "--run-id", run_id]
    if model == "seasonal-naive":
        command += ["--seasonal-period", "60"]
    receipt, stdout = execute(suite, info, key + "-train", command, mode)
    summary = json.loads(stdout)
    receipts = [receipt]
    require(summary == read_json(run / "train-summary.json"), "train summary changed")
    actual_config = tomllib.loads((run / "resolved-config.toml").read_text(encoding="utf-8"))
    validate_config(actual_config, config, horizon, seed, mode, model)
    data_sha = info["data"][actual_config["data"]["path"]]
    require(summary["source_sha256"] == identity["source_sha256"] and summary["data_sha256"] == data_sha, "training identity differs")
    require(summary["access"]["windows"][2] == 0 and summary["access"]["batches"][2] == 0, "training accessed test")
    gate = read_json(suite / "data-gate.json")
    counts = gate["reports"][f"{name}-{horizon}"]["window_counts"]
    if model == "dlinear":
        epochs = summary["epochs"]
        require(1 <= len(epochs) <= 10, "invalid epoch coverage")
        best = min(epochs, key=lambda e: e["validation_loss"])
        require(summary["best_epoch"] == best["epoch"], "selected epoch differs")
        require(len(epochs) == 10 or len(epochs) - best["epoch"] == 3, "training stopped without patience")
        for i, epoch in enumerate(epochs):
            require(epoch["epoch"] == i+1 and epoch["valid_labels"] == counts[0]*horizon*len(actual_config["data"]["target_columns"]), "epoch coverage differs")
            require(math.isclose(epoch["learning_rate"], 1e-4 * 0.5**max(0,i-1), rel_tol=1e-14), "type1 rate differs")
            require(math.isfinite(epoch["train_loss"]) and math.isfinite(epoch["validation_loss"]), "non-finite epoch metric")
        require(summary["access"]["windows"] == [counts[0]*len(epochs), counts[1]*len(epochs), 0], "train/validation coverage differs")
    else:
        require(summary["optimizer_steps"] == 0 and summary["epochs"] == [], "parameter-free baseline pretended to train")
    bundle = bundle_at(run)
    bundle_at(run, "last")
    manifest = read_json(bundle / "manifest.json")
    require(manifest["provenance"]["source_sha256"] == identity["source_sha256"] and manifest["data_sha256"] == data_sha and manifest["epoch"] == summary["best_epoch"] and manifest["upstream_sha"] == TSLIB_SHA, "checkpoint identity differs")
    evaluation_path = run / "evaluation.json"
    receipt, stdout = execute(suite, info, key+"-evaluate", [binary, "evaluate", "--checkpoint", str(run / "best"), "--device", device, "--output", str(evaluation_path)], mode)
    receipts.append(receipt)
    evaluation = json.loads(stdout)
    require(evaluation == read_json(evaluation_path), "evaluation output differs")
    require(evaluation["windows"] == counts[2] and evaluation["data_sha256"] == data_sha and evaluation["checkpoint_sha256"] == sha256(bundle / "manifest.json"), "evaluation identity/coverage differs")
    metric = evaluation["metrics"]["standardized"]["overall"]
    require(metric["count"] == counts[2]*horizon*len(actual_config["data"]["target_columns"]), "evaluation element coverage differs")
    require(all(math.isfinite(metric[k]) for k in ["mse","mae","rmse"]), "nonfinite evaluation metric")
    if predict:
        prediction_path = run / "predictions.csv.gz"
        receipt, stdout = execute(suite, info, key+"-predict", [binary, "predict", "--checkpoint", str(run / "best"), "--device", device, "--output", str(prediction_path)], mode)
        receipts.append(receipt)
        prediction = json.loads(stdout)
        require(prediction["metrics"] == evaluation["metrics"] and prediction["checkpoint_sha256"] == evaluation["checkpoint_sha256"] and prediction["prediction_rows"] == metric["count"], "checkpoint prediction roundtrip differs")
        with gzip.open(prediction_path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            count, previous = 0, None
            columns = actual_config["data"]["target_columns"]
            for row in reader:
                group = count // (horizon*len(columns))
                h = (count // len(columns)) % horizon+1
                j = count % len(columns)
                origin = dt.datetime.fromisoformat(row["forecast_origin"])
                target = dt.datetime.fromisoformat(row["target_time"])
                require(row["run_id"] == run_id and row["series_id"] == "0" and row["valid"] == "true" and int(row["horizon"]) == h and row["target_name"] == columns[j], "prediction key mismatch")
                require(target == origin + dt.timedelta(seconds=(h-1)*actual_config["data"]["interval_seconds"]), "prediction time mismatch")
                if previous is not None:
                    require((group == previous[0] and origin == previous[1]) or (group == previous[0]+1 and origin > previous[1]), "prediction origin order mismatch")
                require(math.isfinite(float(row["y_true"])) and math.isfinite(float(row["y_pred"])), "nonfinite prediction")
                previous = (group, origin)
                count += 1
            require(count == metric["count"], "prediction rows missing")
        write_json(run / "prediction-summary.json", prediction)
    inventory = artifacts([p for p in run.rglob("*") if p.is_file()])
    entry = dict(identity=identity, run=str(run), summary=summary, evaluation=evaluation, receipts=receipts, artifacts=inventory)
    write_json(entry_path, entry)
    print(f"PASS {key}: mse={metric['mse']:.9g} mae={metric['mae']:.9g}", flush=True)
    return entry


def ett_cpu(args: argparse.Namespace) -> None:
    suite, info = suite_context(args)
    rust_run(suite, info, "ETTh1", 96, 2021, "cpu", predict=True)


def official_arguments(name: str, horizon: int, model_id: str) -> list[str]:
    return ["--task_name","long_term_forecast","--is_training","1","--root_path","data/ETT-small/","--data_path",f"{name}.csv","--model_id",model_id,"--model","DLinear","--data",name,"--features","M","--target","OT","--seq_len","96","--label_len","48","--pred_len",str(horizon),"--enc_in","7","--dec_in","7","--c_out","7","--moving_avg","25","--embed","timeF","--freq","h","--batch_size","32","--train_epochs","10","--patience","3","--learning_rate","0.0001","--lradj","type1","--loss","MSE","--num_workers","0","--itr","1","--use_gpu","--gpu","0","--gpu_type","cuda","--des","P1"]


def ett_reference(args: argparse.Namespace) -> None:
    import numpy as np
    suite, info = suite_context(args)
    upstream = ROOT / "tools/reference/.cache/Time-Series-Library"
    require(subprocess.run(["git","diff","--quiet","HEAD","--","*.py"],cwd=upstream).returncode == 0, "official Python source is modified")
    expected_counts = read_json(suite / "data-gate.json")["reports"]
    for name in ETT:
        for horizon in HORIZONS:
            for seed in SEEDS:
                key = f"official-{name}-h{horizon}-s{seed}"
                identity = dict(dataset=name,horizon=horizon,seed=seed,upstream_sha=TSLIB_SHA,data_sha256=info["data"][f"data/ETT-small/{name}.csv"],reference_files=info["reference_files"])
                entry_path = suite / "entries" / f"{key}.json"
                if entry_path.exists():
                    entry = read_json(entry_path)
                    require(entry["identity"] == identity, "official completed identity differs")
                    verify_artifacts(entry["artifacts"])
                    print(f"VERIFIED existing {key}", flush=True)
                    continue
                tag = hashlib.sha256(str(suite).encode()).hexdigest()[:4]
                model_id = f"P1{tag}{list(ETT).index(name)}{horizon}s{seed}"
                run_args = official_arguments(name,horizon,model_id)
                command = [sys.executable,str(ROOT / "tools/reference/run_tslib.py"),"--seed",str(seed),"--",*run_args]
                receipt, stdout = execute(suite, info, key, command)
                runs = re.findall(r"^run dir: (.+)$", stdout, re.M)
                require(len(runs) == 1, "official runner did not report one directory")
                run = Path(runs[0].strip())
                result = read_json(run / "metrics.json")
                require(result["exit_code"] == 0 and result["upstream_sha"] == TSLIB_SHA and result["seed"] == seed and len(result["tests"]) == 1, "official result identity differs")
                log = (run / "stdout.log").read_text(encoding="utf-8")
                settings = re.findall(r">+testing : (.+?)<+",log)
                require(len(settings) == 1, "official test setting missing")
                result_dir = upstream / "results" / settings[0]
                values = np.load(result_dir / "metrics.npy",allow_pickle=False)
                metrics = dict(mae=float(values[0]),mse=float(values[1]),rmse=float(values[2]))
                require(all(math.isfinite(x) for x in metrics.values()), "official metrics are nonfinite")
                count = expected_counts[f"{name}-{horizon}"]["window_counts"][2]
                for filename in ("pred.npy","true.npy"):
                    array = np.load(result_dir / filename,mmap_mode="r",allow_pickle=False)
                    require(array.shape == (count,horizon,7) and array.dtype == np.float32 and np.isfinite(array).all(), "official output coverage/dtype differs")
                for flag,count in zip(["train","val","test"],expected_counts[f"{name}-{horizon}"]["window_counts"]):
                    require(re.search(rf"^{flag} {count}$",log,re.M) is not None, "official split coverage differs")
                require("Use GPU: cuda:0" in log or "cuda:0" in log, "official GPU identity missing")
                require("--individual" not in result["run_args"] and "--use_amp" not in result["run_args"], "official individual/AMP mismatch")
                files = [p for p in run.rglob("*") if p.is_file()] + list(result_dir.glob("*.npy"))
                entry = dict(identity=identity, run=str(run), run_args=run_args, result=result, metrics=metrics, receipts=[receipt], artifacts=artifacts(files))
                write_json(entry_path,entry)
                print(f"PASS {key}: mse={metrics['mse']:.9g} mae={metrics['mae']:.9g}",flush=True)
    print("PASS official matrix: 48/48",flush=True)


def import_reference(args: argparse.Namespace) -> None:
    """Reuse a complete formal matrix after checking every recorded identity."""
    suite, info = suite_context(args)
    previous = (ROOT / args.from_suite).resolve()
    require(ROOT / 'runs/p1-acceptance' in previous.parents and previous != suite, 'invalid reference suite')
    old = read_json(previous / 'suite.json')
    for key in ('upstream_sha', 'data', 'configs', 'reference_files', 'seeds', 'horizons', 'ddof'):
        require(old[key] == info[key], f'reference reuse identity differs: {key}')
    for key in ('os', 'python', 'torch', 'cuda', 'gpu'):
        require(old['environment'][key] == info['environment'][key], f'reference environment differs: {key}')
    upstream = ROOT / 'tools/reference/.cache/Time-Series-Library'
    require(subprocess.check_output(['git','rev-parse','HEAD'], cwd=upstream, text=True).strip() == TSLIB_SHA, 'official HEAD changed')
    require(subprocess.run(['git','diff','--quiet','HEAD','--','*.py'], cwd=upstream).returncode == 0, 'official Python source changed')
    entries = verified_matrix(previous, old, 'official')
    imported = {}
    for (name, horizon, seed), entry in entries.items():
        key = f'official-{name}-h{horizon}-s{seed}'
        source = previous / 'entries' / f'{key}.json'
        destination = suite / 'entries' / source.name
        require(not destination.exists(), 'reference destination already exists')
        for receipt in entry['receipts']:
            require(receipt['data'] == old['data'] and receipt['environment'] == old['environment'] and receipt['upstream_sha'] == TSLIB_SHA, 'reference receipt identity differs')
            directory = Path(receipt['directory'])
            require(read_json(directory / 'receipt.json') == receipt, 'reference receipt changed')
            for filename, expected in receipt['logs'].items():
                require(sha256(directory / filename) == expected, 'reference log changed')
        entry['reuse'] = dict(source_suite=str(previous), source_entry_sha256=sha256(source), approved_reason='Rust-only RNG correction; all official identities and artifacts unchanged')
        write_json(destination, entry)
        imported[key] = entry['reuse']
    write_json(suite / 'reference-reuse.json', dict(source_suite=str(previous), source_suite_sha256=sha256(previous/'suite.json'), entries=imported))
    print('PASS verified formal reference reuse: 48/48', flush=True)


def ett_rust(args: argparse.Namespace) -> None:
    suite, info = suite_context(args)
    for name in ETT:
        for horizon in HORIZONS:
            for seed in SEEDS:
                rust_run(suite,info,name,horizon,seed,"cuda")
    print("PASS Rust CUDA matrix: 48/48",flush=True)


def matrix_by_key(entries: list[dict], kind: str) -> dict:
    expected = {(n,h,s) for n in ETT for h in HORIZONS for s in SEEDS}
    found = {}
    for entry in entries:
        identity = entry["identity"]
        key = (identity["dataset"],identity["horizon"],identity["seed"])
        require(key in expected and key not in found, f"{kind} unexpected/duplicate matrix key: {key}")
        metric = entry["metrics"] if kind == "official" else entry["evaluation"]["metrics"]["standardized"]["overall"]
        require(all(isinstance(metric[k],(int,float)) and math.isfinite(metric[k]) and metric[k]>=0 for k in ("mse","mae")), "invalid matrix metric")
        found[key] = entry
    require(set(found) == expected, f"{kind} matrix coverage is {len(found)}/48")
    return found


def d4_values(official: list[float], rust: list[float]) -> dict:
    require(len(official)==3 and len(rust)==3 and all(math.isfinite(v) and v>=0 for v in official+rust), "D4 requires three finite nonnegative values on both sides")
    mean_o, mean_r = statistics.fmean(official), statistics.fmean(rust)
    std_o, std_r = statistics.pstdev(official), statistics.pstdev(rust)
    threshold = max(2*std_o,0.02*abs(mean_o))
    difference = abs(mean_r-mean_o)
    return dict(official=official,rust=rust,official_mean=mean_o,official_std=std_o,rust_mean=mean_r,rust_std=std_r,threshold=threshold,difference=difference,passed=difference<=threshold)


def verified_matrix(suite: Path, info: dict, kind: str) -> dict:
    pattern = "official-*.json" if kind == "official" else "rust-ETT*-dlinear-cuda-*.json"
    entries = matrix_by_key([read_json(p) for p in (suite / "entries").glob(pattern)], kind)
    for (name,horizon,seed), entry in entries.items():
        verify_artifacts(entry["artifacts"])
        identity = entry["identity"]
        data_sha = info["data"][f"data/ETT-small/{name}.csv"]
        if kind == "official":
            require(identity["upstream_sha"]==TSLIB_SHA and identity["data_sha256"]==data_sha and identity["reference_files"]==info["reference_files"], "official fingerprint differs")
            args = entry["run_args"]
            model_id = args[args.index("--model_id")+1]
            require(args == official_arguments(name,horizon,model_id), "official arguments differ")
            normalized = list(args)
            normalized[normalized.index("--root_path")+1] = str((ROOT / "data/ETT-small").resolve())+os.sep
            normalized += ["--checkpoints",str(Path(entry["run"])/"checkpoints")+os.sep]
            require(entry["result"]["run_args"]==normalized and entry["result"]["seed"]==seed and entry["result"]["exit_code"]==0, "official actual arguments/seed/exit differ")
            require(entry['result'] == read_json(Path(entry['run'])/'metrics.json'), 'official result record differs')
            import numpy as np
            metric_files = [Path(p) for p in entry['artifacts'] if Path(p).name == 'metrics.npy']
            require(len(metric_files) == 1, 'official raw metric inventory differs')
            values = np.load(metric_files[0], allow_pickle=False)
            require(entry['metrics'] == dict(mae=float(values[0]), mse=float(values[1]), rmse=float(values[2])), 'official metric values differ from NumPy artifact')
        else:
            config = f"configs/reference/{ETT[name]}.toml"
            require(identity["source_sha256"]==info["source"]["source_sha256"] and identity["binary_sha256"]==info["binaries"]["cuda"]["sha256"] and identity["config_sha256"]==info["configs"][config] and identity["mode"]=="cuda" and identity["model"]=="dlinear", "Rust matrix identity differs")
            actual = tomllib.loads((Path(entry["run"])/"resolved-config.toml").read_text(encoding="utf-8"))
            validate_config(actual,config,horizon,seed,"cuda","dlinear")
            require(entry["evaluation"]["data_sha256"]==data_sha and entry["evaluation"]["seed"]==seed, "Rust result fingerprint/seed differs")
            require(entry['evaluation'] == read_json(Path(entry['run'])/'evaluation.json') and entry['summary'] == read_json(Path(entry['run'])/'train-summary.json'), 'Rust result record differs')
        require(all(r["exit_code"]==0 for r in entry["receipts"]), "matrix has failed receipt")
        for receipt in entry['receipts']:
            verify_receipt(receipt, info, current_source=kind != 'official')
    return entries


def d4_gate(args: argparse.Namespace) -> None:
    suite, info = suite_context(args)
    official = verified_matrix(suite,info,"official")
    rust = verified_matrix(suite,info,"rust")
    rows = []
    for name in ETT:
        for horizon in HORIZONS:
            for metric in ["mse","mae"]:
                o = [official[(name,horizon,s)]["metrics"][metric] for s in SEEDS]
                r = [rust[(name,horizon,s)]["evaluation"]["metrics"]["standardized"]["overall"][metric] for s in SEEDS]
                rows.append(dict(dataset=name,horizon=horizon,metric=metric,**d4_values(o,r)))
    passed = all(r["passed"] for r in rows)
    write_json(suite/"d4.json",dict(passed=passed,ddof=0,rows=rows,official_runs=48,rust_runs=48,source=info["source"]))
    lines = ["# P1 DLinear ETT comparison","",f"Suite: {suite.relative_to(ROOT).as_posix()}. Official CUDA runs: 48. Rust CUDA runs: 48.",f"Result: {sum(r['passed'] for r in rows)}/32 D4 checks passed. Seeds: 2021, 2022, 2023. Population standard deviation (ddof=0).","","Each bound is max(2 * official_std, 0.02 * abs(official_mean)). The absolute mean difference must be at most that bound. Comparisons use full-precision JSON/NumPy values. Display values use nine significant digits.","",f"Upstream: {TSLIB_SHA}. Runtime source snapshot: {info['source']['source_sha256']}. Build HEAD: {info['source']['code_sha']}; dirty={info['source']['dirty']}.",f"Machine: {info['environment']['os']}; {info['environment']['gpu']}; torch {info['environment']['torch']}; CUDA {info['environment']['cuda']}. Native Rust environment: docs/environment.md.","","Both sides use L96, label_len48 for the official decoder interface, all 7 variables, shared DLinear, moving_avg25, FP32, Adam lr1e-4, batch32, epochs<=10, patience3 and epoch-end type1. The official path retains shuffled validation, mean batch loss and per-epoch test logs. Rust uses element-weighted validation, fixed validation order and test isolation. These differences do not change the D4 bound.","","| Dataset | H | Metric | Official seeds 2021/2022/2023 | Rust seeds 2021/2022/2023 | Official mean / std | Rust mean / std | Bound | Difference | Pass |","| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        values=lambda xs: " / ".join(format(x,".9g") for x in xs)
        lines.append(f"| {r['dataset']} | {r['horizon']} | {r['metric']} | {values(r['official'])} | {values(r['rust'])} | {values([r['official_mean'],r['official_std']])} | {values([r['rust_mean'],r['rust_std']])} | {r['threshold']:.9g} | {r['difference']:.9g} | {r['passed']} |")
    path = ROOT / args.report
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    require(passed, f"D4 failed: {sum(r['passed'] for r in rows)}/32; see {path}")
    print("PASS D4: 32/32",flush=True)


def cement_gate(args: argparse.Namespace) -> None:
    suite, info = suite_context(args)
    require(read_json(suite/'d4.json')['passed'], 'D4 must pass before cement acceptance')
    rows = []
    for model in ('last-value', 'seasonal-naive', 'dlinear'):
        for horizon in (15, 30, 60):
            for mode in ('cpu', 'cuda'):
                entry = rust_run(suite, info, 'cement', horizon, 2021, mode, model)
                variables = entry['evaluation']['metrics']['original']['variables']
                config = tomllib.loads((Path(entry['run'])/'resolved-config.toml').read_text(encoding='utf-8'))
                require(set(variables) == set(config['data']['target_columns']) and len(variables) == 18, 'cement variable coverage differs')
                for name, metric in variables.items():
                    require(metric['count'] == CEMENT_COUNTS[horizon][2]*horizon, 'cement valid element coverage differs')
                    require(all(math.isfinite(metric[k]) and metric[k] >= 0 for k in ('mae','mse','rmse')), 'cement invalid metric')
                    rows.append(dict(model=model, horizon=horizon, device=mode, variable=name, **metric, train_s=entry['summary']['seconds'], evaluate_s=entry['evaluation']['seconds']))
    require(len(rows) == 18*18, 'cement matrix coverage differs')
    write_json(suite/'cement.json', dict(passed=True, runs=18, variables=18, source=info['source'], rows=rows))
    lines = ['# P1 cement baseline acceptance', '', f'Suite: {suite.relative_to(ROOT).as_posix()}. Complete runs: 18/18. Seed: 2021.', 'Protocol: cement-twin-v1, including all four approved adjustments. All legal windows are used. DLinear uses L96, batch32, Adam lr1e-4, max_epochs10, patience3, epoch-end type1. SeasonalNaive period is 60.', f'Runtime source SHA256: {info["source"]["source_sha256"]}. Raw data SHA256: {PRIVATE_SHA}.', '', 'All metrics below use original variable units. Reports contain statistics only. Per-run receipts and complete checkpoint identities remain in the ignored suite directory.', '', '| Model | H | Device | Variable | Count | MAE | RMSE | Train seconds | Evaluate seconds |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['model']} | {row['horizon']} | {row['device']} | {row['variable']} | {row['count']} | {row['mae']:.10g} | {row['rmse']:.10g} | {row['train_s']:.3f} | {row['evaluate_s']:.3f} |")
    (ROOT/args.report).write_text(chr(10).join(lines)+chr(10), encoding='utf-8')
    print('PASS cement full matrix: 18/18, all 18 variables', flush=True)


def recompute_gate(args: argparse.Namespace) -> None:
    suite, info = suite_context(args)
    require(read_json(suite/'cement.json')['passed'], 'cement matrix must pass before recomputation')
    entry = read_json(suite/'entries/rust-cement-dlinear-cuda-h60-s2021.json')
    verify_artifacts(entry['artifacts'])
    run = Path(entry['run'])
    columns = ['窑喂料反馈', '窑头煤反馈', '分解炉内温度1反馈']
    directory = suite/'recompute'/dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory.mkdir(parents=True, exist_ok=False)
    prediction_path = directory/'predictions.csv.gz'
    receipt, stdout = execute(suite, info, 'cement-three-variable-export', [info['binaries']['cuda']['path'], 'predict', '--checkpoint', str(run/'best'), '--device', 'cuda:0', '--columns', ','.join(columns), '--output', str(prediction_path)])
    prediction = json.loads(stdout)
    require(prediction['metrics'] == entry['evaluation']['metrics'] and prediction['checkpoint_sha256'] == entry['evaluation']['checkpoint_sha256'] and prediction['data_sha256'] == PRIVATE_SHA, 'export changed model, data or evaluation metrics')
    require(prediction['prediction_rows'] == 18574560, 'three-variable export coverage differs')
    write_json(directory/'prediction-summary.json', prediction)
    output = directory/'independent.json'
    second, _ = execute(suite, info, 'cement-independent-recompute', [sys.executable, str(ROOT/'tools/validation/recompute_predictions.py'), '--predictions', str(prediction_path), '--evaluation', str(run/'evaluation.json'), '--columns', *columns, '--horizon', '60', '--output', str(output)], 'cpu')
    result = read_json(output)
    require(result['passed'] and result['rows'] == 18574560 and len(result['checks']) == 6, 'independent recomputation failed')
    result.update(source=info['source'], data_sha256=PRIVATE_SHA, checkpoint_sha256=prediction['checkpoint_sha256'], receipts=[receipt, second], artifacts=artifacts([p for p in directory.iterdir() if p.is_file()]), script_sha256=sha256(ROOT/'tools/validation/recompute_predictions.py'))
    write_json(suite/'recompute.json', result)
    print('PASS prediction export and independent recomputation: 6/6', flush=True)


def verify_receipt(receipt: dict, info: dict, current_source: bool = True) -> None:
    require(receipt['exit_code'] == 0, 'failed experiment receipt')
    path = Path(receipt['directory'])
    require(read_json(path/'receipt.json') == receipt, 'receipt content differs')
    require(receipt['data'] == info['data'] and receipt['configs'] == info['configs'] and receipt['upstream_sha'] == TSLIB_SHA, 'receipt identity differs')
    if current_source:
        require(receipt['source'] == info['source']['source_sha256'] and receipt['binaries'] == info['binaries'], 'receipt source or binary differs')
    for key in ('os','python','torch','cuda','gpu'):
        require(receipt['environment'][key] == info['environment'][key], 'receipt environment differs')
    for name, expected in receipt['logs'].items():
        require(sha256(path/name) == expected, 'receipt log hash differs')


QUALITY_COMMANDS = {
    'cpu-fmt': ['fmt','--all','--check'],
    'cpu-build': ['build','--workspace'],
    'cpu-clippy': ['clippy','--workspace','--all-targets','--','-D','warnings'],
    'cpu-test': ['test','--workspace'],
    'cuda-build': ['build','--workspace','--features','cuda'],
    'cuda-clippy': ['clippy','--workspace','--all-targets','--features','cuda','--','-D','warnings'],
    'cuda-test': ['test','--workspace','--features','cuda'],
    'cuda-doctor': ['run','-p','ferrots-cli','--features','cuda','--','doctor','--device','cuda:0','--training-smoke-test'],
    'cuda-invalid-device': ['run','-p','ferrots-cli','--features','cuda','--','doctor','--device','cuda:9'],
    'python-tests': ['run','--frozen','--project','tools/reference','python','-m','unittest','discover','-s','tools/validation/tests','-p','test_*.py'],
    'task-validate': ['.trellis/scripts/task.py','validate','.trellis/tasks/09-28-p1-data-train-loop'],
    'private-data': ['tools/check_private_data.sh'],
    'diff-check': ['diff','--check'],
}


def validate_quality(quality: dict) -> dict:
    records = quality['records']
    require(len(records) == len(QUALITY_COMMANDS) and {r['key'] for r in records} == set(QUALITY_COMMANDS), 'full quality command coverage differs')
    require(quality['script_sha256'] == sha256(ROOT/'tools/validation/p1_quality.ps1'), 'quality runner changed')
    require(bool(quality['inputs']), 'quality input fingerprint inventory missing')
    for path, expected in quality['inputs'].items():
        require(sha256(ROOT/path) == expected, f'quality input changed: {path}')
    totals = {}
    for record in records:
        key = record['key']
        require(record['arguments'] == QUALITY_COMMANDS[key], f'quality command differs: {key}')
        require(record['rustup_auto_install'] == '0' and record['rustup_toolchain'] == '1.98.0' and '1.98.0' in record['rustc'] and '1.98.0' in record['rustdoc'], 'quality toolchain differs')
        require(sha256(Path(record['log'])) == record['log_sha256'], 'quality log changed')
        log = Path(record['log']).read_text(encoding='utf-8-sig')
        expected_failure = key == 'cuda-invalid-device'
        require(record['expected_failure'] == expected_failure and ((record['exit_code'] != 0) if expected_failure else (record['exit_code'] == 0)), f'quality command failed: {key}')
        if expected_failure:
            require('requested_device: cuda:9' in log and 'device initialization' in log, 'cuda:9 did not reach device validation')
        if key.startswith('cuda-'):
            require(record['rustflags'] == '-C target-feature=+crt-static' and record['nvcc_append_flags'] == '-Xcompiler=/MT' and record['cl'] == '/Zc:preprocessor /utf-8', 'CUDA static CRT flags differ')
        if key in ('cpu-test','cuda-test'):
            results = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', log)
            require(bool(results), 'test summary missing')
            counts = [sum(int(r[i]) for r in results) for i in range(3)]
            require(counts[0] > 0 and counts[1:] == [0,0], 'failed or ignored tests')
            totals[key] = dict(passed=counts[0], failed=counts[1], ignored=counts[2])
        if key == 'python-tests':
            match = re.search(r'Ran (\d+) tests', log)
            require(match is not None and re.search(r'^OK$', log, re.M) is not None, 'Python tests did not pass')
            totals[key] = dict(passed=int(match[1]), failed=0, ignored=0)
    return totals


def verify_rust_entry(entry: dict, info: dict, reports: dict, name: str, horizon: int, mode: str, model: str, predict: bool = False) -> None:
    config = 'configs/engineering/cement-twin.toml' if name == 'cement' else f'configs/reference/{ETT[name]}.toml'
    expected = dict(dataset=name, horizon=horizon, seed=2021, mode=mode, model=model, source_sha256=info['source']['source_sha256'], config_sha256=info['configs'][config], binary_sha256=info['binaries'][mode]['sha256'], predict=predict)
    require(entry['identity'] == expected, 'Rust acceptance entry identity differs')
    verify_artifacts(entry['artifacts'])
    run = Path(entry['run'])
    actual = tomllib.loads((run/'resolved-config.toml').read_text(encoding='utf-8'))
    validate_config(actual, config, horizon, 2021, mode, model)
    summary, evaluation = entry['summary'], entry['evaluation']
    require(summary == read_json(run/'train-summary.json') and evaluation == read_json(run/'evaluation.json'), 'Rust saved output differs')
    counts = reports[f'{name}-{horizon}']['window_counts']
    data_sha = info['data'][actual['data']['path']]
    require(summary['source_sha256'] == expected['source_sha256'] and summary['data_sha256'] == data_sha and evaluation['data_sha256'] == data_sha, 'Rust source/data identity differs')
    require(summary['access']['windows'][2] == 0 and summary['access']['batches'][2] == 0, 'Trainer visited test split')
    require(evaluation['windows'] == counts[2] and evaluation['metrics']['standardized']['overall']['count'] == counts[2]*horizon*len(actual['data']['target_columns']), 'Rust evaluation coverage differs')
    if model == 'dlinear':
        epochs = summary['epochs']
        require(1 <= len(epochs) <= 10 and summary['access']['windows'] == [counts[0]*len(epochs),counts[1]*len(epochs),0], 'Rust training coverage differs')
        best = min(epochs, key=lambda e:e['validation_loss'])['epoch']
        require(best == summary['best_epoch'] and (len(epochs) == 10 or len(epochs)-best == 3), 'Rust early stopping differs')
        for i, epoch in enumerate(epochs):
            require(epoch['epoch'] == i+1 and epoch['valid_labels'] == counts[0]*horizon*len(actual['data']['target_columns']), 'Rust epoch coverage differs')
            require(math.isclose(epoch['learning_rate'],1e-4*0.5**max(0,i-1),rel_tol=1e-14), 'Rust epoch rate differs')
    else:
        require(summary['optimizer_steps'] == 0 and summary['epochs'] == [], 'parameter-free model training differs')
    manifest = read_json(bundle_at(run)/'manifest.json')
    bundle_at(run, 'last')
    require(manifest['epoch'] == summary['best_epoch'] and manifest['provenance']['source_sha256'] == info['source']['source_sha256'] and manifest['data_sha256'] == data_sha, 'best checkpoint identity differs')
    require(evaluation['checkpoint_sha256'] == sha256(bundle_at(run)/'manifest.json'), 'evaluation checkpoint differs')
    if predict:
        prediction = read_json(run/'prediction-summary.json')
        require(prediction['metrics'] == evaluation['metrics'] and prediction['prediction_rows'] == evaluation['metrics']['standardized']['overall']['count'], 'CPU closure prediction differs')
    for receipt in entry['receipts']:
        verify_receipt(receipt, info)


def final_gate(args: argparse.Namespace) -> None:
    suite, info = suite_context(args)
    data = read_json(suite/'data-gate.json')
    require(data['passed'], 'data gate missing or failed')
    reports = data['reports']
    required_data = {f'{name}-{h}' for name in ETT for h in HORIZONS} | {'exchange'} | {f'cement-{h}' for h in CEMENT_COUNTS}
    require(required_data <= reports.keys(), 'data matrix coverage missing')
    for receipt in data['receipts']:
        verify_receipt(receipt, info)
    d4_gate(argparse.Namespace(suite=args.suite, report='docs/results/p1-dlinear-ett.md'))
    cpu = read_json(suite/'entries/rust-ETTh1-dlinear-cpu-h96-s2021.json')
    verify_rust_entry(cpu, info, reports, 'ETTh1', 96, 'cpu', 'dlinear', True)
    cement = read_json(suite/'cement.json')
    require(cement['passed'] and cement['runs'] == 18 and cement['variables'] == 18 and len(cement['rows']) == 324, 'cement full gate missing')
    keys = {(m,h,d) for m in ('last-value','seasonal-naive','dlinear') for h in CEMENT_COUNTS for d in ('cpu','cuda')}
    paths = list((suite/'entries').glob('rust-cement-*.json'))
    entries = [read_json(p) for p in paths]
    require(len(entries) == 18 and {(e['identity']['model'],e['identity']['horizon'],e['identity']['mode']) for e in entries} == keys, 'cement exact matrix coverage differs')
    for entry in entries:
        identity = entry['identity']
        verify_rust_entry(entry, info, reports, 'cement', identity['horizon'], identity['mode'], identity['model'])
        variables = entry['evaluation']['metrics']['original']['variables']
        require(len(variables) == 18, 'cement variable coverage differs')
        for name, metric in variables.items():
            require(metric['count'] == CEMENT_COUNTS[identity['horizon']][2]*identity['horizon'] and all(math.isfinite(metric[k]) for k in ('mae','rmse')), 'cement metric coverage differs')
            row = [r for r in cement['rows'] if (r['model'],r['horizon'],r['device'],r['variable']) == (identity['model'],identity['horizon'],identity['mode'],name)]
            require(len(row) == 1 and all(row[0][k] == metric[k] for k in ('count','mae','rmse')), 'cement report differs from evaluation')
    recompute = read_json(suite/'recompute.json')
    require(recompute['passed'] and recompute['rows'] == 18574560 and recompute['origins'] == 103192 and len(recompute['checks']) == 6, 'recomputation coverage differs')
    require(recompute['source']['source_sha256'] == info['source']['source_sha256'] and recompute['data_sha256'] == PRIVATE_SHA and recompute['script_sha256'] == sha256(ROOT/'tools/validation/recompute_predictions.py'), 'recomputation identity differs')
    require(all(c['passed'] and math.isfinite(c['difference']) and c['difference'] < 1e-6 and c['count'] == 6191520 for c in recompute['checks']), 'recomputation metric bound failed')
    verify_artifacts(recompute['artifacts'])
    for receipt in recompute['receipts']:
        verify_receipt(receipt, info)
    quality = read_json(suite/'quality.json')
    totals = validate_quality(quality)
    result = dict(passed=True, suite=str(suite), source=info['source'], official_runs=48, rust_cuda_runs=48, d4_checks=32, cement_runs=18, prediction_rows=18574560, recompute_checks=6, max_recompute_difference=max(c['difference'] for c in recompute['checks']), tests=totals, quality_commands=len(quality['records']), gates=artifacts([suite/n for n in ('suite.json','data-gate.json','d4.json','cement.json','recompute.json','quality.json','reference-reuse.json')]))
    write_json(suite/'final.json', result)
    lines = ['# P1 final acceptance', '', f'Suite: {suite.relative_to(ROOT).as_posix()}. Result: PASS.', f'Runtime source SHA256: {info["source"]["source_sha256"]}. Build HEAD: {info["source"]["code_sha"]}; dirty={info["source"]["dirty"]}. The build HEAD predates the uncommitted P1 source; the source snapshot identifies the tested implementation.', '', '| Gate | Result |', '| --- | --- |', '| Data | ETT 16 combinations; Exchange 7,588 records; cement raw/scaler/segments/windows and zero-parse cache hit |', '| ETT CPU | Complete ETTh1 H96 seed2021 train/evaluate/predict |', '| ETT CUDA | Official 48 verified formal runs + 48 fresh Rust runs |', '| D4 | 32/32, unchanged two-sided bounds, ddof=0 |', '| Cement | 18/18 full runs, all 18 original-unit variables |', '| Independent predictions | 18,574,560 rows; 103,192 origins; 6/6 metrics below 1e-6 |']
    for name, counts in totals.items():
        lines.append(f'| {name} | {counts["passed"]} passed; {counts["failed"]} failed; {counts["ignored"]} ignored |')
    lines += [f'| Maximum recompute difference | {result["max_recompute_difference"]:.17g} |', '| Quality | fmt, both builds and Clippy -D warnings, full workspace tests, cuda:0 smoke, real cuda:9 rejection, Python tests, task validation, privacy and diff checks |', '', '## Evidence and decisions', '', '- Complete per-seed D4 results: docs/results/p1-dlinear-ett.md.', '- Complete 18-variable cement statistics: docs/results/p1-cement-baselines.md.', '- Original 31/32 D4 failure, cross-loading diagnosis and approved RNG correction: research/d4-diagnosis.md. The original suite is retained.', '- Formal official reuse is explicit in reference-reuse.json and each entry; original code, arguments, data, environment, logs and artifact hashes were checked.', '- Exchange count correction and all four cement protocol adjustments remain approved. Seeds, hyperparameters and numerical bounds were not relaxed.', '- Receipts and raw artifacts remain under ignored runs/. No private rows or prediction rows are included in tracked reports.', '- P1 inference bundles do not support training resume. P2, P3, P4, push and remote CI remain outside this task. P1 finish-work remains pending after the local commit.']
    (ROOT/args.report).write_text(chr(10).join(lines)+chr(10), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    entry = commands.add_parser("preflight")
    entry.add_argument("--suite", required=True)
    entry.add_argument("--cpu-bin", required=True)
    entry.add_argument("--cuda-bin", required=True)
    entry.set_defaults(action=preflight)
    entry = commands.add_parser("data")
    entry.add_argument("--suite", required=True)
    entry.set_defaults(action=data_gate)
    entry = commands.add_parser("ett-cpu")
    entry.add_argument("--suite", required=True)
    entry.set_defaults(action=ett_cpu)
    entry = commands.add_parser("ett-reference")
    entry.add_argument("--suite", required=True)
    entry.set_defaults(action=ett_reference)
    entry = commands.add_parser('import-reference')
    entry.add_argument('--suite', required=True)
    entry.add_argument('--from-suite', required=True)
    entry.set_defaults(action=import_reference)
    entry = commands.add_parser("ett-rust")
    entry.add_argument("--suite", required=True)
    entry.set_defaults(action=ett_rust)
    entry = commands.add_parser("d4")
    entry.add_argument("--suite", required=True)
    entry.add_argument("--report", required=True)
    entry.set_defaults(action=d4_gate)
    entry = commands.add_parser('cement')
    entry.add_argument('--suite', required=True)
    entry.add_argument('--report', required=True)
    entry.set_defaults(action=cement_gate)
    entry = commands.add_parser('recompute')
    entry.add_argument('--suite', required=True)
    entry.set_defaults(action=recompute_gate)
    entry = commands.add_parser('final')
    entry.add_argument('--suite', required=True)
    entry.add_argument('--report', required=True)
    entry.set_defaults(action=final_gate)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
