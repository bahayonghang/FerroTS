"""Run the fixed P2 CUDA acceptance matrix with immutable attempt receipts."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from gpu_observer import DeviceMemoryObserver
from p1_acceptance import (ROOT, PRIVATE, PRIVATE_SHA, ETT, HORIZONS, SEEDS, CEMENT_COUNTS,
    sha256, write_json, read_json, require, git, source_snapshot, artifacts, verify_artifacts,
    bundle_at, d4_values, matrix_by_key)

UPSTREAM_SHA = "c2426e68ca13f74aaec08045c5c724d8ad328124"
UPSTREAM = ROOT / "tools/reference/.cache/iTransformer"
P1_SUITE = ROOT / "runs/p1-acceptance/20260928-0842-rng"


def config_path(name, horizon):
    return (f"configs/engineering/itransformer-cement-h{horizon}.toml" if name == "cement"
            else f"configs/reference/itransformer/{ETT[name]}-h{horizon}.toml")


def model_profile(name, horizon):
    dim = 512 if name == "cement" or (name == "ETTh1" and horizon >= 336) else 256 if name == "ETTh1" else 128
    return dict(name="itransformer", d_model=dim, d_ff=2048 if name == "cement" else dim,
        n_heads=8, e_layers=2, dropout=0.1, activation="gelu", use_norm=True)


def validate_config(actual, name, horizon, seed):
    expected = tomllib.loads((ROOT / config_path(name, horizon)).read_text(encoding="utf-8"))
    require(actual["model"] == model_profile(name, horizon), "model configuration differs")
    require(actual["window"] == expected["window"], "window configuration differs")
    for key, value in expected["data"].items():
        if key == "missing":
            value = dict(value, tokens=value.get("tokens", ["", "NaN", "nan"]))
        require(actual["data"].get(key) == value, f"data configuration differs: {key}")
    contract = dict(seed=seed, device="cuda:0", dtype="f32", batch_size=32, accumulation_steps=1,
        max_epochs=10, patience=3, learning_rate=0.0001, beta1=0.9, beta2=0.999, eps=1e-8,
        weight_decay=0., lr_schedule="type1", drop_last=name != "cement")
    for key, value in contract.items():
        require(actual["train"].get(key, False if key == "drop_last" else None) == value,
                f"training configuration differs: {key}")


def imported_context(info):
    """Validate the immutable parent of an explicit, fully audited tool transition."""
    origin = info.get("imported_evidence")
    if origin is None:
        return None
    require(origin["format_version"] == 1, "unsupported evidence transition")
    verify_artifacts(origin["files"])
    parent = Path(origin["suite"]).resolve()
    parent_path = parent/"suite.json"
    require(str(parent_path) in origin["files"], "parent suite is not bound")
    previous = read_json(parent_path)
    require("imported_evidence" not in previous, "nested evidence transitions are not supported")
    for key in ("upstream_sha", "source", "data", "configs", "reference_files", "binaries", "environment",
                "seeds", "horizons", "ddof", "policy", "formal_cuda_runs", "p1_baseline_files"):
        require(previous[key] == info[key], f"imported training identity differs: {key}")
    audit_path = str(Path(origin["verification"]).resolve())
    require(audit_path in origin["files"], "parent verification is not bound")
    audit = read_json(Path(audit_path))
    require(audit["passed"] and audit["official_cuda"] == 48 and audit["rust_ett_cuda"] == 48
        and audit["d4_passed"] == 32 and audit["suite_sha256"] == sha256(parent_path)
        and audit["source_sha256"] == info["source"]["source_sha256"], "parent verification differs")
    require(set(origin["gates"]) == {"preflight", "data-gate", "numerics", "resume"}, "imported prerequisite set differs")
    require(origin["receipts"] == audit["receipts"] and origin["gates"] == audit["gates"], "imported audit inventory differs")
    return parent, origin


def suite_context(args):
    suite = (ROOT / args.suite).resolve()
    require((ROOT / "runs/p2-acceptance").resolve() in suite.parents, "P2 suite must be inside runs/p2-acceptance")
    info = read_json(suite / "suite.json")
    require(info["upstream_sha"] == UPSTREAM_SHA and info["seeds"] == SEEDS and info["horizons"] == HORIZONS, "suite protocol changed")
    require(info["source"]["source_sha256"] == source_snapshot()["source_sha256"], "runtime source changed; rebuild and rerun affected experiments")
    for group in ("configs", "reference_files", "validation_files", "data"):
        for name, expected in info[group].items():
            require(sha256(ROOT / name) == expected, f"{group} fingerprint changed: {name}")
    for item in info["binaries"].values():
        require(sha256(Path(item["path"])) == item["sha256"], "binary changed")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip() == UPSTREAM_SHA, "upstream HEAD changed")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=UPSTREAM, text=True).strip(), "upstream checkout changed")
    verify_artifacts(info["p1_baseline_files"])
    imported_context(info)
    return suite, info


def execute(suite, info, key, command, observe=False):
    directory = suite / "commands" / key / dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    tick = time.perf_counter()
    print(f"START {key}", flush=True)
    from contextlib import nullcontext
    record = dict(key=key, command=command, exit_code=None, started_at=started,
        source_sha256=info["source"]["source_sha256"], suite_sha256=sha256(suite/"suite.json"),
        directory=str(directory))
    try:
        with DeviceMemoryObserver(directory / "memory") if observe else nullcontext():
            with (directory / "stdout.log").open("w", encoding="utf-8") as out, (directory / "stderr.log").open("w", encoding="utf-8") as err:
                result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"), stdout=out, stderr=err)
        record["exit_code"] = result.returncode
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_s=time.perf_counter()-tick,
            files=artifacts([p for p in directory.rglob("*") if p.is_file()]))
        write_json(directory / "receipt.json", record)
        print(f"END {key} exit={record['exit_code']} seconds={record['elapsed_s']:.3f}", flush=True)
    require(record["exit_code"] == 0, f"command failed; original evidence retained in {directory}")
    if observe:
        memory = read_json(directory/"memory/device-memory.json")
        require(memory["samples"] > 0 and memory["peak_used_mib"] is not None and not memory["errors"], "device memory observation failed")
    return record, (directory/"stdout.log").read_text(encoding="utf-8")


def verify_receipt(receipt, info, suite):
    path = Path(receipt["directory"])
    require(read_json(path/"receipt.json") == receipt and receipt["exit_code"] == 0, "receipt changed or failed")
    expected_suite = sha256(suite/"suite.json")
    if receipt["suite_sha256"] != expected_suite:
        context = imported_context(info)
        require(context is not None, "receipt suite differs without an explicit import")
        parent, origin = context
        location = str((path/"receipt.json").resolve())
        require(location in origin["receipts"], "receipt is not in the audited import inventory")
        verify_artifacts({location: origin["receipts"][location]})
        expected_suite = sha256(parent/"suite.json")
    require(receipt["suite_sha256"] == expected_suite and receipt["source_sha256"] == info["source"]["source_sha256"], "receipt source identity differs")
    verify_artifacts(receipt["files"])
    if (path/"memory/device-memory.json").exists():
        memory = read_json(path/"memory/device-memory.json")
        require(memory["interval_ms"] == 50 and memory["device"] == 0 and memory["samples"] > 0
            and not memory["errors"] and memory["peak_used_mib"] >= memory["baseline_used_mib"] >= 0,
            "device memory evidence is invalid")


def preflight(args):
    require(os.environ.get("RUSTUP_AUTO_INSTALL") == "0" and os.environ.get("RUSTUP_TOOLCHAIN") == "1.98.0", "fixed CUDA environment required")
    binary = (ROOT / args.cuda_bin).resolve()
    require(binary.is_file(), "CUDA binary missing")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip() == UPSTREAM_SHA, "upstream SHA mismatch")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=UPSTREAM, text=True).strip(), "upstream is dirty")
    manifest = tomllib.loads((ROOT/"docs/datasets/manifest.toml").read_text(encoding="utf-8"))
    fingerprints = {}
    for item in manifest["file"]:
        path = ROOT/item["path"]
        require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], "public data fingerprint mismatch")
        fingerprints[item["path"]] = item["sha256"]
    require(PRIVATE.stat().st_size == 1068005224 and sha256(PRIVATE) == PRIVATE_SHA, "private data fingerprint mismatch")
    fingerprints[PRIVATE.relative_to(ROOT).as_posix()] = PRIVATE_SHA
    import torch
    require(torch.cuda.is_available(), "official CUDA device missing")
    require(torch.__version__ == "2.14.0+cu130" and torch.version.git_version == "08187d9e0fba026dc8217405802ab5381dc88d90", "pinned torch identity differs")
    suite = (ROOT/args.suite).resolve()
    require((ROOT/"runs/p2-acceptance").resolve() in suite.parents, "suite must be inside runs/p2-acceptance")
    suite.mkdir(parents=True, exist_ok=False)
    info = dict(format_version=2, upstream_sha=UPSTREAM_SHA, source=source_snapshot(), data=fingerprints,
        seeds=SEEDS, horizons=HORIZONS, ddof=0, policy="cuda-first-2026-09-29", formal_cuda_runs=105,
        original_cpu_control="not executed under the later user instruction",
        binaries={"cuda":dict(path=str(binary), sha256=sha256(binary))},
        environment=dict(os=platform.platform(), python=sys.version.split()[0], torch=torch.__version__,
            torch_sha=torch.version.git_version, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    require(read_json(P1_SUITE/"final.json")["passed"], "accepted P1 baseline evidence missing")
    info["p1_baseline_files"] = artifacts([P1_SUITE/"final.json", P1_SUITE/"cement.json", P1_SUITE/"data-gate.json"] +
        list((P1_SUITE/"entries").glob("rust-cement-*-cuda-*.json")))
    for field, paths in [("configs",(ROOT/"configs").rglob("*.toml")),
                         ("reference_files",(ROOT/"tools/reference").glob("*.py")),
                         ("validation_files",(ROOT/"tools/validation").glob("*"))]:
        info[field] = {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(paths) if p.is_file() and p.suffix in (".toml",".py",".ps1")}
    write_json(suite/"suite.json", info)
    with DeviceMemoryObserver(suite/"preflight-memory"):
        time.sleep(.25)
    receipt, output = execute(suite, info, "preflight-doctor", [str(binary),"doctor","--device","cuda:0","--training-smoke-test"])
    require(info["source"]["source_sha256"] in output, "binary does not match runtime source inventory")
    memory = read_json(suite/"preflight-memory/device-memory.json")
    require(memory["samples"] > 0 and not memory["errors"], "preflight device sampling failed")
    write_gate(suite, "preflight", dict(passed=True, receipts=[receipt],
        artifacts=artifacts(list((suite/"preflight-memory").iterdir()))))
    print(json.dumps(dict(suite=str(suite),status="preflight_passed")))


def official_arguments(name, horizon, model_id):
    profile = model_profile(name,horizon)
    pairs = dict(is_training=1, root_path="data/ETT-small/", data_path=f"{name}.csv", model_id=model_id,
        model="iTransformer", data=name, features="M", target="OT", seq_len=96, label_len=48,
        pred_len=horizon, enc_in=7, dec_in=7, c_out=7, d_model=profile["d_model"], d_ff=profile["d_ff"],
        n_heads=8, e_layers=2, dropout=0.1, activation="gelu", use_norm=1, embed="timeF", freq="h",
        batch_size=32, train_epochs=10, patience=3, learning_rate=0.0001, lradj="type1", loss="MSE",
        num_workers=0, itr=1, use_gpu="True", gpu=0, des="P2")
    return [part for key,value in pairs.items() for part in ("--"+key,str(value))]


def validate_epochs(epochs, train_windows, horizon, channels, official=False):
    require(1 <= len(epochs) <= 10, "invalid epoch count")
    best, stale = None, 0
    for index,row in enumerate(epochs):
        require(stale < 3, "training continued after patience")
        require(math.isfinite(row["validation_loss"]) and row["validation_loss"] >= 0, "invalid validation loss")
        improved = best is None or row["validation_loss"] < best["validation_loss"] or (official and row["validation_loss"] == best["validation_loss"])
        if improved:
            best, stale = row, 0
        else:
            stale += 1
        require(row["epoch"] == index+1 and math.isclose(row["learning_rate"],1e-4*0.5**max(0,index-1),rel_tol=1e-14), "epoch/rate differs")
        require(row["training_windows"] == train_windows, "training coverage differs")
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in ("train_seconds","validation_seconds","steady_seconds")), "invalid performance timing")
        require(row["steady_seconds"] > 0 and row["steady_windows"] == train_windows-32, "invalid steady throughput coverage")
        if not official:
            require(row["valid_labels"] == train_windows*horizon*channels, "effective training labels differ")
    require(len(epochs) == 10 or stale == 3, "short training did not terminate by patience")
    return best


def verify_rust(entry, suite, info):
    identity=entry["identity"];name,horizon,seed=identity["dataset"],identity["horizon"],identity["seed"]
    require(name in [*ETT,"cement"] and seed in SEEDS and horizon in (CEMENT_COUNTS if name=="cement" else HORIZONS), "invalid Rust matrix key")
    config=config_path(name,horizon)
    require(identity == dict(dataset=name,horizon=horizon,seed=seed,model="itransformer",mode="cuda",
        source_sha256=info["source"]["source_sha256"],config_sha256=info["configs"][config],binary_sha256=info["binaries"]["cuda"]["sha256"]), "Rust identity differs")
    verify_artifacts(entry["artifacts"])
    run=Path(entry["run"]);actual=tomllib.loads((run/"resolved-config.toml").read_text(encoding="utf-8"))
    validate_config(actual,name,horizon,seed)
    summary,evaluation=entry["summary"],entry["evaluation"]
    require(summary == read_json(run/"train-summary.json") and evaluation == read_json(run/"evaluation.json"), "Rust outputs changed")
    require(evaluation["seed"] == seed and all(math.isfinite(v) and v >= 0 for k,v in evaluation["metrics"]["standardized"]["overall"].items()
        if k in ("mse", "mae", "rmse")), "Rust evaluation seed or metrics differ")
    counts=read_json(suite/"data-gate.json")["reports"][f"{name}-{horizon}"]["window_counts"]
    train_windows=counts[0] if name=="cement" else counts[0]//32*32
    best=validate_epochs(summary["epochs"],train_windows,horizon,len(actual["data"]["target_columns"]))
    require(summary["best_epoch"]==best["epoch"] and summary["access"]["windows"]==[train_windows*len(summary["epochs"]),counts[1]*len(summary["epochs"]),0], "selection or split coverage differs")
    data_sha=info["data"][actual["data"]["path"]]
    require(summary["source_sha256"]==identity["source_sha256"] and summary["data_sha256"]==data_sha and evaluation["data_sha256"]==data_sha, "Rust source/data differs")
    require(evaluation["windows"]==counts[2] and evaluation["metrics"]["standardized"]["overall"]["count"]==counts[2]*horizon*len(actual["data"]["target_columns"]), "test coverage differs")
    best_dir=bundle_at(run);last=bundle_at(run,"last");manifest=read_json(best_dir/"manifest.json")
    require(manifest["format_version"]==2 and manifest["epoch"]==summary["best_epoch"] and manifest["upstream_sha"]==UPSTREAM_SHA
        and manifest["provenance"]["source_sha256"]==identity["source_sha256"] and evaluation["checkpoint_sha256"]==sha256(best_dir/"manifest.json"), "v2 best identity differs")
    require(read_json(last/"trainer-state.json")["phase"]=="complete", "training did not complete")
    performance=[json.loads(line) for line in (run/"performance.jsonl").read_text().splitlines()]
    require(len(performance)==len(summary["epochs"]) and all(row["checkpoint_seconds"]>=0 and row["steady_samples_per_second"]>0 for row in performance), "performance coverage differs")
    for row, epoch in zip(performance, summary["epochs"]):
        require(all(row[k] == epoch[k] for k in ("epoch", "train_seconds", "validation_seconds", "training_windows", "steady_seconds", "steady_windows")), "performance and epoch records differ")
    require((Path(entry["receipts"][0]["directory"])/"memory/device-memory.json").is_file(), "Rust memory evidence missing")
    for receipt in entry["receipts"]: verify_receipt(receipt,info,suite)


def rust_run(suite,info,name,horizon,seed):
    key=f"rust-{name}-itransformer-cuda-h{horizon}-s{seed}"
    destination=suite/"entries"/f"{key}.json"
    if destination.exists():
        entry=read_json(destination);verify_rust(entry,suite,info);print(f"VERIFIED {key}",flush=True);return entry
    config=config_path(name,horizon);binary=info["binaries"]["cuda"]["path"]
    run_id=key+"-"+dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f");run=suite/"rust"/run_id
    command=[binary,"train","--config",config,"--seed",str(seed),"--cache-dir",read_json(suite/"data-gate.json")["cache_dir"],"--output-root",str(run.parent),"--run-id",run_id]
    train_receipt,stdout=execute(suite,info,key+"-train",command,observe=True);summary=json.loads(stdout)
    evaluate_receipt,stdout=execute(suite,info,key+"-evaluate",[binary,"evaluate","--checkpoint",str(run/"best"),"--output",str(run/"evaluation.json")])
    entry=dict(identity=dict(dataset=name,horizon=horizon,seed=seed,model="itransformer",mode="cuda",
        source_sha256=info["source"]["source_sha256"],config_sha256=info["configs"][config],binary_sha256=info["binaries"]["cuda"]["sha256"]),
        run=str(run),summary=summary,evaluation=json.loads(stdout),receipts=[train_receipt,evaluate_receipt],
        artifacts=artifacts([p for p in run.rglob("*") if p.is_file()]))
    verify_rust(entry,suite,info);write_json(destination,entry)
    metric=entry["evaluation"]["metrics"]["standardized"]["overall"]
    print(f"PASS {key}: mse={metric['mse']:.9g} mae={metric['mae']:.9g}",flush=True);return entry


def inspect_data(suite, info, name, horizon, cache_dir, suffix=""):
    config = "configs/reference/exchange.toml" if name == "exchange" else config_path(name, horizon)
    receipt, stdout = execute(suite, info, f"data-{name}-h{horizon}{suffix}",
        [info["binaries"]["cuda"]["path"], "data", "inspect", "--config", config,
         "--horizon", str(horizon), "--cache-dir", str(cache_dir)])
    report = json.loads(stdout)
    audit = report["window_audit"]
    require(audit["all_same_segment"] and audit["all_target_split"] and audit["checked"] == sum(report["window_counts"]), "window audit failed")
    source = tomllib.loads((ROOT/config).read_text(encoding="utf-8"))["data"]["path"]
    require(report["cache"]["source_sha256"] == info["data"][source], "inspected data hash differs")
    return report, receipt


def data_gate(args):
    suite, info = suite_context(args)
    require_gate(suite, info, "preflight")
    if (suite/"data-gate.json").exists():
        require_gate(suite, info, "data-gate")
        print("VERIFIED existing data gate", flush=True)
        return
    reports, receipts = {}, []
    cache_dir = suite/"cache"/dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for expected in read_json(ROOT/"tests/fixtures/p1/ett.json")["datasets"]:
        name = expected["name"]
        for horizon in HORIZONS:
            actual, receipt = inspect_data(suite, info, name, horizon, cache_dir)
            reports[f"{name}-{horizon}"] = actual
            receipts.append(receipt)
            require(actual["raw"]["rows"] == expected["rows"], "ETT rows differ")
            require(actual["window_counts"] == [expected["counts"][str(horizon)][k] for k in ("train", "val", "test")], "ETT window counts differ")
            for i, col in enumerate(actual["scaler"]["columns"]):
                require(col["name"] == expected["columns"][i] and abs(col["mean"]-expected["mean"][i]) < 1e-9
                    and abs(col["scale"]-expected["scale"][i]) < 1e-9, "ETT scaler differs")
    exchange, receipt = inspect_data(suite, info, "exchange", 96, cache_dir)
    receipts.append(receipt)
    reports["exchange"] = exchange
    require(exchange["raw"]["rows"] == 7588 and exchange["raw"]["start"] == "1990-01-01 00:00:00.000000"
        and exchange["raw"]["end"] == "2010-10-10 00:00:00.000000" and exchange["prepared"]["usable_rows"] == [5311, 760, 1517], "Exchange protocol differs")
    for horizon in CEMENT_COUNTS:
        actual, receipt = inspect_data(suite, info, "cement", horizon, cache_dir)
        reports[f"cement-{horizon}"] = actual
        receipts.append(receipt)
        raw, prepared = actual["raw"], actual["prepared"]
        require([raw[k] for k in ("rows", "non_increasing", "duplicate_groups", "duplicate_rows", "duplicate_removed")] == [6914135, 7, 7, 14, 7], "cement raw counters differ")
        require(raw["start"] == "2023-01-01 00:00:05.000000" and raw["end"] == "2025-05-21 09:42:02.000000", "cement raw times differ")
        require([c["missing"] for c in raw["columns"]] == [620602,0,0,1,0,0,1,0,0,0,0,0,895122,0,0,0,0,0]
            and all(c["non_numeric"] == 0 for c in raw["columns"]), "cement raw missing/nonnumeric counts differ")
        require([prepared[k] for k in ("rows", "running_segments", "segment_rows")] == [1254823,48,529531]
            and prepared["split_segments"] == [37,4,7] and prepared["usable_rows"] == [303009,58246,104051]
            and prepared["imputed_per_column"] == [13]*18, "cement prepared counters differ")
        require(actual["window_counts"] == CEMENT_COUNTS[horizon], "cement windows differ")
        require(len(actual["scaler"]["columns"]) == 18 and all(c["count"] == 303009 for c in actual["scaler"]["columns"]), "cement scaler fit differs")
        baseline = read_json(P1_SUITE/"data-gate.json")["reports"][f"cement-{horizon}"]["scaler"]["columns"]
        for current, previous in zip(actual["scaler"]["columns"], baseline):
            require(current["name"] == previous["name"] and abs(current["mean"]-previous["mean"]) < 1e-9
                and abs(current["scale"]-previous["scale"]) < 1e-9, "cement accepted scaler values differ")
        if horizon != 15:
            require(actual["scaler"] == reports["cement-15"]["scaler"], "cement scaler changed with horizon")
    second, receipt = inspect_data(suite, info, "cement", 15, cache_dir, "-repeat")
    receipts.append(receipt)
    reports["cement-repeat"] = second
    require(not reports["cement-15"]["cache"]["hit"] and reports["cement-15"]["cache"]["parse_count"] == 1, "fresh cement cache required")
    require(second["cache"]["hit"] and second["cache"]["parse_count"] == 0, "repeat cement load parsed CSV")
    write_gate(suite, "data-gate", dict(passed=True, cache_dir=str(cache_dir), reports=reports, receipts=receipts))
    print("PASS data: ETT16, Exchange7588, cement3 and fresh/hit cache", flush=True)


def write_gate(suite, name, value):
    value["suite_sha256"] = sha256(suite/"suite.json")
    write_json(suite/(name+".json"), value)


def require_gate(suite, info, name):
    value = read_json(suite/(name+".json"))
    expected_suite = sha256(suite/"suite.json")
    if value["suite_sha256"] != expected_suite:
        context = imported_context(info)
        require(context is not None, f"gate has no explicit import: {name}")
        parent, origin = context
        require(name in origin["gates"], f"gate is not in the audited import inventory: {name}")
        original = parent/(name+".json")
        verify_artifacts({str(original): origin["gates"][name]})
        require(sha256(suite/(name+".json")) == sha256(original), f"imported gate changed: {name}")
        expected_suite = sha256(parent/"suite.json")
    require(value["passed"] and value["suite_sha256"] == expected_suite, f"gate missing or stale: {name}")
    for receipt in value.get("receipts", []):
        verify_receipt(receipt, info, suite)
    if value.get("artifacts"):
        verify_artifacts(value["artifacts"])
    return value


def cargo_gate(args, kind):
    suite, info = suite_context(args)
    require_gate(suite, info, "preflight")
    arguments = (["test", "-p", "ferrots-train", "--features", "cuda", "--test", "p2_numerics", "cuda_all_parameters", "--", "--nocapture"]
        if kind == "numerics" else ["test", "-p", "ferrots-cli", "--features", "cuda", "--test", "resume_process", "--", "--nocapture"])
    script = ". ./tools/validation/p1_env.ps1 -Mode cuda; $p2Args=@(" + ",".join("'"+a+"'" for a in arguments) + "); Invoke-P1Cargo @p2Args"
    receipt, stdout = execute(suite, info, kind, ["pwsh", "-NoProfile", "-Command", script])
    stderr = (Path(receipt["directory"])/"stderr.log").read_text(encoding="utf-8")
    pattern = r"P2 complete numerical report: (.+)" if kind == "numerics" else r"resume process report: (.+)"
    paths = re.findall(pattern, stdout+"\n"+stderr)
    require(len(paths) == 1, f"{kind} report missing or duplicated")
    path = Path(paths[0].strip())
    report = read_json(path)
    if kind == "numerics":
        require(report["device"] == "cuda" and report["acceptance_version"] == 2 and not report["failures"]
            and len(report["cases"]) == 9, "full CUDA numerical gate did not pass")
    else:
        require(report["fresh_cli_processes"] and report["dropout"] == 0.1 and len(report["cases"]) == 10
            and all(c["device"] == "cuda:0" for c in report["cases"]), "full CUDA process recovery gate did not pass")
    write_gate(suite, kind, dict(passed=True, receipts=[receipt], artifacts=artifacts([path]), report=report))


def official_identity(info, name, horizon, seed):
    return dict(dataset=name, horizon=horizon, seed=seed, upstream_sha=UPSTREAM_SHA,
        data_sha256=info["data"][f"data/ETT-small/{name}.csv"], reference_files=info["reference_files"])


def verify_official(entry, suite, info):
    import numpy as np
    identity = entry["identity"]
    name, horizon, seed = identity["dataset"], identity["horizon"], identity["seed"]
    require(name in ETT and horizon in HORIZONS and seed in SEEDS, "invalid official matrix key")
    require(identity == official_identity(info, name, horizon, seed), "official identity differs")
    verify_artifacts(entry["artifacts"])
    run = Path(entry["run"])
    result = read_json(run/"receipt.json")
    require(result == entry["result"] and result["status"] == "completed" and result["exit_code"] == 0
        and result["seed"] == seed and result["upstream_sha"] == UPSTREAM_SHA
        and result["data_sha256"] == identity["data_sha256"], "official receipt differs")
    require(result["wrapper_sha256"] == info["reference_files"]["tools/reference/run_itransformer.py"]
        and result["entry_sha256"] == sha256(UPSTREAM/"run.py"), "official wrapper or entry differs")
    for path, expected in result["source_hashes"].items():
        require(sha256(UPSTREAM/path) == expected, "official source changed")
    for path, expected in result["artifacts"].items():
        verify_artifacts({str(run/path): expected})
    arguments = entry["run_args"]
    model_id = arguments[arguments.index("--model_id")+1]
    require(arguments == official_arguments(name, horizon, model_id), "official arguments differ")
    normalized = list(arguments)
    normalized[normalized.index("--root_path")+1] = str((ROOT/"data/ETT-small").resolve())+os.sep
    normalized += ["--checkpoints", str(run/"checkpoints")+os.sep]
    require(result["arguments"] == normalized, "official executed arguments differ")
    resolved = read_json(run/"resolved-args.json")
    for key in ("use_amp", "use_multi_gpu", "inverse", "output_attention"):
        require(not resolved[key], f"unexpected official flag: {key}")
    require(resolved["use_gpu"] and resolved["gpu"] == 0, "official GPU differs")
    environment = read_json(run/"environment.json")
    require(environment["torch"] == info["environment"]["torch"] and environment["torch_sha"] == info["environment"]["torch_sha"]
        and environment["cuda"] == info["environment"]["cuda"] and environment["device"] == info["environment"]["gpu"], "official environment differs")
    counts = read_json(suite/"data-gate.json")["reports"][f"{name}-{horizon}"]["window_counts"]
    log = (run/"stdout.log").read_text(encoding="utf-8")
    settings = re.findall(r">+testing : (.+?)<+", log)
    require(len(settings) == 1, "official test setting missing or duplicated")
    for flag, count in zip(("train", "val", "test"), counts):
        require(re.search(rf"^{flag} {count}$", log, re.M), "official split coverage differs")
    result_dir = run/"results"/settings[0]
    metrics = np.load(result_dir/"metrics.npy", allow_pickle=False)
    require(entry["metrics"] == dict(mae=float(metrics[0]), mse=float(metrics[1]), rmse=float(metrics[2])), "official metrics changed")
    require(all(math.isfinite(v) and v >= 0 for v in entry["metrics"].values()), "official metric invalid")
    for filename in ("pred.npy", "true.npy"):
        array = np.load(result_dir/filename, mmap_mode="r", allow_pickle=False)
        require(array.shape == (counts[2], horizon, 7) and array.dtype == np.float32, "official output coverage/dtype differs")
        for begin in range(0, len(array), 256):
            require(np.isfinite(array[begin:begin+256]).all(), "nonfinite official output")
    require((run/"checkpoints"/settings[0]/"checkpoint.pth").is_file(), "official best checkpoint missing")
    rows = [json.loads(line) for line in (run/"performance.jsonl").read_text().splitlines()]
    validate_epochs(rows, counts[0]//32*32, horizon, 7, official=True)
    require(all(r["checkpoint_seconds"] >= 0 and r["allocator_max_allocated_bytes"] > 0
        and r["allocator_max_reserved_bytes"] >= r["allocator_max_allocated_bytes"] for r in rows), "official performance missing")
    require((Path(entry["receipts"][0]["directory"])/"memory/device-memory.json").is_file(), "official memory evidence missing")
    for receipt in entry["receipts"]:
        verify_receipt(receipt, info, suite)


def training_prerequisites(suite, info):
    for name in ("preflight", "data-gate", "numerics", "resume"):
        require_gate(suite, info, name)
    validate_quality(read_json(suite/"quality.json"))


def ett_reference(args):
    import numpy as np
    suite, info = suite_context(args)
    training_prerequisites(suite, info)
    for name in ETT:
        for horizon in HORIZONS:
            for seed in SEEDS:
                key = f"official-{name}-h{horizon}-s{seed}"
                destination = suite/"entries"/(key+".json")
                if destination.exists():
                    verify_official(read_json(destination), suite, info)
                    print(f"VERIFIED {key}", flush=True)
                    continue
                run = suite/"official"/(key+"-"+dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
                run_args = official_arguments(name, horizon, f"P2{name}h{horizon}s{seed}")
                receipt, stdout = execute(suite, info, key, [sys.executable,
                    str(ROOT/"tools/reference/run_itransformer.py"), "--seed", str(seed), "--output", str(run), "--", *run_args], observe=True)
                settings = re.findall(r">+testing : (.+?)<+", stdout)
                require(len(settings) == 1, "official test setting missing or duplicated")
                values = np.load(run/"results"/settings[0]/"metrics.npy", allow_pickle=False)
                entry = dict(identity=official_identity(info,name,horizon,seed), run=str(run), run_args=run_args,
                    result=read_json(run/"receipt.json"), metrics=dict(mae=float(values[0]),mse=float(values[1]),rmse=float(values[2])),
                    receipts=[receipt], artifacts=artifacts([p for p in run.rglob("*") if p.is_file()]))
                verify_official(entry, suite, info)
                write_json(destination, entry)
                print(f"PASS {key}: mse={entry['metrics']['mse']:.9g} mae={entry['metrics']['mae']:.9g}", flush=True)
    verified_matrix(suite, info, "official")
    print("PASS official CUDA ETT: 48/48", flush=True)


def verified_matrix(suite, info, kind):
    pattern = "official-*.json" if kind == "official" else "rust-ETT*-itransformer-cuda-*.json"
    entries = matrix_by_key([read_json(p) for p in (suite/"entries").glob(pattern)], kind)
    for entry in entries.values():
        (verify_official if kind == "official" else verify_rust)(entry, suite, info)
    return entries


def ett_rust(args):
    suite, info = suite_context(args)
    training_prerequisites(suite, info)
    verified_matrix(suite, info, "official")
    for name in ETT:
        for horizon in HORIZONS:
            for seed in SEEDS:
                rust_run(suite, info, name, horizon, seed)
    verified_matrix(suite, info, "rust")
    print("PASS Rust CUDA ETT: 48/48", flush=True)


def comparison_rows(official, rust):
    rows = []
    for name in ETT:
        for horizon in HORIZONS:
            for metric in ("mse", "mae"):
                o = [official[(name,horizon,s)]["metrics"][metric] for s in SEEDS]
                r = [rust[(name,horizon,s)]["evaluation"]["metrics"]["standardized"]["overall"][metric] for s in SEEDS]
                rows.append(dict(dataset=name, horizon=horizon, metric=metric, **d4_values(o,r)))
    return rows


def d4_gate(args):
    suite, info = suite_context(args)
    rows = comparison_rows(verified_matrix(suite,info,"official"), verified_matrix(suite,info,"rust"))
    passed = all(r["passed"] for r in rows)
    write_gate(suite, "d4", dict(passed=passed, ddof=0, rows=rows, official_runs=48, rust_runs=48))
    lines = ["# P2 iTransformer ETT comparison", "",
        f"Suite: {suite.relative_to(ROOT).as_posix()}. Upstream: {UPSTREAM_SHA}.",
        f"Runtime source: {info['source']['source_sha256']}. CUDA: {info['environment']['gpu']}.",
        f"D4: {sum(r['passed'] for r in rows)}/32. Official CUDA 48 runs; Rust CUDA 48 runs. Seeds: 2021, 2022, 2023.", "",
        "Population standard deviation uses ddof=0. The bound is max(2 * official_std, 0.02 * abs(official_mean)). Full precision values determine each result.",
        "Both paths use the fixed official profiles listed in docs/models/itransformer.md. Official validation retains shuffled batches and drop_last. Rust validation uses all windows in fixed order and element-weighted loss. Rust training does not read test data.",
        "The original full CPU control was not executed under the later CUDA-first user instruction. Existing CPU numerical and recovery evidence remains separate.", "",
        "| Dataset | H | Metric | Official seeds | Rust seeds | Official mean / std | Rust mean / std | Bound | Difference | Pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    show = lambda xs: " / ".join(format(x, ".9g") for x in xs)
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['horizon']} | {row['metric']} | {show(row['official'])} | {show(row['rust'])} | {show([row['official_mean'],row['official_std']])} | {show([row['rust_mean'],row['rust_std']])} | {row['threshold']:.9g} | {row['difference']:.9g} | {row['passed']} |")
    (ROOT/"docs/results/p2-itransformer-ett.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    require(passed, f"D4 failed: {sum(r['passed'] for r in rows)}/32; original matrix retained")
    print("PASS D4: 32/32", flush=True)


def cement_entries(suite, info):
    entries = [read_json(p) for p in (suite/"entries").glob("rust-cement-*.json")]
    expected = {(h,s) for h in CEMENT_COUNTS for s in SEEDS}
    found = {}
    for entry in entries:
        identity = entry["identity"]
        key = identity["horizon"], identity["seed"]
        require(key in expected and key not in found, "duplicate or unexpected cement matrix key")
        verify_rust(entry, suite, info)
        variables = entry["evaluation"]["metrics"]["original"]["variables"]
        targets = tomllib.loads((ROOT/config_path("cement",key[0])).read_text(encoding="utf-8"))["data"]["target_columns"]
        require(isinstance(variables, dict) and len(variables) == 18 and set(variables) == set(targets), "cement must evaluate all 18 variables")
        require(all(v["count"] == CEMENT_COUNTS[key[0]][2]*key[0] for v in variables.values()), "cement variable coverage differs")
        require(all(isinstance(v[metric], (int, float)) and math.isfinite(v[metric]) and v[metric] >= 0
            for v in variables.values() for metric in ("mae", "mse", "rmse")), "cement invalid metric")
        found[key] = entry
    require(set(found) == expected, "cement matrix must contain exactly 9 runs")
    return found


def cement_gate(args):
    suite, info = suite_context(args)
    training_prerequisites(suite, info)
    require_gate(suite, info, "d4")
    for horizon in CEMENT_COUNTS:
        for seed in SEEDS:
            rust_run(suite, info, "cement", horizon, seed)
    entries = cement_entries(suite, info)
    rows = []
    for horizon in CEMENT_COUNTS:
        targets = tomllib.loads((ROOT/config_path("cement",horizon)).read_text(encoding="utf-8"))["data"]["target_columns"]
        for variable in targets:
            for metric in ("mae", "rmse"):
                values = [entries[(horizon,s)]["evaluation"]["metrics"]["original"]["variables"][variable][metric] for s in SEEDS]
                rows.append(dict(horizon=horizon, variable=variable, metric=metric, seeds=values,
                    mean=statistics.fmean(values), std=statistics.pstdev(values)))
    write_gate(suite, "cement", dict(passed=True, runs=9, ddof=0, rows=rows))
    cement_report(suite, info, rows)
    print("PASS CUDA cement: 9/9; all 18 variables", flush=True)


def cement_report(suite, info, rows):
    verify_artifacts(info["p1_baseline_files"])
    baseline = {}
    for model in ("last-value", "seasonal-naive", "dlinear"):
        for horizon in CEMENT_COUNTS:
            entry = read_json(P1_SUITE/"entries"/f"rust-cement-{model}-cuda-h{horizon}-s2021.json")
            verify_artifacts(entry["artifacts"])
            require(entry["evaluation"]["data_sha256"] == PRIVATE_SHA, "P1 baseline data differs")
            for name, variable in entry["evaluation"]["metrics"]["original"]["variables"].items():
                for metric in ("mae", "rmse"):
                    baseline[(model,horizon,name,metric)] = variable[metric]
    lines = ["# P2 iTransformer cement comparison", "",
        f"Suite: {suite.relative_to(ROOT).as_posix()}. Runtime source: {info['source']['source_sha256']}.",
        "Nine complete CUDA runs use H15/30/60 and seeds 2021/2022/2023. All 18 variables and all legal test windows are evaluated in original units.",
        f"P1 baseline suite: {P1_SUITE.relative_to(ROOT).as_posix()}. Baseline CUDA results use seed2021 (n=1; standard deviation is not applicable). Artifacts were rechecked before this report.", "",
        "| H | Variable | Metric | iTransformer seeds 2021 / 2022 / 2023 | Mean | Population std | P1 LastValue n=1 | P1 SeasonalNaive n=1 | P1 DLinear n=1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        values = " / ".join(format(v,".9g") for v in row["seeds"])
        references = " | ".join(format(baseline[(m,row["horizon"],row["variable"],row["metric"])],".9g") for m in ("last-value","seasonal-naive","dlinear"))
        lines.append(f"| {row['horizon']} | {row['variable']} | {row['metric']} | {values} | {row['mean']:.9g} | {row['std']:.9g} | {references} |")
    (ROOT/"docs/results/p2-itransformer-cement.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def performance_report(suite, entries):
    rows = []
    for entry in entries:
        identity = entry["identity"]
        run = Path(entry["run"])
        epochs = [json.loads(line) for line in (run/"performance.jsonl").read_text().splitlines()]
        memory = read_json(Path(entry["receipts"][0]["directory"])/"memory/device-memory.json")
        seconds = sum(e["steady_seconds"] for e in epochs)
        rows.append(dict(dataset=identity["dataset"],horizon=identity["horizon"],seed=identity["seed"],
            engine="official" if "upstream_sha" in identity else "rust", epochs=epochs, memory=memory,
            steady_samples_per_second=sum(e["steady_windows"] for e in epochs)/seconds))
    write_gate(suite, "performance", dict(passed=True, runs=rows))
    lines = ["# P2 CUDA performance evidence", "",
        f"Suite: {suite.relative_to(ROOT).as_posix()}. Each row represents one complete training run.",
        "Stage boundaries synchronize CUDA. Steady throughput excludes the first optimizer group of each epoch (batch32, accumulation1). The aggregate divides all steady windows by all steady seconds.",
        "Memory is device-wide used memory sampled by nvidia-smi every 50 ms. Other processes contribute to these values. Process memory is unavailable under the current observer. The estimated dropped count measures sample gaps. Official allocator values are measured by torch. Rust allocator values are unavailable.",
        "Full per-epoch times, sample counts, allocator peaks and observer timestamps are in performance.json and each immutable run artifact.", "",
        "| Engine | Dataset | H | Seed | Epochs | Train s | Validation s | Checkpoint s | Steady windows/s | Baseline MiB | Device peak MiB | Samples | Estimated dropped | Allocated peak bytes | Reserved peak bytes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        epochs, memory = row["epochs"], row["memory"]
        allocated = max((e.get("allocator_max_allocated_bytes",0) for e in epochs),default=0) or "unavailable"
        reserved = max((e.get("allocator_max_reserved_bytes",0) for e in epochs),default=0) or "unavailable"
        lines.append(f"| {row['engine']} | {row['dataset']} | {row['horizon']} | {row['seed']} | {len(epochs)} | {sum(e['train_seconds'] for e in epochs):.3f} | {sum(e['validation_seconds'] for e in epochs):.3f} | {sum(e['checkpoint_seconds'] for e in epochs):.3f} | {row['steady_samples_per_second']:.3f} | {memory['baseline_used_mib']} | {memory['peak_used_mib']} | {memory['samples']} | {memory['estimated_dropped_samples']} | {allocated} | {reserved} |")
    (ROOT/"docs/results/p2-itransformer-performance.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def recompute_gate(args):
    suite, info = suite_context(args)
    require_gate(suite, info, "cement")
    entry = cement_entries(suite, info)[(60,2021)]
    run = Path(entry["run"])
    columns = ["窑喂料反馈", "窑头煤反馈", "分解炉内温度1反馈"]
    directory = suite/"recompute"/dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=False)
    prediction_path = directory/"predictions.csv.gz"
    receipt, stdout = execute(suite, info, "cement-three-variable-export", [info["binaries"]["cuda"]["path"],
        "predict", "--checkpoint", str(run/"best"), "--device", "cuda:0", "--columns", ",".join(columns), "--output", str(prediction_path)])
    prediction = json.loads(stdout)
    require(prediction["metrics"] == entry["evaluation"]["metrics"] and prediction["checkpoint_sha256"] == entry["evaluation"]["checkpoint_sha256"]
        and prediction["data_sha256"] == PRIVATE_SHA and prediction["prediction_rows"] == 18574560, "prediction identity or coverage differs")
    write_json(directory/"prediction-summary.json", prediction)
    output = directory/"independent.json"
    second, _ = execute(suite, info, "cement-independent-recompute", [sys.executable,
        str(ROOT/"tools/validation/recompute_predictions.py"), "--predictions", str(prediction_path),
        "--evaluation", str(run/"evaluation.json"), "--columns", *columns, "--horizon", "60", "--output", str(output)])
    result = read_json(output)
    require(result["passed"] and result["rows"] == 18574560 and len(result["checks"]) == 6, "full independent recomputation failed")
    result.update(data_sha256=PRIVATE_SHA, checkpoint_sha256=prediction["checkpoint_sha256"], receipts=[receipt,second],
        artifacts=artifacts([p for p in directory.iterdir() if p.is_file()]), script_sha256=sha256(ROOT/"tools/validation/recompute_predictions.py"))
    write_gate(suite, "recompute", result)
    print("PASS independent recomputation: 18,574,560 rows, 6/6 metrics", flush=True)


QUALITY_COMMANDS = {
    "fmt": ["fmt","--all","--check"],
    "cuda-build": ["build","--workspace","--features","cuda"],
    "cuda-clippy": ["clippy","--workspace","--all-targets","--features","cuda","--","-D","warnings"],
    "cuda-test": ["test","--workspace","--features","cuda"],
    "cuda-doctor": ["run","-p","ferrots-cli","--features","cuda","--","doctor","--device","cuda:0","--training-smoke-test"],
    "cuda-invalid-device": ["run","-p","ferrots-cli","--features","cuda","--","doctor","--device","cuda:9"],
    "python-tests": ["run","--frozen","--project","tools/reference","python","-m","unittest","discover","-s","tools/validation/tests","-p","test_*.py"],
    "task-validate": ["run","--frozen","--project","tools/reference","python",".trellis/scripts/task.py","validate",".trellis/tasks/09-28-p2-itransformer"],
    "private-data": ["tools/check_private_data.sh"],
    "diff-check": ["diff","--check"],
}


def validate_quality(quality):
    records = quality["records"]
    require(quality["passed"] and len(records) == len(QUALITY_COMMANDS) and {r["key"] for r in records} == set(QUALITY_COMMANDS), "full quality command coverage differs")
    require(quality["script_sha256"] == sha256(ROOT/"tools/validation/p2_quality.ps1"), "quality script changed")
    require(bool(quality["inputs"]), "quality input inventory missing")
    for path, expected in quality["inputs"].items():
        require(sha256(ROOT/path) == expected, f"quality input changed: {path}")
    totals = {}
    for row in records:
        key = row["key"]
        require(row["arguments"] == QUALITY_COMMANDS[key], f"quality command differs: {key}")
        require(row["rustup_auto_install"] == "0" and row["rustup_toolchain"] == "1.98.0"
            and "1.98.0" in row["rustc"] and "1.98.0" in row["rustdoc"], "quality toolchain differs")
        require(row["rustflags"] == "-C target-feature=+crt-static" and row["nvcc_append_flags"] == "-Xcompiler=/MT"
            and row["cl"] == "/Zc:preprocessor /utf-8", "CUDA CRT flags differ")
        require(sha256(Path(row["log"])) == row["log_sha256"], "quality log changed")
        log = Path(row["log"]).read_text(encoding="utf-8-sig")
        expected_failure = key == "cuda-invalid-device"
        require(row["expected_failure"] == expected_failure and row["exit_code"] == (1 if expected_failure else 0), f"quality exit differs: {key}")
        if expected_failure:
            require("requested_device: cuda:9" in log and "device initialization" in log, "cuda:9 did not reach device validation")
        if key == "cuda-test":
            results = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", log)
            counts = [sum(int(r[i]) for r in results) for i in range(3)]
            require(counts[0] >= 71 and counts[1:] == [0,0], "missing, failed or ignored Rust tests")
            totals[key] = dict(passed=counts[0], failed=0, ignored=0)
        if key == "python-tests":
            match = re.search(r"Ran (\d+) tests", log)
            require(match is not None and int(match[1]) >= 19 and re.search(r"^OK\s*$", log, re.M), "Python tests did not pass")
            totals[key] = dict(passed=int(match[1]), failed=0, ignored=0)
    return totals


def final_gate(args):
    suite, info = suite_context(args)
    training_prerequisites(suite, info)
    official, rust = verified_matrix(suite, info, "official"), verified_matrix(suite, info, "rust")
    rows = comparison_rows(official, rust)
    require(all(r["passed"] for r in rows) and rows == require_gate(suite,info,"d4")["rows"], "D4 record differs")
    cement = cement_entries(suite, info)
    require_gate(suite, info, "cement")
    require_gate(suite, info, "recompute")
    require(len(list((suite/"entries").glob("*.json"))) == 105, "formal run set is not exactly 105")
    totals = validate_quality(read_json(suite/"quality.json"))
    performance_report(suite, [*official.values(), *rust.values(), *cement.values()])
    write_gate(suite, "final", dict(passed=True, official_cuda=len(official), rust_ett_cuda=len(rust),
        cement_cuda=len(cement), d4_passed=32, recomputed_rows=18574560, quality=totals,
        cpu_control="not executed under the later CUDA-first user instruction",
        source_sha256=info["source"]["source_sha256"]))
    print("PASS final CUDA acceptance: 105 runs, D4 32/32, independent metrics 6/6", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight","numerics","resume","data","ett-reference","ett-rust","d4","cement","recompute","final"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--cuda-bin", default="target/windows-static/x86_64-pc-windows-msvc/debug/ferrots.exe")
    args = parser.parse_args()
    if args.stage in ("numerics", "resume"):
        cargo_gate(args, args.stage)
    else:
        {"preflight":preflight, "data":data_gate, "ett-reference":ett_reference, "ett-rust":ett_rust,
         "d4":d4_gate, "cement":cement_gate, "recompute":recompute_gate, "final":final_gate}[args.stage](args)


if __name__ == "__main__":
    main()
