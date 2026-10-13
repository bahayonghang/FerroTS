"""Statistical reports for verified P3 entries; never launches training."""
from __future__ import annotations

import json
import math
import statistics
import tomllib
from pathlib import Path

from p1_acceptance import (
    CEMENT_COUNTS, ETT, HORIZONS, ROOT, SEEDS, d4_values, read_json,
    require, sha256, write_json,
)


def _values(entries):
    return list(entries.values()) if isinstance(entries, dict) else list(entries)


def _matrix(entries, engine, cement=False, complete=True):
    expected = ({("cement", h, s) for h in CEMENT_COUNTS for s in SEEDS}
                if cement else {(d, h, s) for d in ETT for h in HORIZONS for s in SEEDS})
    found = {}
    for entry in _values(entries):
        identity = entry["identity"]
        key = identity["dataset"], identity["horizon"], identity["seed"]
        require(identity["engine"] == engine and identity["model"] == "informer",
                "report entry engine/model differs")
        require(key in expected and key not in found, "duplicate or unexpected matrix cell")
        found[key] = entry
    if complete:
        require(set(found) == expected, f"{engine} matrix missing cells: {len(found)}/{len(expected)}")
    return found


def d4_rows(official_entries, rust_entries):
    """Exactly 32 unchanged D4 comparisons from the two complete ETT matrices."""
    official = _matrix(official_entries, "official")
    rust = _matrix(rust_entries, "rust")
    rows = []
    for dataset in ETT:
        for horizon in HORIZONS:
            for metric in ("mse", "mae"):
                o = [official[(dataset, horizon, s)]["metrics"][metric] for s in SEEDS]
                r = [rust[(dataset, horizon, s)]["evaluation"]["metrics"]["standardized"]["overall"][metric]
                     for s in SEEDS]
                rows.append(dict(dataset=dataset, horizon=horizon, metric=metric, **d4_values(o, r)))
    return rows


def cement_rows(entries):
    """108 named original-unit MAE/RMSE cells, aggregated over three seeds."""
    matrix = _matrix(entries, "rust", cement=True)
    rows = []
    for horizon in CEMENT_COUNTS:
        config = ROOT / f"configs/engineering/informer-cement-h{horizon}.toml"
        targets = tomllib.loads(config.read_text(encoding="utf-8"))["data"]["target_columns"]
        require(len(targets) == len(set(targets)) == 18, "config requires all 18 variables")
        by_seed = {}
        for seed in SEEDS:
            variables = matrix[("cement", horizon, seed)]["evaluation"]["metrics"]["original"]["variables"]
            require(isinstance(variables, dict) and set(variables) == set(targets),
                    "cement must evaluate all 18 variables by name")
            for metrics in variables.values():
                require(metrics["count"] == CEMENT_COUNTS[horizon][2] * horizon,
                        "cement variable coverage differs")
                require(all(isinstance(metrics.get(m), (int, float))
                            and not isinstance(metrics[m], bool)
                            and math.isfinite(metrics[m]) and metrics[m] >= 0
                            for m in ("mae", "mse", "rmse")), "invalid original-unit metric")
            by_seed[seed] = variables
        for name in targets:
            for metric in ("mae", "rmse"):
                values = [by_seed[s][name][metric] for s in SEEDS]
                rows.append(dict(horizon=horizon, variable=name, metric=metric, seeds=values,
                                 mean=statistics.fmean(values), std=statistics.pstdev(values)))
    return rows


def _historical_baselines(rows):
    """Read published numbers only; historical receipts are not current acceptance."""
    directory = ROOT / "docs/results"
    p1, p2 = directory / "p1-cement-baselines.md", directory / "p2-itransformer-cement.md"
    baseline = {}
    for line in p1.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or cells[0] not in ("last-value", "seasonal-naive", "dlinear"):
            continue
        model, horizon, device, name, _count, mae, rmse = cells[:7]
        if device == "cuda":
            for metric, value in (("mae", mae), ("rmse", rmse)):
                key = model, int(horizon), name, metric
                require(key not in baseline, "duplicate historical P1 cell")
                baseline[key] = dict(mean=float(value), std=None, n=1)
    for line in p2.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or cells[0] not in {str(h) for h in CEMENT_COUNTS}:
            continue
        horizon, name, metric, _seeds, mean, std = cells[:6]
        key = "itransformer", int(horizon), name, metric
        require(key not in baseline, "duplicate historical P2 cell")
        baseline[key] = dict(mean=float(mean), std=float(std), n=3)
    expected = {(m, r["horizon"], r["variable"], r["metric"])
                for r in rows for m in ("last-value", "seasonal-naive", "dlinear", "itransformer")}
    require(set(baseline) == expected, "historical published baseline coverage differs")
    for values in baseline.values():
        require(math.isfinite(values["mean"]) and values["mean"] >= 0
                and (values["std"] is None or math.isfinite(values["std"]) and values["std"] >= 0),
                "invalid historical published metric")
    enriched = [dict(row, historical={m: baseline[(m, row["horizon"], row["variable"], row["metric"])]
                                    for m in ("last-value", "seasonal-naive", "dlinear", "itransformer")})
                for row in rows]
    provenance = {str(path.relative_to(ROOT)): sha256(path) for path in (p1, p2)}
    return enriched, provenance


def _performance_rows(entries):
    rows = []
    for entry in entries:
        identity, run = entry["identity"], Path(entry["run"])
        epochs = [json.loads(line) for line in (run / "performance.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        require(bool(epochs), "performance epochs are missing")
        require([e["epoch"] for e in epochs] == list(range(1, len(epochs) + 1)),
                "performance epoch progression differs")
        for epoch in epochs:
            for field in ("train_seconds", "validation_seconds", "checkpoint_seconds", "steady_seconds", "steady_windows"):
                value = epoch[field]
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value >= 0, f"invalid performance {field}")
        memory_path = Path(entry["receipts"][0]["directory"]) / "memory/device-memory.json"
        memory = read_json(memory_path)
        require(memory["interval_ms"] == 50 and memory["device"] == int(identity["device"].split(":")[1]),
                "performance observer device/interval differs")
        require(memory["samples"] > 0 and not memory["errors"], "performance memory observer failed")
        require(all(isinstance(memory[k], (int, float)) and math.isfinite(memory[k]) and memory[k] >= 0
                    for k in ("baseline_used_mib", "peak_used_mib", "estimated_dropped_samples")),
                "invalid device memory measurement")
        seconds = sum(e["steady_seconds"] for e in epochs)
        require(seconds > 0, "performance steady time must be positive")
        allocated = reserved = None
        if identity["engine"] == "official":
            for field in ("allocator_max_allocated_bytes", "allocator_max_reserved_bytes"):
                require(all(isinstance(e.get(field), (int, float)) and math.isfinite(e[field]) and e[field] >= 0
                            for e in epochs), "official allocator measurement missing")
            allocated = max(e["allocator_max_allocated_bytes"] for e in epochs)
            reserved = max(e["allocator_max_reserved_bytes"] for e in epochs)
        rows.append(dict(engine=identity["engine"], dataset=identity["dataset"],
                         horizon=identity["horizon"], seed=identity["seed"], device=identity["device"],
                         epochs=epochs, memory=memory, run=str(run),
                         steady_samples_per_second=sum(e["steady_windows"] for e in epochs) / seconds,
                         allocator_max_allocated_bytes=allocated, allocator_max_reserved_bytes=reserved))
    return rows


def _show(values):
    return " / ".join(format(v, ".9g") for v in values)


def write_reports(suite, info, entries):
    """Write complete reports when available, preserving failed D4 evidence.

    Callers verify immutable receipts and artifacts before passing entries. This
    layer checks coverage and computes statistics, not run authenticity.
    """
    suite, entries = Path(suite), _values(entries)
    official = [e for e in entries if e["identity"]["engine"] == "official"]
    rust = [e for e in entries if e["identity"]["engine"] == "rust" and e["identity"]["dataset"] != "cement"]
    cement = [e for e in entries if e["identity"]["engine"] == "rust" and e["identity"]["dataset"] == "cement"]
    require(len(official) + len(rust) + len(cement) == len(entries), "unexpected report engine")
    for items, engine, private in ((official, "official", False), (rust, "rust", False), (cement, "rust", True)):
        _matrix(items, engine, cement=private, complete=False)
    docs = ROOT / "docs/results"
    docs.mkdir(parents=True, exist_ok=True)
    source = info["source"]
    caption = f"Suite: {suite}. Runtime source: {source['source_sha256']}."
    d4 = dict(complete=len(official) == len(rust) == 48, passed=False, ddof=0,
              official_runs=len(official), rust_runs=len(rust), source=source, rows=[])
    if d4["complete"]:
        d4["rows"] = d4_rows(official, rust)
        d4["passed"] = all(r["passed"] for r in d4["rows"])
        lines = ["# P3 Informer ETT comparison", "", caption,
                 f"D4: {sum(r['passed'] for r in d4['rows'])}/32. Official CUDA 48; Rust CUDA 48; seeds 2021/2022/2023.", "",
                 "Population standard deviation uses ddof=0. Each bound is max(2 * official_std, 0.02 * abs(official_mean)); the absolute difference must be at most the bound. Full precision determines the result. Failed cells remain in the report.",
                 "Profiles and documented initialization, sampling and validation differences are in docs/models/informer.md. Historical P1/P2 results are not receipts for this runtime.", "",
                 "| Dataset | H | Metric | Official seeds | Rust seeds | Official mean / std | Rust mean / std | Bound | Difference | Pass |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in d4["rows"]:
            lines.append(f"| {r['dataset']} | {r['horizon']} | {r['metric']} | {_show(r['official'])} | {_show(r['rust'])} | {_show([r['official_mean'], r['official_std']])} | {_show([r['rust_mean'], r['rust_std']])} | {r['threshold']:.9g} | {r['difference']:.9g} | {r['passed']} |")
        (docs / "p3-informer-ett.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    private = dict(complete=len(cement) == 9, passed=False, runs=len(cement), ddof=0, source=source, rows=[])
    if private["complete"]:
        rows, history = _historical_baselines(cement_rows(cement))
        private.update(passed=True, rows=rows, historical_published_files=history)
        lines = ["# P3 Informer cement comparison", "", caption,
                 "Nine complete CUDA runs: H15/30/60, seeds 2021/2022/2023. All 18 variables and legal test windows use original units. Each of the 108 cells contains a three-seed mean and population standard deviation (ddof=0).", "",
                 "P1 values are the published CUDA seed2021 results (n=1; standard deviation unavailable). P2 values are published three-seed means/std. Historical numbers are rounded as published; historical receipts are not loaded or accepted as current runtime evidence.",
                 "Sources: docs/results/p1-cement-baselines.md and docs/results/p2-itransformer-cement.md; their hashes are in cement.json.", "",
                 "| H | Variable | Metric | Informer seeds | Informer mean | Informer population std | P1 LastValue n=1 | P1 SeasonalNaive n=1 | P1 DLinear n=1 | P2 iTransformer mean / std |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            h = r["historical"]
            baseline = " | ".join(f"{h[m]['mean']:.9g}" for m in ("last-value", "seasonal-naive", "dlinear"))
            lines.append(f"| {r['horizon']} | {r['variable']} | {r['metric']} | {_show(r['seeds'])} | {r['mean']:.9g} | {r['std']:.9g} | {baseline} | {_show([h['itransformer']['mean'], h['itransformer']['std']])} |")
        (docs / "p3-informer-cement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    performance = dict(complete=len(entries) == 105, passed=len(entries) == 105,
                       source=source, environment=info.get("environment", {}),
                       runs=_performance_rows(entries))
    lines = ["# P3 CUDA performance evidence", "", caption,
             "Machine / GPU / toolchain: " + json.dumps(info.get("environment", {}), ensure_ascii=False, sort_keys=True),
             f"Complete training runs recorded: {len(entries)}/105. Device is shown per run; official and Rust engines remain separate.",
             "Stage boundaries synchronize CUDA. Steady throughput excludes the first optimizer group of each epoch, and aggregates total steady windows / total steady seconds. Per-epoch training, validation, checkpoint, optional test times and counts are retained in performance.json.",
             "Memory is whole-device used memory sampled by nvidia-smi every 50 ms; concurrent processes contribute. Baseline, peak, timestamps, sample counts, estimated dropped samples and errors are retained. Process memory is unavailable. Official allocator peaks are measured by torch; Rust allocator peaks are unavailable and are not substituted with device totals.", "",
             "| Engine | Dataset | H | Seed | Device | Epochs | Train s | Validation s | Checkpoint s | Steady windows/s | Baseline MiB | Device peak MiB | Samples | Estimated dropped | Torch allocated peak bytes | Torch reserved peak bytes |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in performance["runs"]:
        es, m = r["epochs"], r["memory"]
        lines.append(f"| {r['engine']} | {r['dataset']} | {r['horizon']} | {r['seed']} | {r['device']} | {len(es)} | {sum(e['train_seconds'] for e in es):.3f} | {sum(e['validation_seconds'] for e in es):.3f} | {sum(e['checkpoint_seconds'] for e in es):.3f} | {r['steady_samples_per_second']:.3f} | {m['baseline_used_mib']} | {m['peak_used_mib']} | {m['samples']} | {m['estimated_dropped_samples']} | {r['allocator_max_allocated_bytes'] if r['allocator_max_allocated_bytes'] is not None else 'unavailable'} | {r['allocator_max_reserved_bytes'] if r['allocator_max_reserved_bytes'] is not None else 'unavailable'} |")
    (docs / "p3-informer-performance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = {"d4": d4, "cement": private, "performance": performance}
    for name, report in output.items():
        write_json(suite / f"{name}.json", report)
        report["path"] = str(suite / f"{name}.json")
        report["report"] = str(docs / f"p3-informer-{'ett' if name == 'd4' else name}.md") if report["complete"] or name == "performance" else None
    output["passed"] = all(output[name]["passed"] for name in ("d4", "cement", "performance"))
    return output
