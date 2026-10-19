"""Verify and compare retained bounded official Informer CUDA trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from p1_acceptance import sha256, write_json
from p3_official_trajectory import compare_step, first_difference


def tensor_hash(tensor: torch.Tensor) -> dict:
    array = tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def same_sampling(left: Path, right: Path, rows: list[dict]) -> dict:
    with np.load(left) as a, np.load(right) as b:
        names = sorted(set(a.files) | set(b.files))
        for archive, row in zip((a, b), rows):
            recorded = {item["name"]: item for item in row["items"]
                        if item["name"].startswith(("sample.", "dropout."))}
            if set(recorded) != set(archive.files):
                raise ValueError("sampling archive keys differ from trajectory row")
            for name, item in recorded.items():
                array = archive[name]
                if name.startswith("dropout."):
                    count = int(np.prod(item["shape"]))
                    array = np.unpackbits(array, count=count).reshape(item["shape"]).astype(bool)
                digest = hashlib.sha256(array.tobytes()).hexdigest()
                if digest != item["value"]["sha256"]:
                    raise ValueError(f"sampling payload hash differs from row: {name}")
        difference = next((name for name in names if name not in a.files or name not in b.files
                           or not np.array_equal(a[name], b[name])), None)
        return dict(keys=len(names), first_difference=difference,
                    sample_calls=sum(name.startswith("sample.") for name in names),
                    dropout_masks=sum(name.startswith("dropout.") for name in names))


def tensor_differences(left: Path, right: Path) -> dict:
    a = torch.load(left, map_location="cpu", weights_only=True)
    b = torch.load(right, map_location="cpu", weights_only=True)
    result = {}
    for section in ("model", "gradients", "tensors"):
        changed = []
        for name in sorted(set(a[section]) | set(b[section])):
            if name not in a[section] or name not in b[section]:
                changed.append(dict(name=name, issue="missing"))
                continue
            x, y = a[section][name], b[section][name]
            if x.shape != y.shape or x.dtype != y.dtype:
                changed.append(dict(name=name, issue="shape_or_dtype"))
                continue
            if torch.equal(x, y):
                continue
            delta = (x.double() - y.double()).abs()
            changed.append(dict(name=name, unequal=int((x != y).sum()),
                                max_abs=float(delta.max()), mean_abs=float(delta.mean())))
        result[section] = dict(total=len(set(a[section]) | set(b[section])),
                               changed=changed)
    optimizer_changed = []
    if a["optimizer"]["param_groups"] != b["optimizer"]["param_groups"]:
        raise ValueError("optimizer hyperparameters or parameter ids differ")
    names_a, names_b = list(a["gradients"]), list(b["gradients"])
    if names_a != names_b:
        raise ValueError("optimizer parameter names differ")
    ids_a = a["optimizer"]["param_groups"][0]["params"]
    ids_b = b["optimizer"]["param_groups"][0]["params"]
    if len(ids_a) != len(names_a) or len(ids_b) != len(names_b):
        raise ValueError("optimizer parameter ids differ")
    for name, id_a, id_b in zip(names_a, ids_a, ids_b):
        state_a = a["optimizer"]["state"][id_a]
        state_b = b["optimizer"]["state"][id_b]
        if set(state_a) != set(state_b):
            raise ValueError(f"optimizer fields differ: {name}")
        for field in sorted(state_a):
            x, y = state_a[field], state_b[field]
            if not torch.equal(x, y):
                delta = (x.double() - y.double()).abs()
                optimizer_changed.append(dict(name=f"{name}.{field}",
                    unequal=int((x != y).sum()), max_abs=float(delta.max()),
                    mean_abs=float(delta.mean())))
    result["optimizer"] = dict(total=len(a["optimizer"]["state"]), changed=optimizer_changed)
    return result


def verify_initial_snapshot(trace: Path, row: dict) -> None:
    snapshot = torch.load(trace / "initial-state.pth", map_location="cpu", weights_only=True)
    if set(snapshot) != set(row["state"]):
        raise ValueError("initial snapshot state keys differ from row")
    for name, tensor in snapshot.items():
        if tensor_hash(tensor) != row["state"][name]:
            raise ValueError(f"initial snapshot differs from row: {name}")


def verify_step_snapshot(trace: Path, row: dict) -> None:
    snapshot = torch.load(trace / f"step-{row['step']:03d}.pth", map_location="cpu", weights_only=True)
    for name, tensor in snapshot["model"].items():
        if tensor_hash(tensor) != row["state_after_update"][name]:
            raise ValueError(f"updated state snapshot differs from row: {name}")
    for name, tensor in snapshot["gradients"].items():
        if tensor_hash(tensor) != row["gradients"][name]:
            raise ValueError(f"gradient snapshot differs from row: {name}")
    parameters = list(snapshot["gradients"])
    identifiers = snapshot["optimizer"]["param_groups"][0]["params"]
    if len(parameters) != len(identifiers):
        raise ValueError("optimizer parameter mapping cannot be verified")
    for name, identifier in zip(parameters, identifiers):
        recorded = row["optimizer_after"][name]
        saved = snapshot["optimizer"]["state"][identifier]
        if set(recorded) != set(saved):
            raise ValueError(f"optimizer state keys differ from row: {name}")
        for field, value in saved.items():
            observed = tensor_hash(value) if isinstance(value, torch.Tensor) else value
            if observed != recorded[field]:
                raise ValueError(f"optimizer snapshot differs from row: {name}.{field}")


def verify_receipt(attempt: Path) -> dict:
    receipt_path = attempt / "official/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    bad = []
    for name, record in receipt["artifacts"].items():
        path = attempt / "official" / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            bad.append(name)
    if bad:
        raise ValueError(f"receipt artifact mismatch: {attempt}: {bad}")
    if receipt["status"] != "completed" or receipt["exit_code"] != 0:
        raise ValueError(f"official process did not exit cleanly: {attempt}")
    return dict(sha256=sha256(receipt_path), artifacts_verified=len(receipt["artifacts"]),
                upstream_sha=receipt["upstream_sha"], source_hashes=receipt["source_hashes"],
                data_sha256=receipt["data_sha256"], wrapper_sha256=receipt["wrapper_sha256"],
                observer_sha256=receipt["observer_sha256"], elapsed_seconds=receipt["elapsed_seconds"])


def analyze(directory: Path) -> dict:
    directory = directory.resolve()
    if not directory.is_relative_to((ROOT / "runs/p3-development").resolve()):
        raise ValueError("run must be under runs/p3-development")
    identity = json.loads((directory / "identity.json").read_text())
    capture_source = (directory / "capture-source.py") if (directory / "capture-source.py").is_file() else (ROOT / "tools/validation/p3_official_trajectory.py")
    if identity["script_sha256"] != sha256(capture_source):
        raise ValueError("capture script changed since these traces")
    if identity["data_sha256"] != sha256(ROOT / "data/ETT-small/ETTh2.csv"):
        raise ValueError("ETTh2 data changed since these traces")
    attempts = [directory / name for name in ("attempt-001", "attempt-002")]
    receipts = [verify_receipt(path) for path in attempts]
    for key in ("upstream_sha", "source_hashes", "data_sha256", "wrapper_sha256", "observer_sha256"):
        if receipts[0][key] != receipts[1][key]:
            raise ValueError(f"official attempts differ in {key}")
    if receipts[0]["data_sha256"] != identity["data_sha256"]:
        raise ValueError("receipt data differs from identity")
    if receipts[0]["wrapper_sha256"] != identity["wrapper_sha256"] or receipts[0]["observer_sha256"] != identity["observer_sha256"]:
        raise ValueError("official wrapper or observer differs from identity")
    env = [json.loads((path / "official/environment.json").read_text()) for path in attempts]
    if env[0] != env[1]:
        raise ValueError("official environments differ")
    args = [json.loads((path / "official/resolved-args.json").read_text()) for path in attempts]
    for item in args:
        item.pop("checkpoints", None)
    if args[0] != args[1]:
        raise ValueError("official resolved arguments differ beyond checkpoint path")
    initial = [json.loads((path / "trace/initial.json").read_text()) for path in attempts]
    for attempt, row in zip(attempts, initial):
        verify_initial_snapshot(attempt / "trace", row)
    first = first_difference(initial[0], initial[1], "initial")
    divergence = dict(step=0, path=first) if first else None
    step_results = []
    for step in range(1, 101):
        files = [path / "trace" / f"step-{step:03d}.json" for path in attempts]
        if not all(path.is_file() for path in files):
            break
        rows = [json.loads(path.read_text()) for path in files]
        sampling = same_sampling(*(path / "trace" / f"sampling-{step:03d}.npz" for path in attempts), rows)
        if step in (1, 2, 5):
            for attempt, row in zip(attempts, rows):
                verify_step_snapshot(attempt / "trace", row)
        difference = compare_step(*rows)
        if sampling["first_difference"] and not difference:
            raise ValueError(f"sampling payload differs but row hash agrees at step {step}")
        step_results.append(dict(step=step, first_difference=difference,
                                 loss=[row["loss_value"] for row in rows], sampling=sampling,
                                 elapsed_seconds=[row["elapsed_seconds"] for row in rows]))
        if difference and divergence is None:
            divergence = dict(step=step, path=difference)
            break
    if not step_results and divergence is None:
        raise ValueError("no comparable optimizer updates")
    tensors = None
    if divergence and divergence["step"]:
        step = divergence["step"]
        snapshots = [path / "trace" / f"step-{step:03d}.pth" for path in attempts]
        if all(path.is_file() for path in snapshots):
            tensors = tensor_differences(*snapshots)
    manifests = []
    for attempt in attempts:
        manifest = {str(path.relative_to(attempt)): sha256(path) for path in sorted(attempt.rglob("*"))
                    if path.is_file() and path.name != "stop"}
        manifests.append(manifest)
    report = dict(classification="bounded official-to-official CUDA diagnosis; zero formal D4 entries",
                  identity_sha256=sha256(directory / "identity.json"),
                  capture_script_sha256=identity["script_sha256"],
                  capture_source=str(capture_source),
                  comparator_script_sha256=sha256(Path(__file__)),
                  receipts=[{k: v for k, v in receipt.items() if k != "source_hashes"} for receipt in receipts],
                  environment_equal=True, resolved_args_equal_except_checkpoints=True,
                  initial_difference=first, first_difference=divergence,
                  last_compared_step=step_results[-1]["step"] if step_results else 0,
                  steps=step_results, tensors=tensors,
                  attempt_manifest=manifests)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run)
    write_json(args.run / "comparison.json", report)
    print(json.dumps(dict(first_difference=report["first_difference"],
                          last_compared_step=report["last_compared_step"],
                          report=str(args.run / "comparison.json")), sort_keys=True))


if __name__ == "__main__":
    main()
