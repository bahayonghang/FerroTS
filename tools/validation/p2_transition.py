"""Preserve and revalidate P2 ETT evidence across the cement-consumer correction."""
from __future__ import annotations
import argparse
import ast
import copy
import datetime as dt
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/"tools/validation"))
import p2_acceptance as p

GATES = ("preflight", "data-gate", "numerics", "resume")
CHANGED = {"suite_context", "verify_receipt", "require_gate", "cement_entries", "cement_gate", "cement_report"}


def check_validator_transition(before, after):
    old, new = ast.parse(before), ast.parse(after)
    old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n, include_attributes=False) for n in new.body if isinstance(n, ast.FunctionDef)}
    p.require(set(new_functions)-set(old_functions) == {"imported_context"} and not set(old_functions)-set(new_functions), "unexpected validator function set")
    changed = {name for name in old_functions if old_functions[name] != new_functions[name]}
    p.require(changed == CHANGED, "validator change exceeds the cement and provenance boundary")
    other = lambda tree: [ast.dump(n, include_attributes=False) for n in tree.body if not isinstance(n, ast.FunctionDef)]
    p.require(other(old) == other(new), "validator constants or imports changed")
    return sorted(changed)


def parent_matrix(parent, info):
    official = p.verified_matrix(parent, info, "official")
    rust = p.verified_matrix(parent, info, "rust")
    rows = p.comparison_rows(official, rust)
    recorded = p.require_gate(parent, info, "d4")
    p.require(len(rows) == 32 and all(row["passed"] for row in rows) and rows == recorded["rows"], "complete original D4 must pass")
    p.require(len(list((parent/"entries").glob("*.json"))) == 96, "transition requires exactly 96 ETT entries and no cement entries")
    return [*official.values(), *rust.values()]


def prepare(args):
    parent, info = p.suite_context(argparse.Namespace(suite=args.parent))
    p.training_prerequisites(parent, info)
    entries = parent_matrix(parent, info)
    destination = (p.ROOT/args.output).resolve()
    p.require((p.ROOT/"runs").resolve() in destination.parents, "preparation must remain under runs")
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination/"p2_acceptance.before.py"
    shutil.copyfile(p.ROOT/"tools/validation/p2_acceptance.py", snapshot)
    quality = p.read_json(parent/"quality.json")
    receipt_paths = [Path(r["directory"])/"receipt.json" for e in entries for r in e["receipts"]]
    for name in GATES:
        receipt_paths.extend(Path(r["directory"])/"receipt.json" for r in p.read_json(parent/(name+".json")).get("receipts", []))
    files = [parent/"suite.json", parent/"quality.json", parent/"d4.json", snapshot]
    files += [parent/(name+".json") for name in GATES]
    files += list((parent/"entries").glob("*.json"))
    files += [Path(row["log"]) for row in quality["records"]]
    audit = dict(format_version=1, passed=True, time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        suite=str(parent), suite_sha256=p.sha256(parent/"suite.json"), source_sha256=info["source"]["source_sha256"],
        official_cuda=48, rust_ett_cuda=48, d4_passed=32, prior_validator=str(snapshot),
        files=p.artifacts(files), entries=p.artifacts(list((parent/"entries").glob("*.json"))),
        receipts=p.artifacts(receipt_paths), gates={name:p.artifacts([parent/(name+".json")])[str(parent/(name+".json"))] for name in GATES},
        prior_quality=p.validate_quality(quality))
    p.write_json(destination/"verification.json", audit)
    print(f"PASS preserved parent audit: {destination/'verification.json'}", flush=True)


def create(args):
    audit_path=(p.ROOT/args.preparation/"verification.json").resolve()
    audit=p.read_json(audit_path); p.require(audit["passed"], "parent audit did not pass")
    parent=Path(audit["suite"]); info=p.read_json(parent/"suite.json")
    p.require("imported_evidence" not in info, "nested imports are not supported")
    p.verify_artifacts(audit["files"])
    prior=Path(audit["prior_validator"])
    p.require(p.sha256(prior)==info["validation_files"]["tools/validation/p2_acceptance.py"], "old validator snapshot differs")
    changes=check_validator_transition(prior.read_text(encoding="utf-8-sig"),(p.ROOT/"tools/validation/p2_acceptance.py").read_text(encoding="utf-8-sig"))
    p.require(p.source_snapshot()["source_sha256"]==info["source"]["source_sha256"], "runtime source changed")
    for group in ("configs", "reference_files", "data", "validation_files"):
        for name, expected in info[group].items():
            actual = prior if name == "tools/validation/p2_acceptance.py" else p.ROOT/name
            p.require(p.sha256(actual)==expected, f"parent fingerprint differs: {name}")
    for binary in info["binaries"].values():
        p.require(p.sha256(Path(binary["path"]))==binary["sha256"], "parent CUDA binary changed")
    p.verify_artifacts(info["p1_baseline_files"])
    quality=p.read_json(parent/"quality.json")
    for name, expected in quality["inputs"].items():
        actual=prior if name=="tools/validation/p2_acceptance.py" else p.ROOT/name
        p.require(p.sha256(actual)==expected, f"parent quality input changed: {name}")
    parent_matrix(parent, info)
    p.verify_artifacts(audit["receipts"])
    p.require(p.artifacts(list((parent/"entries").glob("*.json")))==audit["entries"], "parent entry inventory changed")
    child=(p.ROOT/args.suite).resolve()
    p.require((p.ROOT/"runs/p2-acceptance").resolve() in child.parents and child!=parent, "new suite must be a distinct P2 acceptance directory")
    new=copy.deepcopy(info)
    new["validation_files"]={path.relative_to(p.ROOT).as_posix():p.sha256(path) for path in sorted((p.ROOT/"tools/validation").glob("*")) if path.is_file() and path.suffix in (".py",".ps1")}
    p.require(set(new["validation_files"])-set(info["validation_files"])=={"tools/validation/p2_transition.py"}, "unexpected added validation tools")
    new["imported_evidence"]=dict(format_version=1,suite=str(parent),verification=str(audit_path),
        files=p.artifacts([parent/"suite.json",audit_path,prior]),receipts=audit["receipts"],gates=audit["gates"],
        reason="Correct named cement metric consumers; preserve fully revalidated ETT and prerequisite evidence",changed_functions=changes)
    child.mkdir(parents=True,exist_ok=False); (child/"entries").mkdir()
    for old in sorted((parent/"entries").glob("*.json")):
        shutil.copyfile(old,child/"entries"/old.name)
    for name in GATES:
        shutil.copyfile(parent/(name+".json"),child/(name+".json"))
    p.write_json(child/"suite.json",new)
    p.suite_context(argparse.Namespace(suite=str(child)))
    p.write_gate(child,"transition",dict(passed=True,parent_suite=str(parent),parent_suite_sha256=p.sha256(parent/"suite.json"),
        imported_official_cuda=48,imported_rust_ett_cuda=48,verification=str(audit_path),artifacts=p.artifacts([audit_path,prior])))
    print(f"PASS versioned suite: {child}; run current CUDA quality, D4, cement, recompute, final quality and final",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="stage",required=True)
    before=sub.add_parser("prepare"); before.add_argument("--parent",required=True); before.add_argument("--output",required=True)
    after=sub.add_parser("create"); after.add_argument("--preparation",required=True); after.add_argument("--suite",required=True)
    args=parser.parse_args(); (prepare if args.stage=="prepare" else create)(args)


if __name__=="__main__":
    main()
