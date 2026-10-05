"""Strict provenance checks for the scoped cement-tool transition."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"tools/validation"))
import p2_acceptance as p


class ImportedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.parent=self.root/"parent"; self.child=self.root/"child"
        self.parent.mkdir(); self.child.mkdir()
        self.old={key:{} for key in ("data","configs","reference_files","binaries","environment","p1_baseline_files")}
        self.old.update(upstream_sha="synthetic",source={"source_sha256":"source"},seeds=[2021,2022,2023],horizons=[96,192,336,720],ddof=0,policy="cuda-first-2026-09-29",formal_cuda_runs=105)
        p.write_json(self.parent/"suite.json",self.old)
        self.command=self.parent/"command"; self.command.mkdir(); log=self.command/"stdout.log"; log.write_text("synthetic success",encoding="utf-8")
        self.receipt=dict(directory=str(self.command),exit_code=0,suite_sha256=p.sha256(self.parent/"suite.json"),source_sha256="source",files=p.artifacts([log]))
        p.write_json(self.command/"receipt.json",self.receipt)
        self.gates={}
        for name in ("preflight","data-gate","numerics","resume"):
            gate=dict(passed=True,suite_sha256=p.sha256(self.parent/"suite.json"),receipts=[self.receipt])
            p.write_json(self.parent/(name+".json"),gate); p.write_json(self.child/(name+".json"),gate)
            self.gates[name]=p.artifacts([self.parent/(name+".json")])[str(self.parent/(name+".json"))]
        self.audit_path=self.root/"verification.json"
        self.audit=dict(passed=True,official_cuda=48,rust_ett_cuda=48,d4_passed=32,suite_sha256=p.sha256(self.parent/"suite.json"),source_sha256="source",receipts=p.artifacts([self.command/"receipt.json"]),gates=self.gates)
        p.write_json(self.audit_path,self.audit)
        self.info=copy.deepcopy(self.old)
        self.info["imported_evidence"]=dict(format_version=1,suite=str(self.parent),verification=str(self.audit_path),files=p.artifacts([self.parent/"suite.json",self.audit_path]),receipts=self.audit["receipts"],gates=self.gates)
        p.write_json(self.child/"suite.json",self.info)

    def test_original_same_suite_receipt_still_passes(self):
        p.verify_receipt(self.receipt,self.old,self.parent)

    def test_exact_imported_receipt_passes(self):
        p.verify_receipt(self.receipt,self.info,self.child)

    def test_current_child_receipt_passes(self):
        directory=self.child/"command"; directory.mkdir()
        receipt=dict(self.receipt,directory=str(directory),suite_sha256=p.sha256(self.child/"suite.json"))
        p.write_json(directory/"receipt.json",receipt)
        p.verify_receipt(receipt,self.info,self.child)

    def test_parent_receipt_without_explicit_import_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"explicit import"):
            p.verify_receipt(self.receipt,self.old,self.child)

    def test_changed_parent_manifest_is_rejected(self):
        (self.parent/"suite.json").write_text("{}",encoding="utf-8")
        with self.assertRaises(ValueError):
            p.verify_receipt(self.receipt,self.info,self.child)

    def test_changed_receipt_and_command_log_are_rejected(self):
        receipt=copy.deepcopy(self.receipt); receipt["unexpected"]=True
        p.write_json(self.command/"receipt.json",receipt)
        with self.assertRaises(ValueError):
            p.verify_receipt(receipt,self.info,self.child)
        p.write_json(self.command/"receipt.json",self.receipt)
        (self.command/"stdout.log").write_text("changed",encoding="utf-8")
        with self.assertRaises(ValueError):
            p.verify_receipt(self.receipt,self.info,self.child)

    def test_unlisted_parent_receipt_is_rejected(self):
        directory=self.parent/"unlisted"; directory.mkdir()
        receipt=dict(self.receipt,directory=str(directory)); p.write_json(directory/"receipt.json",receipt)
        with self.assertRaisesRegex(ValueError,"inventory"):
            p.verify_receipt(receipt,self.info,self.child)

    def test_imported_gate_requires_identical_copy(self):
        p.require_gate(self.child,self.info,"data-gate")
        gate=p.read_json(self.child/"data-gate.json"); gate["unexpected"]=True
        p.write_json(self.child/"data-gate.json",gate)
        with self.assertRaisesRegex(ValueError,"imported gate changed"):
            p.require_gate(self.child,self.info,"data-gate")

    def test_import_does_not_accept_arbitrary_gate(self):
        p.write_json(self.child/"d4.json",dict(passed=True,suite_sha256=p.sha256(self.parent/"suite.json")))
        with self.assertRaisesRegex(ValueError,"inventory"):
            p.require_gate(self.child,self.info,"d4")

    def test_training_identity_changes_are_rejected(self):
        for key,value in [("source",{"source_sha256":"changed"}),("binaries",{"cuda":"changed"}),("configs",{"batch_size":16}),("seeds",[2021])]:
            with self.subTest(key=key):
                info=copy.deepcopy(self.info); info[key]=value
                with self.assertRaisesRegex(ValueError,"training identity"):
                    p.imported_context(info)

    def test_incomplete_or_failed_parent_audit_is_rejected(self):
        for key,value in [("passed",False),("official_cuda",47),("rust_ett_cuda",47),("d4_passed",31)]:
            with self.subTest(key=key):
                audit=copy.deepcopy(self.audit); audit[key]=value; p.write_json(self.audit_path,audit)
                info=copy.deepcopy(self.info); info["imported_evidence"]["files"]=p.artifacts([self.parent/"suite.json",self.audit_path])
                with self.assertRaisesRegex(ValueError,"verification differs"):
                    p.imported_context(info)

    def test_import_inventory_must_match_bound_audit(self):
        info=copy.deepcopy(self.info); info["imported_evidence"]["receipts"]={}
        with self.assertRaisesRegex(ValueError,"audit inventory"):
            p.imported_context(info)


class TransitionCompatibilityTests(unittest.TestCase):
    def setUp(self):
        import p2_transition
        self.check = p2_transition.check_validator_transition
        self.before = "UPSTREAM = 'fixed'\n" + "\n".join(f"def {name}():\n    return 1\n" for name in sorted(p2_transition.CHANGED | {'rust_run'}))
        self.after = "UPSTREAM = 'fixed'\n" + "\n".join(f"def {name}():\n    return {1 if name == 'rust_run' else 2}\n" for name in sorted(p2_transition.CHANGED | {'rust_run'})) + "\ndef imported_context():\n    return None\n"

    def test_only_scoped_functions_can_change(self):
        self.assertEqual(len(self.check(self.before, self.after)), 6)

    def test_training_function_change_is_rejected(self):
        changed = self.after.replace("def rust_run():\n    return 1", "def rust_run():\n    return 3")
        with self.assertRaisesRegex(ValueError, "boundary"):
            self.check(self.before, changed)

    def test_constants_and_imports_cannot_change(self):
        with self.assertRaisesRegex(ValueError, "constants or imports"):
            self.check(self.before, self.after.replace("UPSTREAM = 'fixed'", "UPSTREAM = 'different'"))

    def test_unrelated_helper_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "function set"):
            self.check(self.before, self.after + "\ndef unrelated():\n    pass\n")


if __name__=="__main__":
    unittest.main()
