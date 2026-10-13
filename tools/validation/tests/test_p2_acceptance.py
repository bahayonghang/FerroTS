"""Reject incomplete P2 evidence before formal training or acceptance."""
import copy
import json
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p2_acceptance as acceptance


def epoch(number, loss):
    return dict(epoch=number, validation_loss=loss, learning_rate=1e-4*0.5**max(0,number-2),
        train_seconds=2., validation_seconds=1., steady_seconds=1., steady_windows=64,
        training_windows=96, valid_labels=96*96*7)


class AcceptanceTests(unittest.TestCase):
    def test_all_fixed_profiles_and_horizon_specific_dimensions(self):
        for name, horizons in [(n, acceptance.HORIZONS) for n in acceptance.ETT] + [("cement", acceptance.CEMENT_COUNTS)]:
            for horizon in horizons:
                actual = tomllib.loads((acceptance.ROOT/acceptance.config_path(name,horizon)).read_text(encoding="utf-8"))
                if "missing" in actual["data"]:
                    actual["data"]["missing"].setdefault("tokens", ["", "NaN", "nan"])
                acceptance.validate_config(actual,name,horizon,2021)
                for section,key,value in [("train","seed",2024),("train","max_epochs",3),("train","batch_size",16),
                    ("train","device","cpu"),("model","d_model",64),("window","horizon",1)]:
                    bad = copy.deepcopy(actual)
                    bad[section][key] = value
                    with self.subTest(name=name,horizon=horizon,key=key), self.assertRaises(ValueError):
                        acceptance.validate_config(bad,name,horizon,2021)
        self.assertEqual(acceptance.model_profile("ETTh1",192)["d_model"],256)
        self.assertEqual(acceptance.model_profile("ETTh1",336)["d_ff"],512)

    def test_exact_matrix_coverage_and_seeds(self):
        entries = [dict(identity=dict(dataset=n,horizon=h,seed=s),metrics=dict(mse=.2,mae=.3))
            for n in acceptance.ETT for h in acceptance.HORIZONS for s in acceptance.SEEDS]
        self.assertEqual(len(acceptance.matrix_by_key(entries,"official")),48)
        for bad in (entries[:-1], entries+[entries[0]], entries[1:]+[entries[1]]):
            with self.assertRaises(ValueError):
                acceptance.matrix_by_key(bad,"official")
        entries[0]["identity"]["seed"] = 2024
        with self.assertRaises(ValueError):
            acceptance.matrix_by_key(entries,"official")

    def test_epochs_require_normal_end_and_full_coverage(self):
        complete = [epoch(i,1./i) for i in range(1,11)]
        self.assertEqual(acceptance.validate_epochs(complete,96,96,7)["epoch"],10)
        patience = [epoch(i,1.) for i in range(1,5)]
        self.assertEqual(acceptance.validate_epochs(patience,96,96,7)["epoch"],1)
        with self.assertRaisesRegex(ValueError,"short"):
            acceptance.validate_epochs(patience,96,96,7,official=True)
        for bad in (complete[:3], patience+[epoch(5,1.)]):
            with self.assertRaises(ValueError):
                acceptance.validate_epochs(bad,96,96,7)
        for key,value in [("training_windows",95),("valid_labels",1),("steady_windows",96),
            ("validation_loss",float("nan")),("learning_rate",1e-3),("steady_seconds",0.)]:
            bad = copy.deepcopy(complete)
            bad[3][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                acceptance.validate_epochs(bad,96,96,7)

    def test_source_config_and_binary_changes_rejected(self):
        info = dict(upstream_sha=acceptance.UPSTREAM_SHA,seeds=acceptance.SEEDS,horizons=acceptance.HORIZONS,
            source=dict(source_sha256="old"),configs={"a":"old"},reference_files={},validation_files={},data={},
            binaries={"cuda":dict(path="binary",sha256="old")})
        with patch.object(acceptance,"read_json",return_value=info), patch.object(acceptance,"source_snapshot",return_value={"source_sha256":"new"}):
            with self.assertRaisesRegex(ValueError,"source changed"):
                acceptance.suite_context(Namespace(suite="runs/p2-acceptance/test"))
        with patch.object(acceptance,"read_json",return_value=info), patch.object(acceptance,"source_snapshot",return_value={"source_sha256":"old"}), patch.object(acceptance,"sha256",return_value="new"):
            with self.assertRaisesRegex(ValueError,"configs fingerprint"):
                acceptance.suite_context(Namespace(suite="runs/p2-acceptance/test"))
            info["configs"] = {}
            with self.assertRaisesRegex(ValueError,"binary changed"):
                acceptance.suite_context(Namespace(suite="runs/p2-acceptance/test"))

    def test_failed_command_and_launch_error_retain_receipts_without_retry(self):
        for outcome in (subprocess.CompletedProcess(["synthetic"],1), OSError("launch failure")):
            with tempfile.TemporaryDirectory() as temp:
                suite = Path(temp)
                acceptance.write_json(suite/"suite.json", {"synthetic":True})
                with patch.object(acceptance.subprocess,"run") as execute:
                    if isinstance(outcome,Exception):
                        execute.side_effect = outcome
                    else:
                        execute.return_value = outcome
                    with self.assertRaises((ValueError,OSError)):
                        acceptance.execute(suite,{"source":{"source_sha256":"synthetic"}},"failure",["synthetic","--batch-size","32"])
                    self.assertEqual(execute.call_count,1)
                paths = list((suite/"commands").rglob("receipt.json"))
                self.assertEqual(len(paths),1)
                receipt = acceptance.read_json(paths[0])
                self.assertNotEqual(receipt["exit_code"],0)
                self.assertEqual(receipt["command"][-1],"32")

    def test_receipt_artifacts_reject_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/"synthetic.log"
            file.write_text("passed",encoding="utf-8")
            files = acceptance.artifacts([file])
            acceptance.verify_artifacts(files)
            file.write_text("failed",encoding="utf-8")
            with self.assertRaises(ValueError):
                acceptance.verify_artifacts(files)

    def test_cement_can_use_the_second_gpu_without_changing_ett(self):
        self.assertEqual(set(acceptance.CEMENT_TRAIN_BATCH.values()), {32})
        cement = tomllib.loads((acceptance.ROOT/acceptance.config_path("cement", 60)).read_text(encoding="utf-8"))
        cement["data"]["missing"].setdefault("tokens", ["", "NaN", "nan"])
        cement["train"]["device"] = "cuda:1"
        acceptance.validate_config(cement, "cement", 60, 2021)
        cement["train"]["batch_size"] = 1024
        with self.assertRaises(ValueError):
            acceptance.validate_config(cement, "cement", 60, 2021)
        ett = tomllib.loads((acceptance.ROOT/acceptance.config_path("ETTh1", 96)).read_text(encoding="utf-8"))
        ett["train"]["device"] = "cuda:1"
        with self.assertRaises(ValueError):
            acceptance.validate_config(ett, "ETTh1", 96, 2021)

    def test_steady_coverage_uses_the_resolved_batch(self):
        rows = [epoch(i, 1./i) for i in range(1, 11)]
        for row in rows:
            row["steady_windows"] = 80
        self.assertEqual(acceptance.validate_epochs(rows, 96, 96, 7, batch_size=16)["epoch"], 10)
        with self.assertRaises(ValueError):
            acceptance.validate_epochs(rows, 96, 96, 7)

    def test_published_p1_table_covers_cuda_baselines(self):
        baseline = acceptance.published_p1_cuda_baseline()
        self.assertEqual(len(baseline), 324)
        self.assertEqual(baseline[("last-value", 15, "窑喂料反馈", "mae")], 1.054294337)

    def test_quality_requires_full_cuda_commands(self):
        entries = [dict(key=k) for k in acceptance.QUALITY_COMMANDS]
        for bad in ([],entries[:-1],entries[1:]+[entries[1]]):
            with self.assertRaisesRegex(ValueError,"coverage"):
                acceptance.validate_quality(dict(passed=True,records=bad))
        self.assertFalse(any(key.startswith("cpu") for key in acceptance.QUALITY_COMMANDS))

    def test_official_instrumentation_compiles_and_rejects_anchor_drift(self):
        sys.path.insert(0,str(acceptance.ROOT/"tools/reference"))
        import p2_observer
        import ast
        source = (acceptance.UPSTREAM/"experiments/exp_long_term_forecasting.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        cls = next(node for node in module.body if isinstance(node,ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name == "train")
        lines = source.splitlines(keepends=True)
        original = "".join(lines[method.lineno-1:method.end_lineno])
        instrumented = p2_observer.instrument(original)
        compile(textwrap.dedent(instrumented),"<test-observer>","exec")
        self.assertEqual(instrumented.count("_p2_first_batch(self)"),1)
        with self.assertRaisesRegex(ValueError,"anchor"):
            p2_observer.instrument(original.replace("model_optim.step()","model_optim.changed()"))


if __name__ == "__main__":
    unittest.main()
