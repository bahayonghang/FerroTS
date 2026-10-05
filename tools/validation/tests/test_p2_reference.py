"""Check P2 reference identity, fixture completeness and runner input handling."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/reference"))
import gen_p2_fixtures as fixtures
import run_itransformer as runner


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((fixtures.DESTINATION / "manifest.json").read_text(encoding="utf-8"))

    def test_complete_mapping_gradients_and_optimizer(self):
        self.assertEqual(len(self.manifest["cases"]), 9)
        for case in self.manifest["cases"]:
            tensors = load_file(fixtures.DESTINATION / (case["name"] + ".safetensors"))
            parameters = case["parameters"]
            self.assertEqual(len(parameters), 38)
            for parameter in parameters:
                name = parameter["name"]
                for prefix in ("initial", "gradient", "probe_gradient", "updated", "exp_avg", "exp_avg_sq"):
                    tensor = tensors[f"{prefix}.{name}"]
                    self.assertEqual(list(tensor.shape), parameter["rust_shape"])
                    self.assertEqual(tensor.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(tensor).all())
                self.assertEqual(int(tensors[f"step.{name}"]), 1)
                if case["require_nonzero_probe"]:
                    self.assertGreater(int(torch.count_nonzero(tensors[f"probe_gradient.{name}"])), 0)
            self.assertEqual(list(tensors["output"].shape),
                             [case["batch"], case["config"]["pred_len"], case["config"]["enc_in"]])
            for name in case["traces"]:
                self.assertIn("trace." + name, tensors)

    def test_sampler_seeds_epochs_and_drop_last(self):
        self.assertEqual([c["seed"] for c in self.manifest["initializations"]], [2021, 2022, 2023])
        for case in self.manifest["initializations"]:
            self.assertEqual(len(case["orders"]), 10)
            for order in case["orders"]:
                self.assertEqual(len(order), 128)
                self.assertEqual(len(set(order)), 128)
                self.assertTrue(all(0 <= x < 137 for x in order))

    def test_empty_buffers_roundtrip(self):
        self.assertEqual(load_file(fixtures.DESTINATION / "buffers.safetensors"), {})

    def test_shared_dropout_masks_and_case_integrity(self):
        import hashlib
        masks = load_file(fixtures.DESTINATION / "dropout-masks.safetensors")
        self.assertEqual(len(masks), 18)
        for mask in masks.values():
            self.assertTrue(((mask == 0) | (mask == 1)).all())
            self.assertGreater(int(torch.count_nonzero(mask)), 0)
        self.assertEqual([c["forward_id"] for c in self.manifest["dropout_cases"]], [0, 1])
        for case in self.manifest["dropout_cases"]:
            self.assertEqual(case["dropout_site_version"], 1)
            self.assertEqual(case["mask_source"], "dropout-masks.safetensors")
            self.assertEqual(case["config"]["dropout"], 0.1)
            tensors = load_file(fixtures.DESTINATION / (case["name"] + ".safetensors"))
            for parameter in case["parameters"]:
                self.assertTrue(torch.isfinite(tensors["gradient." + parameter["name"]]).all())
        for name, receipt in self.manifest["files"].items():
            data = (fixtures.DESTINATION / name).read_bytes()
            self.assertEqual(len(data), receipt["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), receipt["sha256"])

    def test_conv_mapping_rejects_nonpointwise_kernel(self):
        with self.assertRaisesRegex(ValueError, "pointwise"):
            fixtures.mapped("encoder.attn_layers.0.conv1.weight", torch.zeros(2, 3, 2))
        self.assertEqual(fixtures.mapped("encoder.attn_layers.0.conv1.weight", torch.zeros(2, 3, 1)).shape, (2, 3))

    def test_wrong_or_dirty_upstream_rejected(self):
        with patch.object(fixtures, "git", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "pinned"):
                fixtures.source_identity()
        with patch.object(fixtures, "git", side_effect=[fixtures.UPSTREAMS["iTransformer"][1], " M run.py"]):
            with self.assertRaisesRegex(ValueError, "dirty"):
                fixtures.source_identity()


class RunnerTests(unittest.TestCase):
    def test_pinned_early_stopping_works_with_numpy2(self):
        import math
        import numpy as np
        sys.path.insert(0, str(runner.UPSTREAM))
        from utils.tools import EarlyStopping
        before = hasattr(np, "Inf")
        try:
            runner.install_numpy_compatibility()
            early = EarlyStopping(patience=3)
            self.assertEqual(early.val_loss_min, math.inf)
            self.assertEqual(early.counter, 0)
            self.assertIsNone(early.best_score)
        finally:
            if not before:
                del np.Inf

    def args(self):
        return ["--root_path", "tests/fixtures/p1", "--data_path", "ett.json",
                "--model_id", "test", "--model", "iTransformer"]

    def test_only_root_path_is_rewritten(self):
        args = self.args() + ["--des", "tests/fixtures/p1"]
        _, actual, path = runner.prepare(args)
        self.assertTrue(Path(actual[1]).is_absolute())
        self.assertEqual(actual[-1], "tests/fixtures/p1")
        self.assertEqual(path, ROOT / "tests/fixtures/p1/ett.json")

    def test_equals_root_syntax(self):
        _, actual, _ = runner.prepare(["--root_path=tests/fixtures/p1", *self.args()[2:]])
        self.assertTrue(actual[0].startswith("--root_path=" + str(ROOT)))

    def test_missing_data_duplicate_args_and_checkpoint_override_rejected(self):
        for extra in (["--root_path", "missing"], ["--checkpoints", "outside"], ["--data_path", "absent.csv"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                runner.prepare(self.args() + extra)
        args = self.args()
        args[3] = "absent.csv"
        with self.assertRaisesRegex(ValueError, "missing"):
            runner.prepare(args)

    def test_unrelated_model_and_unsafe_id_rejected(self):
        args = self.args()
        args[-1] = "DLinear"
        with self.assertRaisesRegex(ValueError, "iTransformer"):
            runner.prepare(args)
        args = self.args()
        args[5] = "../outside"
        with self.assertRaisesRegex(ValueError, "model_id"):
            runner.prepare(args)


if __name__ == "__main__":
    unittest.main()
