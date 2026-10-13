"""Exercise rejection boundaries and exclusive per-GPU matrix scheduling."""
import copy
import json
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/validation"))
import p3_acceptance as p3


class InformerAcceptanceTests(unittest.TestCase):
    def test_resolved_profiles_allow_only_seed_and_device_overrides(self):
        for dataset, horizon in [("ETTh1", 96), ("ETTm1", 720), ("cement", 60)]:
            actual = tomllib.loads((ROOT / p3.config_path(dataset, horizon)).read_text())
            actual["train"].update(seed=2023, device="cuda:1")
            actual["train"].pop("drop_last")  # serde omits its false default.
            p3.validate_config(actual, dataset, horizon, 2023, "cuda:1")
            for section, key, value in [("train", "drop_last", True), ("train", "batch_size", 64),
                    ("model", "factor", 4), ("window", "gap", 1), ("data", "future_known_columns", ["OT"])]:
                changed = copy.deepcopy(actual)
                changed[section][key] = value
                with self.subTest(dataset=dataset, key=key), self.assertRaises(ValueError):
                    p3.validate_config(changed, dataset, horizon, 2023, "cuda:1")

    def test_gate_rejects_foreign_suite_and_verifies_command_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            suite = Path(folder)
            (suite / "suite.json").write_text("{}\n")
            p3.gate(suite, "quality", dict(passed=True, receipts=[{"token": 1}]))
            with patch.object(p3, "verify_receipt") as verify:
                self.assertTrue(p3.checked_gate(suite, {}, "quality")["passed"])
                verify.assert_called_once_with({"token": 1}, {}, suite)
            (suite / "suite.json").write_text('{"different":true}\n')
            with self.assertRaisesRegex(ValueError, "stale gate"):
                p3.checked_gate(suite, {}, "quality")

    def test_scheduler_executes_cells_once_with_one_job_per_gpu(self):
        active, seen, lock = set(), [], threading.Lock()
        def execute(suite, info, cell, device):
            with lock:
                self.assertNotIn(device, active)
                active.add(device)
            time.sleep(0.001)
            with lock:
                seen.append(cell)
                active.remove(device)
        cells = p3.matrix() + p3.matrix(cement=True)
        self.assertEqual(len(set(cells)), 105)
        with patch.object(p3, "run_cell", side_effect=execute):
            p3.run_matrix(None, {"devices": ["cuda:0", "cuda:1"]}, cells)
        self.assertCountEqual(seen, cells)

    def test_failed_worker_prevents_remaining_matrix_from_launching(self):
        cells = p3.matrix()
        invoked = []
        def fail(suite, info, cell, device):
            invoked.append(cell)
            raise ValueError("retained command failure")
        with patch.object(p3, "run_cell", side_effect=fail), self.assertRaisesRegex(ValueError, "retained command"):
            p3.run_matrix(None, {"devices": ["cuda:0", "cuda:1"]}, cells)
        self.assertLessEqual(len(invoked), 2)


if __name__ == "__main__":
    unittest.main()
