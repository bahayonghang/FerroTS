"""Check bounded reference inputs and the proposed matrix without training."""
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/reference"))
sys.path.insert(0, str(ROOT / "tools/validation"))
import run_informer
import p3_smoke


class InformerReferenceTests(unittest.TestCase):
    def test_observer_counts_retained_and_dropped_tail_windows(self):
        class Loader:
            batch_size = 32
            dataset = range(8449)
            def __len__(self):
                return 264 if self.drop_last else 265
        loader = Loader()
        model = SimpleNamespace(_p2_observation={})
        observer = SimpleNamespace(begin=lambda *args: None)
        for drop_last, windows in [(False, 8449), (True, 8448)]:
            loader.drop_last = drop_last
            run_informer.begin_observation(observer, model, 0, loader)
            self.assertEqual(model._p2_observation["training_windows"], windows)
            self.assertEqual(model._p2_observation["steady_windows"], windows - 32)

    def arguments(self, root, filename="sample.csv"):
        return ["--root_path", str(root), "--data_path", filename,
                "--model", "Informer", "--model_id", "test_case"]

    def test_resolves_local_data_for_an_isolated_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sample.csv").write_text("date,x\n2024-01-01,1\n")
            normalized, data = run_informer.prepare(self.arguments(root))
            self.assertEqual(data, root / "sample.csv")
            self.assertEqual(Path(normalized[normalized.index("--root_path") + 1]), root)

    def test_rejects_missing_data_duplicates_other_models_and_owned_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sample.csv").write_text("date,x\n2024-01-01,1\n")
            for arguments in [self.arguments(root, "missing.csv"),
                    self.arguments(root) + ["--model", "Informer"],
                    [v.replace("Informer", "iTransformer") for v in self.arguments(root)],
                    self.arguments(root) + ["--checkpoints", "elsewhere"]]:
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    run_informer.prepare(arguments)

    def test_external_output_is_rejected_before_a_process_or_directory_is_created(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sample.csv").write_text("date,x\n2024-01-01,1\n")
            destination = root / "must_not_exist"
            with patch.object(run_informer, "git", side_effect=lambda *a, **kw: run_informer.UPSTREAMS["Time-Series-Library"][1] if a[0] == "rev-parse" else ""), patch.object(run_informer.subprocess, "Popen", side_effect=AssertionError("must not start")):
                with self.assertRaisesRegex(ValueError, "ignored runs"):
                    run_informer.run(2021, self.arguments(root), destination)
            self.assertFalse(destination.exists())

    def test_model_scanner_preserves_import_names_from_isolated_cwd(self):
        sys.path.insert(0, str(run_informer.UPSTREAM))
        from exp.exp_basic import Exp_Basic
        original = Exp_Basic._scan_models_directory
        try:
            run_informer.install_model_source_path()
            scan = Exp_Basic.__new__(Exp_Basic)._scan_models_directory()
            self.assertEqual(scan["Informer"], "models.Informer")
            import importlib
            module = importlib.import_module(scan["Informer"])
            self.assertEqual(Path(module.__file__).resolve(), run_informer.UPSTREAM / "models/Informer.py")
        finally:
            Exp_Basic._scan_models_directory = original

    def test_formal_matrix_has_unique_complete_public_and_private_cells(self):
        rows = p3_smoke.planned_matrix(ROOT / "target/linux-cuda/debug/ferrots", ROOT / "runs/p3-acceptance/test")
        keys = {(r["engine"], r["dataset"], r["horizon"], r["seed"]) for r in rows}
        self.assertEqual(len(rows), len(keys))
        self.assertEqual(Counter((r["engine"], r["dataset"] == "cement") for r in rows),
                         Counter({("official", False): 48, ("rust", False): 48, ("rust", True): 9}))
        self.assertEqual({r["seed"] for r in rows}, {2021, 2022, 2023})
        for row in rows:
            if row["engine"] == "official":
                command = row["command"]
                self.assertEqual(command[command.index("--train_epochs") + 1], "10")
                self.assertEqual(command[command.index("--pred_len") + 1], str(row["horizon"]))
            else:
                config = p3_smoke.tomllib.loads((ROOT / row["config"]).read_text())
                self.assertEqual(config["model"]["name"], "informer")
                self.assertEqual(config["window"]["horizon"], row["horizon"])
                self.assertFalse(config["train"].get("drop_last", False))


if __name__ == "__main__":
    unittest.main()
