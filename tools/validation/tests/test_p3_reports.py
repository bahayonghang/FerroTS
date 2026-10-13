"""Coverage, statistical and provenance boundaries for P3 reports."""
import copy
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p3_reports as module


def ett_entries(engine, value=1.):
    entries = []
    for name in module.ETT:
        for horizon in module.HORIZONS:
            for seed in module.SEEDS:
                entry = {"identity": dict(engine=engine, model="informer", dataset=name,
                                           horizon=horizon, seed=seed, device="cuda:0")}
                metrics = dict(mse=value, mae=value)
                if engine == "official":
                    entry["metrics"] = metrics
                else:
                    entry["evaluation"] = {"metrics": {"standardized": {"overall": metrics}}}
                entries.append(entry)
    return entries


class D4Tests(unittest.TestCase):
    def test_zero_variance_boundary_is_not_relaxed(self):
        rows = module.d4_rows(ett_entries("official"), ett_entries("rust", 1.03))
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(r["threshold"] == .02 and not r["passed"] for r in rows))

    def test_missing_duplicate_and_wrong_engine_are_rejected(self):
        official, rust = ett_entries("official"), ett_entries("rust")
        for invalid in (official[:-1], official + [official[0]], rust):
            with self.subTest(invalid=len(invalid)):
                with self.assertRaises(ValueError):
                    module.d4_rows(invalid, rust)

    def test_failure_writes_evidence_without_claiming_full_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(module, "ROOT", root), patch.object(module, "_performance_rows", return_value=[]):
                result = module.write_reports(root / "suite", {"source": {"source_sha256": "fixture"}},
                                              ett_entries("official") + ett_entries("rust", 2.))
            evidence = json.loads((root / "suite/d4.json").read_text())
            self.assertTrue(evidence["complete"])
            self.assertFalse(evidence["passed"])
            self.assertEqual(len(evidence["rows"]), 32)
            self.assertFalse(result["passed"])
            self.assertIn("D4: 0/32", (root / "docs/results/p3-informer-ett.md").read_text())
            self.assertFalse((root / "docs/results/p3-informer-cement.md").exists())


class CementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        active = patch.object(module, "ROOT", self.root)
        active.start()
        self.addCleanup(active.stop)
        self.targets = [f"target_{i:02}" for i in reversed(range(18))]
        self.entries = []
        for h in module.CEMENT_COUNTS:
            config = self.root / f"configs/engineering/informer-cement-h{h}.toml"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text("[data]\ntarget_columns = " + json.dumps(self.targets) + "\n")
            for seed in module.SEEDS:
                variables = {}
                for i, name in enumerate(sorted(self.targets)):
                    value = i + (seed - 2020)
                    variables[name] = dict(count=module.CEMENT_COUNTS[h][2] * h,
                                           mae=value, rmse=value + .5, mse=(value + .5) ** 2)
                self.entries.append({"identity": dict(engine="rust", model="informer", dataset="cement",
                                                      horizon=h, seed=seed, device="cuda:1"),
                                     "evaluation": {"metrics": {"original": {"variables": variables}}}})

    def test_named_18_variable_population_statistics(self):
        for entry in self.entries:
            variables = entry["evaluation"]["metrics"]["original"]["variables"]
            entry["evaluation"]["metrics"]["original"]["variables"] = dict(reversed(list(variables.items())))
        rows = module.cement_rows(self.entries)
        self.assertEqual(len(rows), 108)
        self.assertEqual(rows[0]["variable"], "target_17")
        row = next(r for r in rows if (r["horizon"], r["variable"], r["metric"]) == (60, "target_00", "mae"))
        self.assertEqual(row["seeds"], [1, 2, 3])
        self.assertEqual(row["mean"], 2.)
        self.assertAlmostEqual(row["std"], statistics.pstdev([1, 2, 3]))
        self.assertNotAlmostEqual(row["std"], statistics.stdev([1, 2, 3]))

    def test_missing_duplicate_variable_and_non_dictionary_reject(self):
        for mutation in ("missing_run", "duplicate_run", "missing_variable", "extra_variable", "list_variables", "count", "nan"):
            entries = copy.deepcopy(self.entries)
            variables = entries[0]["evaluation"]["metrics"]["original"]["variables"]
            if mutation == "missing_run":
                entries.pop()
            elif mutation == "duplicate_run":
                entries.append(entries[0])
            elif mutation == "missing_variable":
                variables.pop("target_00")
            elif mutation == "extra_variable":
                variables["unexpected"] = variables.pop("target_00")
            elif mutation == "list_variables":
                entries[0]["evaluation"]["metrics"]["original"]["variables"] = list(variables.values())
            elif mutation == "count":
                variables["target_00"]["count"] -= 1
            else:
                variables["target_00"]["mae"] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.cement_rows(entries)

    def test_published_history_is_numbers_only_with_source_hashes(self):
        rows = module.cement_rows(self.entries)
        directory = self.root / "docs/results"
        directory.mkdir(parents=True)
        p1 = [f"| {model} | {h} | cuda | {name} | 1 | 1.5 | 2.5 | 0 | 0 |"
              for model in ("last-value", "seasonal-naive", "dlinear")
              for h in module.CEMENT_COUNTS for name in self.targets]
        p2 = [f"| {r['horizon']} | {r['variable']} | {r['metric']} | 1 / 2 / 3 | 2 | 0.816496581 | 0 | 0 | 0 |" for r in rows]
        (directory / "p1-cement-baselines.md").write_text("\n".join(p1))
        (directory / "p2-itransformer-cement.md").write_text("\n".join(p2))
        enriched, sources = module._historical_baselines(rows)
        self.assertEqual(len(sources), 2)
        self.assertEqual(enriched[0]["historical"]["dlinear"], dict(mean=1.5, std=None, n=1))
        self.assertEqual(enriched[0]["historical"]["itransformer"]["n"], 3)
        self.assertEqual(enriched[0]["historical"]["itransformer"]["mean"], 2.)


class PerformanceTests(unittest.TestCase):
    def test_throughput_weights_time_and_keeps_allocator_scope_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, receipt = root / "run", root / "receipt"
            run.mkdir()
            (receipt / "memory").mkdir(parents=True)
            epochs = [dict(epoch=1, train_seconds=2., validation_seconds=.2, checkpoint_seconds=.1,
                           steady_seconds=1., steady_windows=10),
                      dict(epoch=2, train_seconds=4., validation_seconds=.2, checkpoint_seconds=.1,
                           steady_seconds=3., steady_windows=60)]
            (run / "performance.jsonl").write_text("\n".join(json.dumps(e) for e in epochs))
            memory = dict(interval_ms=50, device=1, samples=10, errors=[], baseline_used_mib=10.,
                          peak_used_mib=30., estimated_dropped_samples=1)
            module.write_json(receipt / "memory/device-memory.json", memory)
            entry = dict(identity=dict(engine="rust", dataset="cement", horizon=60, seed=2021, device="cuda:1"),
                         run=str(run), receipts=[dict(directory=str(receipt))])
            result = module._performance_rows([entry])[0]
            self.assertEqual(result["steady_samples_per_second"], 17.5)
            self.assertIsNone(result["allocator_max_allocated_bytes"])
            self.assertEqual(result["memory"]["device"], 1)
            entry["identity"]["device"] = "cuda:0"
            with self.assertRaisesRegex(ValueError, "device/interval"):
                module._performance_rows([entry])
            entry["identity"].update(device="cuda:1", engine="official")
            with self.assertRaisesRegex(ValueError, "allocator"):
                module._performance_rows([entry])
            for i, epoch in enumerate(epochs):
                epoch.update(allocator_max_allocated_bytes=100 + i, allocator_max_reserved_bytes=200 + i)
            (run / "performance.jsonl").write_text("\n".join(json.dumps(e) for e in epochs))
            result = module._performance_rows([entry])[0]
            self.assertEqual(result["allocator_max_allocated_bytes"], 101)
            self.assertEqual(result["allocator_max_reserved_bytes"], 201)


if __name__ == "__main__":
    unittest.main()
