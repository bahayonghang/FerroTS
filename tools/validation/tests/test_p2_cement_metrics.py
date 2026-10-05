"""Reject incorrect original-unit cement metrics and preserve named aggregation."""
import copy
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p2_acceptance as module


class CementContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = self.root / 'suite'
        (self.suite / 'entries').mkdir(parents=True)
        (self.root / 'docs/results').mkdir(parents=True)
        self.targets = [f'target_{i:02}' for i in reversed(range(18))]
        self.entries = {}
        self.info = {'source': {'source_sha256': 'synthetic'}, 'p1_baseline_files': {}}
        for horizon in module.CEMENT_COUNTS:
            config = self.root / module.config_path('cement', horizon)
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text('[data]\ntarget_columns = ' + json.dumps(self.targets) + '\n', encoding='utf-8')
            for seed in module.SEEDS:
                variables = {}
                for index, name in enumerate(sorted(self.targets)):
                    mae = 1 + index + (seed - 2021) / 10 + horizon / 1000
                    variables[name] = {'count': module.CEMENT_COUNTS[horizon][2] * horizon, 'mae': mae, 'rmse': mae + .5, 'mse': (mae + .5) ** 2}
                self.entries[(horizon, seed)] = {'identity': {'dataset': 'cement', 'horizon': horizon, 'seed': seed}, 'evaluation': {'metrics': {'original': {'variables': variables}}}}
                self.write_entry(horizon, seed)
        self.patches = [patch.object(module, 'ROOT', self.root), patch.object(module, 'P1_SUITE', self.root / 'p1'), patch.object(module, 'verify_rust'), patch.object(module, 'verify_artifacts')]
        for active in self.patches:
            active.start()
            self.addCleanup(active.stop)

    def write_entry(self, horizon, seed):
        path = self.suite / 'entries' / f'rust-cement-h{horizon}-s{seed}.json'
        path.write_text(json.dumps(self.entries[(horizon, seed)]), encoding='utf-8')
        return path

    def test_complete_dictionary_preserves_all_nine_entries(self):
        result = module.cement_entries(self.suite, self.info)
        self.assertEqual(set(result), set(self.entries))
        self.assertEqual(module.verify_rust.call_count, 9)

    def test_dictionary_order_does_not_change_variable_identity(self):
        for key, entry in self.entries.items():
            metrics = entry['evaluation']['metrics']['original']
            metrics['variables'] = dict(reversed(list(metrics['variables'].items())))
            self.write_entry(*key)
        self.assertEqual(len(module.cement_entries(self.suite, self.info)), 9)

    def test_missing_and_extra_variable_names_are_rejected(self):
        key = (15, 2021)
        original = copy.deepcopy(self.entries[key])
        for extra in (False, True):
            with self.subTest(extra=extra):
                self.entries[key] = copy.deepcopy(original)
                variables = self.entries[key]['evaluation']['metrics']['original']['variables']
                variables.pop('target_00')
                if extra:
                    variables['unexpected'] = copy.deepcopy(variables['target_01'])
                self.write_entry(*key)
                with self.assertRaisesRegex(ValueError, 'all 18 variables'):
                    module.cement_entries(self.suite, self.info)

    def test_incomplete_variable_counts_are_rejected(self):
        self.entries[(15, 2021)]['evaluation']['metrics']['original']['variables']['target_00']['count'] -= 1
        self.write_entry(15, 2021)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            module.cement_entries(self.suite, self.info)

    def test_invalid_original_unit_metrics_are_rejected(self):
        key = (15, 2021)
        original = copy.deepcopy(self.entries[key])
        for metric in ('mae', 'mse', 'rmse'):
            for value in (float('nan'), float('inf'), -1., None):
                with self.subTest(metric=metric, value=value):
                    self.entries[key] = copy.deepcopy(original)
                    self.entries[key]['evaluation']['metrics']['original']['variables']['target_00'][metric] = value
                    self.write_entry(*key)
                    with self.assertRaisesRegex(ValueError, 'invalid metric'):
                        module.cement_entries(self.suite, self.info)

    def test_incomplete_and_duplicate_matrix_is_rejected(self):
        entry_path = self.write_entry(15, 2021)
        retained = entry_path.read_bytes()
        entry_path.rename(entry_path.with_suffix('.retained'))
        with self.assertRaisesRegex(ValueError, 'exactly 9'):
            module.cement_entries(self.suite, self.info)
        entry_path.write_bytes(retained)
        (entry_path.parent / 'rust-cement-duplicate.json').write_bytes(retained)
        with self.assertRaisesRegex(ValueError, 'duplicate or unexpected'):
            module.cement_entries(self.suite, self.info)

    def test_gate_aggregates_108_rows_by_name_in_config_order(self):
        with patch.object(module, 'suite_context', return_value=(self.suite, self.info)), patch.object(module, 'training_prerequisites'), patch.object(module, 'require_gate'), patch.object(module, 'rust_run') as train, patch.object(module, 'cement_entries', return_value=self.entries), patch.object(module, 'write_gate') as save, patch.object(module, 'cement_report') as report:
            module.cement_gate(object())
            self.assertEqual(train.call_count, 9)
            result = save.call_args.args[2]
            self.assertEqual(result['runs'], 9)
            self.assertEqual(result['ddof'], 0)
            self.assertEqual(len(result['rows']), 108)
            self.assertEqual(result['rows'][0]['variable'], self.targets[0])
            for row in result['rows']:
                values = [self.entries[(row['horizon'], seed)]['evaluation']['metrics']['original']['variables'][row['variable']][row['metric']] for seed in module.SEEDS]
                self.assertEqual(row['seeds'], values)
                self.assertEqual(row['mean'], statistics.fmean(values))
                self.assertEqual(row['std'], statistics.pstdev(values))
            self.assertEqual(report.call_args.args[2], result['rows'])

    def test_p1_report_uses_named_metrics_and_single_seed_label(self):
        baseline = self.root / 'p1/entries'
        baseline.mkdir(parents=True)
        for model, value in [('last-value', 11.), ('seasonal-naive', 22.), ('dlinear', 33.)]:
            for horizon in module.CEMENT_COUNTS:
                entry = {'artifacts': {}, 'evaluation': {'data_sha256': module.PRIVATE_SHA, 'metrics': {'original': {'variables': {name: {'mae': value, 'rmse': value + .5} for name in self.targets}}}}}
                (baseline / f'rust-cement-{model}-cuda-h{horizon}-s2021.json').write_text(json.dumps(entry), encoding='utf-8')
        rows = [{'horizon': h, 'variable': name, 'metric': metric, 'seeds': [1., 2., 3.], 'mean': 2., 'std': statistics.pstdev([1., 2., 3.])} for h in module.CEMENT_COUNTS for name in self.targets for metric in ('mae', 'rmse')]
        module.cement_report(self.suite, self.info, rows)
        text = (self.root / 'docs/results/p2-itransformer-cement.md').read_text(encoding='utf-8')
        self.assertIn('n=1; standard deviation is not applicable', text)
        self.assertEqual(sum(line.startswith(('| 15 |', '| 30 |', '| 60 |')) for line in text.splitlines()), 108)
        self.assertIn('| 11 | 22 | 33 |', text)
        self.assertIn('| 11.5 | 22.5 | 33.5 |', text)


if __name__ == "__main__":
    unittest.main()
