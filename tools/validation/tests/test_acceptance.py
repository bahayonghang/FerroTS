import copy
import math
import sys
import tempfile
import tomllib
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p1_acceptance as acceptance


def entries():
    return [dict(identity=dict(dataset=n, horizon=h, seed=s), metrics=dict(mse=0.1, mae=0.2))
            for n in acceptance.ETT for h in acceptance.HORIZONS for s in acceptance.SEEDS]


class AcceptanceTests(unittest.TestCase):
    def test_matrix_requires_exact_unique_seed_coverage(self):
        good = entries()
        self.assertEqual(len(acceptance.matrix_by_key(good, 'official')), 48)
        for bad in (good[:-1], good+[good[0]], good[1:]+[good[1]]):
            with self.assertRaises(ValueError):
                acceptance.matrix_by_key(bad, 'official')
        bad = entries()
        bad[0]['identity']['seed'] = 2024
        with self.assertRaises(ValueError):
            acceptance.matrix_by_key(bad, 'official')

    def test_metrics_reject_nonfinite_and_negative_values(self):
        for value in (math.nan, math.inf, -math.inf, -0.1):
            bad = entries()
            bad[0]['metrics']['mse'] = value
            with self.assertRaises(ValueError):
                acceptance.matrix_by_key(bad, 'official')
            with self.assertRaises(ValueError):
                acceptance.d4_values([value, 1, 1], [1, 1, 1])
        with self.assertRaises(ValueError):
            acceptance.d4_values([1, 1], [1, 1, 1])

    def test_d4_two_sided_inclusive_boundary_and_population_std(self):
        for value in (49.0, 51.0):
            self.assertTrue(acceptance.d4_values([50.0]*3, [value]*3)['passed'])
        for value in (math.nextafter(49.0, -math.inf), math.nextafter(51.0, math.inf)):
            self.assertFalse(acceptance.d4_values([50.0]*3, [value]*3)['passed'])
        result = acceptance.d4_values([9, 10, 11], [10, 10, 10])
        self.assertAlmostEqual(result['official_std'], math.sqrt(2/3))
        self.assertAlmostEqual(result['threshold'], 2*math.sqrt(2/3))
        self.assertTrue(acceptance.d4_values([0]*3, [0]*3)['passed'])
        self.assertFalse(acceptance.d4_values([0]*3, [1e-12]*3)['passed'])

    def test_config_overrides_and_explicit_missing_token_default(self):
        for config in ('configs/reference/ett-h1.toml', 'configs/engineering/cement-twin.toml'):
            actual = tomllib.loads((acceptance.ROOT/config).read_text(encoding='utf-8'))
            actual['window']['horizon'] = 96
            actual['train'].update(seed=2021, device='cpu', accumulation_steps=1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
            if 'missing' in actual['data']:
                actual['data']['missing'].setdefault('tokens', ['', 'NaN', 'nan'])
            acceptance.validate_config(actual, config, 96, 2021, 'cpu', 'dlinear')
            for section, key, value in [('train','seed',2022), ('train','learning_rate',0.001), ('window','horizon',95)]:
                bad = copy.deepcopy(actual)
                bad[section][key] = value
                with self.assertRaises(ValueError):
                    acceptance.validate_config(bad, config, 96, 2021, 'cpu', 'dlinear')
            if 'missing' in actual['data']:
                actual['data']['missing']['tokens'] = []
                with self.assertRaises(ValueError):
                    acceptance.validate_config(actual, config, 96, 2021, 'cpu', 'dlinear')

    def test_changed_source_and_artifact_fingerprints_fail(self):
        with patch.object(acceptance, 'read_json', return_value={'source': {'source_sha256': 'old'}}), patch.object(acceptance, 'source_snapshot', return_value={'source_sha256': 'new'}):
            with self.assertRaisesRegex(ValueError, 'runtime sources changed'):
                acceptance.suite_context(Namespace(suite='runs/synthetic-acceptance-test'))
        root = Path(tempfile.mkdtemp(prefix='ferrots-acceptance-test-'))
        p = root/'synthetic.txt'
        p.write_text('synthetic', encoding='utf-8')
        inventory = acceptance.artifacts([p])
        acceptance.verify_artifacts(inventory)
        p.write_text('altered!!', encoding='utf-8')
        with self.assertRaises(ValueError):
            acceptance.verify_artifacts(inventory)
        with self.assertRaises(ValueError):
            acceptance.verify_artifacts({})

    def test_quality_rejects_missing_or_duplicate_commands(self):
        records = [dict(key=key) for key in acceptance.QUALITY_COMMANDS]
        for bad in ([], records[:-1], records[1:]+[records[1]]):
            with self.assertRaisesRegex(ValueError, 'full quality command coverage'):
                acceptance.validate_quality(dict(records=bad))

    def test_final_rejects_failed_or_incomplete_data_gate(self):
        suite = Path('synthetic-suite')
        for data, message in ((dict(passed=False), 'data gate missing or failed'),
                              (dict(passed=True, reports={}), 'data matrix coverage missing')):
            with patch.object(acceptance, 'suite_context', return_value=(suite, {})), patch.object(acceptance, 'read_json', return_value=data):
                with self.assertRaisesRegex(ValueError, message):
                    acceptance.final_gate(Namespace(suite=str(suite), report='unused.md'))


if __name__ == '__main__':
    unittest.main()
