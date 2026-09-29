import copy
import csv
import datetime as dt
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recompute_predictions import HEADER, recompute


class RecomputeTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix='ferrots-recompute-test-'))/'synthetic.csv'
        self.columns = ['a', 'b', 'c']
        self.evaluation = dict(run_id='synthetic', windows=2, metrics={'original': {'variables': {n: dict(count=4, mae=1.5, rmse=math.sqrt(2.5)) for n in self.columns}}})
        self.rows = []
        for minute in (0, 3):
            origin = dt.datetime(2024, 1, 1, 0, minute)
            for h in range(2):
                for name in self.columns:
                    self.rows.append(['synthetic', '0', origin.strftime('%Y-%m-%d %H:%M:%S.%f'), (origin+dt.timedelta(minutes=h)).strftime('%Y-%m-%d %H:%M:%S.%f'), str(h+1), name, '10', str(11+h), 'true'])

    def run_case(self, rows, evaluation=None):
        with self.path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            writer.writerows(rows)
        return recompute(self.path, evaluation or self.evaluation, self.columns, 2)

    def test_success_and_exact_coverage(self):
        result = self.run_case(self.rows)
        self.assertEqual(result['rows'], 12)
        self.assertEqual(len(result['checks']), 6)
        self.assertTrue(all(c['difference'] == 0 for c in result['checks']))
        for rows in (self.rows[:-1], self.rows+self.rows[:1], []):
            with self.assertRaises(ValueError):
                self.run_case(rows)

    def test_bad_keys_nonfinite_missing_and_metric_mismatch(self):
        for index, value in ((0,'wrong'), (1,'9'), (3,'2024-01-02 00:00:00.000000'), (4,'2'), (5,'c'), (6,'NaN'), (7,'inf'), (8,'false')):
            rows = copy.deepcopy(self.rows)
            rows[0][index] = value
            with self.assertRaises(ValueError):
                self.run_case(rows)
        rows = self.rows[:6]+self.rows[:6]
        with self.assertRaisesRegex(ValueError, 'origin'):
            self.run_case(rows)
        evaluation = copy.deepcopy(self.evaluation)
        evaluation['metrics']['original']['variables']['a']['mae'] += 2e-6
        with self.assertRaisesRegex(ValueError, '1e-6'):
            self.run_case(self.rows, evaluation)


if __name__ == '__main__':
    unittest.main()
