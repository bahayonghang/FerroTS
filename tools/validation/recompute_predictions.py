"""Independently validate every prediction key and recompute original-unit errors."""
import argparse
import csv
import datetime as dt
import gzip
import json
import math
from pathlib import Path

HEADER = ['run_id', 'series_id', 'forecast_origin', 'target_time', 'horizon', 'target_name', 'y_true', 'y_pred', 'valid']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recompute(predictions, evaluation, columns, horizon, interval_seconds=60, gap=0):
    require(len(columns) == len(set(columns)) and len(columns) > 0 and horizon > 0, 'invalid export dimensions')
    expected_rows = evaluation['windows'] * horizon * len(columns)
    sums = {name: [0, 0.0, 0.0] for name in columns}
    count, origins, prior_origin = 0, 0, None
    origin_text, targets = None, []
    opener = gzip.open if str(predictions).endswith('.gz') else open
    with opener(predictions, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == HEADER, 'prediction schema differs')
        for row in reader:
            require(len(row) == len(HEADER), 'prediction row width differs')
            offset = count % (horizon * len(columns))
            step, column = divmod(offset, len(columns))
            if offset == 0:
                origin_text = row[2]
                origin = dt.datetime.fromisoformat(origin_text)
                require(prior_origin is None or origin > prior_origin, 'duplicate or unordered origin')
                prior_origin = origin
                targets = [(origin + dt.timedelta(seconds=(gap+h)*interval_seconds)).strftime('%Y-%m-%d %H:%M:%S.%f') for h in range(horizon)]
                origins += 1
            require(row[0] == evaluation['run_id'] and row[1] == '0', 'run or series identity differs')
            require(row[2] == origin_text and row[3] == targets[step], 'origin or target time differs')
            require(row[4] == str(step+1) and row[5] == columns[column], 'horizon or variable order differs')
            require(row[8] == 'true', 'the complete-window protocol requires valid targets')
            truth, prediction = float(row[6]), float(row[7])
            require(math.isfinite(truth) and math.isfinite(prediction), 'nonfinite valid prediction')
            error = prediction-truth
            acc = sums[columns[column]]
            acc[0] += 1
            acc[1] += abs(error)
            acc[2] += error*error
            count += 1
            require(count <= expected_rows, 'extra prediction rows')
            if count % 1000000 == 0:
                print(f'Validated prediction rows: {count}', flush=True)
    require(count == expected_rows and origins == evaluation['windows'], 'prediction coverage differs')
    checks = []
    for name, (n, absolute, squared) in sums.items():
        expected = evaluation['metrics']['original']['variables'][name]
        require(n > 0 and n == expected['count'], 'valid count differs')
        for metric, value in [('mae', absolute/n), ('rmse', math.sqrt(squared/n))]:
            reference = expected[metric]
            require(math.isfinite(value) and math.isfinite(reference), 'nonfinite metric')
            difference = abs(value-reference)
            checks.append(dict(variable=name, metric=metric, count=n, recomputed=value, evaluated=reference, difference=difference, tolerance=1e-6, passed=difference < 1e-6))
    require(all(c['passed'] for c in checks), 'independent metric difference is not below 1e-6')
    return dict(passed=True, rows=count, origins=origins, columns=columns, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--columns', nargs='+', required=True)
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--interval-seconds', type=int, default=60)
    parser.add_argument('--gap', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding='utf-8'))
    result = recompute(args.predictions, evaluation, args.columns, args.horizon, args.interval_seconds, args.gap)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+chr(10), encoding='utf-8')
    print(f"PASS independent recomputation: {result['rows']} rows, {len(result['checks'])} metric checks", flush=True)


if __name__ == '__main__':
    main()
