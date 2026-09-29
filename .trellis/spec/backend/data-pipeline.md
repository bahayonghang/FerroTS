# P1 data pipeline contracts

## Scope and ownership

Read this spec before changing core configuration, the data crate, data inspection, or prediction inputs. Core owns strict serde/TOML schemas. Data owns CSV parsing, raw reports, preprocessing, cache, scaler, calendar features, legal windows and host batches. Data does not depend on Candle. CLI owns explicit overrides and output paths.

## Read and prepare

- Missing files are errors. The production pipeline never downloads data.
- Select columns by exact UTF-8 names. Reject duplicate headers, missing configured columns, unknown TOML keys and unsupported frequencies. Preserve raw counts before sorting or deduplication.
- Sorting is explicit and stable. Duplicate policies are error, preserve, keep_first and keep_last. A duplicate group is one timestamp; duplicate_rows counts every involved row; duplicate_removed counts removed rows.
- Grid means use the configured minimum valid samples per column. Running-state rules use pre-fill observations. Long unknown runs and stopped rows end segments. Trim unknown segment endpoints, then enforce minimum segment duration.
- Causal fill never crosses a segment. Compute staleness from the last observed value; an imputed value must not restart that clock. Keep observed, available and imputed masks distinct.
- Assign whole-segment splits using original segment lengths and cumulative-minute midpoints before complete-window filtering.

## Protocol-specific rules

ETT uses fixed 12/4/4-month row boundaries and only the official prefix: 14,400 hourly rows or 57,600 minute rows. Both ETT families use timeenc=1, freq=h. Other public datasets do not acquire ETT row boundaries.

cement-twin-v1 retains all four approved adjustments in docs/datasets/cement-yangquan.md section 2.2: fill all 18 columns, exclude any still-incomplete minute from windows, keep the first duplicate, and split by full-segment cumulative-minute midpoints. Filled complete values are valid protocol targets. The generic engineering default supervises observed targets only. Keep both policies explicit in configuration.

## Scaler, cache and windows

- Fit f64 means and population variance from training data only. Constant columns use scale=1 and a constant flag. Save the column names, counts and protocol identity. Load the saved scaler for evaluation and prediction; never fit on a new inference file.
- h calendar features match TSLib HourOfDay, DayOfWeek, DayOfMonth and DayOfYear. t prepends MinuteOfHour. Monday is zero. Follow the fixed reference treatment of leap years.
- Cache keys include source content SHA256, preprocessing protocol and version. Verify file hashes, shapes, keys and version on read. A hit reports parse_count=0, although source hashing still occurs. Corruption is an error; bypass is explicit. Retain failed temporary output for diagnosis.
- WindowIndex stores origins and split/segment identity. History is [o-L,o); targets are [o+g,o+g+H). History, gap and target stay in one legal continuous span. Every target belongs to one split. Validation/test history can precede the target split.
- HostBatch preserves tail batches, column order and masks. HostInput contains no labels. Fixed order() is used for evaluation. training_order() maps the pinned reference permutation to train indices only; see training-loop.md.
- Future prediction uses the final complete history span and extrapolated timestamps. It does not apply the original training file's row cutoffs to a new inference file.

## Required checks

Keep csv_report, preprocess, scaler, cache, time_features, windows and batches tests. Cover duplicate counts, ordering, grid sample minimums, 5/6-minute gaps, 29/30-minute segments, 10/11-minute fill ages, constants, missing history, target boundaries, gaps, L=1 and tail batches. Formal data acceptance independently checks public hashes, ETT windows/time/scalers, Exchange 7,588 records, and the cement raw/segment/window/scaler counts. Tests and fixtures contain synthetic/public values only.
