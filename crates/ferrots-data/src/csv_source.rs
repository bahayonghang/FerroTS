//! UTF-8 CSV ingestion. Error messages never include cell values or rows.
use crate::{DataError, Result, ensure};
use chrono::{DateTime, NaiveDateTime};
use ferrots_core::config::{DataSpec, DuplicatePolicy, NumberType, SortPolicy};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnReport {
    pub name: String,
    pub missing: usize,
    pub missing_rate: f64,
    pub non_numeric: usize,
    pub constant: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawReport {
    pub rows: usize,
    pub start: String,
    pub end: String,
    pub non_increasing: usize,
    pub duplicate_groups: usize,
    pub duplicate_rows: usize,
    pub duplicate_removed: usize,
    pub gaps: usize,
    pub max_gap_seconds: f64,
    pub columns: Vec<ColumnReport>,
}

/// Row-major f64 values. Missing positions hold zero and observed=0.
#[derive(Debug, Clone)]
pub struct Frame {
    pub columns: Vec<String>,
    pub timestamps: Vec<i64>,
    pub values: Vec<f64>,
    pub observed: Vec<u8>,
    pub raw: RawReport,
}
impl Frame {
    pub fn rows(&self) -> usize {
        self.timestamps.len()
    }
    pub fn width(&self) -> usize {
        self.columns.len()
    }
}

pub fn parse_timestamp(text: &str, format: &str) -> Result<i64> {
    NaiveDateTime::parse_from_str(text, format)
        .map(|d| d.and_utc().timestamp_micros())
        .map_err(|_| DataError::Invalid("timestamp does not match configured format".into()))
}
pub fn timestamp_text(micros: i64) -> Result<String> {
    DateTime::from_timestamp_micros(micros)
        .map(|d| d.naive_utc().format("%Y-%m-%d %H:%M:%S%.6f").to_string())
        .ok_or_else(|| DataError::Invalid("timestamp outside supported range".into()))
}
fn csv_error(error: csv::Error) -> DataError {
    let row = error.position().map(|p| p.line()).unwrap_or(0);
    DataError::Invalid(format!("CSV structure or encoding error at line {row}"))
}

pub fn read_csv(spec: &DataSpec) -> Result<Frame> {
    ensure(!spec.input_columns.is_empty(), "input columns are empty")?;
    ensure(
        spec.target_columns
            .iter()
            .all(|n| spec.input_columns.contains(n)),
        "target is not a named input column",
    )?;
    let mut reader = csv::ReaderBuilder::new().from_reader(std::fs::File::open(&spec.path)?);
    let headers = reader.headers().map_err(csv_error)?.clone();
    let unique: HashSet<_> = headers.iter().collect();
    ensure(unique.len() == headers.len(), "duplicate CSV column name")?;
    let find = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| DataError::Invalid(format!("missing column: {name}")))
    };
    let time_idx = find(&spec.timestamp_column)?;
    let mut columns = spec.input_columns.clone();
    for name in &spec.future_known_columns {
        if !columns.contains(name) {
            columns.push(name.clone());
        }
    }
    ensure(
        columns.iter().collect::<HashSet<_>>().len() == columns.len(),
        "duplicate configured input column",
    )?;
    let indices: Vec<usize> = columns
        .iter()
        .map(|name| find(name))
        .collect::<Result<_>>()?;
    let c = columns.len();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    let mut observed = Vec::new();
    let mut missing = vec![0; c];
    let mut first: Vec<Option<f64>> = vec![None; c];
    let mut constant = vec![true; c];
    let mut frequencies = HashMap::<i64, usize>::new();
    let mut non_increasing = 0;
    let mut gaps = 0;
    let mut max_gap = 0;
    let mut record = csv::StringRecord::new();
    while reader.read_record(&mut record).map_err(csv_error)? {
        let line = timestamps.len() + 2;
        let time = parse_timestamp(&record[time_idx], &spec.timestamp_format).map_err(|_| {
            DataError::Invalid(format!(
                "invalid timestamp at line {line}, column {}",
                spec.timestamp_column
            ))
        })?;
        if let Some(&previous) = timestamps.last() {
            if time <= previous {
                non_increasing += 1;
            }
            let delta = time - previous;
            if delta > spec.interval_seconds * 1_000_000 {
                gaps += 1;
            }
            max_gap = max_gap.max(delta);
        }
        *frequencies.entry(time).or_default() += 1;
        timestamps.push(time);
        for (j, &index) in indices.iter().enumerate() {
            let token = record[index].trim();
            if spec.missing.tokens.iter().any(|s| s == token) {
                missing[j] += 1;
                values.push(0.0);
                observed.push(0);
                continue;
            }
            let number = token.parse::<f64>().ok().filter(|v| v.is_finite());
            let value = number.ok_or_else(|| {
                DataError::Invalid(format!(
                    "non-finite or nonnumeric value at line {line}, column {}",
                    columns[j]
                ))
            })?;
            ensure(
                spec.column_types.get(&columns[j]) != Some(&NumberType::Integer)
                    || value.fract() == 0.0,
                format!("non-integer value at line {line}, column {}", columns[j]),
            )?;
            match first[j] {
                Some(v) => constant[j] &= v == value,
                None => first[j] = Some(value),
            }
            values.push(value);
            observed.push(1);
        }
    }
    ensure(!timestamps.is_empty(), "CSV contains no data rows")?;
    let duplicate_groups = frequencies.values().filter(|&&n| n > 1).count();
    let duplicate_rows = frequencies.values().filter(|&&n| n > 1).sum::<usize>();
    let raw = RawReport {
        rows: timestamps.len(),
        start: timestamp_text(*timestamps.iter().min().expect("nonempty timestamps"))?,
        end: timestamp_text(*timestamps.iter().max().expect("nonempty timestamps"))?,
        non_increasing,
        duplicate_groups,
        duplicate_rows,
        duplicate_removed: 0,
        gaps,
        max_gap_seconds: max_gap as f64 / 1e6,
        columns: columns
            .iter()
            .enumerate()
            .map(|(j, name)| ColumnReport {
                name: name.clone(),
                missing: missing[j],
                missing_rate: missing[j] as f64 / timestamps.len() as f64,
                non_numeric: 0,
                constant: first[j].is_some() && constant[j],
            })
            .collect(),
    };
    ensure(
        spec.duplicates != DuplicatePolicy::Error || duplicate_groups == 0,
        "duplicate timestamp rejected by policy",
    )?;
    let mut frame = Frame {
        columns,
        timestamps,
        values,
        observed,
        raw,
    };
    let mut order: Vec<usize> = (0..frame.rows()).collect();
    if spec.sort == SortPolicy::Stable {
        order.sort_by_key(|&i| frame.timestamps[i]);
    }
    if matches!(
        spec.duplicates,
        DuplicatePolicy::KeepFirst | DuplicatePolicy::KeepLast
    ) {
        let mut chosen = HashMap::new();
        for &i in &order {
            if spec.duplicates == DuplicatePolicy::KeepFirst {
                chosen.entry(frame.timestamps[i]).or_insert(i);
            } else {
                chosen.insert(frame.timestamps[i], i);
            }
        }
        order.retain(|&i| chosen.get(&frame.timestamps[i]) == Some(&i));
    }
    frame.raw.duplicate_removed = frame.rows() - order.len();
    if frame.raw.duplicate_removed > 0 || spec.sort == SortPolicy::Stable {
        frame.timestamps = order.iter().map(|&i| frame.timestamps[i]).collect();
        frame.values = order
            .iter()
            .flat_map(|&i| frame.values[i * c..(i + 1) * c].iter().copied())
            .collect();
        frame.observed = order
            .iter()
            .flat_map(|&i| frame.observed[i * c..(i + 1) * c].iter().copied())
            .collect();
    }
    Ok(frame)
}
