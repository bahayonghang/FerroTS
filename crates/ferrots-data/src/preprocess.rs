use crate::{Frame, RawReport, Result, csv_source::parse_timestamp, ensure};
use ferrots_core::config::{DataSpec, MissingMode, SplitSpec};
use serde::{Deserialize, Serialize};

pub const TRAIN: u8 = 0;
pub const VALIDATION: u8 = 1;
pub const TEST: u8 = 2;
pub const UNUSED: u8 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: usize,
    pub start: usize,
    pub end: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub segment: usize,
    pub start: usize,
    pub end: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedReport {
    pub rows: usize,
    pub running_segments: usize,
    pub segment_rows: usize,
    pub split_segments: [usize; 3],
    pub usable_rows: [usize; 3],
    pub imputed_per_column: Vec<usize>,
}
#[derive(Debug, Clone)]
pub struct PreparedSeries {
    pub columns: Vec<String>,
    pub timestamps: Vec<i64>,
    pub values: Vec<f64>,
    pub observed: Vec<u8>,
    pub available: Vec<u8>,
    pub imputed: Vec<u8>,
    pub splits: Vec<u8>,
    pub segments: Vec<Segment>,
    pub spans: Vec<Span>,
    pub raw: RawReport,
    pub report: PreparedReport,
}
impl PreparedSeries {
    pub fn rows(&self) -> usize {
        self.timestamps.len()
    }
    pub fn width(&self) -> usize {
        self.columns.len()
    }
}

fn aggregate(frame: Frame, spec: &DataSpec) -> Result<Frame> {
    let Some(grid) = &spec.grid else {
        return Ok(frame);
    };
    ensure(
        frame.timestamps.windows(2).all(|t| t[0] <= t[1]),
        "grid aggregation requires sorted timestamps",
    )?;
    let delta = spec
        .interval_seconds
        .checked_mul(1_000_000)
        .ok_or_else(|| crate::DataError::Invalid("sampling interval overflow".into()))?;
    let first = frame.timestamps[0].div_euclid(delta) * delta;
    let last = frame.timestamps[frame.rows() - 1].div_euclid(delta) * delta;
    let n = usize::try_from((last - first) / delta + 1)
        .map_err(|_| crate::DataError::Invalid("grid length overflow".into()))?;
    let c = frame.width();
    let size = n
        .checked_mul(c)
        .ok_or_else(|| crate::DataError::Invalid("grid size overflow".into()))?;
    let mut values = vec![0.0; size];
    let mut counts = vec![0_usize; size];
    for (i, &time) in frame.timestamps.iter().enumerate() {
        let r = ((time - first) / delta) as usize;
        for j in 0..c {
            if frame.observed[i * c + j] != 0 {
                values[r * c + j] += frame.values[i * c + j];
                counts[r * c + j] += 1;
            }
        }
    }
    let observed = counts
        .iter()
        .enumerate()
        .map(|(k, &count)| {
            if count >= grid.min_samples {
                values[k] /= count as f64;
                1
            } else {
                values[k] = 0.0;
                0
            }
        })
        .collect();
    ensure(
        values.iter().all(|v| v.is_finite()),
        "non-finite grid aggregate",
    )?;
    Ok(Frame {
        columns: frame.columns,
        timestamps: (0..n).map(|r| first + r as i64 * delta).collect(),
        values,
        observed,
        raw: frame.raw,
    })
}

fn segments(frame: &Frame, spec: &DataSpec) -> Result<Vec<Segment>> {
    let c = frame.width();
    let delta = spec.interval_seconds * 1_000_000;
    let running_col = spec
        .running
        .as_ref()
        .map(|r| {
            frame
                .columns
                .iter()
                .position(|n| *n == r.column)
                .ok_or_else(|| crate::DataError::Invalid("running column is absent".into()))
        })
        .transpose()?;
    let minimum = spec
        .running
        .as_ref()
        .map(|r| r.min_duration_seconds)
        .unwrap_or(spec.interval_seconds);
    let max_missing = spec
        .running
        .as_ref()
        .map(|r| r.max_missing_seconds)
        .unwrap_or(spec.missing.max_missing_seconds);
    let mut result = Vec::new();
    let mut start = None;
    let mut last_valid = 0;
    let finish = |start: &mut Option<usize>, end: usize, result: &mut Vec<Segment>| {
        if let Some(s) = start.take()
            && ((end - s) as i64) * spec.interval_seconds >= minimum
        {
            result.push(Segment {
                id: result.len(),
                start: s,
                end,
            });
        }
    };
    for i in 0..frame.rows() {
        if i > 0 && frame.timestamps[i] - frame.timestamps[i - 1] != delta {
            finish(&mut start, last_valid + 1, &mut result);
        }
        let state = if let (Some(rule), Some(j)) = (&spec.running, running_col) {
            if frame.observed[i * c + j] == 0 {
                0
            } else if rule.op.matches(frame.values[i * c + j], rule.threshold) {
                1
            } else {
                -1
            }
        } else if frame.observed[i * c..(i + 1) * c].iter().any(|&m| m != 0) {
            1
        } else {
            0
        };
        match state {
            1 => {
                if start.is_none() {
                    start = Some(i);
                }
                last_valid = i;
            }
            -1 => finish(&mut start, last_valid + 1, &mut result),
            _ => {
                if start.is_some() && (i - last_valid) as i64 * spec.interval_seconds > max_missing
                {
                    finish(&mut start, last_valid + 1, &mut result);
                }
            }
        }
    }
    finish(&mut start, last_valid + 1, &mut result);
    Ok(result)
}

fn assign_splits(frame: &Frame, segments: &[Segment], spec: &DataSpec) -> Result<Vec<u8>> {
    let mut splits = vec![UNUSED; frame.rows()];
    match &spec.split {
        SplitSpec::Segments { train, validation } => {
            let total: usize = segments.iter().map(|s| s.end - s.start).sum();
            let mut cursor = 0;
            for segment in segments {
                let length = segment.end - segment.start;
                let midpoint = (cursor as f64 + length as f64 / 2.0) / total as f64;
                let split = if midpoint < *train {
                    TRAIN
                } else if midpoint < train + validation {
                    VALIDATION
                } else {
                    TEST
                };
                splits[segment.start..segment.end].fill(split);
                cursor += length;
            }
        }
        SplitSpec::Dates {
            train_end,
            validation_end,
            test_end,
        } => {
            let a = parse_timestamp(train_end, &spec.timestamp_format)?;
            let b = parse_timestamp(validation_end, &spec.timestamp_format)?;
            let z = parse_timestamp(test_end, &spec.timestamp_format)?;
            ensure(a < b && b < z, "date split boundaries must increase")?;
            for (s, &t) in splits.iter_mut().zip(&frame.timestamps) {
                *s = if t < a {
                    TRAIN
                } else if t < b {
                    VALIDATION
                } else if t < z {
                    TEST
                } else {
                    UNUSED
                };
            }
        }
        _ => {
            let (a, b, z) = match spec.split {
                SplitSpec::Rows {
                    train_end,
                    validation_end,
                    test_end,
                } => (train_end, validation_end, test_end),
                SplitSpec::Ratio { train, test } => (
                    (frame.rows() as f64 * train).floor() as usize,
                    frame.rows() - (frame.rows() as f64 * test).floor() as usize,
                    frame.rows(),
                ),
                _ => unreachable!(),
            };
            ensure(
                a > 0 && a < b && b < z && z <= frame.rows(),
                "row split outside prepared data",
            )?;
            splits[..a].fill(TRAIN);
            splits[a..b].fill(VALIDATION);
            splits[b..z].fill(TEST);
        }
    }
    Ok(splits)
}

pub fn prepare(frame: Frame, spec: &DataSpec) -> Result<PreparedSeries> {
    prepare_impl(frame, spec, false)
}
pub fn prepare_inference(frame: Frame, spec: &DataSpec) -> Result<PreparedSeries> {
    prepare_impl(frame, spec, true)
}
fn prepare_impl(frame: Frame, spec: &DataSpec, inference: bool) -> Result<PreparedSeries> {
    let mut frame = aggregate(frame, spec)?;
    if spec.missing.mode == MissingMode::Error {
        ensure(
            frame.observed.iter().all(|&m| m != 0),
            "missing values rejected by reproduction protocol",
        )?;
    }
    let segments = segments(&frame, spec)?;
    ensure(!segments.is_empty(), "no valid segments")?;
    let splits = if inference {
        vec![UNUSED; frame.rows()]
    } else {
        assign_splits(&frame, &segments, spec)?
    };
    let c = frame.width();
    let mut available = frame.observed.clone();
    let mut imputed = vec![0; frame.values.len()];
    if spec.missing.mode == MissingMode::Causal {
        for segment in &segments {
            let mut last: Vec<Option<usize>> = vec![None; c];
            for i in segment.start..segment.end {
                for (j, last_row) in last.iter_mut().enumerate() {
                    let k = i * c + j;
                    if frame.observed[k] != 0 {
                        *last_row = Some(i);
                    } else if let Some(previous) = *last_row
                        && frame.timestamps[i] - frame.timestamps[previous]
                            <= spec.missing.max_stale_seconds * 1_000_000
                    {
                        frame.values[k] = frame.values[previous * c + j];
                        available[k] = 1;
                        imputed[k] = 1;
                    }
                }
            }
        }
    }
    let mut spans = Vec::new();
    let mut usable_rows = [0; 3];
    let mut split_segments = [0; 3];
    for segment in &segments {
        let mut split_seen = [false; 3];
        let mut span_start = None;
        for i in segment.start..=segment.end {
            let complete = i < segment.end && available[i * c..(i + 1) * c].iter().all(|&m| m != 0);
            if i < segment.end && splits[i] < UNUSED {
                let s = splits[i] as usize;
                split_seen[s] = true;
                if complete {
                    usable_rows[s] += 1;
                }
            }
            if i < segment.end && (!spec.missing.complete_windows || complete) {
                if span_start.is_none() {
                    span_start = Some(i);
                }
            } else if let Some(start) = span_start.take() {
                spans.push(Span {
                    segment: segment.id,
                    start,
                    end: i,
                });
            }
        }
        for i in 0..3 {
            split_segments[i] += usize::from(split_seen[i]);
        }
    }
    let report = PreparedReport {
        rows: frame.rows(),
        running_segments: segments.len(),
        segment_rows: segments.iter().map(|s| s.end - s.start).sum(),
        split_segments,
        usable_rows,
        imputed_per_column: (0..c)
            .map(|j| {
                imputed
                    .iter()
                    .skip(j)
                    .step_by(c)
                    .filter(|&&m| m != 0)
                    .count()
            })
            .collect(),
    };
    Ok(PreparedSeries {
        columns: frame.columns,
        timestamps: frame.timestamps,
        values: frame.values,
        observed: frame.observed,
        available,
        imputed,
        splits,
        segments,
        spans,
        raw: frame.raw,
        report,
    })
}
