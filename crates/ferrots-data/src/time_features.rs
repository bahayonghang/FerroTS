//! TSLib timeenc=1 for h/t. Monday=0; leap years retain denominator 365.
use crate::{DataError, Result, ensure};
use chrono::{DateTime, Datelike, Timelike};

pub fn width(frequency: &str) -> Result<usize> {
    match frequency {
        "h" => Ok(4),
        "t" => Ok(5),
        _ => Err(DataError::Invalid("time frequency must be h or t".into())),
    }
}
pub fn append(micros: i64, frequency: &str, output: &mut Vec<f32>) -> Result<()> {
    ensure(
        matches!(frequency, "h" | "t"),
        "time frequency must be h or t",
    )?;
    let time = DateTime::from_timestamp_micros(micros)
        .ok_or_else(|| DataError::Invalid("invalid feature timestamp".into()))?;
    if frequency == "t" {
        output.push((time.minute() as f64 / 59.0 - 0.5) as f32);
    }
    output.extend([
        (time.hour() as f64 / 23.0 - 0.5) as f32,
        (time.weekday().num_days_from_monday() as f64 / 6.0 - 0.5) as f32,
        (time.day0() as f64 / 30.0 - 0.5) as f32,
        (time.ordinal0() as f64 / 365.0 - 0.5) as f32,
    ]);
    Ok(())
}
pub fn compute(timestamps: &[i64], frequency: &str) -> Result<Vec<f32>> {
    let mut values = Vec::with_capacity(timestamps.len() * width(frequency)?);
    for &t in timestamps {
        append(t, frequency, &mut values)?;
    }
    Ok(values)
}
