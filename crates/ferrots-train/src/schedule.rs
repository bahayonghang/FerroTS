use crate::{Result, TrainError};
use ferrots_core::config::Schedule;
/// Upstream applies type1 after epoch e, for the following epoch.
pub fn epoch_end_rate(initial: f64, epoch: usize, schedule: Schedule) -> Result<f64> {
    if !initial.is_finite() || initial <= 0.0 || epoch == 0 {
        return Err(TrainError::InvalidState("invalid scheduler input".into()));
    }
    Ok(match schedule {
        Schedule::Constant => initial,
        Schedule::Type1 => initial * 0.5f64.powf((epoch - 1) as f64),
    })
}
