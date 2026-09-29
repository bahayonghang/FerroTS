use crate::{PreparedSeries, Result, ensure, preprocess::UNUSED};
use ferrots_core::config::WindowSpec;
use ferrots_core::reference_rng::ReferenceRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Window {
    pub origin: usize,
    pub segment: usize,
    pub split: u8,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowIndex {
    pub entries: Vec<Window>,
    pub counts: [usize; 3],
    pub excluded_missing_history: usize,
}
impl WindowIndex {
    pub fn build(p: &PreparedSeries, w: &WindowSpec) -> Result<Self> {
        ensure(
            w.lookback > 0 && w.horizon > 0 && w.stride > 0,
            "window lengths and stride must be positive",
        )?;
        let length = w
            .lookback
            .checked_add(w.gap)
            .and_then(|n| n.checked_add(w.horizon))
            .ok_or_else(|| crate::DataError::Invalid("window length overflow".into()))?;
        let mut missing = vec![0usize; p.rows() + 1];
        let mut transitions = vec![0usize; p.rows() + 1];
        for i in 0..p.rows() {
            missing[i + 1] = missing[i]
                + usize::from(p.available[i * p.width()..(i + 1) * p.width()].contains(&0));
            transitions[i + 1] =
                transitions[i] + usize::from(i > 0 && p.splits[i] != p.splits[i - 1]);
        }
        let mut result = Self {
            entries: vec![],
            counts: [0; 3],
            excluded_missing_history: 0,
        };
        for span in &p.spans {
            if span.end - span.start < length {
                continue;
            }
            for origin in (span.start + w.lookback..=span.end - w.gap - w.horizon).step_by(w.stride)
            {
                let a = origin + w.gap;
                let z = a + w.horizon;
                let split = p.splits[a];
                if split >= UNUSED || transitions[z] != transitions[a + 1] {
                    continue;
                }
                if missing[origin] != missing[origin - w.lookback] {
                    result.excluded_missing_history += 1;
                    continue;
                }
                result.entries.push(Window {
                    origin,
                    segment: span.segment,
                    split,
                });
                result.counts[split as usize] += 1;
            }
        }
        ensure(
            !result.entries.is_empty(),
            "no legal windows for configured lengths and splits",
        )?;
        Ok(result)
    }
    pub fn order(&self, split: u8) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, w)| (w.split == split).then_some(i))
            .collect()
    }
    pub fn training_order(&self, after_init: &ReferenceRng, epoch: u64) -> Vec<usize> {
        let entries = self.order(0);
        after_init
            .training_permutation(entries.len(), epoch)
            .into_iter()
            .map(|i| entries[i])
            .collect()
    }
}
