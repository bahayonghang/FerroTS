//! CPU RNG semantics from PyTorch 2.14.0, commit
//! 08187d9e0fba026dc8217405802ab5381dc88d90. See NOTICE.md for licenses.
//! MT19937: Copyright (C) 1997-2002 Makoto Matsumoto and Takuji Nishimura.
//! This stream is separate from the P0 dropout ChaCha8 contract.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceRngState {
    pub format_version: u32,
    pub algorithm: String,
    pub words: Vec<u32>,
    pub index: usize,
}

#[derive(Debug, thiserror::Error)]
#[error("invalid ReferenceRng state: version, algorithm, 624 words or index mismatch")]
pub struct InvalidRngState;

#[derive(Clone)]
pub struct ReferenceRng {
    state: [u32; 624],
    index: usize,
}

impl ReferenceRng {
    pub fn snapshot(&self) -> ReferenceRngState {
        ReferenceRngState {
            format_version: 1,
            algorithm: "mt19937-pytorch-cpu-v1".into(),
            words: self.state.to_vec(),
            index: self.index,
        }
    }

    pub fn from_state(state: &ReferenceRngState) -> Result<Self, InvalidRngState> {
        if state.format_version != 1
            || state.algorithm != "mt19937-pytorch-cpu-v1"
            || state.words.len() != 624
            || state.index > 624
            || state.words.iter().all(|&n| n == 0)
        {
            return Err(InvalidRngState);
        }
        Ok(Self {
            state: state
                .words
                .as_slice()
                .try_into()
                .map_err(|_| InvalidRngState)?,
            index: state.index,
        })
    }

    pub fn new(seed: u64) -> Self {
        let mut state = [0; 624];
        state[0] = seed as u32;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^ (y >> 18)
    }

    fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | self.next_u32() as u64
    }

    /// Equivalent to a scalar CPU int64 Tensor.random_().
    pub fn seed_draw(&mut self) -> u64 {
        self.next_u64() & 0x7fffffffffffffff
    }

    pub fn uniform(&mut self, from: f32, to: f32) -> f32 {
        let unit = (self.next_u32() & 0xffffff) as f32 * (1.0 / 16777216.0);
        unit * (to - from) + from
    }

    /// CPU randperm, including the upstream large-length branch.
    pub fn permutation(&mut self, length: usize) -> Vec<usize> {
        let mut order: Vec<_> = (0..length).collect();
        if length < u32::MAX as usize / 20 {
            for i in 0..length.saturating_sub(1) {
                let offset = self.next_u32() as usize % (length - i);
                order.swap(i, i + offset);
            }
        } else {
            for i in 0..length {
                let j = (self.next_u64() % (i as u64 + 1)) as usize;
                order[i] = order[j];
                order[j] = i;
            }
        }
        order
    }

    /// Called on the state immediately after model initialization. Epoch is zero-based.
    /// Reference train and val each consume base_seed + sampler_seed; test consumes
    /// base_seed only. Advancing that stream never accesses validation/test data.
    pub fn training_permutation(&self, length: usize, epoch: u64) -> Vec<usize> {
        let mut global = self.clone();
        for _ in 0..epoch {
            for _ in 0..5 {
                global.seed_draw();
            }
        }
        global.seed_draw();
        Self::new(global.seed_draw()).permutation(length)
    }
}
