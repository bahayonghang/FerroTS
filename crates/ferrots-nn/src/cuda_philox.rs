//! PyTorch 2.14 CUDA fused dropout for Informer only.
//!
//! The shared [`crate::dropout`] stream stays ChaCha8. This generator follows
//! `aten/src/ATen/native/cuda/Dropout.cu` at torch commit
//! `08187d9e0fba026dc8217405802ab5381dc88d90`: Philox4x32-10 via CURAND,
//! block 256, UNROLL 4, and a grid capped by the device's resident threads.
//! `F.dropout` passes the keep probability into the kernel. Eval and p=0 do
//! not consume the offset; that short-circuit stays in the caller.

use candle_core::{Error, Result};

const BLOCK: u64 = 256;
const UNROLL: u64 = 4;
const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;
const CURAND_2POW32_INV: f32 = 2.3283064e-10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchLimits {
    pub sms: u32,
    pub max_threads_per_sm: u32,
}

#[cfg(feature = "cuda")]
pub fn limits_for(device: &candle_core::Device) -> Result<LaunchLimits> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let ctx = stream.context();
    let sms = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(|error| Error::Msg(error.to_string()))?;
    let threads = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
        .map_err(|error| Error::Msg(error.to_string()))?;
    Ok(LaunchLimits {
        sms: u32::try_from(sms).map_err(|error| Error::Msg(error.to_string()))?,
        max_threads_per_sm: u32::try_from(threads)
            .map_err(|error| Error::Msg(error.to_string()))?,
    })
}

/// Scale applied to kept values. Matches `(float)(1.0 / (float)(1 - p))`.
pub fn keep_scale(probability: f64) -> f64 {
    let keep = (1.0 - probability) as f32;
    f64::from((1.0 / f64::from(keep)) as f32)
}

/// Map a mask generated in contiguous `[B, C, L]` storage onto `[B, L, C]`.
pub fn to_batch_length_channel(physical: &[f32], dims: &[usize]) -> Result<Vec<f32>> {
    let [batch, len, channels] = dims else {
        return Err(Error::Msg(
            "channel-outer dropout expects a rank-3 tensor".into(),
        ));
    };
    if physical.len() != batch * len * channels {
        return Err(Error::Msg(
            "channel-outer dropout mask does not match the tensor".into(),
        ));
    }
    let mut logical = vec![0.0; physical.len()];
    for b in 0..*batch {
        for c in 0..*channels {
            let src = (b * *channels + c) * *len;
            for l in 0..*len {
                logical[(b * *len + l) * *channels + c] = physical[src + l];
            }
        }
    }
    Ok(logical)
}

pub fn mask(
    seed: u64,
    offset: u64,
    nelem: usize,
    probability: f64,
    limits: LaunchLimits,
) -> Result<(Vec<f32>, u64)> {
    if nelem == 0 {
        return Ok((Vec::new(), offset));
    }
    if !offset.is_multiple_of(UNROLL) {
        return Err(Error::Msg(
            "cuda dropout philox offset must be a multiple of 4".into(),
        ));
    }
    let nelem = nelem as u64;
    let (grid, counter_offset) = launch(nelem, limits)?;
    let threads = grid * BLOCK;
    let keep_below = (1.0 - probability) as f32;
    let mut out = vec![0.0; nelem as usize];
    let base = offset / UNROLL;
    match vector_width(nelem) {
        4 => fill_vec4(&mut out, seed, base, threads, nelem, keep_below),
        2 => fill_vec2(&mut out, seed, base, threads, nelem, keep_below),
        _ => fill_scalar(&mut out, seed, base, threads, nelem, keep_below),
    }
    Ok((out, offset + counter_offset))
}

fn launch(nelem: u64, limits: LaunchLimits) -> Result<(u64, u64)> {
    let blocks_per_sm = u64::from(limits.max_threads_per_sm) / BLOCK;
    if blocks_per_sm == 0 {
        return Err(Error::Msg(
            "cuda device has fewer than 256 threads per multiprocessor".into(),
        ));
    }
    let cap = u64::from(limits.sms) * blocks_per_sm;
    let grid = nelem.div_ceil(BLOCK).min(cap);
    if grid == 0 {
        return Err(Error::Msg("cuda dropout launch grid is empty".into()));
    }
    let counter_offset = ((nelem - 1) / (BLOCK * grid * UNROLL) + 1) * UNROLL;
    Ok((grid, counter_offset))
}

fn vector_width(nelem: u64) -> u64 {
    let mut width = 4;
    while width > 1 && !nelem.is_multiple_of(width) {
        width /= 2;
    }
    if nelem.is_multiple_of(width) {
        width
    } else {
        1
    }
}

fn fill_vec4(out: &mut [f32], seed: u64, base: u64, threads: u64, nelem: u64, keep_below: f32) {
    let stride = threads * 4;
    for idx in 0..threads {
        let mut linear = idx * 4;
        let mut call = 0;
        while linear < nelem {
            let words = philox_words(seed, idx, base + call);
            for (ii, word) in words.iter().enumerate() {
                out[(linear as usize) + ii] = keep(*word, keep_below);
            }
            call += 1;
            linear += stride;
        }
    }
}

fn fill_vec2(out: &mut [f32], seed: u64, base: u64, threads: u64, nelem: u64, keep_below: f32) {
    let stride = threads * 2;
    for idx in 0..threads {
        let mut linear = idx * 2;
        let mut call = 0;
        while linear < nelem {
            let words = philox_words(seed, idx, base + call);
            for ii in 0..2 {
                let position = linear + ii;
                if position < nelem {
                    out[position as usize] = keep(words[ii as usize], keep_below);
                }
            }
            call += 1;
            linear += stride;
        }
    }
}

fn fill_scalar(out: &mut [f32], seed: u64, base: u64, threads: u64, nelem: u64, keep_below: f32) {
    let rounded = ((nelem - 1) / (threads * UNROLL) + 1) * threads * UNROLL;
    for idx in 0..threads {
        let mut linear = idx;
        let mut call = 0;
        while linear < rounded {
            let words = philox_words(seed, idx, base + call);
            for (ii, word) in words.iter().enumerate() {
                let position = linear + threads * ii as u64;
                if position < nelem {
                    out[position as usize] = keep(*word, keep_below);
                }
            }
            call += 1;
            linear += threads * UNROLL;
        }
    }
}

fn keep(word: u32, keep_below: f32) -> f32 {
    if uniform(word) < keep_below { 1.0 } else { 0.0 }
}

fn uniform(word: u32) -> f32 {
    (word as f32) * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0)
}

fn philox_words(seed: u64, subsequence: u64, counter: u64) -> [u32; 4] {
    philox10(
        [
            counter as u32,
            (counter >> 32) as u32,
            subsequence as u32,
            (subsequence >> 32) as u32,
        ],
        [seed as u32, (seed >> 32) as u32],
    )
}

fn philox10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 0..9 {
        counter = philox_round(counter, key);
        key[0] = key[0].wrapping_add(PHILOX_W0);
        key[1] = key[1].wrapping_add(PHILOX_W1);
    }
    philox_round(counter, key)
}

fn philox_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (lo0, hi0) = mulhilo(PHILOX_M0, counter[0]);
    let (lo1, hi1) = mulhilo(PHILOX_M1, counter[2]);
    [
        hi1 ^ counter[1] ^ key[0],
        lo1,
        hi0 ^ counter[3] ^ key[1],
        lo0,
    ]
}

fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = u64::from(a) * u64::from(b);
    (product as u32, (product >> 32) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RTX_4090: LaunchLimits = LaunchLimits {
        sms: 128,
        max_threads_per_sm: 1536,
    };

    fn bits(mask: &[f32]) -> String {
        mask.iter()
            .map(|value| if *value == 0.0 { '0' } else { '1' })
            .collect()
    }

    #[test]
    fn rtx4090_masks_match_torch_2_14_cuda_dropout() {
        let (small, offset) = mask(2023, 0, 64, 0.1, RTX_4090).unwrap();
        assert_eq!(offset, 4);
        assert_eq!(
            bits(&small),
            "1111010111111111101111011111011110111111111100111001111101110101"
        );
        let (vec2, _) = mask(2023, 0, 6, 0.1, RTX_4090).unwrap();
        assert_eq!(bits(&vec2), "110111");
        let (scalar, _) = mask(2023, 0, 17, 0.1, RTX_4090).unwrap();
        assert_eq!(bits(&scalar), "10111110111011000");
        let (_, chained) = mask(2023, offset, 8, 0.1, RTX_4090).unwrap();
        assert_eq!(chained, 8);
        let (partial, next) = mask(2023, 0, 802_816, 0.1, RTX_4090).unwrap();
        assert_eq!(next, 8);
        let zeros: Vec<usize> = partial
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value == 0.0).then_some(index))
            .collect();
        assert_eq!(zeros.len(), 80_276);
        assert_eq!(
            zeros.iter().map(|index| *index as u64).sum::<u64>(),
            32_218_339_240
        );
        assert_eq!(bits(&partial[..32]), "11110101111111111011110111110111");
        assert_eq!(bits(&partial[partial.len() - 16..]), "1111111101111111");
        assert_eq!((keep_scale(0.1) as f32).to_bits(), 1_066_285_284);
        let physical = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(
            to_batch_length_channel(&physical, &[1, 3, 2]).unwrap(),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }
}
