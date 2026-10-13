#![cfg(feature = "cuda")]

use candle_core::cuda_backend::cudarc::driver::{result, sys};
use candle_core::{DType, Device, Tensor};
use ferrots_core::config::{Activation, ModelSpec, WindowSpec};
use ferrots_nn::{
    informer::Informer,
    model::{ForwardContext, ModelInput},
};
use std::time::Instant;
type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

// The native CUDA async-pool high watermark observes short-lived allocations,
// unlike periodic device-wide sampling. This target contains a single test so
// no concurrent Rust test in this process can change the pool watermark.
fn used(pool: sys::CUmemoryPool, attr: sys::CUmemPool_attribute) -> R<u64> {
    let mut bytes = 0_u64;
    // SAFETY: pool is borrowed from this process's live CUDA device, and both
    // selected attributes have the documented cuuint64_t value type.
    unsafe {
        result::mem_pool::get_attribute(pool, attr, (&mut bytes as *mut u64).cast())?;
    }
    Ok(bytes)
}
fn reset(pool: sys::CUmemoryPool) -> R {
    let mut zero = 0_u64;
    // SAFETY: resetting the documented high-watermark attribute to zero changes
    // only accounting; the pool is live and no allocation policy is modified.
    unsafe {
        result::mem_pool::set_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
            (&mut zero as *mut u64).cast(),
        )?;
    }
    Ok(())
}

#[test]
fn cuda_sparse_attention_shapes_and_allocator_peak() -> R {
    let device = Device::new_cuda(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    assert!(
        stream.context().has_async_alloc(),
        "pool accounting requires async allocation"
    );
    // SAFETY: device remains alive for the complete test and owns this pool.
    let pool = unsafe { result::device::get_mem_pool(stream.context().cu_device())? };
    for length in [96, 192, 384, 768, 1536] {
        let spec = ModelSpec::Informer {
            d_model: 8,
            n_heads: 2,
            d_ff: 16,
            e_layers: 2,
            d_layers: 1,
            factor: 2,
            distil: true,
            label_len: length / 2,
            dropout: 0.,
            activation: Activation::Gelu,
        };
        let window = WindowSpec {
            lookback: length,
            horizon: length / 2,
            gap: 0,
            stride: 1,
        };
        let model = Informer::new(&spec, &window, 3, 4, 2021, &device)?.freeze_for_eval()?;
        let input = ModelInput {
            history: Tensor::arange(0_u32, (2 * length * 3) as u32, &device)?
                .to_dtype(DType::F32)?
                .reshape((2, length, 3))?
                .affine(0.001, -0.5)?,
            history_time: Some(Tensor::zeros((2, length, 4), DType::F32, &device)?),
            future_time: Some(Tensor::zeros((2, length / 2, 4), DType::F32, &device)?),
        };
        device.synchronize()?;
        let baseline = used(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
        )?;
        reset(pool)?;
        let start = Instant::now();
        let (prediction, trace) =
            model.forward_with_trace(&input, ForwardContext::evaluation_with_id(1), None)?;
        device.synchronize()?;
        let peak = used(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        )?;
        println!(
            "ATTENTION_COMPLEXITY L={length} baseline={baseline} peak={peak} delta={} seconds={:.6}",
            peak - baseline,
            start.elapsed().as_secs_f64()
        );
        assert_eq!(prediction.dims(), [2, length / 2, 3]);
        let mut cases = 0;
        for (name, scores) in &trace {
            if let Some(prefix) = name.strip_suffix(".scores") {
                let queries = trace[&format!("{prefix}.query_projection")].dim(1)?;
                let keys = trace[&format!("{prefix}.key_projection")].dim(1)?;
                let expected = (2 * (queries as f64).ln().ceil() as usize)
                    .max(1)
                    .min(queries);
                assert_eq!(scores.dims(), [2, 2, expected, keys]);
                assert!(expected < queries, "full score matrix: {name}");
                if let Some(mask) = trace.get(&format!("{prefix}.mask")) {
                    assert_eq!(mask.dims(), scores.dims());
                }
                println!(
                    "SPARSE_SHAPE L={length} site={prefix} query={queries} keys={keys} scores={:?}",
                    scores.dims()
                );
                cases += 1;
            }
        }
        assert_eq!(
            cases, 4,
            "encoder and decoder self/cross must all be sparse"
        );
    }
    Ok(())
}
