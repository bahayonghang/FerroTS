//! Differentiable, linear-storage operations used by Informer.

use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};

/// Inclusive prefix sum along `dim`. CUDA keeps the input and gradients on device.
///
/// A single CUDA thread scans each independent row; this intentionally favors a
/// simple linear-storage baseline over a parallel scan with extra workspace.
pub fn prefix_sum(x: &Tensor, dim: usize) -> Result<Tensor> {
    if x.dtype() != DType::F32 || x.dim(dim)? == 0 || x.elem_count() == 0 {
        candle_core::bail!("prefix_sum requires nonempty FP32 input")
    }
    x.contiguous()?.apply_op1(Scan {
        dim,
        reverse: false,
    })
}

/// PyTorch MaxPool1d(kernel_size=3, stride=2, padding=1) for `[B,C,L]`.
/// Argmax chooses the first equal maximum and gather backward adds overlapping
/// contributions. The saved indices never require a host copy.
pub fn max_pool1d(x: &Tensor) -> Result<Tensor> {
    let (batch, channels, length) = x.dims3()?;
    if x.dtype() != DType::F32 || x.elem_count() == 0 {
        candle_core::bail!("max_pool1d requires nonempty FP32 input")
    }
    let flat = x.contiguous()?.reshape((batch * channels, length))?;
    let indices = flat.apply_op1(PoolIndices)?;
    flat.gather(&indices, 1)?
        .reshape((batch, channels, length.div_ceil(2)))
}

/// Replace selected rows of `[B,H,L,D]` with `[B,H,U,D]` updates.
///
/// `indices` is `[B,H,U]` U32, in range and unique within each batch/head, as
/// guaranteed by the attention top-query selection. Indices are discrete. Old
/// values at overwritten rows receive zero gradient; updates receive the
/// corresponding output cotangent.
pub fn replace_rows(base: &Tensor, updates: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let (batch, heads, _, width) = base.dims4()?;
    let (ub, uh, count, ud) = updates.dims4()?;
    if (ub, uh, ud) != (batch, heads, width)
        || indices.dims() != [batch, heads, count]
        || indices.dtype() != DType::U32
        || base.dtype() != DType::F32
        || updates.dtype() != DType::F32
    {
        candle_core::bail!("replace_rows requires matching FP32 tensors and [B,H,U] U32 indices")
    }
    let indices = indices
        .unsqueeze(3)?
        .broadcast_as(updates.shape())?
        .contiguous()?;
    // Candle 0.11 Scatter backward does not mask overwritten initial values.
    // Mask them explicitly before applying scatter, retaining a correct graph.
    let mask = base
        .ones_like()?
        .scatter(&indices, &updates.zeros_like()?, 2)?;
    base.mul(&mask)?.scatter(&indices, updates, 2)
}

struct Scan {
    dim: usize,
    reverse: bool,
}

impl CustomOp1 for Scan {
    fn name(&self) -> &'static str {
        "informer-prefix-sum"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("scan input must be contiguous".into()))?;
        let input = &storage.as_slice::<f32>()?[start..end];
        let length = layout.dims()[self.dim];
        let inner: usize = layout.dims()[self.dim + 1..].iter().product();
        let mut output = vec![0_f32; input.len()];
        for row in 0..input.len() / length {
            let base = row / inner * length * inner + row % inner;
            let mut sum = 0_f32;
            for offset in 0..length {
                let column = if self.reverse {
                    length - 1 - offset
                } else {
                    offset
                };
                let index = base + column * inner;
                sum += input[index];
                output[index] = sum;
            }
        }
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda::scan(storage, layout, self.dim, self.reverse)
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(grad.contiguous()?.apply_op1(Scan {
            dim: self.dim,
            reverse: !self.reverse,
        })?))
    }
}

struct PoolIndices;

impl CustomOp1 for PoolIndices {
    fn name(&self) -> &'static str {
        "informer-max-pool-indices"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("pool input must be contiguous".into()))?;
        let input = &storage.as_slice::<f32>()?[start..end];
        let (rows, length) = layout.shape().dims2()?;
        let output_length = length.div_ceil(2);
        let mut indices = Vec::with_capacity(rows * output_length);
        for row in 0..rows {
            for column in 0..output_length {
                let first = (column * 2).saturating_sub(1);
                let end = (column * 2 + 2).min(length);
                let mut best = first;
                for candidate in first + 1..end {
                    let value = input[row * length + candidate];
                    if value > input[row * length + best] || value.is_nan() {
                        best = candidate;
                    }
                }
                indices.push(u32::try_from(best).map_err(candle_core::Error::wrap)?);
            }
        }
        Ok((CpuStorage::U32(indices), Shape::from((rows, output_length))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda::pool_indices(storage, layout)
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, _grad: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::cuda_backend::{CudaStorage, WrapErr, cudarc};
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use std::sync::OnceLock;

    const SOURCE: &str = r#"
extern "C" __global__ void scan_f32(const float *x, float *y,
    unsigned long long offset, unsigned long long rows,
    unsigned long long length, unsigned long long inner, int reverse) {
    unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    unsigned long long base = row / inner * length * inner + row % inner;
    float sum = 0.0f;
    for (unsigned long long i = 0; i < length; ++i) {
        unsigned long long column = reverse ? length - 1 - i : i;
        unsigned long long index = base + column * inner;
        sum += x[offset + index];
        y[index] = sum;
    }
}
extern "C" __global__ void pool_indices_f32(const float *x, unsigned int *indices,
    unsigned long long offset, unsigned long long count,
    unsigned long long length, unsigned long long output_length) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned long long row = i / output_length, center = (i % output_length) * 2;
    unsigned long long first = center == 0 ? 0 : center - 1;
    unsigned long long end = center + 2 < length ? center + 2 : length;
    unsigned long long best = first;
    float value = x[offset + row * length + first];
    for (unsigned long long j = first + 1; j < end; ++j) {
        float candidate = x[offset + row * length + j];
        if (candidate > value || isnan(candidate)) { best = j; value = candidate; }
    }
    indices[i] = (unsigned int)best;
}
"#;

    fn ptx() -> Result<&'static str> {
        static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
        PTX.get_or_init(|| {
            cudarc::nvrtc::compile_ptx(SOURCE)
                .map(|ptx| ptx.to_src())
                .map_err(|error| error.to_string())
        })
        .as_deref()
        .map_err(|error| candle_core::Error::Msg(error.clone()))
    }

    pub(super) fn scan(
        storage: &CudaStorage,
        layout: &Layout,
        dim: usize,
        reverse: bool,
    ) -> Result<(CudaStorage, Shape)> {
        let device = &storage.device;
        let input = storage.as_cuda_slice::<f32>()?;
        let count = layout.shape().elem_count();
        let length = layout.dims()[dim] as u64;
        let rows = count as u64 / length;
        let inner = layout.dims()[dim + 1..].iter().product::<usize>() as u64;
        let offset = layout.start_offset() as u64;
        let reverse = i32::from(reverse);
        let function =
            device.get_or_load_custom_func("scan_f32", "ferrots_informer_ops_v1", ptx()?)?;
        // SAFETY: kernel initializes every output element, reads only the
        // validated contiguous input range, and runs on the owning stream.
        let mut output = unsafe { device.alloc::<f32>(count) }?;
        let mut launch = function.builder();
        launch
            .arg(input)
            .arg(&mut output)
            .arg(&offset)
            .arg(&rows)
            .arg(&length)
            .arg(&inner)
            .arg(&reverse);
        let threads = u32::try_from(rows).map_err(candle_core::Error::wrap)?;
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            layout.shape().clone(),
        ))
    }

    pub(super) fn pool_indices(
        storage: &CudaStorage,
        layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, length) = layout.shape().dims2()?;
        u32::try_from(length).map_err(candle_core::Error::wrap)?;
        let output_length = length.div_ceil(2);
        let count = (rows * output_length) as u64;
        let offset = layout.start_offset() as u64;
        let device = &storage.device;
        let input = storage.as_cuda_slice::<f32>()?;
        let function = device.get_or_load_custom_func(
            "pool_indices_f32",
            "ferrots_informer_ops_v1",
            ptx()?,
        )?;
        // SAFETY: each valid output thread writes one in-range index; source
        // bounds and the contiguous layout are established by max_pool1d.
        let mut output = unsafe { device.alloc::<u32>(count as usize) }?;
        let length_arg = length as u64;
        let output_length_arg = output_length as u64;
        let mut launch = function.builder();
        launch
            .arg(input)
            .arg(&mut output)
            .arg(&offset)
            .arg(&count)
            .arg(&length_arg)
            .arg(&output_length_arg);
        let threads = u32::try_from(count).map_err(candle_core::Error::wrap)?;
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((rows, output_length)),
        ))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::cuda_backend::cudarc::driver::{result, sys};
    use candle_core::{Device, Var};

    fn pool_value(
        pool: sys::CUmemoryPool,
        attr: sys::CUmemPool_attribute,
    ) -> std::result::Result<u64, Box<dyn std::error::Error>> {
        let mut bytes = 0_u64;
        // SAFETY: the live device owns this pool; these memory attributes use
        // cuuint64_t, matching the pointed-to value exactly.
        unsafe {
            result::mem_pool::get_attribute(pool, attr, (&mut bytes as *mut u64).cast())?;
        }
        Ok(bytes)
    }

    #[test]
    fn cuda_prefix_operator_forward_and_vjp_peak()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        assert!(
            stream.context().has_async_alloc(),
            "reliable pool accounting is required"
        );
        // SAFETY: the CUDA device and its pool remain alive throughout the test.
        let pool = unsafe { result::device::get_mem_pool(stream.context().cu_device())? };
        {
            let warm = Var::from_vec(vec![1_f32; 16], 16, &device)?;
            let output = prefix_sum(&warm, 0)?;
            let cotangent = Tensor::ones(16, DType::F32, &device)?;
            Scan {
                dim: 0,
                reverse: false,
            }
            .bwd(&warm, &output, &cotangent)?;
        }
        device.synchronize()?;
        let length = 1_000_000;
        let x = Var::from_vec(vec![1_f32; length], length, &device)?;
        let cotangent = Tensor::ones(length, DType::F32, &device)?;
        device.synchronize()?;
        let baseline = pool_value(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
        )?;
        let mut zero = 0_u64;
        // SAFETY: reset only the documented high watermark to zero; this does
        // not modify allocation policy or free any memory.
        unsafe {
            result::mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                (&mut zero as *mut u64).cast(),
            )?;
        }
        let start = std::time::Instant::now();
        let output = prefix_sum(&x, 0)?;
        // This is the exact production callback invoked by Candle autograd,
        // not an independently reimplemented suffix-sum reference.
        let gradient = Scan {
            dim: 0,
            reverse: false,
        }
        .bwd(&x, &output, &cotangent)?
        .expect("prefix gradient");
        device.synchronize()?;
        let peak = pool_value(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        )?;
        let delta = peak.checked_sub(baseline).expect("peak below baseline");
        let input_bytes = (length * std::mem::size_of::<f32>()) as u64;
        println!(
            "PREFIX_MEMORY measurement_scope=prefix_sum_operator_forward_plus_vjp n={length} dtype=f32 baseline={baseline} peak={peak} delta={delta} input_bytes={input_bytes} ratio={} seconds={:.6} source=CUDA_POOL_USED_MEM_HIGH includes=output+returned_gradient+operator_workspace excludes=preallocated_input+cotangent+outer_loss_graph+GradStore",
            delta as f64 / input_bytes as f64,
            start.elapsed().as_secs_f64()
        );
        assert!(delta > 0, "pool accounting failed to observe the output");
        assert!(delta <= 4 * input_bytes, "AC2 operator peak exceeded");
        // Keep both return values alive through the peak reading. Read values
        // afterwards so host validation allocations cannot affect the measure.
        assert_eq!(
            output.narrow(0, length - 1, 1)?.to_vec1::<f32>()?,
            [length as f32]
        );
        assert_eq!(gradient.narrow(0, 0, 1)?.to_vec1::<f32>()?, [length as f32]);
        Ok(())
    }
}
