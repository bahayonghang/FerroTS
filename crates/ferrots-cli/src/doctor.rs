use candle_core::{DType, Device, Tensor, Var};
use ferrots_core::Device as DeviceSpec;
use ferrots_nn::{resolve_device, train_ops};
use ferrots_train::{NamedParams, ParamGroup, StatefulAdamW};
use std::{error::Error, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn stage<T>(label: &str, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    operation().map_err(|error| format!("{label}: {error}").into())
}
pub fn run(spec: DeviceSpec, smoke: bool) -> Result<()> {
    println!("code_sha: {}", env!("FERROTS_CODE_SHA"));
    println!("source_sha256: {}", env!("FERROTS_SOURCE_SHA256"));
    println!("rustc: {}", env!("FERROTS_RUSTC"));
    println!("target: {}", env!("FERROTS_TARGET"));
    println!("build_features: {}", env!("FERROTS_FEATURES"));
    println!("candle: 0.11.0 (31f35b147389700ed2a178ee66a91c3cc25cc80d)");
    println!("requested_device: {spec}");
    let device = stage("device initialization", || Ok(resolve_device(spec)?))?;
    match &device {
        Device::Cpu => println!(
            "cuda_driver_api: n/a\ncuda_runtime: n/a\ngpu_name: n/a\ngpu_compute_capability: n/a\ngpu_memory_bytes: n/a\ndriver_version: n/a"
        ),
        #[cfg(feature = "cuda")]
        Device::Cuda(cuda) => stage("CUDA diagnostics", || {
            let stream = cuda.cuda_stream();
            let context = stream.context();
            context.bind_to_thread()?;
            let driver = cudarc::runtime::result::version::get_driver_version()?;
            let runtime = cudarc::runtime::result::version::get_runtime_version()?;
            let (major, minor) = context.compute_capability()?;
            println!(
                "cuda_driver_api: {}.{} ({driver})",
                driver / 1000,
                driver % 1000 / 10
            );
            println!(
                "cuda_runtime: {}.{} ({runtime})",
                runtime / 1000,
                runtime % 1000 / 10
            );
            println!("gpu_name: {}", context.name()?);
            println!("gpu_compute_capability: {major}.{minor}");
            println!("gpu_memory_bytes: {}", context.total_mem()?);
            // The CUDA API exposes its supported API version, not the display driver release.
            let release = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=driver_version", "--format=csv,noheader"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.lines().next().map(str::to_owned));
            println!(
                "driver_version: {}",
                release.as_deref().unwrap_or("unavailable (nvidia-smi)")
            );
            Ok(())
        })?,
        _ => return Err("unsupported backend".into()),
    }
    if smoke {
        training_smoke(&device)?;
    }
    Ok(())
}
fn training_smoke(device: &Device) -> Result<()> {
    let params = stage("parameter initialization", || {
        Ok(NamedParams::new(vec![
            (
                "linear1.weight".into(),
                Var::from_slice(
                    &[
                        0.1_f32, -0.2, 0.3, 0.4, -0.1, 0.2, 0.5, -0.3, 0.2, -0.4, 0.3, 0.1,
                    ],
                    (4, 3),
                    device,
                )?,
            ),
            (
                "linear1.bias".into(),
                Var::from_slice(&[0.01_f32, -0.02, 0.03, -0.04], 4, device)?,
            ),
            ("norm.gamma".into(), Var::ones(4, DType::F32, device)?),
            (
                "norm.beta".into(),
                Var::from_slice(&[0.02_f32, -0.01, 0.03, 0.01], 4, device)?,
            ),
            (
                "linear2.weight".into(),
                Var::from_slice(
                    &[0.2_f32, -0.4, 0.1, 0.3, -0.2, 0.1, 0.4, -0.3],
                    (2, 4),
                    device,
                )?,
            ),
            ("linear2.bias".into(), Var::zeros(2, DType::F32, device)?),
        ])?)
    })?;
    let names = params.iter().map(|(n, _)| n.to_owned()).collect();
    let mut group = ParamGroup::adam(names, 0.01);
    group.weight_decay = 0.01;
    let mut optimizer = StatefulAdamW::new(params.clone(), vec![group])?;
    let x = Tensor::from_slice(
        &[1.0_f32, 0.3, -0.5, -0.2, 0.8, 0.1, 0.6, -0.7, 0.4],
        (3, 3),
        device,
    )?;
    let target = Tensor::from_slice(&[0.2_f32, -0.1, 0.4, 0.3, -0.2, 0.1], (3, 2), device)?;
    let mask = Tensor::from_slice(&[1_f32, 1., 1., 0., 0., 1.], (3, 2), device)?;
    let p = |name: &str| params.get(name).expect("fixed smoke-test parameter set");
    device.synchronize()?;
    let start = Instant::now();
    let loss = stage("forward", || {
        let h = x
            .matmul(&p("linear1.weight").t()?)?
            .broadcast_add(p("linear1.bias"))?;
        let h = train_ops::layer_norm(&h, p("norm.gamma"), p("norm.beta"), 1e-5)?;
        let h = train_ops::softmax(&h, 1)?;
        let output = h
            .matmul(&p("linear2.weight").t()?)?
            .broadcast_add(p("linear2.bias"))?;
        let loss = output
            .sub(&target)?
            .sqr()?
            .mul(&mask)?
            .sum_all()?
            .div(&mask.sum_all()?)?;
        device.synchronize()?;
        Ok(loss)
    })?;
    println!("forward_ms: {:.3}", start.elapsed().as_secs_f64() * 1000.0);
    let value = loss.to_scalar::<f32>()?;
    if !value.is_finite() {
        return Err("forward: non-finite loss".into());
    }
    println!("loss: {value:.8}");
    let start = Instant::now();
    let gradients = stage("backward", || {
        let gradients = loss.backward()?;
        device.synchronize()?;
        Ok(gradients)
    })?;
    println!("backward_ms: {:.3}", start.elapsed().as_secs_f64() * 1000.0);
    let mut before = Vec::new();
    stage("gradient validation", || {
        for (name, var) in params.iter() {
            let gradient = gradients
                .get(var)
                .ok_or_else(|| format!("missing gradient for {name}"))?;
            let values = gradient.flatten_all()?.to_vec1::<f32>()?;
            if values.iter().any(|v| !v.is_finite()) || !values.iter().any(|&v| v != 0.0) {
                return Err(format!("invalid or zero gradient for {name}").into());
            }
            before.push((name.to_owned(), var.flatten_all()?.to_vec1::<f32>()?));
        }
        Ok(())
    })?;
    device.synchronize()?;
    let start = Instant::now();
    stage("AdamW update", || {
        optimizer.step(&gradients)?;
        device.synchronize()?;
        Ok(())
    })?;
    println!("update_ms: {:.3}", start.elapsed().as_secs_f64() * 1000.0);
    stage("parameter validation", || {
        for (name, old) in before {
            let new = p(&name).flatten_all()?.to_vec1::<f32>()?;
            if new.iter().any(|v| !v.is_finite()) || new == old {
                return Err(format!("parameter did not change or is non-finite: {name}").into());
            }
        }
        Ok(())
    })?;
    println!("parameters_with_gradients: {}", params.iter().count());
    println!("parameters_updated: {}", params.iter().count());
    println!("training_smoke_test: passed");
    Ok(())
}
