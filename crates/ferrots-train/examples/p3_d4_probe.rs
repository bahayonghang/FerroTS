//! Diagnostic only. Reads fixtures or frozen checkpoints and writes nothing
//! back to a suite directory.
use candle_core::{DType, Device, Tensor, backprop::GradStore};
use ferrots_core::{
    config::{Activation, Config, ModelSpec, Overrides, WindowSpec},
    metrics::MetricGrid,
};
use ferrots_data::batch::Dataset;
use ferrots_nn::{
    Mode,
    informer::{Informer, InformerSamplingSchedule},
    model::{ForecastModel, ForwardContext},
};
use ferrots_train::{
    checkpoint,
    evaluation::{self, EvaluationOptions},
    loss::masked_mse,
    session::{Progress, TrainingSession},
    trainer::{SplitAccess, select_targets, tracked_batch},
};
use std::{collections::BTreeMap, env, path::Path, process::ExitCode};

#[path = "../../ferrots-nn/src/cuda_philox.rs"]
#[allow(dead_code)] // Diagnostic reuses the exact generator without exposing a production API.
mod probe_philox;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn production(dropout: f64) -> (ModelSpec, WindowSpec) {
    (
        ModelSpec::Informer {
            d_model: 512,
            n_heads: 8,
            d_ff: 2048,
            e_layers: 2,
            d_layers: 1,
            factor: 3,
            distil: true,
            label_len: 48,
            dropout,
            activation: Activation::Gelu,
        },
        WindowSpec {
            lookback: 96,
            horizon: 720,
            gap: 0,
            stride: 1,
        },
    )
}

fn cuda_device() -> R<Device> {
    let index: usize = env::var("FERROTS_PROBE_DEVICE")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    let device = Device::new_cuda(index)?;
    let marker = Tensor::new(&[1f32], &device)?;
    println!(
        "probe_device index={index} marker_device={:?} candle_gemm_reduced_precision_f32={}",
        marker.device(),
        candle_core::cuda::gemm_reduced_precision_f32()
    );
    if !marker.device().is_cuda() {
        return Err("probe marker is not on CUDA".into());
    }
    Ok(device)
}

fn load_map(path: &Path, device: &Device) -> R<BTreeMap<String, Tensor>> {
    Ok(candle_core::safetensors::load(path, device)?
        .into_iter()
        .collect())
}

fn subset(all: &BTreeMap<String, Tensor>, prefix: &str) -> BTreeMap<String, Tensor> {
    all.iter()
        .filter_map(|(key, value)| {
            key.strip_prefix(prefix)
                .map(|key| (key.to_owned(), value.clone()))
        })
        .collect()
}

fn diff_line(name: &str, actual: &Tensor, expected: &Tensor) -> R<()> {
    if actual.dims() != expected.dims() || actual.dtype() != expected.dtype() {
        println!(
            "DIFF {name} schema actual={:?}/{:?} expected={:?}/{:?}",
            actual.dims(),
            actual.dtype(),
            expected.dims(),
            expected.dtype()
        );
        return Ok(());
    }
    if actual.dtype() == DType::U32 {
        let a = actual.flatten_all()?.to_vec1::<u32>()?;
        let b = expected.flatten_all()?.to_vec1::<u32>()?;
        let mismatches = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        println!(
            "DIFF {name} dtype=u32 mismatches={mismatches} n={}",
            a.len()
        );
        return Ok(());
    }
    let a = actual.flatten_all()?.to_vec1::<f32>()?;
    let b = expected.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut exceed = 0usize;
    let mut nonfinite = 0usize;
    for (x, y) in a.iter().zip(&b) {
        if !x.is_finite() || !y.is_finite() {
            nonfinite += 1;
            continue;
        }
        let abs = (x - y).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / y.abs().max(1e-12));
        if abs > 1e-5 + 1e-4 * y.abs() {
            exceed += 1;
        }
    }
    println!(
        "DIFF {name} n={} max_abs={max_abs:.8e} max_rel={max_rel:.8e} exceed_1e-5_1e-4={exceed} nonfinite={nonfinite}",
        a.len()
    );
    Ok(())
}

fn compare(path: &Path, device: &Device) -> R<()> {
    let values = load_map(path, device)?;
    let (spec, window) = production(0.0);
    let model = Informer::new(&spec, &window, 7, 4, 2021, device)?;
    let weights = subset(&values, "initial.");
    let buffers = subset(&values, "buffer.initial.");
    let samples = subset(&values, "sample.");
    let rust_names: BTreeMap<_, _> = model
        .weights()
        .iter()
        .map(|(k, v)| (k.clone(), v.dims().to_vec()))
        .collect();
    if rust_names.len() != weights.len() || !rust_names.keys().eq(weights.keys()) {
        println!(
            "NAME_MISMATCH rust={} torch={}",
            rust_names.len(),
            weights.len()
        );
        for name in rust_names.keys() {
            if !weights.contains_key(name) {
                println!("MISSING_TORCH {name}");
            }
        }
        for name in weights.keys() {
            if !rust_names.contains_key(name) {
                println!("MISSING_RUST {name}");
            }
        }
        return Ok(());
    }
    model.load_weights(&weights)?;
    model.load_buffers(&buffers)?;
    let input = ferrots_nn::model::ModelInput {
        history: values["input.history"].clone(),
        history_time: Some(values["input.history_time"].clone()),
        future_time: Some(values["input.future_time"].clone()),
    };
    let output = model.forward_with_samples(
        &input,
        ForwardContext {
            mode: Mode::Train,
            forward_id: 0,
        },
        &samples,
    )?;
    diff_line("output", &output, &values["output"])?;
    let loss = output.sub(&values["input.target"])?.sqr()?.mean_all()?;
    diff_line("loss", &loss, &values["loss"])?;
    let gradients = loss.backward()?;
    for (name, variable) in model.named_parameters() {
        match gradients.get(&variable) {
            Some(gradient) => diff_line(
                &format!("gradient.{name}"),
                gradient,
                &values[&format!("gradient.{name}")],
            )?,
            None => println!("DIFF gradient.{name} missing"),
        }
    }
    let after = model.buffers()?;
    for (name, expected) in subset(&values, "buffer.after.") {
        match after.get(&name) {
            Some(actual) => diff_line(&format!("buffer.{name}"), actual, &expected)?,
            None => println!("DIFF buffer.{name} missing"),
        }
    }
    println!("COMPARE_DONE {}", path.display());
    Ok(())
}

fn init_diff(path: &Path, device: &Device) -> R<()> {
    let values = load_map(path, device)?;
    let weights = subset(&values, "initial.");
    let (spec, window) = production(0.1);
    let model = Informer::new(&spec, &window, 7, 4, 2021, device)?;
    println!("init_seed=2021 dropout=0.1 note=independent streams, not a claimed match");
    for (name, variable) in model.named_parameters() {
        match weights.get(&name) {
            Some(expected) => diff_line(&format!("init.{name}"), variable.as_tensor(), expected)?,
            None => println!("DIFF init.{name} missing_from_torch"),
        }
    }
    Ok(())
}

fn dataset_for(loaded: &checkpoint::Checkpoint) -> R<Dataset> {
    let data = Dataset::load(
        &loaded.config.data,
        &loaded.config.window,
        Some(loaded.scaler.clone()),
    )?;
    if data.cache.source_sha256 != loaded.manifest.data_sha256 {
        return Err("dataset fingerprint differs from checkpoint".into());
    }
    Ok(data)
}

fn eval_identity(
    loaded: &checkpoint::Checkpoint,
    data: &Dataset,
    device: &Device,
    split: u8,
    identity: u64,
    policy: &str,
) -> R<(f64, f64, u64)> {
    let batch_size = loaded.config.train.batch_size;
    let mut metrics = MetricGrid::new(data.spec.target_columns.clone(), data.window.horizon)?;
    let mut access = SplitAccess::default();
    let c = data.target_indices.len();
    for (batch_id, chunk) in data.index.order(split).chunks(batch_size).enumerate() {
        let first = match policy {
            "evaluation" => (2u64 << 62) | ((split as u64) << 60) | (identity << 56),
            "validation" => (1u64 << 62) | (identity << 56),
            other => return Err(format!("unknown policy {other}").into()),
        };
        let host = tracked_batch(data, split, chunk, &mut access)?;
        let batch = ferrots_train::trainer::DeviceBatch::from_host(&host, device)?;
        let pred = select_targets(
            &loaded.model.forward_with_context(
                &batch.input,
                ForwardContext::evaluation_with_id(first | batch_id as u64),
            )?,
            &data.target_indices,
        )?;
        let values = pred.flatten_all()?.to_vec1::<f32>()?;
        for (b, origin) in host.origins.iter().enumerate() {
            for h in 0..data.window.horizon {
                let row = origin.row + data.window.gap + h;
                for (j, &col) in data.target_indices.iter().enumerate() {
                    let k = (b * data.window.horizon + h) * c + j;
                    let valid = host.supervision.valid[k] != 0;
                    let prediction = f64::from(values[k]);
                    let target = f64::from(host.supervision.targets[k]);
                    metrics.add(
                        h,
                        j,
                        (prediction, target),
                        (
                            data.scaler.inverse(col, prediction),
                            data.series.values[row * data.series.width() + col],
                        ),
                        valid,
                    )?;
                }
            }
        }
    }
    let report = metrics.finish();
    let overall = &report.standardized.overall;
    Ok((
        overall.mse.ok_or_else(|| "empty mse".to_string())?,
        overall.mae.ok_or_else(|| "empty mae".to_string())?,
        overall.count,
    ))
}

fn variance(path: &Path, device: &Device) -> R<()> {
    let loaded = checkpoint::load(path, device)?;
    if !loaded.model.is_frozen() {
        return Err("checkpoint model is not frozen".into());
    }
    let data = dataset_for(&loaded)?;
    let canonical = evaluation::evaluate(
        &loaded,
        &data,
        device,
        &EvaluationOptions {
            run_id: "p3-d4-probe",
            split: 2,
            batch_size: loaded.config.train.batch_size,
            export: None,
            columns: &[],
        },
    )?;
    let official = &canonical.metrics.standardized.overall;
    println!(
        "CANONICAL_EVAL mse={:.12} mae={:.12} count={} first_forward_id={}",
        official.mse.unwrap_or(f64::NAN),
        official.mae.unwrap_or(f64::NAN),
        official.count,
        canonical.first_forward_id
    );
    for identity in 0..5u64 {
        let (mse, mae, count) = eval_identity(&loaded, &data, device, 2, identity, "evaluation")?;
        println!(
            "IDENTITY policy=evaluation split=2 id={identity} mse={mse:.12} mae={mae:.12} count={count}"
        );
    }
    Ok(())
}

fn rank(run: &Path, device: &Device) -> R<()> {
    let mut bundles = Vec::new();
    for entry in std::fs::read_dir(run.join("checkpoints"))? {
        let path = entry?.path();
        if path.join("manifest.json").is_file() {
            bundles.push(path);
        }
    }
    bundles.sort();
    let mut seen = BTreeMap::<(usize, String), std::path::PathBuf>::new();
    for path in bundles {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json"))?)?;
        let epoch = manifest["epoch"].as_u64().ok_or("epoch missing")? as usize;
        let loss = manifest["validation_loss"]
            .as_f64()
            .ok_or("validation_loss missing")?;
        let bytes = std::fs::read(path.join("model.safetensors"))?;
        let digest = format!("{:x}", md5_fallback(&bytes));
        println!(
            "CHECKPOINT epoch={epoch} stored_val={loss:.12} bytes={} path={}",
            bytes.len(),
            path.display()
        );
        seen.entry((epoch, digest)).or_insert(path);
    }
    let mut data_cache: Option<(String, Dataset)> = None;
    for ((epoch, _), path) in &seen {
        let loaded = checkpoint::load(path, device)?;
        let key = loaded.manifest.data_sha256.clone();
        if data_cache.as_ref().is_none_or(|(k, _)| k != &key) {
            data_cache = Some((key, dataset_for(&loaded)?));
        }
        let data = &data_cache.as_ref().unwrap().1;
        let (val_mse, val_mae, val_count) =
            eval_identity(&loaded, data, device, 1, 0, "validation")?;
        let (test_mse, test_mae, test_count) =
            eval_identity(&loaded, data, device, 2, 0, "evaluation")?;
        println!(
            "RANK epoch={epoch} fresh_val_mse={val_mse:.12} fresh_val_mae={val_mae:.12} val_count={val_count} test_mse={test_mse:.12} test_mae={test_mae:.12} test_count={test_count}"
        );
    }
    Ok(())
}

struct DualValidation {
    weighted: f64,
    equal: f64,
    batches: usize,
    tail_windows: usize,
    tail_mean: f64,
    head_mean: f64,
}

/// Label-weighted sum/count and TSLib's unweighted mean of per-batch means.
/// Forward ids follow `validation_loss`: epoch 1 starts at 0 and each later
/// epoch continues through `windows.div_ceil(batch_size)` batches.
fn dual_validation(
    loaded: &checkpoint::Checkpoint,
    data: &Dataset,
    device: &Device,
    id_offset: usize,
) -> R<DualValidation> {
    let batch_size = loaded.config.train.batch_size;
    let model = loaded.model.freeze_for_eval()?;
    let mut access = SplitAccess::default();
    access.batches[1] = id_offset;
    let mut sum = 0f64;
    let mut count = 0u64;
    let mut mean_sum = 0f64;
    let mut batches = 0usize;
    let mut tail_windows = 0usize;
    let mut tail_mean = 0f64;
    let mut head_mean = None;
    for chunk in data.index.order(1).chunks(batch_size) {
        let context = ForwardContext::evaluation_with_id((1u64 << 62) | access.batches[1] as u64);
        let host = tracked_batch(data, 1, chunk, &mut access)?;
        let batch = ferrots_train::trainer::DeviceBatch::from_host(&host, device)?;
        let pred = select_targets(
            &model.forward_with_context(&batch.input, context)?,
            &data.target_indices,
        )?;
        if pred.track_op() {
            return Err("validation output tracks a graph".into());
        }
        let loss = masked_mse(&pred, &batch.supervision.target, &batch.supervision.mask)?;
        if loss.valid_count == 0 {
            return Err("validation batch has no valid labels".into());
        }
        let batch_sum = loss.sum.to_scalar::<f32>()?;
        if !batch_sum.is_finite() {
            return Err("non-finite batch loss".into());
        }
        sum += batch_sum as f64;
        count += loss.valid_count;
        let batch_mean = (batch_sum / loss.valid_count as f32) as f64;
        mean_sum += batch_mean;
        batches += 1;
        if head_mean.is_none() {
            head_mean = Some(batch_mean);
        }
        tail_windows = host.batch_size;
        tail_mean = batch_mean;
    }
    if count == 0 || batches == 0 {
        return Err("validation has no valid labels".into());
    }
    Ok(DualValidation {
        weighted: sum / count as f64,
        equal: mean_sum / batches as f64,
        batches,
        tail_windows,
        tail_mean,
        head_mean: head_mean.ok_or("validation produced no batch mean")?,
    })
}

fn first_min(rows: &[(usize, DualValidation)], equal: bool) -> (usize, f64) {
    let mut best_epoch = rows[0].0;
    let mut best_value = if equal {
        rows[0].1.equal
    } else {
        rows[0].1.weighted
    };
    for (epoch, row) in rows.iter().skip(1) {
        let value = if equal { row.equal } else { row.weighted };
        if value < best_value {
            best_epoch = *epoch;
            best_value = value;
        }
    }
    (best_epoch, best_value)
}

fn agg(run: &Path, device: &Device) -> R<()> {
    let mut bundles = Vec::new();
    for entry in std::fs::read_dir(run.join("checkpoints"))? {
        let path = entry?.path();
        if path.join("manifest.json").is_file() {
            bundles.push(path);
        }
    }
    bundles.sort();
    if bundles.is_empty() {
        return Err("no checkpoint generations".into());
    }
    let mut seen = BTreeMap::<usize, String>::new();
    let mut data_cache: Option<(String, Dataset)> = None;
    let mut rows = Vec::new();
    for path in bundles {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json"))?)?;
        let epoch = manifest["epoch"].as_u64().ok_or("epoch missing")? as usize;
        let stored = manifest["validation_loss"]
            .as_f64()
            .ok_or("validation_loss missing")?;
        let bytes = std::fs::read(path.join("model.safetensors"))?;
        let digest = format!("{:x}", md5_fallback(&bytes));
        if let Some(prior) = seen.get(&epoch) {
            if prior != &digest {
                return Err(format!("epoch {epoch} has two different weight digests").into());
            }
            continue;
        }
        seen.insert(epoch, digest);
        let loaded = checkpoint::load(&path, device)?;
        let key = loaded.manifest.data_sha256.clone();
        if data_cache.as_ref().is_none_or(|(cached, _)| cached != &key) {
            data_cache = Some((key, dataset_for(&loaded)?));
        }
        let data = &data_cache.as_ref().unwrap().1;
        let windows = data.index.counts[1];
        let n_batches = windows.div_ceil(loaded.config.train.batch_size);
        if epoch == 0 || n_batches == 0 {
            return Err("aggregation probe expects a positive training epoch".into());
        }
        let id_offset = (epoch - 1) * n_batches;
        let measured = dual_validation(&loaded, data, device, id_offset)?;
        println!(
            "AGG epoch={epoch} stored_val={stored:.12} weighted={:.12} equal={:.12} batches={} tail_windows={} tail_mean={:.12} head_mean={:.12} weighted_minus_equal={:.12} id_offset={id_offset} windows={windows}",
            measured.weighted,
            measured.equal,
            measured.batches,
            measured.tail_windows,
            measured.tail_mean,
            measured.head_mean,
            measured.weighted - measured.equal,
        );
        rows.push((epoch, measured));
    }
    rows.sort_by_key(|(epoch, _)| *epoch);
    if rows.is_empty() {
        return Err("no scored checkpoints".into());
    }
    let (weighted_epoch, weighted_value) = first_min(&rows, false);
    let (equal_epoch, equal_value) = first_min(&rows, true);
    println!(
        "ARGMIN weighted_epoch={weighted_epoch} weighted={weighted_value:.12} equal_epoch={equal_epoch} equal={equal_value:.12} rank_flip={}",
        weighted_epoch != equal_epoch
    );
    if let (Some((_, first)), Some((_, third))) = (
        rows.iter().find(|(epoch, _)| *epoch == 1),
        rows.iter().find(|(epoch, _)| *epoch == 3),
    ) {
        println!(
            "COMPARE epoch3_weighted_higher={} epoch3_equal_higher={} pair_flip={}",
            third.weighted > first.weighted,
            third.equal > first.equal,
            (third.weighted > first.weighted) != (third.equal > first.equal)
        );
    }
    Ok(())
}

struct WindowBound {
    weighted: f64,
    equal: f64,
    windows: usize,
    batches: usize,
    batch_size: usize,
    tail_windows: usize,
    tail_mean: f64,
    count_min: u64,
    count_max: u64,
    mean_min: f64,
    mean_max: f64,
    split_gap: f32,
    tail_window_gap: f64,
}

/// Fixed-order per-window MSE. With a one-window tail and equal label counts,
/// `E(t) = (n * W + (batch_size - 1) * t) / (batch_size * n_batches)`.
/// Predictions stay tied to these forward ids.
fn equal_at_tail(windows: usize, batch_size: usize, weighted: f64, tail: f64) -> f64 {
    let n = windows as f64;
    let width = batch_size as f64;
    let n_batches = windows.div_ceil(batch_size) as f64;
    (n * weighted + (width - 1.0) * tail) / (width * n_batches)
}

fn per_window(pred: &Tensor, target: &Tensor, mask: &Tensor) -> R<(Vec<f32>, Vec<f32>)> {
    if pred.dims() != target.dims() || mask.dims() != target.dims() || pred.rank() != 3 {
        return Err(format!("window loss shape {:?} mask {:?}", pred.dims(), mask.dims()).into());
    }
    let mask_f = mask.to_dtype(DType::F32)?;
    let sums = pred.sub(target)?.sqr()?.mul(&mask_f)?.sum(2)?.sum(1)?;
    let counts = mask_f.sum(2)?.sum(1)?;
    Ok((sums.to_vec1()?, counts.to_vec1()?))
}

fn window_bound(
    loaded: &checkpoint::Checkpoint,
    data: &Dataset,
    device: &Device,
    id_offset: usize,
) -> R<WindowBound> {
    let batch_size = loaded.config.train.batch_size;
    let model = loaded.model.freeze_for_eval()?;
    let mut access = SplitAccess::default();
    access.batches[1] = id_offset;
    let mut sum = 0f64;
    let mut count = 0u64;
    let mut mean_sum = 0f64;
    let mut batches = 0usize;
    let mut windows = 0usize;
    let mut tail_windows = 0usize;
    let mut tail_mean = 0f64;
    let mut tail_window_mean = None;
    let mut count_min = u64::MAX;
    let mut count_max = 0u64;
    let mut mean_min = f64::INFINITY;
    let mut mean_max = f64::NEG_INFINITY;
    let mut split_gap = 0f32;
    for chunk in data.index.order(1).chunks(batch_size) {
        let context = ForwardContext::evaluation_with_id((1u64 << 62) | access.batches[1] as u64);
        let host = tracked_batch(data, 1, chunk, &mut access)?;
        let batch = ferrots_train::trainer::DeviceBatch::from_host(&host, device)?;
        let pred = select_targets(
            &model.forward_with_context(&batch.input, context)?,
            &data.target_indices,
        )?;
        if pred.track_op() {
            return Err("validation output tracks a graph".into());
        }
        let loss = masked_mse(&pred, &batch.supervision.target, &batch.supervision.mask)?;
        if loss.valid_count == 0 {
            return Err("validation batch has no valid labels".into());
        }
        let batch_sum = loss.sum.to_scalar::<f32>()?;
        if !batch_sum.is_finite() {
            return Err("non-finite batch loss".into());
        }
        let (sums, counts) = per_window(&pred, &batch.supervision.target, &batch.supervision.mask)?;
        if sums.len() != host.batch_size || counts.len() != host.batch_size {
            return Err("per-window loss length does not match the batch".into());
        }
        let mut piece = 0f32;
        let mut piece_count = 0u64;
        let mut last_mean = 0f64;
        for (window_sum, window_count) in sums.iter().zip(&counts) {
            if !window_sum.is_finite() || !window_count.is_finite() || *window_count <= 0.0 {
                return Err("non-finite or empty window loss".into());
            }
            let count_u = *window_count as u64;
            if (*window_count - count_u as f32).abs() != 0.0 {
                return Err("window valid count is not an integer".into());
            }
            count_min = count_min.min(count_u);
            count_max = count_max.max(count_u);
            last_mean = f64::from(window_sum / window_count);
            mean_min = mean_min.min(last_mean);
            mean_max = mean_max.max(last_mean);
            piece += window_sum;
            piece_count += count_u;
            windows += 1;
        }
        if piece_count != loss.valid_count {
            return Err(format!(
                "window counts {piece_count} disagree with batch count {}",
                loss.valid_count
            )
            .into());
        }
        split_gap = split_gap.max((piece - batch_sum).abs());
        sum += batch_sum as f64;
        count += loss.valid_count;
        let batch_mean = (batch_sum / loss.valid_count as f32) as f64;
        mean_sum += batch_mean;
        batches += 1;
        tail_windows = host.batch_size;
        tail_mean = batch_mean;
        tail_window_mean = Some(last_mean);
    }
    if count == 0 || batches == 0 || windows == 0 {
        return Err("validation has no valid labels".into());
    }
    let tail_window_gap = if tail_windows == 1 {
        (tail_window_mean.ok_or("missing tail window")? - tail_mean).abs()
    } else {
        0.0
    };
    Ok(WindowBound {
        weighted: sum / count as f64,
        equal: mean_sum / batches as f64,
        windows,
        batches,
        batch_size,
        tail_windows,
        tail_mean,
        count_min,
        count_max,
        mean_min,
        mean_max,
        split_gap,
        tail_window_gap,
    })
}

fn bounds(run: &Path, device: &Device) -> R<()> {
    let mut bundles = Vec::new();
    for entry in std::fs::read_dir(run.join("checkpoints"))? {
        let path = entry?.path();
        if path.join("manifest.json").is_file() {
            bundles.push(path);
        }
    }
    bundles.sort();
    if bundles.is_empty() {
        return Err("no checkpoint generations".into());
    }
    let mut seen = BTreeMap::<usize, String>::new();
    let mut data_cache: Option<(String, Dataset)> = None;
    let mut rows = Vec::new();
    for path in bundles {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json"))?)?;
        let epoch = manifest["epoch"].as_u64().ok_or("epoch missing")? as usize;
        let stored = manifest["validation_loss"]
            .as_f64()
            .ok_or("validation_loss missing")?;
        let bytes = std::fs::read(path.join("model.safetensors"))?;
        let digest = format!("{:x}", md5_fallback(&bytes));
        if let Some(prior) = seen.get(&epoch) {
            if prior != &digest {
                return Err(format!("epoch {epoch} has two different weight digests").into());
            }
            continue;
        }
        seen.insert(epoch, digest);
        let loaded = checkpoint::load(&path, device)?;
        let key = loaded.manifest.data_sha256.clone();
        if data_cache.as_ref().is_none_or(|(cached, _)| cached != &key) {
            data_cache = Some((key, dataset_for(&loaded)?));
        }
        let data = &data_cache.as_ref().unwrap().1;
        let n_batches = data.index.counts[1].div_ceil(loaded.config.train.batch_size);
        if epoch == 0 || n_batches == 0 {
            return Err("bounds probe expects a positive training epoch".into());
        }
        let id_offset = (epoch - 1) * n_batches;
        let measured = window_bound(&loaded, data, device, id_offset)?;
        if (measured.weighted - stored).abs() > 1e-8 {
            return Err(format!(
                "epoch {epoch} weighted {:.12} does not match stored {stored:.12}",
                measured.weighted
            )
            .into());
        }
        if measured.split_gap > 1e-2 || measured.tail_window_gap > 1e-5 {
            return Err(format!(
                "epoch {epoch} window reduction disagrees: split_gap={} tail_gap={}",
                measured.split_gap, measured.tail_window_gap
            )
            .into());
        }
        let uniform = measured.count_min == measured.count_max;
        let remainder_one = measured.tail_windows == 1;
        let (at_min, at_max, formula_gap) = if uniform && remainder_one {
            let at_min = equal_at_tail(
                measured.windows,
                measured.batch_size,
                measured.weighted,
                measured.mean_min,
            );
            let at_max = equal_at_tail(
                measured.windows,
                measured.batch_size,
                measured.weighted,
                measured.mean_max,
            );
            let formula_gap = (equal_at_tail(
                measured.windows,
                measured.batch_size,
                measured.weighted,
                measured.tail_mean,
            ) - measured.equal)
                .abs();
            (at_min, at_max, formula_gap)
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        if formula_gap.is_finite() && formula_gap > 1e-6 {
            return Err(format!(
                "epoch {epoch} equal-weight formula gap {formula_gap:.3e} exceeds 1e-6"
            )
            .into());
        }
        println!(
            "BOUND epoch={epoch} stored_val={stored:.12} weighted={:.12} equal={:.12} windows={} batches={} batch_size={} tail_windows={} count_min={} count_max={} mean_min={:.12} mean_max={:.12} e_at_min={at_min:.12} e_at_max={at_max:.12} formula_gap={formula_gap:.3e} split_gap={:.3e} tail_window_gap={:.3e} id_offset={id_offset} uniform={uniform} remainder_one={remainder_one}",
            measured.weighted,
            measured.equal,
            measured.windows,
            measured.batches,
            measured.batch_size,
            measured.tail_windows,
            measured.count_min,
            measured.count_max,
            measured.mean_min,
            measured.mean_max,
            measured.split_gap,
            measured.tail_window_gap,
        );
        rows.push((epoch, measured, at_min, at_max, uniform, remainder_one));
    }
    let Some(row3) = rows.iter().find(|(epoch, ..)| *epoch == 3) else {
        return Err("epoch 3 checkpoint missing".into());
    };
    let Some(row1) = rows.iter().find(|(epoch, ..)| *epoch == 1) else {
        return Err("epoch 1 checkpoint missing".into());
    };
    let (e3_min, uniform3, rem3) = (row3.2, row3.4, row3.5);
    let (e1_max, uniform1, rem1) = (row1.3, row1.4, row1.5);
    if !(uniform1 && uniform3 && rem1 && rem3) {
        println!("SHUFFLE can_flip=unknown reason=count_or_remainder");
        return Ok(());
    }
    let gap = e3_min - e1_max;
    println!(
        "SHUFFLE e3_at_min={e3_min:.12} e1_at_max={e1_max:.12} e3_at_min_minus_e1_at_max={gap:.12} can_flip={}",
        e3_min < e1_max
    );
    Ok(())
}

fn max_abs(actual: &Tensor, expected: &Tensor) -> R<f32> {
    if actual.dims() != expected.dims() || actual.dtype() != expected.dtype() {
        return Err(format!(
            "schema {:?} {:?} vs {:?} {:?}",
            actual.dims(),
            actual.dtype(),
            expected.dims(),
            expected.dtype()
        )
        .into());
    }
    let left = actual.flatten_all()?.to_vec1::<f32>()?;
    let right = expected.flatten_all()?.to_vec1::<f32>()?;
    Ok(left
        .iter()
        .zip(&right)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max))
}

fn init_gap(path: &Path, device: &Device) -> R<()> {
    let seed: u64 = env::var("FERROTS_INIT_SEED")
        .unwrap_or_else(|_| "2023".into())
        .parse()?;
    let values = load_map(path, device)?;
    let (spec, window) = production(0.1);
    let model = Informer::new(&spec, &window, 7, 4, seed, device)?;
    let mut rows = Vec::new();
    for (name, tensor) in model.weights() {
        let Some(expected) = values.get(name) else {
            println!("INIT_MISSING_TORCH {name}");
            continue;
        };
        let gap = max_abs(tensor, expected)?;
        rows.push((gap, name.clone()));
    }
    let buffers = model.buffers()?;
    for (name, tensor) in &buffers {
        if tensor.dtype() != DType::F32 {
            println!(
                "INIT_BUFFER {name} dtype={:?} skipped_float_gap",
                tensor.dtype()
            );
            continue;
        }
        let Some(expected) = values.get(name) else {
            println!("INIT_MISSING_TORCH {name}");
            continue;
        };
        let gap = max_abs(tensor, expected)?;
        rows.push((gap, name.clone()));
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (gap, name) in rows.iter().take(8) {
        println!("INIT_GAP {name} max_abs={gap:.8e}");
    }
    let worst = rows.first().map(|(gap, _)| *gap).unwrap_or(0.0);
    println!(
        "INIT_SUMMARY seed={seed} tensors={} worst={worst:.8e}",
        rows.len()
    );
    Ok(())
}

fn md5_fallback(bytes: &[u8]) -> u64 {
    // Short identity for de-duplicating identical weight files in one process.
    // Not a security hash and not compared with suite SHA-256 receipts.
    bytes.iter().fold(0xcbf29ce484222325u64, |acc, b| {
        (acc ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}

fn score_split(
    model: &ForecastModel,
    data: &Dataset,
    batch_size: usize,
    device: &Device,
    split: u8,
) -> R<(f64, f64, u64)> {
    let frozen = model.freeze_for_eval()?;
    let mut metrics = MetricGrid::new(data.spec.target_columns.clone(), data.window.horizon)?;
    let mut access = SplitAccess::default();
    let c = data.target_indices.len();
    let first = (2u64 << 62) | ((split as u64) << 60);
    for (batch_id, chunk) in data.index.order(split).chunks(batch_size).enumerate() {
        let host = tracked_batch(data, split, chunk, &mut access)?;
        let batch = ferrots_train::trainer::DeviceBatch::from_host(&host, device)?;
        let pred = select_targets(
            &frozen.forward_with_context(
                &batch.input,
                ForwardContext::evaluation_with_id(first | batch_id as u64),
            )?,
            &data.target_indices,
        )?;
        let values = pred.flatten_all()?.to_vec1::<f32>()?;
        for (b, origin) in host.origins.iter().enumerate() {
            for h in 0..data.window.horizon {
                let row = origin.row + data.window.gap + h;
                for (j, &col) in data.target_indices.iter().enumerate() {
                    let k = (b * data.window.horizon + h) * c + j;
                    let valid = host.supervision.valid[k] != 0;
                    let prediction = f64::from(values[k]);
                    let target = f64::from(host.supervision.targets[k]);
                    metrics.add(
                        h,
                        j,
                        (prediction, target),
                        (
                            data.scaler.inverse(col, prediction),
                            data.series.values[row * data.series.width() + col],
                        ),
                        valid,
                    )?;
                }
            }
        }
    }
    let report = metrics.finish();
    let overall = &report.standardized.overall;
    Ok((
        overall.mse.ok_or_else(|| "empty mse".to_string())?,
        overall.mae.ok_or_else(|| "empty mae".to_string())?,
        overall.count,
    ))
}

fn batch0(config_path: &Path, device: &Device) -> R<()> {
    let config = Config::load(config_path, Overrides::default())?;
    let data = Dataset::load(&config.data, &config.window, None)?;
    let session = TrainingSession::new(&config, &data, device)?;
    let entries = session.next_entries().to_vec();
    let shown = entries.len().min(4);
    println!(
        "BATCH0_ORDER head={:?} batch={} seed={}",
        &entries[..shown],
        entries.len(),
        config.train.seed
    );
    let mut access = SplitAccess::default();
    let host = tracked_batch(&data, 0, &entries, &mut access)?;
    let batch = ferrots_train::trainer::DeviceBatch::from_host(&host, device)?;
    let history = batch.input.history.flatten_all()?.to_vec1::<f32>()?;
    let target = batch.supervision.target.flatten_all()?.to_vec1::<f32>()?;
    let history_mean = history.iter().map(|v| f64::from(*v)).sum::<f64>() / history.len() as f64;
    let target_mean = target.iter().map(|v| f64::from(*v)).sum::<f64>() / target.len() as f64;
    let time_mean = match &batch.input.history_time {
        Some(time) => {
            let values = time.flatten_all()?.to_vec1::<f32>()?;
            values.iter().map(|v| f64::from(*v)).sum::<f64>() / values.len() as f64
        }
        None => f64::NAN,
    };
    let eval_mode = env::var("FERROTS_BATCH0_EVAL").ok().as_deref() == Some("1");
    let context = ForwardContext {
        mode: if eval_mode { Mode::Eval } else { Mode::Train },
        forward_id: 0,
    };
    let injected = match env::var("FERROTS_BATCH0_SAMPLES") {
        Ok(path) => Some(load_map(Path::new(&path), device)?),
        Err(_) => None,
    };
    let owned;
    let samples = if let Some(injected) = &injected {
        Some(injected)
    } else if let ModelSpec::Informer {
        factor,
        e_layers,
        d_layers,
        distil,
        label_len,
        ..
    } = &config.model
    {
        let schedule = InformerSamplingSchedule {
            lookback: config.window.lookback,
            label_len: *label_len,
            horizon: config.window.horizon,
            factor: *factor,
            e_layers: *e_layers,
            d_layers: *d_layers,
            distil: *distil,
            batch_size: config.train.batch_size,
            train_windows: data.index.counts[0],
            val_windows: data.index.counts[1],
            test_windows: data.index.counts[2],
            train_drop_last: config.train.drop_last,
        };
        let mut rng = schedule.rng_before_train_batch(session.model.sampling_rng(), 0, 0);
        owned = schedule.draw_train_indices(&mut rng, device)?;
        println!("BATCH0_INDICES source=torch_randint sites={}", owned.len());
        Some(&owned)
    } else {
        None
    };
    let prediction = match (&session.model, samples) {
        (ForecastModel::Informer(model), Some(samples)) => {
            model.forward_with_samples(&batch.input, context, samples)?
        }
        (_, Some(_)) => return Err("batch0 samples require Informer".into()),
        (_, None) => session.model.forward_with_context(&batch.input, context)?,
    };
    if samples.is_some() {
        println!(
            "BATCH0_SAMPLES {}",
            samples.as_ref().map(|m| m.len()).unwrap_or(0)
        );
    }
    let pred = select_targets(&prediction, &data.target_indices)?;
    let loss = masked_mse(&pred, &batch.supervision.target, &batch.supervision.mask)?;
    let sum = f64::from(loss.sum.to_scalar::<f32>()?);
    println!(
        "BATCH0 mode={} history_mean={history_mean:.8e} time_mean={time_mean:.8e} target_mean={target_mean:.8e} loss={:.10} count={}",
        if eval_mode { "eval" } else { "train" },
        sum / loss.valid_count as f64,
        loss.valid_count
    );
    Ok(())
}

fn step_diagnostic(
    config_path: &Path,
    initial_path: &Path,
    steps: usize,
    device: &Device,
) -> R<()> {
    if !(1..=5).contains(&steps) {
        return Err("step diagnostic needs 1..5 steps".into());
    }
    let reduced_precision =
        env::var("FERROTS_PROBE_REDUCED_PRECISION").ok().as_deref() == Some("1");
    candle_core::cuda::set_gemm_reduced_precision_f32(reduced_precision);
    let config = Config::load(config_path, Overrides::default())?;
    let data = Dataset::load(&config.data, &config.window, None)?;
    let model = ForecastModel::from_config(&config, device)?;
    let initial = load_map(initial_path, device)?;
    let weights = model
        .weights()
        .keys()
        .map(|name| {
            Ok((
                name.clone(),
                initial
                    .get(name)
                    .ok_or(format!("missing initial weight {name}"))?
                    .clone(),
            ))
        })
        .collect::<R<BTreeMap<_, _>>>()?;
    model.load_weights(&weights)?;
    let buffers = model
        .buffers()?
        .into_iter()
        .map(|(name, value)| (name.clone(), initial.get(&name).cloned().unwrap_or(value)))
        .collect();
    model.load_buffers(&buffers)?;
    let mut session = TrainingSession::from_model(&config, &data, device, model)?;
    let ModelSpec::Informer {
        factor,
        e_layers,
        d_layers,
        distil,
        label_len,
        ..
    } = &config.model
    else {
        return Err("step diagnostic requires Informer".into());
    };
    let schedule = InformerSamplingSchedule {
        lookback: config.window.lookback,
        label_len: *label_len,
        horizon: config.window.horizon,
        factor: *factor,
        e_layers: *e_layers,
        d_layers: *d_layers,
        distil: *distil,
        batch_size: config.train.batch_size,
        train_windows: data.index.counts[0],
        val_windows: data.index.counts[1],
        test_windows: data.index.counts[2],
        train_drop_last: config.train.drop_last,
    };
    let entries = data.index.order(0);
    let order: Vec<usize> = schedule
        .permutation(session.model.sampling_rng(), 0)
        .into_iter()
        .map(|index| entries[index])
        .collect();
    let mut rng = schedule.rng_before_train_batch(session.model.sampling_rng(), 0, 0);
    let output_dir = if reduced_precision {
        Path::new("runs/p3-development/20261017-informer-steps-reduced")
    } else {
        Path::new("runs/p3-development/20261017-informer-steps")
    };
    std::fs::create_dir_all(output_dir)?;
    println!(
        "SETTINGS candle_gemm_reduced_precision_f32={} steps={steps}",
        candle_core::cuda::gemm_reduced_precision_f32()
    );
    for step in 0..steps {
        let offset = step * config.train.batch_size;
        let end = (offset + config.train.batch_size).min(order.len());
        let batch_entries = &order[offset..end];
        let mut access = SplitAccess::default();
        let host = tracked_batch(&data, 0, batch_entries, &mut access)?;
        let batch = ferrots_train::trainer::DeviceBatch::from_host(&host, device)?;
        let samples = schedule.draw_train_indices(&mut rng, device)?;
        let ForecastModel::Informer(informer) = &session.model else {
            unreachable!()
        };
        let (prediction, traces) = informer.forward_with_trace(
            &batch.input,
            ForwardContext {
                mode: Mode::Train,
                forward_id: step as u64,
            },
            Some(&samples),
        )?;
        let reference = output_dir.join(format!("official-step{step}.safetensors"));
        if reference.exists() && !reduced_precision {
            let official = load_map(&reference, device)?;
            let mut offset = if step == 0 {
                0
            } else {
                let previous = load_map(
                    &output_dir.join(format!("rust-step{}.safetensors", step - 1)),
                    device,
                )?;
                u64::try_from(previous["philox_offset"].to_vec1::<i64>()?[0])?
            };
            let mut mask_sites = 0;
            for (name, expected) in &official {
                if !name.starts_with("mask.") {
                    continue;
                }
                let (mask, next) = probe_philox::mask(
                    config.train.seed,
                    offset,
                    expected.elem_count(),
                    if let ModelSpec::Informer { dropout, .. } = config.model {
                        dropout
                    } else {
                        unreachable!()
                    },
                    probe_philox::limits_for(device)?,
                )?;
                let actual: Vec<u8> = mask.into_iter().map(|value| value as u8).collect();
                let expected = expected.flatten_all()?.to_vec1::<u8>()?;
                let unequal = actual
                    .iter()
                    .zip(expected.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                let reordered = if official[name].rank() == 3 {
                    let physical: Vec<f32> = actual.iter().map(|value| *value as f32).collect();
                    let logical =
                        probe_philox::to_batch_length_channel(&physical, official[name].dims())?;
                    logical
                        .iter()
                        .zip(expected.iter())
                        .filter(|(a, b)| **a as u8 != **b)
                        .count()
                } else {
                    usize::MAX
                };
                println!(
                    "MASK {name} shape={:?} n={} offset={offset}->{next} direct_unequal={unequal} reordered_unequal={reordered}",
                    official[name].dims(),
                    actual.len()
                );
                offset = next;
                mask_sites += 1;
            }
            println!(
                "MASK_SUMMARY sites={mask_sites} generated_offset={offset} model_offset={}",
                informer.cuda_philox_offset()?
            );
        }
        let pred = select_targets(&prediction, &data.target_indices)?;
        let loss = masked_mse(&pred, &batch.supervision.target, &batch.supervision.mask)?;
        let loss_mean = loss.sum.to_scalar::<f32>()? / loss.valid_count as f32;
        let raw_gradients = loss.sum.backward()?;
        let mut normalized = GradStore::default();
        let mut values = BTreeMap::new();
        for (name, sample) in samples {
            values.insert(format!("sample.{name}"), sample);
        }
        values.insert("loss".into(), Tensor::new(&[loss_mean], device)?);
        values.insert("prediction".into(), pred.detach());
        values.insert(
            "philox_offset".into(),
            Tensor::new(&[i64::try_from(informer.cuda_philox_offset()?)?], device)?,
        );
        for (name, value) in traces {
            values.insert(format!("trace.{name}"), value.detach());
        }
        for (name, var) in session.model.named_parameters() {
            let gradient = raw_gradients
                .get(&var)
                .ok_or(format!("missing gradient {name}"))?
                .affine(1.0 / loss.valid_count as f64, 0.0)?;
            values.insert(format!("gradient.{name}"), gradient.detach());
            normalized.insert(&var, gradient);
        }
        session.optimizer.step(&normalized)?;
        for (name, var) in session.model.named_parameters() {
            values.insert(format!("updated.{name}"), var.detach());
            let state = session
                .optimizer
                .state(&name)
                .ok_or(format!("missing Adam state {name}"))?;
            values.insert(format!("exp_avg.{name}"), state.exp_avg.clone());
            values.insert(format!("exp_avg_sq.{name}"), state.exp_avg_sq.clone());
        }
        for (name, buffer) in session.model.buffers()? {
            values.insert(format!("buffer.{name}"), buffer);
        }
        let output = output_dir.join(format!("rust-step{step}.safetensors"));
        candle_core::safetensors::save(&values.into_iter().collect(), output)?;
        println!(
            "STEP {step} order_head={:?} count={} loss={loss_mean:.12}",
            &batch_entries[..batch_entries.len().min(4)],
            loss.valid_count
        );
    }
    Ok(())
}

fn orders(config_path: &Path, out: &Path, device: &Device) -> R<()> {
    let config = Config::load(config_path, Overrides::default())?;
    let data = Dataset::load(&config.data, &config.window, None)?;
    let model = ForecastModel::from_config(&config, device)?;
    let ids = data.index.order(0);
    let identity = ids.iter().enumerate().all(|(i, id)| i == *id);
    println!(
        "TRAIN_ENTRIES count={} first={:?} last={:?} identity={identity} data_sha256={}",
        ids.len(),
        ids.first().copied(),
        ids.last().copied(),
        data.cache.source_sha256
    );
    let mut text = String::new();
    for epoch in 0..2 {
        let order = data.index.training_order(model.sampling_rng(), epoch);
        let bytes: Vec<u8> = order
            .iter()
            .flat_map(|n| (*n as u64).to_le_bytes())
            .collect();
        println!(
            "ORDER_SHA epoch={epoch} len={} sha256={} head={:?} tail={:?}",
            order.len(),
            ferrots_data::cache::hash_bytes(&bytes),
            &order[..order.len().min(4)],
            &order[order.len().saturating_sub(4)..]
        );
        text.push_str(&format!("epoch {epoch} len {}\n", order.len()));
        for (i, n) in order.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            text.push_str(&n.to_string());
        }
        text.push('\n');
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, text)?;
    println!("ORDERS_WROTE {}", out.display());
    Ok(())
}

fn load_orders(path: &Path) -> R<Vec<Vec<usize>>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|line| !line.is_empty());
    let mut orders = Vec::new();
    while let Some(header) = lines.next() {
        let mut parts = header.split_whitespace();
        if parts.next() != Some("epoch") || parts.next().is_none() || parts.next() != Some("len") {
            return Err(format!("bad order header: {header}").into());
        }
        let len: usize = parts.next().ok_or("order header missing length")?.parse()?;
        let body = lines.next().ok_or("order body missing")?;
        let order = body
            .split_whitespace()
            .map(|value| value.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?;
        if order.len() != len {
            return Err(format!("order length {} != {len}", order.len()).into());
        }
        orders.push(order);
    }
    if orders.is_empty() {
        return Err("no training orders".into());
    }
    Ok(orders)
}

fn apply_order(session: &mut TrainingSession<'_>, orders: &[Vec<usize>], index: usize) -> R<()> {
    let order = orders
        .get(index)
        .ok_or("missing official order for the next epoch")?
        .clone();
    let shown = order.len().min(4);
    let head = order[..shown].to_vec();
    session.replace_training_order(order)?;
    let next = session.next_entries();
    println!(
        "ORDER_APPLIED index={index} head={head:?} next={:?}",
        &next[..next.len().min(shown)]
    );
    Ok(())
}

fn train1(
    config_path: &Path,
    weights: Option<&Path>,
    orders: Option<&[Vec<usize>]>,
    device: &Device,
) -> R<()> {
    let config = Config::load(config_path, Overrides::default())?;
    let data = Dataset::load(&config.data, &config.window, None)?;
    let model = ForecastModel::from_config(&config, device)?;
    let replaced = if let Some(path) = weights {
        let values = load_map(path, device)?;
        let initial = subset(&values, "initial.");
        let count = initial.len();
        model.load_weights(&initial)?;
        count
    } else {
        0
    };
    println!(
        "TRAIN1 seed={} max_epochs={} batch={} weights_replaced={replaced} data_sha256={}",
        config.train.seed,
        config.train.max_epochs,
        config.train.batch_size,
        data.cache.source_sha256
    );
    let mut session = TrainingSession::from_model(&config, &data, device, model)?;
    if let Some(orders) = orders {
        apply_order(&mut session, orders, 0)?;
    }
    loop {
        match session.advance()? {
            Progress::Training { updated } => {
                if updated && session.state.global_optimizer_step.is_multiple_of(64) {
                    println!("STEP {}", session.state.global_optimizer_step);
                }
            }
            Progress::Epoch { record, improved } => {
                println!(
                    "EPOCH epoch={} train_loss={:.12} validation_loss={:.12} steps={} labels={} improved={improved} seconds={:.3}",
                    record.epoch,
                    record.train_loss,
                    record.validation_loss,
                    record.optimizer_steps,
                    record.valid_labels,
                    record.seconds
                );
                let (mse, mae, count) =
                    score_split(&session.model, &data, config.train.batch_size, device, 2)?;
                println!("TEST_CANONICAL mse={mse:.12} mae={mae:.12} count={count}");
                if record.epoch >= config.train.max_epochs {
                    break;
                }
                if let Some(orders) = orders {
                    apply_order(&mut session, orders, record.epoch)?;
                }
            }
            Progress::Complete => {
                println!("TRAIN1_COMPLETE_WITHOUT_EPOCH");
                break;
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("probe error: {error}");
            ExitCode::from(1)
        }
    }
}

fn run() -> R<()> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or(
        "usage: p3_d4_probe tf32|compare|init|variance|rank|agg|bounds|init-gap|batch0|step-diagnostic|orders|train1|train-order <path> [extra]",
    )?;
    let device = cuda_device()?;
    match command.as_str() {
        "tf32" => Ok(()),
        "compare" => compare(
            Path::new(&args.next().ok_or("compare needs a fixture")?),
            &device,
        ),
        "init" => init_diff(
            Path::new(&args.next().ok_or("init needs a fixture")?),
            &device,
        ),
        "variance" => variance(
            Path::new(&args.next().ok_or("variance needs a checkpoint")?),
            &device,
        ),
        "rank" => rank(
            Path::new(&args.next().ok_or("rank needs a run directory")?),
            &device,
        ),
        "agg" => agg(
            Path::new(&args.next().ok_or("agg needs a run directory")?),
            &device,
        ),
        "bounds" => bounds(
            Path::new(&args.next().ok_or("bounds needs a run directory")?),
            &device,
        ),
        "init-gap" => init_gap(
            Path::new(&args.next().ok_or("init-gap needs a safetensors file")?),
            &device,
        ),
        "batch0" => batch0(
            Path::new(&args.next().ok_or("batch0 needs a config")?),
            &device,
        ),
        "step-diagnostic" => {
            let config = args.next().ok_or("step-diagnostic needs a config")?;
            let initial = args.next().ok_or("step-diagnostic needs initial weights")?;
            let steps: usize = args.next().ok_or("step-diagnostic needs steps")?.parse()?;
            step_diagnostic(Path::new(&config), Path::new(&initial), steps, &device)
        }
        "orders" => {
            let config = args.next().ok_or("orders needs a config")?;
            let out = args.next().ok_or("orders needs an output path")?;
            orders(Path::new(&config), Path::new(&out), &device)
        }
        "train1" => {
            let config = args.next().ok_or("train1 needs a config")?;
            let weights = args.next();
            train1(
                Path::new(&config),
                weights.as_deref().map(Path::new),
                None,
                &device,
            )
        }
        "train-order" => {
            let config = args.next().ok_or("train-order needs a config")?;
            let orders = args.next().ok_or("train-order needs an order file")?;
            let loaded = load_orders(Path::new(&orders))?;
            train1(Path::new(&config), None, Some(&loaded), &device)
        }
        other => Err(format!("unknown command {other}").into()),
    }
}
