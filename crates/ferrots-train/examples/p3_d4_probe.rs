//! Diagnostic only. Reads fixtures or frozen checkpoints and writes nothing
//! back to a suite directory.
use candle_core::{DType, Device, Tensor};
use ferrots_core::{
    config::{Activation, Config, ModelSpec, Overrides, WindowSpec},
    metrics::MetricGrid,
};
use ferrots_data::batch::Dataset;
use ferrots_nn::{
    Mode,
    informer::Informer,
    model::{ForecastModel, ForwardContext},
};
use ferrots_train::{
    checkpoint,
    evaluation::{self, EvaluationOptions},
    session::{Progress, TrainingSession},
    trainer::{SplitAccess, select_targets, tracked_batch},
};
use std::{collections::BTreeMap, env, path::Path, process::ExitCode};

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
        "usage: p3_d4_probe tf32|compare|init|variance|rank|orders|train1|train-order <path> [extra]",
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
