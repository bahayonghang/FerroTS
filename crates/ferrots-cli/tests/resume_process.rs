mod common;
use candle_core::{DType, Device, Tensor};
use ferrots_core::config::{Activation, Config, MissingMode, ModelSpec, SplitSpec};
use ferrots_data::batch::Dataset;
use ferrots_nn::{Mode, RngStream, model::ForwardContext};
use ferrots_train::{checkpoint, session::TrainingState, trainer::DeviceBatch};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};
type R = Result<(), Box<dyn std::error::Error>>;

fn fixture(device: &str, accumulation: usize, empty: bool, ties: bool) -> (PathBuf, Config) {
    let (root, mut c) = common::fixture(ModelSpec::Itransformer {
        d_model: 8,
        n_heads: 2,
        d_ff: 16,
        e_layers: 2,
        dropout: 0.1,
        activation: Activation::Gelu,
        use_norm: true,
    });
    let mut csv = "date,x,y\n".to_owned();
    for i in 0..30 {
        let y = if empty && i < 16 && i % 4 == 3 {
            String::new()
        } else {
            (i as f32 * 0.3).sin().to_string()
        };
        csv.push_str(&format!(
            "2024-01-{:02} 00:00:00,{},{}\n",
            i + 1,
            i as f32 * 0.2,
            y
        ));
    }
    fs::write(&c.data.path, csv).unwrap();
    c.data.target_columns = vec!["y".into()];
    c.data.split = SplitSpec::Rows {
        train_end: 16,
        validation_end: 23,
        test_end: 30,
    };
    c.window.lookback = if empty { 1 } else { 2 };
    c.window.horizon = 1;
    c.window.stride = if empty { 2 } else { 1 };
    c.data.missing.mode = MissingMode::Causal;
    c.data.missing.max_stale_seconds = 0;
    c.train.device = device.into();
    c.train.accumulation_steps = accumulation;
    c.train.batch_size = if empty { 1 } else { 3 };
    c.train.max_epochs = 12;
    c.train.patience = 12;
    if ties {
        c.train.learning_rate = 1e-45;
        c.train.patience = 2;
    }
    common::write_config(&root, &c);
    (root, c)
}
fn run_dir(summary: &Value) -> PathBuf {
    PathBuf::from(summary["run_dir"].as_str().unwrap())
}
fn train(root: &Path, name: &str, steps: Option<usize>) -> Value {
    let config = root.join("config.toml");
    let mut args = vec![
        "train",
        "--config",
        config.to_str().unwrap(),
        "--run-id",
        name,
    ];
    let count = steps.map(|n| n.to_string());
    if let Some(n) = &count {
        args.extend(["--stop-after-steps", n]);
    }
    common::json(&args)
}
fn resume(path: &Path, steps: Option<usize>) -> Value {
    let mut args = vec!["resume", "--checkpoint", path.to_str().unwrap()];
    let count = steps.map(|n| n.to_string());
    if let Some(n) = &count {
        args.extend(["--stop-after-steps", n]);
    }
    common::json(&args)
}
fn tensor_error(a: &Tensor, b: &Tensor, tolerance: f64) -> Result<f64, Box<dyn std::error::Error>> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    let mut max = 0f64;
    for (x, y) in a.iter().zip(&b) {
        assert!(x.is_finite() && y.is_finite());
        if tolerance == 0. {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        let delta = (*x as f64 - *y as f64).abs();
        assert!(delta <= tolerance, "tensor delta {delta} > {tolerance}");
        max = max.max(delta);
    }
    Ok(max)
}
fn normalized(s: &TrainingState) -> Value {
    let mut v = serde_json::to_value(s).unwrap();
    v["epoch_seconds"] = 0.into();
    v["steady_seconds"] = 0.into();
    v["best_generation"] = s.best_generation.is_some().into();
    for e in v["epochs"].as_array_mut().unwrap() {
        for name in [
            "seconds",
            "train_seconds",
            "validation_seconds",
            "steady_seconds",
        ] {
            e[name] = 0.into();
        }
    }
    v
}
fn compare_json(a: &Value, b: &Value, tolerance: f64) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len());
            for (k, v) in a {
                compare_json(v, &b[k], tolerance);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b) {
                compare_json(x, y, tolerance);
            }
        }
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => assert!(
            (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= tolerance,
            "state {a} != {b}"
        ),
        _ => assert_eq!(a, b),
    }
}
fn compare(
    left: &Path,
    right: &Path,
    device: &Device,
) -> Result<Value, Box<dyn std::error::Error>> {
    let a = checkpoint::load(left, device)?;
    let b = checkpoint::load(right, device)?;
    let da = Dataset::load(&a.config.data, &a.config.window, Some(a.scaler.clone()))?;
    let db = Dataset::load(&b.config.data, &b.config.window, Some(b.scaler.clone()))?;
    let sa = checkpoint::load_session(left, &a.config, &da, device, &a.manifest.provenance)?;
    let sb = checkpoint::load_session(right, &b.config, &db, device, &b.manifest.provenance)?;
    let tolerance = if device.is_cpu() { 0. } else { 1e-6 };
    compare_json(&normalized(&sa.state), &normalized(&sb.state), tolerance);
    assert_eq!(
        serde_json::to_value(&sa.scheduler)?,
        serde_json::to_value(&sb.scheduler)?
    );
    assert_eq!(
        serde_json::to_value(sa.rng_state())?,
        serde_json::to_value(sb.rng_state())?
    );
    assert_eq!(sa.next_entries(), sb.next_entries());
    assert_eq!(
        a.model.weights().keys().collect::<Vec<_>>(),
        b.model.weights().keys().collect::<Vec<_>>()
    );
    let mut max_error = 0f64;
    for (name, t) in a.model.weights() {
        max_error = max_error.max(tensor_error(t, &b.model.weights()[name], tolerance)?);
        let x = sa.optimizer.state(name).unwrap();
        let y = sb.optimizer.state(name).unwrap();
        assert_eq!(x.step, y.step);
        max_error = max_error.max(tensor_error(&x.exp_avg, &y.exp_avg, tolerance)?);
        max_error = max_error.max(tensor_error(&x.exp_avg_sq, &y.exp_avg_sq, tolerance)?);
    }
    let entries = if sa.next_entries().is_empty() {
        vec![da.index.order(1)[0]]
    } else {
        sa.next_entries().to_vec()
    };
    let batch = DeviceBatch::from_host(&da.batch(&entries)?, device)?;
    let pa = sa.model.forward_with_context(
        &batch.input,
        ForwardContext {
            mode: Mode::Train,
            forward_id: sa.state.next_forward_id,
        },
    )?;
    let pb = sb.model.forward_with_context(
        &batch.input,
        ForwardContext {
            mode: Mode::Train,
            forward_id: sb.state.next_forward_id,
        },
    )?;
    max_error = max_error.max(tensor_error(&pa, &pb, tolerance)?);
    max_error = max_error.max(tensor_error(
        &a.model.forward(&batch.input)?,
        &b.model.forward(&batch.input)?,
        tolerance,
    )?);
    for layer_id in 0..9 {
        let ones = Tensor::ones((2, 3, 8), DType::F32, device)?;
        let x = ferrots_nn::dropout(
            &ones,
            0.1,
            Mode::Train,
            RngStream {
                seed: a.config.train.seed,
                run_step: sa.state.next_forward_id,
                layer_id,
            },
        )?;
        let y = ferrots_nn::dropout(
            &ones,
            0.1,
            Mode::Train,
            RngStream {
                seed: b.config.train.seed,
                run_step: sb.state.next_forward_id,
                layer_id,
            },
        )?;
        tensor_error(&x, &y, 0.)?;
    }
    if sa.state.best_generation.is_some() {
        let ra = a.directory.parent().unwrap().parent().unwrap().join("best");
        let rb = b.directory.parent().unwrap().parent().unwrap().join("best");
        let ba = checkpoint::load(&ra, device)?;
        let bb = checkpoint::load(&rb, device)?;
        assert_eq!(ba.manifest.epoch, bb.manifest.epoch);
        for (name, t) in ba.model.weights() {
            max_error = max_error.max(tensor_error(t, &bb.model.weights()[name], tolerance)?);
        }
    }
    Ok(
        json!({"left":a.directory,"right":b.directory,"parameters":a.model.weights().len(),"max_tensor_error":max_error,
        "atol":tolerance,"rtol":0,"optimizer_steps":sa.state.global_optimizer_step,"next_forward_id":sa.state.next_forward_id,
        "phase":sa.state.phase,"epoch":sa.state.epoch,"best_epoch":sa.state.early.best_epoch,"stale":sa.state.early.stale,
        "source_sha256":a.manifest.provenance.source_sha256,"empty_batches_in_completed_epochs":sa.state.epochs.iter().map(|r|r.skipped_empty_batches).sum::<usize>(),
        "scheduler_applied_epoch":sa.scheduler.applied_epoch,"test_windows":sa.state.access.windows[2]}),
    )
}
#[test]
fn dropout_training_matches_across_fresh_cli_processes() -> R {
    let devices = if cfg!(feature = "cuda") {
        vec!["cuda:0"]
    } else {
        vec!["cpu"]
    };
    let mut cases = vec![];
    for name in devices {
        let device = ferrots_nn::resolve_device(name.parse()?)?;
        for accumulation in [1, 2] {
            for n in [3, 7] {
                for empty in [false, true] {
                    let (root, _) = fixture(name, accumulation, empty, false);
                    let continuous = train(&root, "continuous", Some(2 * n));
                    let paused = train(&root, "interrupted", Some(n));
                    assert_eq!(paused["optimizer_steps"], n);
                    assert_eq!(continuous["optimizer_steps"], 2 * n);
                    let restored = resume(&run_dir(&paused).join("last"), Some(n));
                    assert_eq!(restored["optimizer_steps"], 2 * n);
                    let mut result = compare(
                        &run_dir(&continuous).join("last"),
                        &run_dir(&restored).join("last"),
                        &device,
                    )?;
                    result["device"] = name.into();
                    result["n"] = n.into();
                    result["accumulation"] = accumulation.into();
                    result["empty_labels"] = empty.into();
                    if empty {
                        assert!(
                            result["next_forward_id"].as_u64().unwrap()
                                > result["optimizer_steps"].as_u64().unwrap()
                        );
                    }
                    eprintln!(
                        "resume case device={name} n={n} accumulation={accumulation} empty={empty} passed"
                    );
                    cases.push(result);
                }
            }
            let (root, _) = fixture(name, accumulation, false, true);
            let continuous = train(&root, "continuous", None);
            let n = if accumulation == 1 { 11 } else { 7 };
            let paused = train(&root, "interrupted", Some(n));
            let restored = resume(&run_dir(&paused).join("last"), None);
            assert_eq!(continuous["best_epoch"], 1);
            assert_eq!(continuous["epochs"].as_array().unwrap().len(), 3);
            let mut result = compare(
                &run_dir(&continuous).join("last"),
                &run_dir(&restored).join("last"),
                &device,
            )?;
            assert_eq!(result["stale"], 2);
            assert_eq!(result["scheduler_applied_epoch"], 2);
            result["device"] = name.into();
            result["accumulation"] = accumulation.into();
            result["patience_case"] = true.into();
            cases.push(result);
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runs/p2-development");
    fs::create_dir_all(&root)?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let path = root.join(format!(
        "resume-process-{}-{nonce}.json",
        if cfg!(feature = "cuda") {
            "cuda"
        } else {
            "cpu"
        }
    ));
    fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"format_version":1,"fresh_cli_processes":true,"dropout":0.1,"cases":cases}),
        )?,
    )?;
    eprintln!("resume process report: {}", path.display());
    Ok(())
}
