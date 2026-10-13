#![cfg(feature = "cuda")]
//! Persistent Informer state is part of both continuation and inference.
use candle_core::{DType, Device, Tensor};
use ferrots_core::config::{Activation, Config, ModelSpec, SplitSpec};
use ferrots_data::{batch::Dataset, cache::fingerprint};
use ferrots_nn::model::ForecastModel;
use ferrots_train::{
    checkpoint::{self, Provenance},
    evaluation::{EvaluationOptions, evaluate, predict_future},
    session::TrainingSession,
    trainer::validation_loss,
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn directory() -> PathBuf {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "ferrots-p3-buffers-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    path
}

fn example() -> R<(Config, Dataset, Provenance)> {
    let root = directory();
    let mut csv = "date,x,y\n".to_owned();
    for i in 0..50 {
        csv.push_str(&format!(
            "2024-01-01 00:{i:02}:00,{},{}\n",
            (i as f32 * 0.4).sin() + i as f32 * 0.05,
            (i as f32 * 0.3).cos()
        ));
    }
    let mut config = Config::default();
    config.data.path = root.join("synthetic.csv");
    fs::write(&config.data.path, csv)?;
    config.data.cache_dir = root.join("cache");
    config.data.input_columns = vec!["x".into(), "y".into()];
    config.data.target_columns = vec!["x".into(), "y".into()];
    config.data.interval_seconds = 60;
    config.data.split = SplitSpec::Rows {
        train_end: 30,
        validation_end: 40,
        test_end: 50,
    };
    config.window.lookback = 8;
    config.window.horizon = 2;
    config.model = ModelSpec::Informer {
        d_model: 8,
        n_heads: 2,
        d_ff: 16,
        e_layers: 2,
        d_layers: 1,
        factor: 1,
        distil: true,
        label_len: 4,
        dropout: 0.2,
        activation: Activation::Gelu,
    };
    config.train.device = "cuda:0".into();
    config.train.batch_size = 4;
    config.train.accumulation_steps = 2;
    config.train.max_epochs = 2;
    config.output.root = root.join("runs");
    let data = Dataset::load(&config.data, &config.window, None)?;
    let provenance = Provenance {
        code_sha: "synthetic".into(),
        dirty: true,
        source_sha256: "a".repeat(64),
        environment: BTreeMap::from([("test".into(), "p3-buffers".into())]),
    };
    Ok((config, data, provenance))
}

fn values(tensor: &Tensor) -> R<Vec<f64>> {
    Ok(tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?)
}

fn assert_tensors(a: &BTreeMap<String, Tensor>, b: &BTreeMap<String, Tensor>, atol: f64) -> R {
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    for (name, left) in a {
        let right = &b[name];
        assert_eq!(left.dims(), right.dims(), "{name}");
        assert_eq!(left.dtype(), right.dtype(), "{name}");
        let tolerance = if left.dtype().is_float() { atol } else { 0. };
        for (x, y) in values(left)?.into_iter().zip(values(right)?) {
            assert!((x - y).abs() <= tolerance, "{name}: {x} != {y}");
        }
    }
    Ok(())
}

#[test]
fn cuda_snapshot_checkpoint_resume_and_inference_keep_buffers() -> R {
    let device = Device::new_cuda(0)?;
    let (config, data, provenance) = example()?;
    let mut session = TrainingSession::new(&config, &data, &device)?;
    session.advance()?;
    let snapshot = session.snapshot()?;
    assert!(!snapshot.buffers.is_empty());
    let captured = snapshot
        .buffers
        .iter()
        .map(|(name, tensor)| Ok((name.clone(), tensor.copy()?)))
        .collect::<R<BTreeMap<_, _>>>()?;
    for name in captured.keys() {
        assert!(
            session.optimizer.state(name).is_none(),
            "buffer entered Adam"
        );
    }
    let run = directory();
    let bundle = checkpoint::publish_training(&run, "last", &mut session, &provenance, None)?;
    let loaded = checkpoint::load(&bundle, &device)?;
    assert_tensors(&captured, &loaded.model.buffers()?, 0.)?;
    let mut restored = checkpoint::load_session(&bundle, &config, &data, &device, &provenance)?;
    let memory_restored = TrainingSession::resume(&config, &data, &device, snapshot)?;
    assert_tensors(&captured, &memory_restored.model.buffers()?, 0.)?;
    session.advance()?;
    restored.advance()?;
    assert_tensors(session.model.weights(), restored.model.weights(), 1e-6)?;
    assert_tensors(&session.model.buffers()?, &restored.model.buffers()?, 1e-6)?;
    assert_tensors(&captured, &memory_restored.model.buffers()?, 0.)?;
    assert_tensors(&captured, &loaded.model.buffers()?, 0.)?;
    assert_eq!(
        session.state.next_forward_id,
        restored.state.next_forward_id
    );
    assert_eq!(
        serde_json::to_value(session.rng_state())?,
        serde_json::to_value(restored.rng_state())?
    );
    for name in session.model.weights().keys() {
        let left = session.optimizer.state(name).unwrap();
        let right = restored.optimizer.state(name).unwrap();
        assert_eq!(left.step, right.step);
        for (a, b) in [
            (&left.exp_avg, &right.exp_avg),
            (&left.exp_avg_sq, &right.exp_avg_sq),
        ] {
            for (x, y) in values(a)?.into_iter().zip(values(b)?) {
                assert!((x - y).abs() <= 1e-6, "{name}: Adam moment differs");
            }
        }
    }

    let state_before = session.snapshot()?;
    let frozen = session.model.freeze_for_eval()?;
    let mut access = session.state.access.clone();
    let mut repeated_access = access.clone();
    let first = validation_loss(&frozen, &data, &config, &device, &mut access)?;
    let second = validation_loss(&frozen, &data, &config, &device, &mut repeated_access)?;
    assert_eq!(first, second);
    assert_eq!(data.index.counts[1] % config.train.batch_size, 1);
    assert_tensors(&state_before.weights, session.model.weights(), 0.)?;
    assert_tensors(&state_before.buffers, &session.model.buffers()?, 0.)?;
    assert_tensors(&state_before.buffers, &frozen.buffers()?, 0.)?;
    assert_eq!(
        serde_json::to_value(&state_before.rng)?,
        serde_json::to_value(session.rng_state())?
    );

    let options = EvaluationOptions {
        run_id: "synthetic",
        split: 2,
        batch_size: 4,
        export: None,
        columns: &[],
    };
    let first = evaluate(&loaded, &data, &device, &options)?;
    let second = evaluate(&loaded, &data, &device, &options)?;
    assert_eq!(
        serde_json::to_value(first.metrics)?,
        serde_json::to_value(second.metrics)?
    );
    assert_eq!(first.sampling_policy, "evaluation-split-batch-v1");
    assert_eq!(first.first_forward_id, (2 << 62) | (2 << 60));
    assert_eq!(first.batch_size, 4);
    assert_eq!(first.windows, data.index.counts[2]);
    let prediction = predict_future(
        &loaded,
        None,
        &device,
        "synthetic",
        &run.join("future.csv"),
        &[],
    )?;
    assert_eq!(prediction.rows, 4);
    assert_eq!(prediction.seed, config.train.seed);
    assert_eq!(prediction.forward_id, 3 << 62);
    assert_tensors(&captured, &loaded.model.buffers()?, 0.)?;
    Ok(())
}

fn repair_buffers_hash(bundle: &Path) -> R {
    let path = bundle.join("manifest.json");
    let mut manifest: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    manifest["files"]["buffers.safetensors"]["bytes"] =
        fs::metadata(bundle.join("buffers.safetensors"))?
            .len()
            .into();
    manifest["files"]["buffers.safetensors"]["sha256"] =
        fingerprint(bundle.join("buffers.safetensors"))?.into();
    fs::write(path, serde_json::to_vec(&manifest)?)?;
    Ok(())
}

#[test]
fn cuda_corrupt_buffers_fail_before_mutating_the_model_or_reference() -> R {
    let device = Device::new_cuda(0)?;
    let (config, data, provenance) = example()?;
    let mut session = TrainingSession::new(&config, &data, &device)?;
    session.advance()?;
    let valid = session.model.buffers()?;
    let run = directory();
    let original = checkpoint::publish_training(&run, "last", &mut session, &provenance, None)?;
    for kind in ["missing", "extra", "shape", "dtype", "nonfinite"] {
        let mut buffers = session.model.buffers()?;
        let name = buffers
            .keys()
            .find(|name| name.ends_with("running_mean"))
            .unwrap()
            .clone();
        match kind {
            "missing" => {
                buffers.remove(&name);
            }
            "extra" => {
                buffers.insert("unexpected".into(), buffers[&name].clone());
            }
            "shape" => {
                buffers.insert(name, Tensor::zeros(99, DType::F32, &device)?);
            }
            "dtype" => {
                let wrong = buffers[&name].to_dtype(DType::F64)?;
                buffers.insert(name, wrong);
            }
            "nonfinite" => {
                let wrong = Tensor::full(f32::NAN, buffers[&name].shape(), &device)?;
                buffers.insert(name, wrong);
            }
            _ => unreachable!(),
        }
        assert!(session.model.load_buffers(&buffers).is_err(), "{kind}");
        assert_tensors(&valid, &session.model.buffers()?, 0.)?;
        let broken = directory();
        for entry in fs::read_dir(&original)? {
            let entry = entry?;
            fs::copy(entry.path(), broken.join(entry.file_name()))?;
        }
        let payload: std::collections::HashMap<_, _> = buffers.into_iter().collect();
        candle_core::safetensors::save(&payload, broken.join("buffers.safetensors"))?;
        repair_buffers_hash(&broken)?;
        assert!(checkpoint::load(&broken, &device).is_err(), "{kind}");
        assert!(
            checkpoint::load_session(&broken, &config, &data, &device, &provenance).is_err(),
            "{kind}"
        );
        assert_eq!(checkpoint::resolve(&run.join("last"))?, original);
    }
    Ok(())
}

#[test]
fn cuda_informer_rejects_v1_and_stateless_models_reject_buffers() -> R {
    let device = Device::new_cuda(0)?;
    let (config, data, provenance) = example()?;
    let model = ForecastModel::from_config(&config, &device)?;
    let run = directory();
    assert!(
        checkpoint::publish(
            &run,
            "last",
            &checkpoint::Snapshot {
                model: &model,
                config: &config,
                scaler: &data.scaler,
                provenance: &provenance,
                data_sha256: &data.cache.source_sha256,
                epoch: 1,
                validation_loss: 1.,
            }
        )
        .is_err()
    );
    assert!(!run.join(".staging").exists());
    let mut baseline_config = config.clone();
    baseline_config.model = ModelSpec::LastValue;
    let baseline = ForecastModel::from_config(&baseline_config, &device)?;
    assert!(baseline.load_buffers(&model.buffers()?).is_err());
    assert!(baseline.buffers()?.is_empty());
    Ok(())
}

#[test]
fn cuda_changed_supervision_cannot_enter_informer_decoder() -> R {
    let device = Device::new_cuda(0)?;
    let (config, data, _) = example()?;
    let model = ForecastModel::from_config(&config, &device)?.freeze_for_eval()?;
    let host = data.batch(&data.index.order(1)[..2])?;
    let mut batch = ferrots_train::trainer::DeviceBatch::from_host(&host, &device)?;
    let context = ferrots_nn::model::ForwardContext::evaluation_with_id(29);
    let before = model.forward_with_context(&batch.input, context)?;
    batch.supervision.target = Tensor::full(1e6_f32, batch.supervision.target.shape(), &device)?;
    batch.supervision.mask = batch.supervision.mask.zeros_like()?;
    let after = model.forward_with_context(&batch.input, context)?;
    assert_tensors(
        &BTreeMap::from([("prediction".into(), before)]),
        &BTreeMap::from([("prediction".into(), after)]),
        0.,
    )?;
    let ForecastModel::Informer(informer) = &model else {
        panic!("expected Informer")
    };
    let (_, trace) = informer.forward_with_trace(&batch.input, context, None)?;
    let ModelSpec::Informer { label_len, .. } = config.model else {
        panic!("expected Informer")
    };
    assert!(
        trace["decoder_input"]
            .narrow(1, label_len, config.window.horizon)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| *x == 0.)
    );
    Ok(())
}

#[test]
fn cuda_select_targets_accepts_noncontiguous_decoder_forecasts() -> R {
    let device = Device::new_cuda(0)?;
    let decoder = Tensor::arange(0_u32, 24_u32, &device)?
        .to_dtype(DType::F32)?
        .reshape((2, 4, 3))?;
    let forecast = decoder.narrow(1, 1, 2)?;
    assert!(!forecast.is_contiguous());
    let selected = ferrots_train::trainer::select_targets(&forecast, &[2])?;
    assert_eq!(selected.dims(), [2, 2, 1]);
    assert_eq!(
        selected.flatten_all()?.to_vec1::<f32>()?,
        [5., 8., 17., 20.]
    );
    Ok(())
}
