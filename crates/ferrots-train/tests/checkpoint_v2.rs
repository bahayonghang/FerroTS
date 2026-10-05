use candle_core::{DType, Device, Tensor};
use ferrots_core::config::{Activation, Config, ModelSpec, SplitSpec};
use ferrots_data::{batch::Dataset, cache::fingerprint};
use ferrots_train::{
    TrainError,
    checkpoint::{self, Provenance, PublishStage},
    session::{Phase, Progress, TrainingSession},
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
type R = Result<(), Box<dyn std::error::Error>>;
#[test]
fn legacy_inference_bundle_cannot_resume() -> R {
    let device = Device::Cpu;
    let (c, d, p) = example(&device)?;
    let model = ferrots_nn::model::ForecastModel::from_config(&c, &device)?;
    let run = directory();
    let bundle = checkpoint::publish(
        &run,
        "last",
        &checkpoint::Snapshot {
            model: &model,
            config: &c,
            scaler: &d.scaler,
            provenance: &p,
            data_sha256: &d.cache.source_sha256,
            epoch: 1,
            validation_loss: 1.,
        },
    )?;
    checkpoint::load(&bundle, &device)?;
    let error = checkpoint::load_session(&bundle, &c, &d, &device, &p)
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("v1 checkpoint is inference-only")
    );
    Ok(())
}
fn directory() -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let p = std::env::temp_dir().join(format!(
        "ferrots-v2-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&p).unwrap();
    p
}
fn example(device: &Device) -> Result<(Config, Dataset, Provenance), Box<dyn std::error::Error>> {
    let root = directory();
    let mut csv = "date,x,y\n".to_owned();
    for i in 0..30 {
        csv.push_str(&format!(
            "2024-01-{:02} 00:00:00,{},{}\n",
            i + 1,
            i as f32 * 0.2,
            (i as f32 * 0.3).sin()
        ));
    }
    let mut c = Config::default();
    c.data.path = root.join("synthetic.csv");
    fs::write(&c.data.path, csv)?;
    c.data.cache_dir = root.join("cache");
    c.data.input_columns = vec!["x".into(), "y".into()];
    c.data.target_columns = vec!["y".into()];
    c.data.interval_seconds = 86400;
    c.data.split = SplitSpec::Rows {
        train_end: 18,
        validation_end: 24,
        test_end: 30,
    };
    c.window.lookback = 2;
    c.window.horizon = 1;
    c.model = ModelSpec::Itransformer {
        d_model: 8,
        n_heads: 2,
        d_ff: 16,
        e_layers: 2,
        dropout: 0.1,
        activation: Activation::Gelu,
        use_norm: true,
    };
    c.train.device = if device.is_cpu() { "cpu" } else { "cuda:0" }.into();
    c.train.batch_size = 3;
    c.train.accumulation_steps = 2;
    c.train.max_epochs = 2;
    c.output.root = root.join("runs");
    let d = Dataset::load(&c.data, &c.window, None)?;
    let p = Provenance {
        code_sha: "synthetic".into(),
        dirty: true,
        source_sha256: "a".repeat(64),
        environment: BTreeMap::from([("test".into(), "fixed".into())]),
    };
    Ok((c, d, p))
}
fn repair(bundle: &Path, name: &str) -> R {
    let p = bundle.join("manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&fs::read(&p)?)?;
    m["files"][name]["bytes"] = fs::metadata(bundle.join(name))?.len().into();
    m["files"][name]["sha256"] = fingerprint(bundle.join(name))?.into();
    fs::write(p, serde_json::to_vec(&m)?)?;
    Ok(())
}
fn edit(bundle: &Path, name: &str, f: impl FnOnce(&mut serde_json::Value)) -> R {
    let p = bundle.join(name);
    let mut j = serde_json::from_slice(&fs::read(&p)?)?;
    f(&mut j);
    fs::write(p, serde_json::to_vec(&j)?)?;
    if name != "manifest.json" {
        repair(bundle, name)?;
    }
    Ok(())
}
fn assert_session(a: &TrainingSession<'_>, b: &TrainingSession<'_>) -> R {
    assert_eq!(
        serde_json::to_value(&a.state)?,
        serde_json::to_value(&b.state)?
    );
    assert_eq!(
        serde_json::to_value(&a.scheduler)?,
        serde_json::to_value(&b.scheduler)?
    );
    assert_eq!(
        serde_json::to_value(a.rng_state())?,
        serde_json::to_value(b.rng_state())?
    );
    assert_eq!(a.next_entries(), b.next_entries());
    for (n, t) in a.model.weights() {
        assert_eq!(
            t.flatten_all()?.to_vec1::<f32>()?,
            b.model.weights()[n].flatten_all()?.to_vec1::<f32>()?
        );
        let x = a.optimizer.state(n).unwrap();
        let y = b.optimizer.state(n).unwrap();
        assert_eq!(x.step, y.step);
        assert_eq!(
            x.exp_avg.flatten_all()?.to_vec1::<f32>()?,
            y.exp_avg.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            x.exp_avg_sq.flatten_all()?.to_vec1::<f32>()?,
            y.exp_avg_sq.flatten_all()?.to_vec1::<f32>()?
        );
    }
    Ok(())
}
#[test]
fn v2_roundtrips_every_phase_and_preserves_best_and_null_observation() -> R {
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    for device in devices {
        let (c, d, p) = example(&device)?;
        let run = directory();
        let mut s = TrainingSession::new(&c, &d, &device)?;
        let initial = checkpoint::publish_training(&run, "last", &mut s, &p, None)?;
        assert_eq!(fs::read_dir(&initial)?.count(), 12);
        let first = checkpoint::load(&initial, &device)?;
        assert_eq!(first.manifest.validation_loss, None);
        assert!(!run.join("best").exists());
        assert!(
            candle_core::safetensors::load(initial.join("buffers.safetensors"), &device)?
                .is_empty()
        );
        let mut phases = std::collections::BTreeSet::new();
        loop {
            phases.insert(format!("{:?}", s.state.phase));
            let resumed = checkpoint::load_session(run.join("last"), &c, &d, &device, &p)?;
            assert_session(&s, &resumed)?;
            if s.state.phase == Phase::Complete {
                break;
            }
            if let Progress::Epoch { improved: true, .. } = s.advance()? {
                checkpoint::publish_training(&run, "best", &mut s, &p, None)?;
            }
            checkpoint::publish_training(&run, "last", &mut s, &p, None)?;
        }
        assert_eq!(phases.len(), 3);
        assert_eq!(
            checkpoint::load(run.join("best"), &device)?.manifest.epoch,
            s.state.early.best_epoch
        );
        let changed = Provenance {
            code_sha: "new-commit-same-source".into(),
            dirty: false,
            ..p.clone()
        };
        checkpoint::load_session(run.join("last"), &c, &d, &device, &changed)?;
        let bad = Provenance {
            source_sha256: "b".repeat(64),
            ..p.clone()
        };
        assert!(checkpoint::load_session(run.join("last"), &c, &d, &device, &bad).is_err());
        let mut bad = p.clone();
        bad.environment.insert("test".into(), "changed".into());
        assert!(checkpoint::load_session(run.join("last"), &c, &d, &device, &bad).is_err());
    }
    Ok(())
}
#[test]
fn corrupt_v2_payloads_fail_without_changing_live_state() -> R {
    let device = Device::Cpu;
    let (c, d, p) = example(&device)?;
    let run = directory();
    let mut s = TrainingSession::new(&c, &d, &device)?;
    s.advance()?;
    let original = checkpoint::publish_training(&run, "last", &mut s, &p, None)?;
    let before = s.snapshot()?;
    for kind in [
        "extra",
        "missing",
        "hash",
        "mapping",
        "parent",
        "schema",
        "cursor",
        "phase",
        "rng",
        "rng_version",
        "lr",
        "groups",
        "step",
        "negative_moment",
        "nonfinite_moment",
        "tensor_names",
        "shape",
        "dtype",
        "nonfinite_weights",
        "buffers",
        "environment",
        "config",
        "source",
        "data",
        "aggregate",
        "unknown",
    ] {
        let b = directory();
        for entry in fs::read_dir(&original)? {
            let e = entry?;
            if kind == "missing" && e.file_name() == "rng-state.json" {
                continue;
            }
            fs::copy(e.path(), b.join(e.file_name()))?;
        }
        match kind {
            "extra" => {
                fs::write(b.join("extra.json"), b"{}")?;
            }
            "missing" => {}
            "hash" => {
                fs::write(b.join("rng-state.json"), b"{}")?;
            }
            "mapping" => edit(&b, "manifest.json", |j| {
                j["training"]["mapping_version"] = 99.into()
            })?,
            "parent" => {
                edit(
                    &b,
                    "manifest.json",
                    |j| j["training"]["parent"] = serde_json::json!({"generation":"../outside","manifest_sha256":"a".repeat(64)}),
                )?
            }
            "schema" => edit(&b, "data-schema.json", |j| {
                j["input_columns"] = serde_json::json!(["y", "x"])
            })?,
            "cursor" => edit(&b, "trainer-state.json", |j| {
                j["next_batch_offset"] = 999.into()
            })?,
            "phase" => edit(&b, "trainer-state.json", |j| j["phase"] = "validate".into())?,
            "rng" => edit(&b, "rng-state.json", |j| {
                j["post_init"]["index"] = 625.into()
            })?,
            "rng_version" => edit(&b, "rng-state.json", |j| j["format_version"] = 99.into())?,
            "lr" => edit(&b, "scheduler-state.json", |j| j["current_lr"] = 0.9.into())?,
            "groups" => edit(&b, "optimizer-state.json", |j| {
                j["groups"][0]["beta1"] = 0.8.into()
            })?,
            "step" => edit(&b, "optimizer-state.json", |j| {
                j["parameters"][0]["step"] = 999.into()
            })?,
            "environment" => edit(&b, "environment.json", |j| j["test"] = "different".into())?,
            "config" => edit(&b, "manifest.json", |j| {
                j["config_sha256"] = "b".repeat(64).into()
            })?,
            "source" => edit(&b, "manifest.json", |j| {
                j["provenance"]["source_sha256"] = "b".repeat(64).into()
            })?,
            "data" => edit(&b, "manifest.json", |j| {
                j["data_sha256"] = "b".repeat(64).into()
            })?,
            "aggregate" => edit(&b, "trainer-state.json", |j| j["train_sum"] = (-1).into())?,
            "unknown" => edit(&b, "trainer-state.json", |j| j["unrecognized"] = 1.into())?,
            "buffers" => {
                let tensors = std::collections::HashMap::from([(
                    "unexpected".to_owned(),
                    Tensor::zeros(1, DType::F32, &device)?,
                )]);
                candle_core::safetensors::save(&tensors, b.join("buffers.safetensors"))?;
                repair(&b, "buffers.safetensors")?;
            }
            other => {
                let file = if other.ends_with("moment") {
                    "optimizer.safetensors"
                } else {
                    "model.safetensors"
                };
                let mut tensors = candle_core::safetensors::load(b.join(file), &device)?;
                let name = tensors
                    .keys()
                    .find(|n| file == "model.safetensors" || n.ends_with("exp_avg_sq"))
                    .unwrap()
                    .clone();
                let t = &tensors[&name];
                let new = match other {
                    "shape" => Tensor::zeros(99, DType::F32, &device)?,
                    "dtype" => t.to_dtype(DType::F64)?,
                    "negative_moment" => Tensor::full(-1f32, t.shape(), &device)?,
                    "nonfinite_moment" | "nonfinite_weights" => {
                        Tensor::full(f32::NAN, t.shape(), &device)?
                    }
                    "tensor_names" => {
                        tensors.insert("unexpected".into(), t.clone());
                        tensors.remove(&name).unwrap()
                    }
                    _ => unreachable!(),
                };
                if other != "tensor_names" {
                    tensors.insert(name, new);
                }
                candle_core::safetensors::save(&tensors, b.join(file))?;
                repair(&b, file)?;
            }
        }
        assert!(
            checkpoint::load_session(&b, &c, &d, &device, &p).is_err(),
            "{kind}"
        );
        assert_eq!(checkpoint::resolve(&run.join("last"))?, original);
    }
    let old = TrainingSession::resume(&c, &d, &device, before)?;
    assert_session(&s, &old)?;
    Ok(())
}
#[test]
fn publish_faults_keep_last_committed_generation_and_live_state() -> R {
    let device = Device::Cpu;
    let (c, d, p) = example(&device)?;
    let run = directory();
    let mut s = TrainingSession::new(&c, &d, &device)?;
    let original = checkpoint::publish_training(&run, "last", &mut s, &p, None)?;
    s.advance()?;
    let before = serde_json::to_value(&s.state)?;
    for target in [
        PublishStage::StagingCreated,
        PublishStage::PayloadsWritten,
        PublishStage::ManifestWritten,
        PublishStage::Validated,
        PublishStage::GenerationCommitted,
        PublishStage::ReferencePending,
    ] {
        let result =
            checkpoint::publish_training_with_hook(&run, "last", &mut s, &p, None, |stage| {
                if stage == target {
                    Err(TrainError::InvalidState(
                        "injected publication fault".into(),
                    ))
                } else {
                    Ok(())
                }
            });
        assert!(result.is_err());
        assert_eq!(checkpoint::resolve(&run.join("last"))?, original);
        checkpoint::load_session(run.join("last"), &c, &d, &device, &p)?;
        assert_eq!(serde_json::to_value(&s.state)?, before);
    }
    assert_eq!(fs::read_dir(run.join(".staging"))?.count(), 4);
    assert_eq!(fs::read_dir(run.join("checkpoints"))?.count(), 3);
    fs::write(run.join("last/00000000000000000002.json"), b"corrupt")?;
    assert!(checkpoint::load_session(run.join("last"), &c, &d, &device, &p).is_err());
    Ok(())
}
