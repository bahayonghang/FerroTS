use candle_core::Device;
use ferrots_core::config::{Config, MissingMode, ModelSpec, SplitSpec};
use ferrots_data::batch::Dataset;
use ferrots_train::{
    checkpoint::{self, Provenance},
    trainer::{EarlyStopping, SplitAccess, Trainer, validation_loss},
};
use std::{collections::BTreeMap, fs};
type R = Result<(), Box<dyn std::error::Error>>;
fn example(
    model: ModelSpec,
    empty_split: Option<u8>,
) -> Result<(Config, Dataset, Provenance), Box<dyn std::error::Error>> {
    let root = std::env::temp_dir().join(format!(
        "ferrots-trainer-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    fs::create_dir_all(&root)?;
    let mut text = "date,x,y\n".to_string();
    for i in 0..30 {
        let split = if i < 18 {
            0
        } else if i < 24 {
            1
        } else {
            2
        };
        let y = if empty_split == Some(split) && i % 2 == 1 {
            String::new()
        } else {
            (i as f64 * 0.2).to_string()
        };
        text.push_str(&format!(
            "2024-01-{:02} 00:00:00,{},{}\n",
            i + 1,
            i as f64 * 0.1,
            y
        ));
    }
    let path = root.join("synthetic.csv");
    fs::write(&path, text)?;
    let mut c = Config::default();
    c.data.path = path;
    c.data.input_columns = vec!["x".into(), "y".into()];
    c.data.target_columns = vec!["y".into()];
    c.data.interval_seconds = 86400;
    c.data.split = SplitSpec::Rows {
        train_end: 18,
        validation_end: 24,
        test_end: 30,
    };
    c.data.cache_dir = root.join("cache");
    c.data.missing.mode = MissingMode::Causal;
    c.data.missing.max_stale_seconds = 0;
    c.window.lookback = 1;
    c.window.horizon = 1;
    c.window.stride = 2;
    c.model = model;
    c.train.batch_size = 2;
    c.train.accumulation_steps = 2;
    c.train.max_epochs = 2;
    c.output.root = root.join("runs");
    c.output.run_id = Some("case".into());
    let data = Dataset::load(&c.data, &c.window, None)?;
    let p = Provenance {
        code_sha: "synthetic".into(),
        dirty: true,
        source_sha256: "a".repeat(64),
        environment: BTreeMap::new(),
    };
    Ok((c, data, p))
}
#[test]
fn fit_retains_tail_freezes_validation_and_never_iterates_test() -> R {
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    for d in devices {
        let (mut c, data, p) = example(
            ModelSpec::Dlinear {
                individual: false,
                moving_avg: 25,
            },
            None,
        )?;
        c.train.device = if d.is_cpu() { "cpu" } else { "cuda:0" }.into();
        let result = Trainer {
            config: &c,
            data: &data,
            device: &d,
            provenance: &p,
        }
        .fit()?;
        assert_eq!(result.epochs.len(), 2);
        assert_eq!(result.access.windows[0], data.index.counts[0] * 2);
        assert_eq!(result.access.windows[1], data.index.counts[1] * 2);
        assert_eq!(result.access.windows[2], 0);
        let batches = data.index.counts[0].div_ceil(c.train.batch_size);
        assert_eq!(result.next_forward_id, batches as u64 * 2);
        assert_eq!(result.optimizer_steps, batches.div_ceil(2) as u64 * 2);
        let best = checkpoint::load(result.run_dir.join("best"), &d)?;
        assert!(best.model.is_frozen());
        let mut access = SplitAccess::default();
        assert_eq!(
            validation_loss(&best.model, &data, &c, &d, &mut access)?,
            result.best_validation_loss
        );
        assert!(
            Trainer {
                config: &c,
                data: &data,
                device: &d,
                provenance: &p
            }
            .fit()
            .is_err()
        );
    }
    Ok(())
}
#[test]
fn no_parameter_fit_has_no_optimizer_or_fake_training_curve() -> R {
    for model in [ModelSpec::LastValue, ModelSpec::SeasonalNaive { period: 1 }] {
        let (c, data, p) = example(model, None)?;
        let r = Trainer {
            config: &c,
            data: &data,
            device: &Device::Cpu,
            provenance: &p,
        }
        .fit()?;
        assert_eq!(r.optimizer_steps, 0);
        assert!(r.epochs.is_empty());
        assert_eq!(r.access.windows[0], 0);
        assert_eq!(r.access.windows[2], 0);
    }
    Ok(())
}

#[test]
fn explicit_drop_last_changes_train_only() -> R {
    let (mut c, data, p) = example(ModelSpec::Linear { individual: false }, None)?;
    c.train.drop_last = true;
    let result = Trainer {
        config: &c,
        data: &data,
        device: &Device::Cpu,
        provenance: &p,
    }
    .fit()?;
    assert_eq!(
        result.access.windows[0],
        data.index.counts[0] / c.train.batch_size * c.train.batch_size * result.epochs.len()
    );
    assert_eq!(
        result.access.windows[1],
        data.index.counts[1] * result.epochs.len()
    );
    assert_eq!(result.access.windows[2], 0);
    Ok(())
}
#[test]
fn zero_label_epoch_or_validation_fails_and_ties_count_patience() -> R {
    for split in [0, 1] {
        let (c, data, p) = example(ModelSpec::Linear { individual: false }, Some(split))?;
        let error = Trainer {
            config: &c,
            data: &data,
            device: &Device::Cpu,
            provenance: &p,
        }
        .fit()
        .unwrap_err()
        .to_string();
        assert!(error.contains("no valid labels"), "{error}");
    }
    let mut e = EarlyStopping::new(2)?;
    assert!(e.observe(1, 1.)?);
    assert!(!e.observe(2, 1.)?);
    assert!(!e.should_stop());
    assert!(!e.observe(3, 1.)?);
    assert!(e.should_stop());
    assert_eq!(e.best_epoch, 1);
    assert!(e.observe(4, f64::NAN).is_err());
    Ok(())
}

#[test]
fn dropout_forward_count_includes_empty_microbatches_and_is_independent_of_updates() -> R {
    use ferrots_core::config::Activation;
    for accumulation_steps in [1, 2] {
        let (mut config, mut data, provenance) = example(
            ModelSpec::Itransformer {
                d_model: 8,
                n_heads: 2,
                d_ff: 16,
                e_layers: 2,
                dropout: 0.1,
                activation: Activation::Gelu,
                use_norm: true,
            },
            None,
        )?;
        config.train.accumulation_steps = accumulation_steps;
        let entries = data.index.order(0);
        for &index in entries.iter().skip(1) {
            let target_row = data.index.entries[index].origin;
            data.series.observed[target_row * 2 + 1] = 0;
        }
        let result = Trainer {
            config: &config,
            data: &data,
            device: &Device::Cpu,
            provenance: &provenance,
        }
        .fit()?;
        let forwards = entries.len().div_ceil(config.train.batch_size) as u64;
        assert_eq!(
            result.next_forward_id,
            forwards * result.epochs.len() as u64
        );
        assert_eq!(result.optimizer_steps, result.epochs.len() as u64);
        assert!(result.epochs.iter().all(|e| e.skipped_empty_batches > 0));
        for e in &result.epochs {
            assert_eq!(e.next_forward_id, forwards * e.epoch as u64);
        }
        assert_eq!(result.access.windows[2], 0);
    }
    Ok(())
}

fn state_without_timing(state: &ferrots_train::session::TrainingState) -> serde_json::Value {
    let mut value = serde_json::to_value(state).unwrap();
    value.as_object_mut().unwrap().remove("epoch_seconds");
    value.as_object_mut().unwrap().remove("steady_seconds");
    for epoch in value["epochs"].as_array_mut().unwrap() {
        for name in [
            "seconds",
            "train_seconds",
            "validation_seconds",
            "steady_seconds",
        ] {
            epoch.as_object_mut().unwrap().remove(name);
        }
    }
    value
}

#[test]
fn every_in_memory_group_and_phase_resumes_without_replay() -> R {
    use ferrots_core::config::Activation;
    use ferrots_train::session::{Phase, TrainingSession};
    for accumulation in [1, 2] {
        let (mut config, _, _) = example(
            ModelSpec::Itransformer {
                d_model: 8,
                n_heads: 2,
                d_ff: 16,
                e_layers: 2,
                dropout: 0.1,
                activation: Activation::Gelu,
                use_norm: true,
            },
            None,
        )?;
        config.window.lookback = 3;
        config.window.horizon = 2;
        config.window.stride = 1;
        config.train.accumulation_steps = accumulation;
        config.train.max_epochs = 3;
        config.train.patience = 4;
        let data = Dataset::load(&config.data, &config.window, None)?;
        let mut direct = TrainingSession::new(&config, &data, &Device::Cpu)?;
        let mut resumed = TrainingSession::new(&config, &data, &Device::Cpu)?;
        let mut phases = std::collections::BTreeSet::new();
        while direct.state.phase != Phase::Complete {
            direct.advance()?;
            resumed.advance()?;
            assert!(direct.at_boundary() && resumed.at_boundary());
            phases.insert(format!("{:?}", resumed.state.phase));
            let snapshot = resumed.snapshot()?;
            resumed = TrainingSession::resume(&config, &data, &Device::Cpu, snapshot)?;
            assert_eq!(
                state_without_timing(&direct.state),
                state_without_timing(&resumed.state)
            );
            assert_eq!(direct.next_entries(), resumed.next_entries());
            assert_eq!(
                serde_json::to_value(direct.rng_state())?,
                serde_json::to_value(resumed.rng_state())?
            );
            for (name, value) in direct.model.weights() {
                assert_eq!(
                    value.flatten_all()?.to_vec1::<f32>()?,
                    resumed.model.weights()[name]
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
                let a = direct.optimizer.state(name).unwrap();
                let b = resumed.optimizer.state(name).unwrap();
                assert_eq!(a.step, b.step);
                assert_eq!(
                    a.exp_avg.flatten_all()?.to_vec1::<f32>()?,
                    b.exp_avg.flatten_all()?.to_vec1::<f32>()?
                );
                assert_eq!(
                    a.exp_avg_sq.flatten_all()?.to_vec1::<f32>()?,
                    b.exp_avg_sq.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        assert_eq!(
            phases,
            ["Train", "Validate", "Complete"]
                .into_iter()
                .map(str::to_owned)
                .collect()
        );
        assert_eq!(
            direct
                .state
                .epochs
                .iter()
                .map(|r| r.learning_rate)
                .collect::<Vec<_>>(),
            [
                config.train.learning_rate,
                config.train.learning_rate,
                config.train.learning_rate / 2.
            ]
        );
        assert_eq!(direct.state.access.windows[2], 0);
        assert!(matches!(
            resumed.advance()?,
            ferrots_train::session::Progress::Complete
        ));
    }
    Ok(())
}

#[test]
fn memory_resume_rejects_cursor_rng_scheduler_and_optimizer_state_corruption() -> R {
    use ferrots_train::session::TrainingSession;
    let (config, data, _) = example(ModelSpec::Linear { individual: false }, None)?;
    let session = TrainingSession::new(&config, &data, &Device::Cpu)?;
    for kind in ["cursor", "rng", "scheduler", "step", "nonfinite", "config"] {
        let mut snapshot = session.snapshot()?;
        match kind {
            "cursor" => snapshot.trainer.next_batch_offset = 1,
            "rng" => snapshot.rng.post_init.index = 625,
            "scheduler" => snapshot.scheduler.current_lr *= 2.,
            "step" => snapshot.optimizer.state.values_mut().next().unwrap().step = 1,
            "nonfinite" => snapshot.trainer.train_sum = f64::NAN,
            _ => snapshot.config_sha256 = "bad".into(),
        }
        assert!(
            TrainingSession::resume(&config, &data, &Device::Cpu, snapshot).is_err(),
            "{kind}"
        );
    }
    session.validate_state()?;
    Ok(())
}

#[test]
fn restored_patience_ties_keep_first_best_and_complete_without_extra_schedule() -> R {
    use ferrots_train::session::{EndReason, Phase, TrainingSession};
    let (mut config, data, _) = example(ModelSpec::Linear { individual: false }, None)?;
    config.train.learning_rate = 1e-45;
    config.train.patience = 2;
    config.train.max_epochs = 10;
    let mut session = TrainingSession::new(&config, &data, &Device::Cpu)?;
    while session.state.phase != Phase::Complete {
        session.advance()?;
        session = TrainingSession::resume(&config, &data, &Device::Cpu, session.snapshot()?)?;
    }
    assert_eq!(session.state.early.best_epoch, 1);
    assert_eq!(session.state.early.stale, 2);
    assert_eq!(session.state.epoch, 3);
    assert_eq!(session.state.end_reason, Some(EndReason::EarlyStopping));
    assert_eq!(session.scheduler.applied_epoch, 2);
    Ok(())
}
