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
