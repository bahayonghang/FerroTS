use candle_core::Device;
use ferrots_core::config::{Activation, Config, MissingMode, ModelSpec, SplitSpec, WindowSpec};
use ferrots_data::{batch::Dataset, cache::hash_bytes};
use ferrots_nn::informer::{Informer, InformerSamplingSchedule};
use ferrots_train::session::{Progress, TrainingSession};
use std::fs;

type R = Result<(), Box<dyn std::error::Error>>;

fn sha(order: &[usize]) -> String {
    let bytes: Vec<u8> = order
        .iter()
        .flat_map(|index| (*index as u64).to_le_bytes())
        .collect();
    hash_bytes(&bytes)
}

fn h720_schedule() -> InformerSamplingSchedule {
    InformerSamplingSchedule {
        lookback: 96,
        label_len: 48,
        horizon: 720,
        factor: 3,
        e_layers: 2,
        d_layers: 1,
        distil: true,
        batch_size: 32,
        train_windows: 7825,
        val_windows: 2161,
        test_windows: 2161,
        train_drop_last: false,
    }
}

#[test]
fn etth1_h720_seed2022_epoch1_order_matches_official_loader() -> R {
    let schedule = h720_schedule();
    assert_eq!(
        schedule.randint_sites(),
        vec![(96, 1_440), (49, 588), (768, 16_128), (49, 9_216)]
    );
    let spec = ModelSpec::Informer {
        d_model: 512,
        n_heads: 8,
        d_ff: 2048,
        e_layers: 2,
        d_layers: 1,
        factor: 3,
        distil: true,
        label_len: 48,
        dropout: 0.1,
        activation: Activation::Gelu,
    };
    let window = WindowSpec {
        lookback: 96,
        horizon: 720,
        gap: 0,
        stride: 1,
    };
    let model = Informer::new(&spec, &window, 7, 4, 2022, &Device::Cpu)?;
    let rng = model.sampling_rng().clone();
    drop(model);
    let epoch0 = schedule.permutation(&rng, 0);
    let epoch1 = schedule.permutation(&rng, 1);
    assert_eq!(epoch0, rng.training_permutation(7825, 0));
    assert_ne!(epoch1, rng.training_permutation(7825, 1));
    assert_eq!(&epoch0[..4], &[3230, 4482, 6428, 980]);
    assert_eq!(
        sha(&epoch0),
        "63a5adfa3e9174942f9bc3c58ec82e69261aa877845f041d857cb8e7e5fdd3d7"
    );
    assert_eq!(&epoch1[..4], &[424, 5564, 5235, 5596]);
    assert_eq!(
        sha(&epoch1),
        "45a15072381dafa7c6a8f4801eb52e5c63e30945abb3456ed8009278e9da210f"
    );
    Ok(())
}

#[test]
fn session_epoch2_order_uses_the_randint_schedule() -> R {
    let root = std::env::temp_dir().join(format!(
        "ferrots-informer-order-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    fs::create_dir_all(&root)?;
    let mut text = "date,x,y\n".to_string();
    for i in 0..80 {
        let day = 1 + i / 24;
        let hour = i % 24;
        text.push_str(&format!(
            "2024-01-{day:02} {hour:02}:00:00,{},{}\n",
            i as f64 * 0.1,
            i as f64 * 0.2
        ));
    }
    let path = root.join("synthetic.csv");
    fs::write(&path, text)?;
    let mut config = Config::default();
    config.data.path = path;
    config.data.input_columns = vec!["x".into(), "y".into()];
    config.data.target_columns = vec!["y".into()];
    config.data.interval_seconds = 3600;
    config.data.time_features = "h".into();
    config.data.split = SplitSpec::Rows {
        train_end: 48,
        validation_end: 64,
        test_end: 80,
    };
    config.data.cache_dir = root.join("cache");
    config.data.missing.mode = MissingMode::Causal;
    config.data.missing.max_stale_seconds = 0;
    config.window.lookback = 8;
    config.window.horizon = 4;
    config.window.stride = 1;
    config.model = ModelSpec::Informer {
        d_model: 8,
        n_heads: 2,
        d_ff: 16,
        e_layers: 2,
        d_layers: 1,
        factor: 3,
        distil: true,
        label_len: 4,
        dropout: 0.0,
        activation: Activation::Gelu,
    };
    config.train.device = "cpu".into();
    config.train.seed = 2022;
    config.train.batch_size = 4;
    config.train.accumulation_steps = 1;
    config.train.max_epochs = 2;
    config.train.patience = 3;
    config.output.root = root.join("runs");
    config.output.run_id = Some("order".into());
    let data = Dataset::load(&config.data, &config.window, None)?;
    let device = Device::Cpu;
    let mut session = TrainingSession::new(&config, &data, &device)?;
    let mut finished = false;
    for _ in 0..64 {
        match session.advance()? {
            Progress::Epoch { record, .. } => {
                assert_eq!(record.epoch, 1);
                finished = true;
                break;
            }
            Progress::Training { .. } => {}
            Progress::Complete => panic!("session stopped before rebuilding the next order"),
        }
    }
    assert!(finished, "epoch did not finish");
    assert_eq!(session.state.epoch, 2);
    let ModelSpec::Informer {
        factor,
        e_layers,
        d_layers,
        distil,
        label_len,
        ..
    } = &config.model
    else {
        panic!("expected Informer");
    };
    let entries = data.index.order(0);
    let schedule = InformerSamplingSchedule {
        lookback: config.window.lookback,
        label_len: *label_len,
        horizon: config.window.horizon,
        factor: *factor,
        e_layers: *e_layers,
        d_layers: *d_layers,
        distil: *distil,
        batch_size: config.train.batch_size,
        train_windows: entries.len(),
        val_windows: data.index.counts[1],
        test_windows: data.index.counts[2],
        train_drop_last: config.train.drop_last,
    };
    let order: Vec<usize> = schedule
        .permutation(session.model.sampling_rng(), 1)
        .into_iter()
        .map(|index| entries[index])
        .collect();
    let previous: Vec<usize> = session
        .model
        .sampling_rng()
        .training_permutation(entries.len(), 1)
        .into_iter()
        .map(|index| entries[index])
        .collect();
    assert_ne!(order, previous);
    assert_eq!(session.state.order_sha256, sha(&order));
    Ok(())
}
