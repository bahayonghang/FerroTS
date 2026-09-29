mod common;
use ferrots_core::config::{DuplicatePolicy, MissingMode, ModelSpec};
use std::{fs, io::Write};

#[test]
fn single_element_constant_and_tail_batches_cover_every_window() {
    let devices = if cfg!(feature = "cuda") {
        vec!["cpu", "cuda:0"]
    } else {
        vec!["cpu"]
    };
    for device in devices {
        for batch in [1, 4] {
            let (root, mut config) = common::fixture(ModelSpec::Dlinear {
                individual: false,
                moving_avg: 25,
            });
            let mut text = String::from("date,x,y\n");
            for i in 0..22 {
                text.push_str(&format!("2024-01-{:02} 00:00:00,{},7\n", i + 1, i));
            }
            fs::write(&config.data.path, text).unwrap();
            config.window.lookback = 1;
            config.window.horizon = 1;
            config.train.device = device.into();
            config.train.batch_size = batch;
            config.train.max_epochs = 1;
            common::write_config(&root, &config);
            let path = root.join("config.toml");
            let report = common::json(&["data", "inspect", "--config", path.to_str().unwrap()]);
            assert_eq!(report["window_counts"], serde_json::json!([11, 5, 5]));
            assert!(
                report["raw"]["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|column| column["name"] == "y" && column["constant"] == true)
            );
            let trained = common::json(&["train", "--config", path.to_str().unwrap()]);
            assert_eq!(trained["optimizer_steps"], 11_usize.div_ceil(batch));
            assert_eq!(trained["access"]["windows"], serde_json::json!([11, 5, 0]));
            let best = config.output.root.join("model/best");
            let metrics = common::json(&["evaluate", "--checkpoint", best.to_str().unwrap()]);
            assert_eq!(metrics["metrics"]["standardized"]["overall"]["count"], 10);
        }
    }
}

#[test]
fn duplicate_policy_and_cross_segment_windows_are_explicit() {
    let (root, mut config) = common::fixture(ModelSpec::LastValue);
    let path = root.join("config.toml");
    fs::OpenOptions::new()
        .append(true)
        .open(&config.data.path)
        .unwrap()
        .write_all(b"2024-01-22 00:00:00,21,42\n")
        .unwrap();
    let bad = common::run(&["data", "inspect", "--config", path.to_str().unwrap()]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("duplicate timestamp"));
    config.data.duplicates = DuplicatePolicy::KeepFirst;
    common::write_config(&root, &config);
    let report = common::json(&["data", "inspect", "--config", path.to_str().unwrap()]);
    assert_eq!(report["raw"]["duplicate_groups"], 1);
    assert_eq!(report["raw"]["duplicate_rows"], 2);
    assert_eq!(report["raw"]["duplicate_removed"], 1);
    let mut text = String::from("date,x,y\n");
    for i in 0..22 {
        let day = if i < 10 { i + 1 } else { i + 4 };
        text.push_str(&format!("2024-01-{day:02} 00:00:00,{},{}\n", i, i * 2));
    }
    fs::write(&config.data.path, text).unwrap();
    let report = common::json(&["data", "inspect", "--config", path.to_str().unwrap()]);
    assert_eq!(report["window_counts"], serde_json::json!([7, 4, 4]));
    assert_eq!(report["window_audit"]["checked"], 15);
    assert_eq!(report["window_audit"]["all_same_segment"], true);
    assert_eq!(report["window_audit"]["all_target_split"], true);
}

#[test]
fn entirely_missing_training_labels_fail_without_updates() {
    let (root, mut config) = common::fixture(ModelSpec::Linear { individual: false });
    let mut text = String::from("date,x,y\n");
    for i in 0..22 {
        let label = if i % 2 == 1 {
            String::new()
        } else {
            i.to_string()
        };
        text.push_str(&format!("2024-01-{:02} 00:00:00,{i},{label}\n", i + 1));
    }
    fs::write(&config.data.path, text).unwrap();
    config.data.target_columns = vec!["y".into()];
    config.data.missing.mode = MissingMode::Causal;
    config.data.missing.max_stale_seconds = 0;
    config.window.lookback = 1;
    config.window.horizon = 1;
    config.window.stride = 2;
    common::write_config(&root, &config);
    let bad = common::run(&[
        "train",
        "--config",
        root.join("config.toml").to_str().unwrap(),
    ]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("training epoch has no valid labels"));
    assert!(!config.output.root.join("model/best").exists());
}

#[test]
fn reordered_columns_invalid_target_missing_file_and_corrupt_bundle() {
    let (root, mut config) = common::fixture(ModelSpec::LastValue);
    let path = root.join("config.toml");
    common::json(&["train", "--config", path.to_str().unwrap()]);
    let best = config.output.root.join("model/best");
    let expected = common::json(&["evaluate", "--checkpoint", best.to_str().unwrap()]);
    let mut text = String::from("y,date,x\n");
    for i in 0..22 {
        text.push_str(&format!("{},2024-01-{:02} 00:00:00,{}\n", i * 2, i + 1, i));
    }
    let reordered = root.join("reordered.csv");
    fs::write(&reordered, text).unwrap();
    let actual = common::json(&[
        "predict",
        "--checkpoint",
        best.to_str().unwrap(),
        "--input",
        reordered.to_str().unwrap(),
        "--output",
        root.join("predictions.csv").to_str().unwrap(),
    ]);
    assert_eq!(expected["metrics"], actual["metrics"]);
    config.data.target_columns = vec!["absent".into()];
    common::write_config(&root, &config);
    let bad = common::run(&["data", "inspect", "--config", path.to_str().unwrap()]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("target"));
    config.data.target_columns = vec!["x".into()];
    config.data.path = root.join("absent.csv");
    common::write_config(&root, &config);
    let bad = common::run(&["data", "inspect", "--config", path.to_str().unwrap()]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("data inspect failed"));
    let reference = fs::read_dir(&best).unwrap().next().unwrap().unwrap().path();
    let value: serde_json::Value = serde_json::from_slice(&fs::read(reference).unwrap()).unwrap();
    let model = config
        .output
        .root
        .join("model/checkpoints")
        .join(value["generation"].as_str().unwrap())
        .join("model.safetensors");
    fs::OpenOptions::new()
        .append(true)
        .open(model)
        .unwrap()
        .write_all(b"corrupt")
        .unwrap();
    let bad = common::run(&["evaluate", "--checkpoint", best.to_str().unwrap()]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("evaluate failed"));
}

#[test]
fn unavailable_device_requests_never_fall_back() {
    let device = if cfg!(feature = "cuda") {
        "cuda:9"
    } else {
        "cuda:0"
    };
    let bad = common::run(&["doctor", "--device", device]);
    assert!(!bad.status.success());
    let error = String::from_utf8_lossy(&bad.stderr);
    assert!(error.contains("CUDA") || error.contains("cuda"), "{error}");
}
