mod common;
use ferrots_core::config::{Activation, ModelSpec};
use ferrots_train::checkpoint;
use std::{
    fs,
    path::{Path, PathBuf},
};

fn model() -> ModelSpec {
    ModelSpec::Itransformer {
        d_model: 8,
        n_heads: 2,
        d_ff: 16,
        e_layers: 2,
        dropout: 0.1,
        activation: Activation::Gelu,
        use_norm: true,
    }
}
fn run_dir(j: &serde_json::Value) -> PathBuf {
    PathBuf::from(j["run_dir"].as_str().unwrap())
}
fn inventory(root: &Path) -> std::collections::BTreeMap<PathBuf, String> {
    let mut result = std::collections::BTreeMap::new();
    for e in fs::read_dir(root).unwrap() {
        let e = e.unwrap();
        if e.file_type().unwrap().is_dir() {
            result.extend(inventory(&e.path()));
        } else {
            result.insert(
                e.path(),
                ferrots_data::cache::fingerprint(e.path()).unwrap(),
            );
        }
    }
    result
}
#[test]
fn pause_before_validation_resume_and_complete_without_mutating_parent() {
    let (root, c) = common::fixture(model());
    let config = root.join("config.toml");
    let pause = common::json(&[
        "train",
        "--config",
        config.to_str().unwrap(),
        "--stop-after-steps",
        "1",
    ]);
    assert_eq!(pause["status"], "paused");
    assert_eq!(pause["optimizer_steps"], 1);
    assert!(pause["best_validation_loss"].is_null());
    assert!(!run_dir(&pause).join("best").exists());
    let parent = run_dir(&pause);
    let old = inventory(&parent);
    let result = common::json(&[
        "resume",
        "--checkpoint",
        parent.join("last").to_str().unwrap(),
    ]);
    let run = run_dir(&result);
    assert_ne!(run, parent);
    assert_eq!(inventory(&parent), old);
    assert_eq!(
        fs::read(run.join("resolved-config.toml")).unwrap(),
        fs::read(parent.join("resolved-config.toml")).unwrap()
    );
    assert_eq!(
        result["epochs"].as_array().unwrap().len(),
        c.train.max_epochs
    );
    assert_eq!(result["access"]["windows"][2], 0);
    assert!(run.join("continuation.json").is_file());
    let metrics = common::json(&[
        "evaluate",
        "--checkpoint",
        run.join("best").to_str().unwrap(),
    ]);
    let predictions = common::json(&[
        "predict",
        "--checkpoint",
        run.join("best").to_str().unwrap(),
        "--output",
        root.join("prediction.csv").to_str().unwrap(),
    ]);
    assert_eq!(metrics["metrics"], predictions["metrics"]);
    let csv = fs::read_to_string(root.join("prediction.csv")).unwrap();
    assert!(
        csv.lines()
            .nth(1)
            .unwrap()
            .starts_with(result["run_id"].as_str().unwrap())
    );
    let repeated = common::json(&[
        "resume",
        "--checkpoint",
        run.join("last").to_str().unwrap(),
        "--stop-after-steps",
        "1",
    ]);
    assert_eq!(repeated["optimizer_steps"], result["optimizer_steps"]);
    assert_eq!(repeated["epochs"], result["epochs"]);
}
#[test]
fn continuation_inherits_verified_best_before_next_improvement() {
    let (root, _) = common::fixture(model());
    let pause = common::json(&[
        "train",
        "--config",
        root.join("config.toml").to_str().unwrap(),
        "--stop-after-steps",
        "4",
    ]);
    let parent = run_dir(&pause);
    let selected = checkpoint::resolve(&parent.join("best")).unwrap();
    let old = inventory(&parent);
    let continuation = common::json(&[
        "resume",
        "--checkpoint",
        parent.join("last").to_str().unwrap(),
        "--stop-after-steps",
        "1",
    ]);
    assert_eq!(continuation["status"], "paused");
    let inherited = checkpoint::resolve(&run_dir(&continuation).join("best")).unwrap();
    assert_ne!(selected, inherited);
    assert_eq!(selected.file_name(), inherited.file_name());
    for e in fs::read_dir(selected).unwrap() {
        let e = e.unwrap();
        assert_eq!(
            fs::read(e.path()).unwrap(),
            fs::read(inherited.join(e.file_name())).unwrap()
        );
    }
    assert_eq!(inventory(&parent), old);
}
#[test]
fn invalid_limits_overrides_v1_and_changed_data_are_rejected() {
    let (root, c) = common::fixture(model());
    let config = root.join("config.toml");
    let fail = common::run(&[
        "train",
        "--config",
        config.to_str().unwrap(),
        "--stop-after-steps",
        "0",
    ]);
    assert!(!fail.status.success());
    assert!(!c.output.root.join("model").exists());
    let paused = common::json(&[
        "train",
        "--config",
        config.to_str().unwrap(),
        "--stop-after-steps",
        "1",
    ]);
    let last = run_dir(&paused).join("last");
    for (arg, value) in [
        ("--seed", "999"),
        ("--batch-size", "1"),
        ("--device", "cpu"),
        ("--stop-after-steps", "0"),
    ] {
        assert!(
            !common::run(&["resume", "--checkpoint", last.to_str().unwrap(), arg, value])
                .status
                .success()
        );
    }
    let mut changed = fs::read_to_string(&c.data.path).unwrap();
    changed = changed.replace("2024-01-01 00:00:00,0,0", "2024-01-01 00:00:00,0,1");
    fs::write(&c.data.path, changed).unwrap();
    assert!(
        !common::run(&["resume", "--checkpoint", last.to_str().unwrap()])
            .status
            .success()
    );
    let (root, c) = common::fixture(ModelSpec::LastValue);
    common::json(&[
        "train",
        "--config",
        root.join("config.toml").to_str().unwrap(),
    ]);
    let fail = common::run(&[
        "resume",
        "--checkpoint",
        c.output.root.join("model/last").to_str().unwrap(),
    ]);
    assert!(!fail.status.success());
    assert!(String::from_utf8_lossy(&fail.stderr).contains("inference-only"));
}
