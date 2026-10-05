use ferrots_core::config::{Config, ModelSpec, SplitSpec};
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};
static FIXTURE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
pub fn fixture(model: ModelSpec) -> (PathBuf, Config) {
    let root = std::env::temp_dir().join(format!(
        "ferrots-cli-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        FIXTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let mut text = "date,x,y\n".to_string();
    for i in 0..22 {
        text.push_str(&format!("2024-01-{:02} 00:00:00,{},{}\n", i + 1, i, i * 2));
    }
    fs::write(root.join("synthetic.csv"), text).unwrap();
    let mut c = Config::default();
    c.data.path = root.join("synthetic.csv");
    c.data.input_columns = vec!["x".into(), "y".into()];
    c.data.target_columns = c.data.input_columns.clone();
    c.data.interval_seconds = 86400;
    c.data.cache_dir = root.join("cache");
    c.data.split = SplitSpec::Rows {
        train_end: 12,
        validation_end: 17,
        test_end: 22,
    };
    c.window.lookback = 2;
    c.window.horizon = 2;
    c.model = model;
    c.train.max_epochs = 2;
    c.train.batch_size = 4;
    c.output.root = root.join("runs");
    c.output.run_id = Some("model".into());
    write_config(&root, &c);
    (root, c)
}
pub fn write_config(root: &std::path::Path, c: &Config) {
    fs::write(root.join("config.toml"), toml::to_string_pretty(c).unwrap()).unwrap();
}
pub fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ferrots"))
        .args(args)
        .output()
        .unwrap()
}
pub fn json(args: &[&str]) -> serde_json::Value {
    let o = run(args);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&o.stdout).unwrap()
}
