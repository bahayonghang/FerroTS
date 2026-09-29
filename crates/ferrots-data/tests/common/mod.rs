#![allow(dead_code)]
use ferrots_core::config::DataSpec;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
pub fn directory() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "ferrots-p1-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}
pub fn csv(text: &str, names: &[&str]) -> DataSpec {
    let dir = directory();
    let path = dir.join("synthetic.csv");
    std::fs::write(&path, text).unwrap();
    DataSpec {
        path,
        input_columns: names.iter().map(|s| (*s).into()).collect(),
        target_columns: vec![names[0].into()],
        cache_dir: dir.join("cache"),
        ..DataSpec::default()
    }
}
