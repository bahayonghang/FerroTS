use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};
use std::{env, process::Command};
fn git(root: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .expect("git metadata");
    assert!(output.status.success(), "git metadata failed");
    String::from_utf8(output.stdout)
        .expect("UTF8 git metadata")
        .trim()
        .to_string()
}
fn source_files(root: &Path, path: &Path, files: &mut BTreeMap<String, String>) {
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("source directory") {
            source_files(root, &entry.expect("source entry").path(), files);
        }
    } else {
        let name = path
            .strip_prefix(root)
            .expect("source below root")
            .to_string_lossy()
            .replace('\\', "/");
        if name.split('/').any(|p| p == "tests") {
            return;
        }
        if path.extension().is_some_and(|s| s == "rs")
            || path.file_name().is_some_and(|s| s == "Cargo.toml")
        {
            println!("cargo:rerun-if-changed={}", path.display());
            files.insert(
                name,
                format!("{:x}", Sha256::digest(fs::read(path).expect("source file"))),
            );
        }
    }
}
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    let root =
        std::path::PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"))
            .join("../..")
            .canonicalize()
            .expect("workspace root");
    let mut files = BTreeMap::new();
    source_files(&root, &root.join("crates"), &mut files);
    for name in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"] {
        let path = root.join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        files.insert(
            name.into(),
            format!(
                "{:x}",
                Sha256::digest(fs::read(path).expect("workspace manifest"))
            ),
        );
    }
    let hash = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&files).expect("source identity"))
    );
    println!("cargo:rustc-env=FERROTS_SOURCE_SHA256={hash}");
    println!(
        "cargo:rustc-env=FERROTS_CODE_SHA={}",
        git(&root, &["rev-parse", "HEAD"])
    );
    println!(
        "cargo:rustc-env=FERROTS_DIRTY={}",
        !git(&root, &["status", "--porcelain"]).is_empty()
    );
    for name in ["HEAD", "index"] {
        println!(
            "cargo:rerun-if-changed={}",
            root.join(git(&root, &["rev-parse", "--git-path", name]))
                .display()
        );
    }
    let branch = git(&root, &["symbolic-ref", "HEAD"]);
    println!(
        "cargo:rerun-if-changed={}",
        root.join(git(&root, &["rev-parse", "--git-path", &branch]))
            .display()
    );
    let output = Command::new(env::var_os("RUSTC").expect("Cargo sets RUSTC"))
        .arg("--version")
        .output()
        .expect("read rustc version");
    assert!(output.status.success(), "rustc --version failed");
    println!(
        "cargo:rustc-env=FERROTS_RUSTC={}",
        String::from_utf8(output.stdout)
            .expect("rustc output is UTF-8")
            .trim()
    );
    println!(
        "cargo:rustc-env=FERROTS_TARGET={}",
        env::var("TARGET").expect("Cargo sets TARGET")
    );
    println!(
        "cargo:rustc-env=FERROTS_FEATURES={}",
        if env::var_os("CARGO_FEATURE_CUDA").is_some() {
            "cuda"
        } else {
            "cpu-only"
        }
    );
}
