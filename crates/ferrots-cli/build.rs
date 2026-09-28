use std::{env, process::Command};
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
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
