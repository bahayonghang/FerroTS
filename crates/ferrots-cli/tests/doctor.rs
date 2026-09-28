use std::process::Command;
#[test]
fn cpu_training_smoke_reports_build_and_updates() {
    let output = Command::new(env!("CARGO_BIN_EXE_ferrots"))
        .args(["doctor", "--device", "cpu", "--training-smoke-test"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    for field in [
        "rustc:",
        "target:",
        "build_features:",
        "candle:",
        "cuda_driver_api:",
        "cuda_runtime:",
        "gpu_name:",
        "gpu_compute_capability:",
        "gpu_memory_bytes:",
        "forward_ms:",
        "backward_ms:",
        "update_ms:",
        "parameters_with_gradients: 6",
        "parameters_updated: 6",
        "training_smoke_test: passed",
    ] {
        assert!(stdout.contains(field), "missing {field}: {stdout}");
    }
}
#[test]
fn explicit_unavailable_or_invalid_devices_fail() {
    for spec in ["cuda:999999", "cuda:18446744073709551615", "gpu", "cuda:-1"] {
        let output = Command::new(env!("CARGO_BIN_EXE_ferrots"))
            .args(["doctor", "--device", spec])
            .output()
            .unwrap();
        assert!(!output.status.success(), "unexpected success for {spec}");
        assert!(!String::from_utf8_lossy(&output.stdout).contains("training_smoke_test: passed"));
    }
    #[cfg(not(feature = "cuda"))]
    {
        let output = Command::new(env!("CARGO_BIN_EXE_ferrots"))
            .args(["doctor", "--device", "cuda:0"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("without the cuda feature"));
    }
}
