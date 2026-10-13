use std::process::Command;

#[test]
fn model_list_filters_task_and_build_device() {
    let output = Command::new(env!("CARGO_BIN_EXE_ferrots"))
        .args(["models", "list", "--task", "forecast", "--device", "cuda:0"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let models = result["models"].as_array().unwrap();
    assert_eq!(models.len(), if cfg!(feature = "cuda") { 6 } else { 0 });
    if cfg!(feature = "cuda") {
        let model = models.iter().find(|m| m["name"] == "itransformer").unwrap();
        assert_eq!(model["parameters"].as_array().unwrap().len(), 7);
        assert_eq!(model["history_time"], true);
        assert_eq!(model["future_known"], false);
        assert_eq!(model["source_sha"], ferrots_nn::itransformer::UPSTREAM_SHA);
    }
    if cfg!(feature = "cuda") {
        let model = models.iter().find(|m| m["name"] == "informer").unwrap();
        assert_eq!(model["reproduction_status"], "preview");
        assert_eq!(model["source_sha"], ferrots_nn::informer::UPSTREAM_SHA);
        assert_eq!(model["history_time"], true);
    }
    let output = Command::new(env!("CARGO_BIN_EXE_ferrots"))
        .args(["models", "list", "--task", "classification"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(result["models"].as_array().unwrap().is_empty());
}
