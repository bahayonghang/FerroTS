use std::{fs, path::Path};
fn inspect(path: &Path, violations: &mut Vec<String>) {
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            inspect(&path, violations);
        } else if path.extension().is_some_and(|x| x == "rs")
            && path.file_name().unwrap() != "infer_ops.rs"
        {
            let text = fs::read_to_string(&path).unwrap();
            let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
            for forbidden in [
                "softmax_last_dim",
                "candle_nn::LayerNorm",
                "rms_norm",
                "apply_op1_no_bwd",
                "apply_op3_no_bwd",
            ] {
                if compact.contains(forbidden) {
                    violations.push(format!("{}: {forbidden}", path.display()));
                }
            }
            // Remove the local differentiable path before checking qualified backend calls.
            if compact
                .replace("train_ops::layer_norm", "")
                .contains("ops::layer_norm")
            {
                violations.push(format!("{}: ops::layer_norm", path.display()));
            }
        }
    }
}
#[test]
fn no_fused_in_train() {
    let mut violations = Vec::new();
    inspect(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut violations,
    );
    assert!(
        violations.is_empty(),
        "forbidden inference operations in training source: {violations:?}"
    );
}
