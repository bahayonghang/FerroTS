use candle_core::{Device, Tensor};
use ferrots_core::config::{Config, ModelSpec};
use ferrots_data::{
    cache::{fingerprint, protocol_hash},
    scaler::{ColumnScale, Scaler},
};
use ferrots_nn::baselines::{Baseline, ModelInput};
use ferrots_train::checkpoint::{self, Provenance, Snapshot};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
type R = Result<(), Box<dyn std::error::Error>>;
fn dir() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "ferrots-checkpoint-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}
fn config(model: ModelSpec) -> (Config, Scaler, Provenance) {
    let mut c = Config::default();
    c.data.path = "synthetic.csv".into();
    c.data.input_columns = vec!["x".into()];
    c.data.target_columns = vec!["x".into()];
    c.window.lookback = 2;
    c.window.horizon = 1;
    c.model = model;
    let s = Scaler {
        format_version: 1,
        protocol_id: c.data.protocol_id.clone(),
        protocol_fingerprint: protocol_hash(&c.data).unwrap(),
        fit_split: "train".into(),
        columns: vec![ColumnScale {
            name: "x".into(),
            count: 3,
            mean: 1.,
            scale: 2.,
            constant: false,
        }],
    };
    (
        c,
        s,
        Provenance {
            code_sha: "synthetic".into(),
            dirty: true,
            source_sha256: "a".repeat(64),
            environment: BTreeMap::new(),
        },
    )
}
fn repair_hash(bundle: &Path, name: &str) -> R {
    let path = bundle.join("manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    m["files"][name]["sha256"] = fingerprint(bundle.join(name))?.into();
    m["files"][name]["bytes"] = fs::metadata(bundle.join(name))?.len().into();
    fs::write(path, serde_json::to_vec(&m)?)?;
    Ok(())
}
#[test]
fn bundles_roundtrip_empty_and_trainable_and_keep_previous_reference() -> R {
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    for device in devices {
        for spec in [
            ModelSpec::LastValue,
            ModelSpec::Dlinear {
                individual: true,
                moving_avg: 25,
            },
        ] {
            let (c, s, p) = config(spec);
            let model = Baseline::new(&c.model, &c.window, 1, 1, &device)?;
            let run = dir();
            let snapshot = Snapshot {
                model: &model,
                config: &c,
                scaler: &s,
                provenance: &p,
                data_sha256: &"b".repeat(64),
                epoch: 1,
                validation_loss: 1.,
            };
            let bundle = checkpoint::publish(&run, "best", &snapshot)?;
            for name in [
                "model.safetensors",
                "scaler.json",
                "data-schema.json",
                "resolved-config.toml",
                "manifest.json",
            ] {
                assert!(bundle.join(name).is_file());
            }
            let loaded = checkpoint::load(run.join("best"), &device)?;
            let input = ModelInput::history(Tensor::from_vec(vec![1f32, 2.], (1, 2, 1), &device)?);
            let a = model
                .freeze_for_eval()?
                .forward(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let b = loaded
                .model
                .forward(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(a, b);
            assert!(!loaded.model.forward(&input)?.track_op());
            let partial = run.join(".staging/interrupted");
            fs::create_dir_all(&partial)?;
            fs::write(partial.join("model.safetensors"), b"partial")?;
            fs::write(run.join("best/.00000000000000000002.pending"), b"partial")?;
            assert_eq!(
                checkpoint::load(run.join("best"), &device)?.directory,
                bundle
            );
            checkpoint::publish(&run, "last", &snapshot)?;
            assert!(
                checkpoint::load(run.join("last"), &device)?
                    .manifest
                    .inference_only
            );
            fs::write(run.join("best/00000000000000000002.json"), b"corrupt")?;
            assert!(checkpoint::load(run.join("best"), &device).is_err());
        }
    }
    Ok(())
}
#[test]
fn corrupt_files_schema_shape_dtype_and_missing_files_fail() -> R {
    let (c, s, p) = config(ModelSpec::Linear { individual: false });
    let model = Baseline::new(&c.model, &c.window, 1, 1, &Device::Cpu)?;
    for kind in ["hash", "schema", "shape", "dtype", "missing"] {
        let run = dir();
        let bundle = checkpoint::publish(
            &run,
            "best",
            &Snapshot {
                model: &model,
                config: &c,
                scaler: &s,
                provenance: &p,
                data_sha256: &"b".repeat(64),
                epoch: 1,
                validation_loss: 1.,
            },
        )?;
        match kind {
            "hash" => fs::write(bundle.join("scaler.json"), b"bad")?,
            "schema" => {
                let f = bundle.join("data-schema.json");
                let mut j: serde_json::Value = serde_json::from_slice(&fs::read(&f)?)?;
                j["input_columns"] = serde_json::json!(["wrong"]);
                fs::write(&f, serde_json::to_vec(&j)?)?;
                repair_hash(&bundle, "data-schema.json")?;
            }
            "shape" | "dtype" => {
                let mut weights =
                    candle_core::safetensors::load(bundle.join("model.safetensors"), &Device::Cpu)?;
                let name = weights.keys().next().unwrap().clone();
                let bad = if kind == "shape" {
                    Tensor::zeros(99, candle_core::DType::F32, &Device::Cpu)?
                } else {
                    weights[&name].to_dtype(candle_core::DType::F64)?
                };
                weights.insert(name, bad);
                candle_core::safetensors::save(&weights, bundle.join("model.safetensors"))?;
                repair_hash(&bundle, "model.safetensors")?;
            }
            _ => {
                let missing = dir();
                fs::copy(bundle.join("manifest.json"), missing.join("manifest.json"))?;
                assert!(checkpoint::load(&missing, &Device::Cpu).is_err());
                continue;
            }
        }
        assert!(checkpoint::load(&bundle, &Device::Cpu).is_err(), "{kind}");
    }
    Ok(())
}
