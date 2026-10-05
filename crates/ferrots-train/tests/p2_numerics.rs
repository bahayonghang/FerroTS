//! Full official iTransformer parameter gate. Tolerances are fixed by the P2 plan.
use candle_core::{Device, Tensor, backprop::GradStore};
use ferrots_core::config::{Activation, ModelSpec, WindowSpec};
use ferrots_nn::{
    itransformer::ITransformer,
    model::{ForwardContext, ModelInput},
};
use ferrots_train::{NamedParams, ParamGroup, StatefulAdam};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

type R = Result<(), Box<dyn std::error::Error>>;

fn comparison(
    actual: &Tensor,
    expected: &Tensor,
    atol: f32,
    rtol: f32,
) -> candle_core::Result<Value> {
    let a = actual.flatten_all()?.to_vec1::<f32>()?;
    let e = expected.flatten_all()?.to_vec1::<f32>()?;
    let finite = a.iter().chain(&e).all(|v| v.is_finite());
    let shape = actual.shape() == expected.shape();
    let mut max_error = 0f32;
    let mut failed_elements = 0;
    let mut worst_index = 0;
    for (i, (a, e)) in a.iter().zip(&e).enumerate() {
        let error = (a - e).abs();
        if error > max_error {
            max_error = error;
            worst_index = i;
        }
        if !a.is_finite() || !e.is_finite() || error > atol + rtol * e.abs() {
            failed_elements += 1;
        }
    }
    Ok(json!({"pass": shape && finite && failed_elements == 0,
        "shape_match": shape, "finite": finite, "nonzero": a.iter().any(|&v| v != 0.),
        "max_error": max_error, "failed_elements": failed_elements, "worst_index": worst_index,
        "actual": a[worst_index], "reference": e[worst_index]}))
}

fn check(device: &Device, label: &str) -> R {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/p2/manifest.json"))?;
    let mut report = vec![];
    let mut failures = vec![];
    let mut update_diagnostics = vec![];
    for case in manifest["cases"].as_array().unwrap() {
        let id = case["name"].as_str().unwrap();
        let cfg = &case["config"];
        let n = |key: &str| cfg[key].as_u64().unwrap() as usize;
        let model = ITransformer::new(
            &ModelSpec::Itransformer {
                d_model: n("d_model"),
                n_heads: n("n_heads"),
                d_ff: n("d_ff"),
                e_layers: n("e_layers"),
                dropout: cfg["dropout"].as_f64().unwrap(),
                activation: if cfg["activation"] == "gelu" {
                    Activation::Gelu
                } else {
                    Activation::Relu
                },
                use_norm: cfg["use_norm"].as_bool().unwrap(),
            },
            &WindowSpec {
                lookback: n("seq_len"),
                horizon: n("pred_len"),
                gap: 0,
                stride: 1,
            },
            n("enc_in"),
            case["time_channels"].as_u64().unwrap() as usize,
            case["seed"].as_u64().unwrap(),
            device,
        )?;
        let values = candle_core::safetensors::load(
            root.join(format!("tests/fixtures/p2/{id}.safetensors")),
            device,
        )?;
        model.load_weights(
            &model
                .weights()
                .keys()
                .map(|name| (name.clone(), values[&format!("initial.{name}")].clone()))
                .collect(),
        )?;
        let history = if case["noncontiguous_input"] == true {
            values["input.history"]
                .transpose(1, 2)?
                .contiguous()?
                .transpose(1, 2)?
        } else {
            values["input.history"].clone()
        };
        let (prediction, trace) = model.forward_with_trace(
            &ModelInput {
                history,
                history_time: values.get("input.history_time").cloned(),
                future_time: None,
            },
            ForwardContext::evaluation(),
        )?;
        let loss = prediction.sub(&values["input.target"])?.sqr()?.mean_all()?;
        let gradients = loss.backward()?;
        let mut probe = loss.clone();
        for (name, tensor) in &trace {
            if name.ends_with(".key_projection.0") {
                probe = probe.add(&tensor.sqr()?.mean_all()?)?;
            }
        }
        let probe_gradients = probe.backward()?;
        let params = NamedParams::new(model.named_parameters())?;
        let mut optimizer = StatefulAdam::new(
            params.clone(),
            vec![ParamGroup::adam(
                params.iter().map(|(n, _)| n.into()).collect(),
                1e-4,
            )],
        )?;
        let mut case_report = BTreeMap::new();
        for (name, var) in params.iter() {
            let mut checks = BTreeMap::new();
            for (kind, gradients) in [
                ("gradient", &gradients),
                ("probe_gradient", &probe_gradients),
            ] {
                let mut result = if let Some(gradient) = gradients.get(var) {
                    comparison(gradient, &values[&format!("{kind}.{name}")], 1e-5, 1e-4)?
                } else {
                    json!({"pass":false,"present":false})
                };
                result["present"] = json!(gradients.get(var).is_some());
                if kind == "probe_gradient"
                    && case["require_nonzero_probe"] == true
                    && result["nonzero"] != true
                {
                    result["pass"] = json!(false);
                }
                if result["pass"] != true {
                    failures.push(format!("{label}/{id}/{kind}/{name}: {result}"));
                }
                checks.insert(kind.to_owned(), result);
            }
            case_report.insert(name.to_owned(), checks);
        }
        optimizer.step(&gradients)?;
        for (name, var) in params.iter() {
            let state = optimizer.state(name).unwrap();
            let checks = case_report.get_mut(name).unwrap();
            for (kind, actual) in [
                ("updated", var.as_tensor()),
                ("exp_avg", &state.exp_avg),
                ("exp_avg_sq", &state.exp_avg_sq),
            ] {
                let result = comparison(actual, &values[&format!("{kind}.{name}")], 1e-6, 0.)?;
                if result["pass"] != true {
                    update_diagnostics.push(format!("{label}/{id}/{kind}/{name}: {result}"));
                }
                if result["finite"] != true {
                    failures.push(format!("{label}/{id}/{kind}/{name}: non-finite update"));
                }
                checks.insert(kind.into(), result);
            }
            let expected_step = values[&format!("step.{name}")].to_scalar::<i64>()? as u64;
            if state.step != expected_step {
                failures.push(format!("{label}/{id}/step/{name}"));
            }
            checks.insert("step".into(),json!({"pass":state.step==expected_step,"actual":state.step,"reference":expected_step}));
        }
        assert_eq!(
            case_report.len(),
            case["parameters"].as_array().unwrap().len()
        );
        assert_eq!(case_report.len(), 38);
        // Control: identical official gradients isolate Adam from model backward.
        // AC3 revision approved 2026-09-29: all MSE gradients remain required;
        // independent-backward updates remain diagnostics with unchanged bounds.
        model.load_weights(
            &model
                .weights()
                .keys()
                .map(|name| (name.clone(), values[&format!("initial.{name}")].clone()))
                .collect(),
        )?;
        let mut official_gradients = GradStore::default();
        for (name, var) in params.iter() {
            official_gradients.insert(var, values[&format!("gradient.{name}")].clone());
        }
        let mut control = StatefulAdam::new(
            params.clone(),
            vec![ParamGroup::adam(
                params.iter().map(|(n, _)| n.into()).collect(),
                1e-4,
            )],
        )?;
        control.step(&official_gradients)?;
        let mut control_report = BTreeMap::new();
        for (name, var) in params.iter() {
            let state = control.state(name).unwrap();
            let mut checks = BTreeMap::new();
            for (kind, actual) in [
                ("updated", var.as_tensor()),
                ("exp_avg", &state.exp_avg),
                ("exp_avg_sq", &state.exp_avg_sq),
            ] {
                let result = comparison(actual, &values[&format!("{kind}.{name}")], 1e-6, 0.)?;
                if result["pass"] != true {
                    failures.push(format!(
                        "{label}/{id}/optimizer_control/{kind}/{name}: {result}"
                    ));
                }
                checks.insert(kind, result);
            }
            checks.insert(
                "step",
                json!({"pass":state.step==1,"actual":state.step,"reference":1}),
            );
            if state.step != 1 {
                failures.push(format!("{label}/{id}/optimizer_control/step/{name}"));
            }
            control_report.insert(name, checks);
        }
        report.push(json!({"case":id,"parameters":case_report,"optimizer_control":control_report}));
    }
    let directory = root.join("runs/p2-development");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!(
        "numerics-{label}-{}.json",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({"acceptance_version":2,"device":label,
            "cases":report,"failures":failures,"independent_backward_update_diagnostics":update_diagnostics}))?,
    )?;
    eprintln!("P2 complete numerical report: {}", path.display());
    assert!(
        failures.is_empty(),
        "{} parameter checks failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

#[test]
fn cpu_all_parameters_gradients_probes_and_adam() -> R {
    check(&Device::Cpu, "cpu")
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_all_parameters_gradients_probes_and_adam() -> R {
    check(&Device::new_cuda(0)?, "cuda")
}
