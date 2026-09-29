use candle_core::{Device, Tensor, Var};
use ferrots_core::config::{ModelSpec, WindowSpec};
use ferrots_nn::baselines::{Baseline, ModelInput, decompose};
use std::collections::BTreeMap;
type R = Result<(), Box<dyn std::error::Error>>;
fn close(actual: &Tensor, expected: &Tensor) -> R {
    assert_eq!(actual.shape(), expected.shape());
    for (a, b) in actual
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(expected.flatten_all()?.to_vec1::<f32>()?)
    {
        assert!(
            a.is_finite() && (a - b).abs() <= 1e-5 + 1e-4 * b.abs(),
            "actual={a}, expected={b}"
        );
    }
    Ok(())
}
#[test]
fn official_shared_individual_length_one_forward_and_all_gradients() -> R {
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    let info: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/p1/dlinear.json"))?;
    for device in devices {
        let fixture = candle_core::safetensors::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/p1/dlinear.safetensors"),
            &device,
        )?;
        for case in info["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let n = |k: &str| case[k].as_u64().unwrap() as usize;
            let spec = ModelSpec::Dlinear {
                individual: case["individual"].as_bool().unwrap(),
                moving_avg: n("moving_avg"),
            };
            let w = WindowSpec {
                lookback: n("lookback"),
                horizon: n("horizon"),
                gap: 0,
                stride: 1,
            };
            let model = Baseline::new(&spec, &w, n("channels"), 1, &device)?;
            let weights: BTreeMap<_, _> = model
                .weights()
                .keys()
                .map(|name| {
                    (
                        name.clone(),
                        fixture[&format!("{id}.initial.{name}")].clone(),
                    )
                })
                .collect();
            model.load_weights(&weights)?;
            let input = Var::from_tensor(&fixture[&format!("{id}.input")])?;
            let (seasonal, trend) = decompose(&input, n("moving_avg"))?;
            close(&seasonal, &fixture[&format!("{id}.seasonal")])?;
            close(&trend, &fixture[&format!("{id}.trend")])?;
            let prediction = model.forward(&ModelInput::history(input.as_tensor().clone()))?;
            close(&prediction, &fixture[&format!("{id}.prediction")])?;
            let loss = prediction
                .sub(&fixture[&format!("{id}.target")])?
                .sqr()?
                .mean_all()?;
            close(&loss.reshape(1)?, &fixture[&format!("{id}.loss")])?;
            let grads = loss.backward()?;
            close(
                grads.get(&input).expect("input gradient"),
                &fixture[&format!("{id}.input_grad")],
            )?;
            for (name, var) in model.named_parameters() {
                close(
                    grads.get(&var).expect("parameter gradient"),
                    &fixture[&format!("{id}.grad.{name}")],
                )?;
            }
        }
    }
    Ok(())
}
