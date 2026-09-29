use candle_core::{Device, Tensor};
use ferrots_core::config::{ModelSpec, WindowSpec};
use ferrots_nn::baselines::{Baseline, ModelInput};
use ferrots_train::{
    NamedParams, ParamGroup, StatefulAdam, accumulation::GradientAccumulator, loss::masked_mse,
};
#[test]
fn upstream_single_adam_update() -> Result<(), Box<dyn std::error::Error>> {
    let info: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/p1/dlinear.json"))?;
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    for d in devices {
        let fixture = candle_core::safetensors::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/p1/dlinear.safetensors"),
            &d,
        )?;
        for case in info["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let n = |s: &str| case[s].as_u64().unwrap() as usize;
            let w = WindowSpec {
                lookback: n("lookback"),
                horizon: n("horizon"),
                gap: 0,
                stride: 1,
            };
            let model = Baseline::new(
                &ModelSpec::Dlinear {
                    individual: case["individual"].as_bool().unwrap(),
                    moving_avg: n("moving_avg"),
                },
                &w,
                n("channels"),
                1,
                &d,
            )?;
            model.load_weights(
                &model
                    .weights()
                    .keys()
                    .map(|name| {
                        (
                            name.clone(),
                            fixture[&format!("{id}.initial.{name}")].clone(),
                        )
                    })
                    .collect(),
            )?;
            let params = NamedParams::new(model.named_parameters())?;
            let group = ParamGroup::adam(params.iter().map(|(n, _)| n.into()).collect(), 0.001);
            let mut opt = StatefulAdam::new(params.clone(), vec![group])?;
            let mut acc = GradientAccumulator::new(params);
            let pred = model.forward(&ModelInput::history(
                fixture[&format!("{id}.input")].clone(),
            ))?;
            let target = &fixture[&format!("{id}.target")];
            let mask = Tensor::ones(target.shape(), candle_core::DType::U8, &d)?;
            acc.add(&masked_mse(&pred, target, &mask)?)?;
            acc.step(&mut opt)?;
            for (name, var) in model.named_parameters() {
                for (a, b) in var.flatten_all()?.to_vec1::<f32>()?.iter().zip(
                    fixture[&format!("{id}.updated.{name}")]
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                ) {
                    assert!((a - b).abs() <= 1e-6, "{id} {name}: {a} {b}");
                }
            }
        }
    }
    Ok(())
}
