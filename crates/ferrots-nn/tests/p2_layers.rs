use candle_core::{Device, Tensor, Var};
use ferrots_nn::train_ops;
type R = Result<(), Box<dyn std::error::Error>>;

fn close(actual: &Tensor, expected: &Tensor) -> R {
    assert_eq!(actual.shape(), expected.shape());
    for (a, e) in actual
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .zip(expected.flatten_all()?.to_vec1::<f32>()?)
    {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "{a} vs {e}"
        );
    }
    Ok(())
}

#[test]
fn official_linear_gelu_layer_norm_forward_and_gradients() -> R {
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    for device in devices {
        let fixtures = candle_core::safetensors::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/p2/layers.safetensors"),
            &device,
        )?;
        let mut variables = std::collections::BTreeMap::new();
        for name in ["input", "weight", "bias", "gamma", "beta"] {
            variables.insert(name, Var::from_tensor(&fixtures[name])?);
        }
        let projected = train_ops::linear(
            &variables["input"].transpose(0, 1)?,
            &variables["weight"],
            &variables["bias"],
        )?;
        let activated = train_ops::gelu(&projected)?;
        let normalized =
            train_ops::layer_norm(&activated, &variables["gamma"], &variables["beta"], 1e-5)?;
        close(&projected, &fixtures["projected"])?;
        close(&activated, &fixtures["activated"])?;
        close(&normalized, &fixtures["normalized"])?;
        let gradients = normalized.sqr()?.mean_all()?.backward()?;
        for (name, variable) in variables {
            close(
                gradients.get(&variable).expect("missing gradient"),
                &fixtures[&format!("gradient.{name}")],
            )?;
        }
    }
    Ok(())
}
