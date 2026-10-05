use candle_core::{Device, Tensor};
use ferrots_core::config::{Activation, ModelSpec, WindowSpec};
use ferrots_nn::{
    itransformer::ITransformer,
    model::{ForwardContext, ModelInput},
};
use serde_json::Value;
type R = Result<(), Box<dyn std::error::Error>>;

fn model(case: &Value, device: &Device) -> candle_core::Result<ITransformer> {
    let c = &case["config"];
    let n = |name: &str| c[name].as_u64().unwrap() as usize;
    ITransformer::new(
        &ModelSpec::Itransformer {
            d_model: n("d_model"),
            n_heads: n("n_heads"),
            d_ff: n("d_ff"),
            e_layers: n("e_layers"),
            dropout: c["dropout"].as_f64().unwrap(),
            activation: if c["activation"] == "gelu" {
                Activation::Gelu
            } else {
                Activation::Relu
            },
            use_norm: c["use_norm"].as_bool().unwrap(),
        },
        &WindowSpec {
            lookback: n("seq_len"),
            horizon: n("pred_len"),
            gap: 0,
            stride: 1,
        },
        n("enc_in"),
        case["time_channels"].as_u64().unwrap_or(4) as usize,
        case["seed"].as_u64().unwrap(),
        device,
    )
}
fn manifest() -> Value {
    serde_json::from_str(include_str!("../../../tests/fixtures/p2/manifest.json")).unwrap()
}
fn fixture(
    name: &str,
    device: &Device,
) -> candle_core::Result<std::collections::HashMap<String, Tensor>> {
    candle_core::safetensors::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../tests/fixtures/p2/{name}.safetensors")),
        device,
    )
}
fn close(name: &str, actual: &Tensor, expected: &Tensor) -> R {
    assert_eq!(actual.shape(), expected.shape(), "{name}");
    for (i, (a, e)) in actual
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .zip(expected.flatten_all()?.to_vec1::<f32>()?)
        .enumerate()
    {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "{name}[{i}]: {a} vs {e}"
        );
    }
    Ok(())
}

#[test]
fn three_seed_initialization_and_ten_sampler_epochs_match_official_bits() -> R {
    let fixture = fixture("initialization", &Device::Cpu)?;
    for case in manifest()["initializations"].as_array().unwrap() {
        let m = model(case, &Device::Cpu)?;
        let seed = case["seed"].as_u64().unwrap();
        for (name, value) in m.weights() {
            let expected = &fixture[&format!("seed{seed}.{name}")];
            let bits = |tensor: &Tensor| -> candle_core::Result<Vec<u32>> {
                Ok(tensor
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(bits(value)?, bits(expected)?, "seed {seed} {name}");
        }
        for (epoch, expected) in case["orders"].as_array().unwrap().iter().enumerate() {
            let order = m.sampling_rng().training_permutation(137, epoch as u64);
            let expected: Vec<_> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(&order[..128], expected, "seed {seed}, epoch {epoch}");
        }
    }
    Ok(())
}

#[test]
fn official_nine_cases_all_layer_outputs_and_independent_freeze() -> R {
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    for device in devices {
        for case in manifest()["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let values = fixture(name, &device)?;
            let m = model(case, &device)?;
            m.load_weights(
                &m.weights()
                    .keys()
                    .map(|key| (key.clone(), values[&format!("initial.{key}")].clone()))
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
            let input = ModelInput {
                history,
                history_time: values.get("input.history_time").cloned(),
                future_time: None,
            };
            let (output, trace) = m.forward_with_trace(&input, ForwardContext::evaluation())?;
            close(name, &output, &values["output"])?;
            assert_eq!(trace.len(), case["traces"].as_array().unwrap().len());
            for name in case["traces"].as_array().unwrap() {
                let name = name.as_str().unwrap();
                close(name, &trace[name], &values[&format!("trace.{name}")])?;
            }
            let frozen = m.freeze_for_eval()?;
            for (_, variable) in m.named_parameters() {
                variable.set(&variable.zeros_like()?)?;
            }
            let frozen_output = frozen.forward(&input, ForwardContext::evaluation())?;
            assert!(!frozen_output.track_op());
            close(name, &frozen_output, &values["output"])?;
            let bad = ModelInput::history(input.history.narrow(1, 0, 0)?);
            assert!(m.forward(&bad, ForwardContext::evaluation()).is_err());
        }
    }
    Ok(())
}

#[test]
fn official_mapping_roundtrips_and_rejects_invalid_sets_before_assignment() -> R {
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    for device in devices {
        let info = manifest();
        let case = &info["cases"][0];
        let m = model(case, &device)?;
        let original = m.official_weights()?;
        for p in case["parameters"].as_array().unwrap() {
            let name = p["name"].as_str().unwrap();
            let shape: Vec<usize> = p["official_shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(original[name].dims(), shape);
        }
        for (_, variable) in m.named_parameters() {
            variable.set(&variable.zeros_like()?)?;
        }
        m.load_official_weights(&original)?;
        for (name, tensor) in m.official_weights()? {
            assert_eq!(
                tensor.flatten_all()?.to_vec1::<f32>()?,
                original[&name].flatten_all()?.to_vec1::<f32>()?
            );
        }
        let key = "projector.bias";
        for kind in [
            "missing",
            "extra",
            "shape",
            "dtype",
            "nonfinite",
            "conv_kernel",
        ] {
            let mut bad = original.clone();
            match kind {
                "missing" => {
                    bad.remove(key);
                }
                "extra" => {
                    bad.insert("unknown".into(), original[key].clone());
                }
                "shape" => {
                    bad.insert(key.into(), original[key].unsqueeze(0)?);
                }
                "dtype" => {
                    bad.insert(key.into(), original[key].to_dtype(candle_core::DType::F64)?);
                }
                "nonfinite" => {
                    bad.insert(key.into(), original[key].affine(f64::NAN, 0.)?);
                }
                _ => {
                    let conv = "encoder.attn_layers.0.conv1.weight";
                    bad.insert(conv.into(), original[conv].squeeze(2)?);
                }
            }
            // A valid changed early parameter must also remain unassigned.
            let early = "enc_embedding.value_embedding.bias";
            bad.insert(early.into(), original[early].zeros_like()?);
            assert!(m.load_official_weights(&bad).is_err(), "{kind}");
            for (name, tensor) in m.official_weights()? {
                assert_eq!(
                    tensor.flatten_all()?.to_vec1::<f32>()?,
                    original[&name].flatten_all()?.to_vec1::<f32>()?,
                    "{kind}: {name}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn history_and_time_bindings_reject_wrong_dimensions_and_dtype() -> R {
    let info = manifest();
    for case in info["cases"].as_array().unwrap().iter().take(3) {
        let m = model(case, &Device::Cpu)?;
        let values = fixture(case["name"].as_str().unwrap(), &Device::Cpu)?;
        let input = ModelInput {
            history: values["input.history"].clone(),
            history_time: values.get("input.history_time").cloned(),
            future_time: None,
        };
        for history in [
            input.history.narrow(0, 0, 0)?,
            input.history.narrow(1, 0, 1)?,
            input.history.narrow(2, 0, 1)?,
            input.history.to_dtype(candle_core::DType::F64)?,
        ] {
            assert!(
                m.forward(
                    &ModelInput {
                        history,
                        history_time: input.history_time.clone(),
                        future_time: None
                    },
                    ForwardContext::evaluation()
                )
                .is_err()
            );
        }
        let times = input.history_time.clone().unwrap_or(Tensor::zeros(
            (2, 12, 4),
            candle_core::DType::F32,
            &Device::Cpu,
        )?);
        for history_time in [
            Some(times.narrow(2, 0, 1)?),
            Some(times.to_dtype(candle_core::DType::F64)?),
        ] {
            assert!(
                m.forward(
                    &ModelInput {
                        history: input.history.clone(),
                        history_time,
                        future_time: None
                    },
                    ForwardContext::evaluation()
                )
                .is_err()
            );
        }
        if m.time_channels > 0 {
            assert!(
                m.forward(
                    &ModelInput::history(input.history),
                    ForwardContext::evaluation()
                )
                .is_err()
            );
        }
    }
    Ok(())
}

#[test]
fn dropout_masks_pin_all_sites_and_match_cpu_cuda() -> R {
    use ferrots_nn::{Mode, RngStream, dropout};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/p2/dropout-masks.safetensors");
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    let mut expected = None;
    for device in devices {
        let mut masks = std::collections::HashMap::new();
        for forward_id in [0, 1] {
            for site in 0..9 {
                let shape: Vec<usize> = if site == 0 {
                    vec![2, 11, 16]
                } else {
                    match (site - 1) % 4 {
                        0 => vec![2, 4, 11, 11],
                        2 => vec![2, 32, 11],
                        _ => vec![2, 11, 16],
                    }
                };
                let ones = Tensor::ones(shape, candle_core::DType::F32, &device)?;
                let mask = dropout(
                    &ones,
                    0.1,
                    Mode::Train,
                    RngStream {
                        seed: 2021,
                        run_step: forward_id,
                        layer_id: site,
                    },
                )?
                .gt(0f32)?
                .to_dtype(candle_core::DType::F32)?;
                masks.insert(format!("forward{forward_id}.site{site}"), mask);
            }
        }
        if device.is_cpu()
            && std::env::var_os("P2_EXPORT_DROPOUT_MASKS").is_some()
            && !path.exists()
        {
            candle_core::safetensors::save(&masks, &path)?;
        }
        let pinned = candle_core::safetensors::load(&path, &Device::Cpu)?;
        assert_eq!(pinned.len(), 18);
        let mut bits = std::collections::BTreeMap::new();
        for (name, mask) in masks {
            assert_eq!(mask.shape(), pinned[&name].shape());
            let values = mask.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(values, pinned[&name].flatten_all()?.to_vec1::<f32>()?);
            bits.insert(name, values);
        }
        if let Some(expected) = &expected {
            assert_eq!(&bits, expected);
        } else {
            expected = Some(bits);
        }
    }
    Ok(())
}

#[test]
fn shared_dropout_masks_match_official_forward_and_all_gradients() -> R {
    let info = manifest();
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    for device in devices {
        for case in info["dropout_cases"]
            .as_array()
            .expect("shared mask fixtures")
        {
            let m = model(case, &device)?;
            let tensors = fixture(case["name"].as_str().unwrap(), &device)?;
            m.load_weights(
                &m.weights()
                    .keys()
                    .map(|n| (n.clone(), tensors[&format!("initial.{n}")].clone()))
                    .collect(),
            )?;
            let input = ModelInput {
                history: tensors["input.history"].clone(),
                history_time: tensors.get("input.history_time").cloned(),
                future_time: None,
            };
            let context = ForwardContext {
                mode: ferrots_nn::Mode::Train,
                forward_id: case["forward_id"].as_u64().unwrap(),
            };
            let (output, trace) = m.forward_with_trace(&input, context)?;
            close("shared dropout output", &output, &tensors["output"])?;
            for name in case["traces"].as_array().unwrap() {
                let name = name.as_str().unwrap();
                close(name, &trace[name], &tensors[&format!("trace.{name}")])?;
            }
            let loss = output.sub(&tensors["input.target"])?.sqr()?.mean_all()?;
            let gradients = loss.backward()?;
            for (name, var) in m.named_parameters() {
                close(
                    &name,
                    gradients.get(&var).expect("shared mask parameter gradient"),
                    &tensors[&format!("gradient.{name}")],
                )?;
            }
            assert_eq!(
                output.flatten_all()?.to_vec1::<f32>()?,
                m.forward(&input, context)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
            assert_ne!(
                output.flatten_all()?.to_vec1::<f32>()?,
                m.forward(&input, ForwardContext::evaluation())?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
        }
    }
    Ok(())
}
