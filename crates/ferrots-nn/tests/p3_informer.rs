#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use ferrots_core::config::{Activation, ModelSpec, WindowSpec};
use ferrots_nn::{
    Mode,
    informer::Informer,
    model::{ForwardContext, ModelInput},
};
use std::{collections::BTreeMap, path::PathBuf};
type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn spec(dropout: f64) -> ModelSpec {
    ModelSpec::Informer {
        d_model: 8,
        n_heads: 2,
        d_ff: 16,
        e_layers: 2,
        d_layers: 1,
        factor: 2,
        distil: true,
        label_len: 6,
        dropout,
        activation: Activation::Gelu,
    }
}
fn window() -> WindowSpec {
    WindowSpec {
        lookback: 12,
        horizon: 4,
        gap: 0,
        stride: 1,
    }
}
fn fixture(name: &str, device: &Device) -> R<BTreeMap<String, Tensor>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/p3");
    Ok(
        candle_core::safetensors::load(root.join(format!("{name}.safetensors")), device)?
            .into_iter()
            .collect(),
    )
}
fn subset(all: &BTreeMap<String, Tensor>, prefix: &str) -> BTreeMap<String, Tensor> {
    all.iter()
        .filter_map(|(key, value)| {
            key.strip_prefix(prefix)
                .map(|key| (key.to_owned(), value.clone()))
        })
        .collect()
}
fn close(name: &str, actual: &Tensor, expected: &Tensor, atol: f32, rtol: f32) -> R {
    assert_eq!(actual.dims(), expected.dims(), "shape: {name}");
    assert_eq!(actual.dtype(), expected.dtype(), "dtype: {name}");
    if actual.dtype() == DType::U32 {
        assert_eq!(
            actual.flatten_all()?.to_vec1::<u32>()?,
            expected.flatten_all()?.to_vec1::<u32>()?,
            "{name}"
        );
        return Ok(());
    }
    let a = actual.flatten_all()?.to_vec1::<f32>()?;
    let b = expected.flatten_all()?.to_vec1::<f32>()?;
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (*a - b).abs() <= atol + rtol * b.abs(),
            "{name}[{i}]: actual={a}, expected={b}, diff={}",
            (*a - b).abs()
        );
    }
    Ok(())
}

#[test]
fn cuda_informer_official_layers_real_loss_gradients_and_buffers() -> R {
    let device = Device::new_cuda(0)?;
    for (name, mode) in [
        ("standard_eval", Mode::Eval),
        ("standard_train", Mode::Train),
    ] {
        let values = fixture(name, &device)?;
        let samples = fixture(&format!("{name}.samples"), &device)?;
        let model = Informer::new(&spec(0.), &window(), 3, 4, 2021, &device)?;
        model.load_weights(&subset(&values, "initial."))?;
        model.load_buffers(&subset(&values, "buffer.initial."))?;
        let input = ModelInput {
            history: values["input.history"].clone(),
            history_time: Some(values["input.history_time"].clone()),
            future_time: Some(values["input.future_time"].clone()),
        };
        let (output, trace) = model.forward_with_trace(
            &input,
            ForwardContext {
                mode,
                forward_id: 0,
            },
            Some(&samples),
        )?;
        for (key, expected) in subset(&values, "trace.") {
            close(
                &format!("{name}:trace:{key}"),
                trace
                    .get(&key)
                    .unwrap_or_else(|| panic!("missing trace {key}")),
                &expected,
                1e-5,
                1e-4,
            )?;
        }
        close(
            &format!("{name}:decoder_input"),
            &trace["decoder_input"],
            &values["input.decoder"],
            0.,
            0.,
        )?;
        close(
            &format!("{name}:output"),
            &output,
            &values["output"],
            1e-5,
            1e-4,
        )?;
        let loss = output.sub(&values["input.target"])?.sqr()?.mean_all()?;
        close(&format!("{name}:loss"), &loss, &values["loss"], 1e-5, 1e-4)?;
        let gradients = loss.backward()?;
        for (key, variable) in model.named_parameters() {
            let gradient = gradients
                .get(&variable)
                .unwrap_or_else(|| panic!("missing gradient {key}"));
            close(
                &format!("{name}:gradient:{key}"),
                gradient,
                &values[&format!("gradient.{key}")],
                1e-5,
                1e-4,
            )?;
        }
        let mut probe = loss;
        for (key, value) in &trace {
            if key.ends_with(".key_projection") {
                probe = probe.add(&value.sqr()?.mean_all()?)?;
            }
        }
        let probe_gradients = probe.backward()?;
        for (key, variable) in model.named_parameters() {
            let gradient = probe_gradients
                .get(&variable)
                .unwrap_or_else(|| panic!("missing probe gradient {key}"));
            assert!(
                gradient
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .any(|v| *v != 0.),
                "disconnected {key}"
            );
            close(
                &format!("{name}:probe:{key}"),
                gradient,
                &values[&format!("probe_gradient.{key}")],
                1e-5,
                1e-4,
            )?;
        }
        for (key, expected) in subset(&values, "buffer.after.") {
            close(
                &format!("{name}:buffer:{key}"),
                &model.buffers()?[&key],
                &expected,
                1e-5,
                1e-4,
            )?;
        }
    }
    Ok(())
}

#[test]
fn cuda_informer_singleton_shapes_sampling_and_frozen_state() -> R {
    let device = Device::new_cuda(0)?;
    for (lookback, horizon, heads) in [(1, 1, 1), (7, 15, 1), (12, 30, 2), (12, 60, 2)] {
        let mut s = spec(0.2);
        if let ModelSpec::Informer {
            ref mut n_heads,
            ref mut label_len,
            ..
        } = s
        {
            *n_heads = heads;
            *label_len = lookback.min(6);
        }
        let w = WindowSpec {
            lookback,
            horizon,
            gap: 0,
            stride: 1,
        };
        let model = Informer::new(&s, &w, 3, 4, 2021, &device)?;
        let history = Tensor::arange(0_u32, (lookback * 3) as u32, &device)?
            .to_dtype(DType::F32)?
            .reshape((1, lookback, 3))?
            .affine(0.1, -0.5)?;
        let input = ModelInput {
            history,
            history_time: Some(Tensor::zeros((1, lookback, 4), DType::F32, &device)?),
            future_time: Some(Tensor::zeros((1, horizon, 4), DType::F32, &device)?),
        };
        let output = model.forward(
            &input,
            ForwardContext {
                mode: Mode::Train,
                forward_id: 3,
            },
        )?;
        assert_eq!(output.dims(), [1, horizon, 3]);
        let gradients = output.sqr()?.mean_all()?.backward()?;
        assert!(
            model
                .named_parameters()
                .iter()
                .all(|(_, p)| gradients.get(p).is_some())
        );
        let frozen = model.freeze_for_eval()?;
        let first = frozen.forward(&input, ForwardContext::evaluation_with_id(19))?;
        assert!(!first.track_op());
        close(
            "repeatable eval",
            &first,
            &frozen.forward(&input, ForwardContext::evaluation_with_id(19))?,
            0.,
            0.,
        )?;
        assert!(
            frozen
                .forward(
                    &input,
                    ForwardContext {
                        mode: Mode::Train,
                        forward_id: 0
                    }
                )
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn cuda_distillation_elu_large_positive_affine_has_finite_gradients() -> R {
    let device = Device::new_cuda(0)?;
    let model = Informer::new(&spec(0.), &window(), 3, 4, 2021, &device)?;
    let mut weights = model
        .weights()
        .iter()
        .map(|(name, value)| Ok((name.clone(), value.detach().copy()?)))
        .collect::<candle_core::Result<BTreeMap<_, _>>>()?;
    weights.insert(
        "encoder.conv_layers.0.norm.bias".into(),
        Tensor::full(100_f32, 8, &device)?,
    );
    model.load_weights(&weights)?;
    let values = fixture("standard_train", &device)?;
    let input = ModelInput {
        history: values["input.history"].clone(),
        history_time: Some(values["input.history_time"].clone()),
        future_time: Some(values["input.future_time"].clone()),
    };
    let output = model.forward(
        &input,
        ForwardContext {
            mode: Mode::Train,
            forward_id: 0,
        },
    )?;
    assert!(
        output
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| x.is_finite())
    );
    let gradients = output
        .sub(&values["input.target"])?
        .sqr()?
        .mean_all()?
        .backward()?;
    for (name, var) in model.named_parameters() {
        assert!(
            gradients
                .get(&var)
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|x| x.is_finite()),
            "non-finite ELU gradient: {name}"
        );
    }
    Ok(())
}

#[test]
fn cuda_top_query_cutoff_ties_are_unique_and_repeatable() -> R {
    let device = Device::new_cuda(0)?;
    let mut s = spec(0.);
    if let ModelSpec::Informer { ref mut factor, .. } = s {
        *factor = 1;
    }
    let model = Informer::new(&s, &window(), 3, 4, 2021, &device)?;
    let mut weights = model
        .weights()
        .iter()
        .map(|(name, value)| Ok((name.clone(), value.detach().copy()?)))
        .collect::<candle_core::Result<BTreeMap<_, _>>>()?;
    for projection in ["query", "key"] {
        for parameter in ["weight", "bias"] {
            let name =
                format!("encoder.attn_layers.0.attention.{projection}_projection.{parameter}");
            weights.insert(name.clone(), weights[&name].zeros_like()?);
        }
    }
    model.load_weights(&weights)?;
    let values = fixture("standard_eval", &device)?;
    let input = ModelInput {
        history: values["input.history"].clone(),
        history_time: Some(values["input.history_time"].clone()),
        future_time: Some(values["input.future_time"].clone()),
    };
    let (output, trace) =
        model.forward_with_trace(&input, ForwardContext::evaluation_with_id(37), None)?;
    let (_, other) =
        model.forward_with_trace(&input, ForwardContext::evaluation_with_id(41), None)?;
    let key = "encoder.attn_layers.0.attention.selected_queries";
    let selected = &trace[key];
    let expected = (12_f64.ln().ceil() as usize).max(1);
    assert_eq!(selected.dims(), [2, 2, expected]);
    assert!(
        expected < window().lookback,
        "cutoff must select a proper subset"
    );
    close(
        "tied selection stable across sampling identities",
        selected,
        &other[key],
        0.,
        0.,
    )?;
    for row in selected.flatten_all()?.to_vec1::<u32>()?.chunks(expected) {
        assert_eq!(
            row.iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            expected
        );
        assert!(row.iter().all(|i| (*i as usize) < window().lookback));
    }
    let gradients = output
        .sub(&values["input.target"])?
        .sqr()?
        .mean_all()?
        .backward()?;
    for (name, var) in model.named_parameters() {
        assert!(
            gradients
                .get(&var)
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite()),
            "nonfinite tie gradient: {name}"
        );
    }
    Ok(())
}
