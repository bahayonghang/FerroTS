//! Opt-in synthetic CUDA timing for the cement-shaped training path.
//! This diagnostic does not produce formal acceptance receipts.
use candle_core::{DType, Device, Tensor};
use ferrots_core::config::{Activation, ModelSpec, WindowSpec};
use ferrots_nn::{
    Mode, RngStream, dropout,
    itransformer::ITransformer,
    model::{ForwardContext, ModelInput},
};
use ferrots_train::{
    NamedParams, ParamGroup, StatefulAdam, accumulation::GradientAccumulator, loss::masked_mse,
};
use serde_json::json;
use std::{collections::HashMap, fs, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("provide a new output directory")?,
    );
    fs::create_dir(&output)?;
    let device = Device::new_cuda(0)?;
    let model = ITransformer::new(
        &ModelSpec::Itransformer {
            d_model: 512,
            n_heads: 8,
            d_ff: 2048,
            e_layers: 2,
            dropout: 0.1,
            activation: Activation::Gelu,
            use_norm: true,
        },
        &WindowSpec {
            lookback: 96,
            horizon: 15,
            gap: 0,
            stride: 1,
        },
        18,
        5,
        2021,
        &device,
    )?;
    let values = |n: usize, scale: f32| {
        (0..n)
            .map(|i| ((i % 211) as f32 * scale).sin())
            .collect::<Vec<_>>()
    };
    let input = ModelInput {
        history: Tensor::from_vec(values(32 * 96 * 18, 0.031), (32, 96, 18), &device)?,
        history_time: Some(Tensor::from_vec(
            values(32 * 96 * 5, 0.017),
            (32, 96, 5),
            &device,
        )?),
        future_time: None,
    };
    let target = Tensor::from_vec(values(32 * 15 * 18, 0.029), (32, 15, 18), &device)?;
    let mask = Tensor::ones((32, 15, 18), DType::U8, &device)?;
    let params = NamedParams::new(model.named_parameters())?;
    let names = params.iter().map(|(name, _)| name.to_owned()).collect();
    let mut optimizer = StatefulAdam::new(params.clone(), vec![ParamGroup::adam(names, 0.0001)])?;
    let mut accumulator = GradientAccumulator::new(params.clone());
    let mut timings = Vec::new();
    let dropout_input = Tensor::ones((32, 23, 2048), DType::F32, &device)?;
    for step in 0..12 {
        device.synchronize()?;
        let start = Instant::now();
        let prediction = model.forward(
            &input,
            ForwardContext {
                mode: Mode::Train,
                forward_id: step,
            },
        )?;
        let loss = masked_mse(&prediction, &target, &mask)?;
        let loss_value = loss.sum.to_scalar::<f32>()? / loss.valid_count as f32;
        device.synchronize()?;
        let forward_loss_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        accumulator.add(&loss)?;
        device.synchronize()?;
        let backward_accumulation_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        assert!(accumulator.step(&mut optimizer)?);
        device.synchronize()?;
        let update_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let evaluation = model.forward(&input, ForwardContext::evaluation())?;
        device.synchronize()?;
        let forward_without_dropout_ms = start.elapsed().as_secs_f64() * 1000.;
        drop(evaluation);
        let start = Instant::now();
        let sample = dropout(
            &dropout_input,
            0.1,
            Mode::Train,
            RngStream {
                seed: 2021,
                run_step: step,
                layer_id: 3,
            },
        )?;
        device.synchronize()?;
        let largest_dropout_ms = start.elapsed().as_secs_f64() * 1000.;
        drop(sample);
        timings.push(json!({
            "step":step, "warmup":step < 2, "loss":loss_value,
            "forward_loss_ms":forward_loss_ms,
            "backward_accumulation_ms":backward_accumulation_ms,
            "update_ms":update_ms,
            "training_step_ms":forward_loss_ms + backward_accumulation_ms + update_ms,
            "forward_without_dropout_ms":forward_without_dropout_ms,
            "largest_dropout_ms":largest_dropout_ms,
        }));
    }
    let mut tensors = HashMap::new();
    for (name, value) in params.iter() {
        tensors.insert(format!("parameter.{name}"), value.detach());
        let state = optimizer.state(name).ok_or("missing optimizer state")?;
        assert_eq!(state.step, 12);
        tensors.insert(format!("exp_avg.{name}"), state.exp_avg.clone());
        tensors.insert(format!("exp_avg_sq.{name}"), state.exp_avg_sq.clone());
    }
    tensors.insert(
        "prediction".into(),
        model.forward(&input, ForwardContext::evaluation())?,
    );
    candle_core::safetensors::save(&tensors, output.join("tensors.safetensors"))?;
    let keys = [
        "forward_loss_ms",
        "backward_accumulation_ms",
        "update_ms",
        "training_step_ms",
        "forward_without_dropout_ms",
        "largest_dropout_ms",
    ];
    let means: serde_json::Map<_, _> = keys
        .into_iter()
        .map(|key| {
            let mean = timings
                .iter()
                .skip(2)
                .map(|row| row[key].as_f64().expect("numeric timing"))
                .sum::<f64>()
                / 10.;
            (key.to_owned(), json!(mean))
        })
        .collect();
    let report = json!({
        "device":"cuda:0", "batch_size":32, "accumulation_steps":1,
        "lookback":96, "horizon":15, "variables":18, "time_features":5,
        "seed":2021, "synthetic":true, "formal_acceptance":false,
        "measured_steps":10, "warmup_steps":2, "mean_ms":means, "samples":timings,
        "boundary":"resident synthetic inputs; synchronized real forward, masked loss, accumulator and Adam; excludes data loading and checkpoints",
    });
    fs::write(
        output.join("profile.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report["mean_ms"])?);
    Ok(())
}
