use candle_core::{DType, Device, Module, Tensor, Var};
use ferrots_nn::{
    infer_ops::{self, InferenceOnly},
    train_ops,
};
type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;
#[test]
fn fused_path_cuts_grad_but_training_ops_preserve_all_parameters() -> TestResult {
    for norm in [false, true] {
        for fused in [false, true] {
            let x = Tensor::from_slice(&[0.4_f32, -0.7, 1.2, 0.2], (2, 2), &Device::Cpu)?;
            let w1 = Var::from_slice(&[0.1_f32, -0.2, 0.3, 0.4, 0.8, -0.5], (2, 3), &Device::Cpu)?;
            let w2 = Var::from_slice(&[0.6_f32, -0.4, 0.2, 0.5, -0.3, 0.9], (3, 2), &Device::Cpu)?;
            let gamma = Var::ones(3, DType::F32, &Device::Cpu)?;
            let beta = Var::zeros(3, DType::F32, &Device::Cpu)?;
            let hidden = x.matmul(&w1)?.contiguous()?;
            let hidden = match (norm, fused) {
                (false, true) => candle_nn::ops::softmax_last_dim(&hidden)?,
                (false, false) => train_ops::softmax(&hidden, 1)?,
                (true, true) => candle_nn::LayerNorm::new(
                    gamma.as_tensor().clone(),
                    beta.as_tensor().clone(),
                    1e-5,
                )
                .forward(&hidden)?,
                (true, false) => train_ops::layer_norm(&hidden, &gamma, &beta, 1e-5)?,
            };
            let gradients = hidden.matmul(&w2)?.sqr()?.sum_all()?.backward()?;
            assert!(gradients.get(&w2).is_some());
            assert_eq!(
                gradients.get(&w1).is_none(),
                fused,
                "norm={norm}, fused={fused}"
            );
            if norm {
                for var in [&gamma, &beta] {
                    assert_eq!(gradients.get(var).is_none(), fused);
                }
            }
        }
    }
    Ok(())
}
#[test]
fn inference_requires_frozen_inputs_and_matches_training_forward() -> TestResult {
    let x = Var::from_slice(&[0.3_f32, -0.7, 2., 0.4, -0.2, 0.9], (2, 3), &Device::Cpu)?;
    let gamma = Var::ones(3, DType::F32, &Device::Cpu)?;
    let beta = Var::zeros(3, DType::F32, &Device::Cpu)?;
    let scope = InferenceOnly::evaluation();
    assert!(infer_ops::softmax_fused(&x, &scope).is_err());
    assert!(infer_ops::softmax_fused(&x.sqr()?, &scope).is_err());
    let frozen = x.detach();
    assert!(infer_ops::layer_norm_fused(&frozen, &gamma, &beta, 1e-5, &scope).is_err());
    let gamma = gamma.detach();
    let beta = beta.detach();
    let pairs = [
        (
            infer_ops::softmax_fused(&frozen, &scope)?,
            train_ops::softmax(&frozen, 1)?,
        ),
        (
            infer_ops::layer_norm_fused(&frozen, &gamma, &beta, 1e-5, &scope)?,
            train_ops::layer_norm(&frozen, &gamma, &beta, 1e-5)?,
        ),
    ];
    for (fast, slow) in pairs {
        assert!(!fast.track_op());
        assert!(fast.sub(&slow)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
    }
    Ok(())
}
