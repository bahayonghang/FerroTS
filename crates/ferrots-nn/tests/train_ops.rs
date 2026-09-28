use candle_core::{Device, Tensor, Var};
use ferrots_nn::train_ops::{self, OpError};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn values(x: &Tensor) -> candle_core::Result<Vec<f64>> {
    x.flatten_all()?.to_vec1()
}
fn objective(y: &Tensor) -> candle_core::Result<Tensor> {
    let weights: Vec<f64> = (0..y.elem_count()).map(|i| (i + 1) as f64 / 7.0).collect();
    y.mul(&Tensor::from_vec(weights, y.shape(), y.device())?)?
        .sum_all()
}
fn finite_difference(
    inputs: &[Vec<f64>],
    shapes: &[Vec<usize>],
    f: impl Fn(&[Tensor]) -> train_ops::Result<Tensor>,
) -> TestResult {
    let vars = inputs
        .iter()
        .zip(shapes)
        .map(|(x, s)| Var::from_slice(x, s.as_slice(), &Device::Cpu))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let tensors: Vec<_> = vars.iter().map(|v| v.as_tensor().clone()).collect();
    let gradients = objective(&f(&tensors)?)?.backward()?;
    for (p, var) in vars.iter().enumerate() {
        let analytic = values(gradients.get(var).expect("missing input gradient"))?;
        for (i, &gradient) in analytic.iter().enumerate() {
            let mut shifted = inputs.to_vec();
            let epsilon = 1e-6;
            shifted[p][i] += epsilon;
            let make = |xs: &[Vec<f64>]| {
                xs.iter()
                    .zip(shapes)
                    .map(|(x, s)| Tensor::from_slice(x, s.as_slice(), &Device::Cpu))
                    .collect::<candle_core::Result<Vec<_>>>()
            };
            let high = objective(&f(&make(&shifted)?)?)?.to_scalar::<f64>()?;
            shifted[p][i] -= 2.0 * epsilon;
            let low = objective(&f(&make(&shifted)?)?)?.to_scalar::<f64>()?;
            let numerical = (high - low) / (2.0 * epsilon);
            assert!(
                (gradient - numerical).abs() <= 2e-7 + 2e-6 * numerical.abs(),
                "parameter {p}, index {i}: {gradient} vs {numerical}"
            );
        }
    }
    Ok(())
}

#[test]
fn train_ops_softmax_arbitrary_axes_and_noncontiguous() -> TestResult {
    let input = vec![0.3, -0.7, 2.0, 0.4, -0.2, 0.9];
    for dim in [0, 1] {
        finite_difference(std::slice::from_ref(&input), &[vec![2, 3]], |xs| {
            train_ops::softmax(&xs[0], dim)
        })?;
    }
    let x = Tensor::from_slice(&input, (2, 3), &Device::Cpu)?.transpose(0, 1)?;
    let y = train_ops::softmax(&x, 1)?;
    for sum in values(&y.sum(1)?)? {
        assert!((sum - 1.0).abs() < 1e-12);
    }
    let large = Tensor::from_slice(&[1000_f64, 1001.], (1, 2), &Device::Cpu)?;
    assert!(
        values(&train_ops::softmax(&large, 1)?)?
            .iter()
            .all(|x| x.is_finite())
    );
    assert!(train_ops::softmax(&large, 2).is_err());
    Ok(())
}

#[test]
fn train_ops_masked_softmax_checks_rows_and_gradients() -> TestResult {
    let mask = Tensor::from_slice(&[1_u8, 0, 1], (1, 3), &Device::Cpu)?;
    finite_difference(
        &[vec![0.3, -0.7, 2., 0.4, -0.2, 0.9]],
        &[vec![2, 3]],
        |xs| train_ops::masked_softmax(&xs[0], &mask, 1),
    )?;
    let x = Tensor::zeros((2, 3), candle_core::DType::F64, &Device::Cpu)?;
    let result = values(&train_ops::masked_softmax(&x, &mask, 1)?)?;
    assert_eq!(result, vec![0.5, 0., 0.5, 0.5, 0., 0.5]);
    let mask = Tensor::from_slice(&[1_u8, 0, 1, 0, 0, 0], (2, 3), &Device::Cpu)?;
    assert!(matches!(
        train_ops::masked_softmax(&x, &mask, 1),
        Err(OpError::FullyMaskedRow)
    ));
    Ok(())
}

#[test]
fn train_ops_layer_norm_all_gradients() -> TestResult {
    finite_difference(
        &[
            vec![0.3, -0.7, 2., 0.4, -0.2, 0.9],
            vec![0.8, 1.2, -0.6],
            vec![0.1, -0.2, 0.3],
        ],
        &[vec![2, 3], vec![3], vec![3]],
        |xs| train_ops::layer_norm(&xs[0], &xs[1], &xs[2], 1e-5),
    )?;
    let x = Tensor::from_slice(&[0.3_f32, -0.7, 2., 0.4, -0.2, 0.9], (2, 3), &Device::Cpu)?;
    let gamma = Tensor::ones(3, candle_core::DType::F32, &Device::Cpu)?;
    let beta = gamma.zeros_like()?;
    let ours = train_ops::layer_norm(&x, &gamma, &beta, 1e-5)?;
    let reference = candle_nn::ops::layer_norm_slow(&x, &gamma, &beta, 1e-5)?;
    assert!(ours.sub(&reference)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
    assert!(train_ops::layer_norm(&x, &gamma, &beta, 0.).is_err());
    Ok(())
}

#[test]
fn train_ops_population_variance_edges() -> TestResult {
    for input in [vec![0.3, -0.7, 2., 0.4, -0.2, 0.9], vec![3.; 6]] {
        finite_difference(&[input], &[vec![2, 3]], |xs| {
            train_ops::pop_var(&xs[0], 1, false)
        })?;
    }
    finite_difference(&[vec![3., -2.]], &[vec![2, 1]], |xs| {
        train_ops::pop_var(&xs[0], 1, true)
    })?;
    let x = Tensor::from_slice(&[1_f64, 3.], (1, 2), &Device::Cpu)?;
    assert_eq!(values(&train_ops::pop_var(&x, 1, false)?)?, vec![1.]);
    Ok(())
}

#[test]
fn train_ops_attention_qkv_gradients() -> TestResult {
    finite_difference(
        &[
            vec![0.3, -0.7, 1.1, 0.2],
            vec![0.4, 0.8, -0.2, 0.1, 0.3, -0.9],
            vec![0.6, -0.4, 0.2, 0.5, 0.3, -0.8],
        ],
        &[vec![1, 1, 2, 2], vec![1, 1, 3, 2], vec![1, 1, 3, 2]],
        |xs| train_ops::attention(&xs[0], &xs[1], &xs[2], None, None),
    )?;
    Ok(())
}
