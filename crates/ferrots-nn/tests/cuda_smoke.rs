#![cfg(feature = "cuda")]

use candle_core::{Device, Result, Tensor, Var};

#[test]
fn cuda_smoke_matmul_backward_update() -> Result<()> {
    // Device creation must fail the test when cuda:0 is unavailable.
    let device = Device::new_cuda(0)?;
    let x = Tensor::from_slice(&[1_f32, 2., 3., -1., 0.5, 2.], (2, 3), &device)?;
    let initial = [0.1_f32, -0.2, 0.3, 0.4, -0.5, 0.6];
    let weights = Var::from_slice(&initial, (3, 2), &device)?;
    let prediction = x.matmul(weights.as_tensor())?;
    let actual = prediction.flatten_all()?.to_vec1::<f32>()?;
    assert_close(&actual, &[-0.8, 2.4, -0.95, 1.6]);

    let loss = prediction.sqr()?.sum_all()?;
    let gradients = loss.backward()?;
    let gradient = gradients.get(&weights).expect("weight gradient is missing");
    let expected_gradient = [0.3_f32, 1.6, -4.15, 11.2, -8.6, 20.8];
    assert_close(
        &gradient.flatten_all()?.to_vec1::<f32>()?,
        &expected_gradient,
    );

    // A direct SGD step checks parameter mutation before the P0 optimizer exists.
    let updated = weights.as_tensor().sub(&gradient.affine(0.01, 0.)?)?;
    weights.set(&updated)?;
    let expected: Vec<f32> = initial
        .iter()
        .zip(expected_gradient)
        .map(|(weight, gradient)| weight - 0.01 * gradient)
        .collect();
    let actual = weights.flatten_all()?.to_vec1::<f32>()?;
    assert_close(&actual, &expected);
    assert!(
        actual
            .iter()
            .zip(initial)
            .all(|(after, before)| *after != before)
    );
    Ok(())
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "non-finite value at {index}: {actual}");
        assert!(
            (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
            "mismatch at {index}: actual={actual}, expected={expected}"
        );
    }
}
