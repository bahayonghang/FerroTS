#![cfg(feature = "cuda")]

use candle_core::{Device, Tensor, Var};
use ferrots_nn::informer_ops::{max_pool1d, prefix_sum, replace_rows};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn close(actual: &Tensor, expected: &[f32]) -> TestResult {
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
            "actual={actual}, expected={expected}"
        );
    }
    Ok(())
}

#[test]
fn cuda_prefix_sum_values_and_suffix_gradient() -> TestResult {
    let device = Device::new_cuda(0)?;
    let x = Var::from_slice(&[1_f32, 2., -3., 4., 5., -6.], (2, 3), &device)?;
    let y = prefix_sum(&x, 1)?;
    close(&y, &[1., 3., 0., 4., 9., 3.])?;
    let cotangent = Tensor::from_slice(&[1_f32, -2., 3., 4., -5., 6.], (2, 3), &device)?;
    let gradients = y.mul(&cotangent)?.sum_all()?.backward()?;
    close(gradients.get(&x).unwrap(), &[2., 1., 3., 5., 1., 6.])?;
    close(&prefix_sum(&x, 0)?, &[1., 2., -3., 5., 7., -9.])?;
    close(
        &prefix_sum(&x.transpose(0, 1)?, 1)?,
        &[1., 5., 2., 7., -3., -9.],
    )?;
    // Contiguous views can still have a nonzero storage offset.
    close(&prefix_sum(&x.narrow(0, 1, 1)?, 1)?, &[4., 9., 3.])?;
    let singleton = Var::from_slice(&[2_f32], (1, 1, 1), &device)?;
    let gradients = prefix_sum(&singleton, 2)?.sum_all()?.backward()?;
    close(gradients.get(&singleton).unwrap(), &[1.])
}

#[test]
fn cuda_prefix_sum_million_elements_has_linear_output() -> TestResult {
    let device = Device::new_cuda(0)?;
    let length = 1_000_000;
    let x = Var::from_vec(vec![1_f32; length], length, &device)?;
    let y = prefix_sum(&x, 0)?;
    let gradients = y.sum_all()?.backward()?;
    let output = y.to_vec1::<f32>()?;
    let gradient = gradients.get(&x).unwrap().to_vec1::<f32>()?;
    for index in [0, 1, length / 2, length - 1] {
        assert_eq!(output[index], (index + 1) as f32);
        assert_eq!(gradient[index], (length - index) as f32);
    }
    // This checks behavior at AC2's size, not the required allocator peak.
    Ok(())
}

#[test]
fn cuda_pool_overlapping_ties_padding_and_singleton() -> TestResult {
    let device = Device::new_cuda(0)?;
    let x = Var::from_slice(
        &[1_f32, 4., 4., 3., 2., -3., -2., -1., -2., -3.],
        (1, 2, 5),
        &device,
    )?;
    let y = max_pool1d(&x)?;
    assert_eq!(y.dims(), [1, 2, 3]);
    close(&y, &[4., 4., 3., -2., -1., -2.])?;
    let cotangent = Tensor::from_slice(&[1_f32, 2., 3., 4., 5., 6.], (1, 2, 3), &device)?;
    let gradients = y.mul(&cotangent)?.sum_all()?.backward()?;
    close(
        gradients.get(&x).unwrap(),
        &[0., 3., 0., 3., 0., 0., 4., 5., 6., 0.],
    )?;
    let singleton = Var::from_slice(&[-2_f32], (1, 1, 1), &device)?;
    let y = max_pool1d(&singleton)?;
    close(&y, &[-2.])?;
    close(y.sum_all()?.backward()?.get(&singleton).unwrap(), &[1.])?;
    let even = Var::from_slice(&[1_f32, 3., 2., 4.], (1, 1, 4), &device)?;
    close(&max_pool1d(&even)?, &[3., 4.])
}

#[test]
fn cuda_replace_rows_clears_old_gradient_and_preserves_updates() -> TestResult {
    let device = Device::new_cuda(0)?;
    let base = Var::from_slice(&[1_f32, 2., 3., 4., 5., 6.], (1, 1, 3, 2), &device)?;
    let updates = Var::from_slice(&[10_f32, 20., 30., 40.], (1, 1, 2, 2), &device)?;
    let indices = Tensor::from_slice(&[2_u32, 0], (1, 1, 2), &device)?;
    let y = replace_rows(&base, &updates, &indices)?;
    close(&y, &[30., 40., 3., 4., 10., 20.])?;
    let cotangent = Tensor::from_slice(&[1_f32, 2., 3., 4., 5., 6.], (1, 1, 3, 2), &device)?;
    let gradients = y.mul(&cotangent)?.sum_all()?.backward()?;
    close(gradients.get(&base).unwrap(), &[0., 0., 3., 4., 0., 0.])?;
    close(gradients.get(&updates).unwrap(), &[5., 6., 1., 2.])
}

#[test]
fn cuda_operators_match_pytorch_forward_and_vjp_fixtures() -> TestResult {
    let device = Device::new_cuda(0)?;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/p3/operators.safetensors");
    let values = candle_core::safetensors::load(root, &device)?;
    for name in ["maxpool_overlap", "maxpool_tie", "maxpool_length_one"] {
        let x = Var::from_tensor(&values[&format!("{name}.input")])?;
        let output = max_pool1d(&x)?;
        close(
            &output,
            &values[&format!("{name}.output")]
                .flatten_all()?
                .to_vec1::<f32>()?,
        )?;
        let gradients = output
            .mul(&values[&format!("{name}.cotangent")])?
            .sum_all()?
            .backward()?;
        close(
            gradients.get(&x).unwrap(),
            &values[&format!("{name}.gradient")]
                .flatten_all()?
                .to_vec1::<f32>()?,
        )?;
    }
    for length in [1, 7] {
        let name = format!("prefix_sum_{length}");
        let x = Var::from_tensor(&values[&format!("{name}.input")])?;
        let output = prefix_sum(&x, 1)?;
        close(
            &output,
            &values[&format!("{name}.output")]
                .flatten_all()?
                .to_vec1::<f32>()?,
        )?;
        let gradients = output
            .mul(&values[&format!("{name}.cotangent")])?
            .sum_all()?
            .backward()?;
        close(
            gradients.get(&x).unwrap(),
            &values[&format!("{name}.gradient")]
                .flatten_all()?
                .to_vec1::<f32>()?,
        )?;
    }
    let base = Var::from_tensor(&values["replace_rows.base"].unsqueeze(1)?)?;
    let updates = Var::from_tensor(&values["replace_rows.updates"].unsqueeze(1)?)?;
    let output = replace_rows(
        &base,
        &updates,
        &values["replace_rows.indices"].unsqueeze(1)?,
    )?;
    close(
        &output,
        &values["replace_rows.output"]
            .flatten_all()?
            .to_vec1::<f32>()?,
    )?;
    let gradients = output
        .mul(&values["replace_rows.cotangent"].unsqueeze(1)?)?
        .sum_all()?
        .backward()?;
    close(
        gradients.get(&base).unwrap(),
        &values["replace_rows.base_gradient"]
            .flatten_all()?
            .to_vec1::<f32>()?,
    )?;
    close(
        gradients.get(&updates).unwrap(),
        &values["replace_rows.updates_gradient"]
            .flatten_all()?
            .to_vec1::<f32>()?,
    )
}
