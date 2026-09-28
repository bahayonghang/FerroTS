#![cfg(feature = "cuda")]
use candle_core::{DType, Device, Tensor, Var};
use ferrots_nn::{Dropout, Mode, RngStream, dropout, train_ops};
type TestResult = Result<(), Box<dyn std::error::Error>>;
fn close(a: &Tensor, b: &Tensor) -> TestResult {
    assert_eq!(a.shape(), b.shape());
    let av = a.flatten_all()?.to_vec1::<f32>()?;
    let bv = b.flatten_all()?.to_vec1::<f32>()?;
    for (a, b) in av.into_iter().zip(bv) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5 + 1e-4 * a.abs(),
            "CPU {a}, CUDA {b}"
        );
    }
    Ok(())
}
fn compare(
    inputs: &[(&[f32], &[usize])],
    operation: impl Fn(&[Tensor], &Device) -> train_ops::Result<Tensor>,
) -> TestResult {
    let gpu = Device::new_cuda(0)?;
    let mut results = Vec::new();
    for device in [&Device::Cpu, &gpu] {
        let vars = inputs
            .iter()
            .map(|(v, s)| Var::from_slice(v, *s, device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let tensors: Vec<_> = vars.iter().map(|v| v.as_tensor().clone()).collect();
        let output = operation(&tensors, device)?;
        let weights = Tensor::from_vec(
            (0..output.elem_count())
                .map(|i| (i + 1) as f32 / 7.)
                .collect::<Vec<_>>(),
            output.shape(),
            device,
        )?;
        let gradients = output.mul(&weights)?.sum_all()?.backward()?;
        let mut result = vec![output];
        for var in vars {
            result.push(
                gradients
                    .get(&var)
                    .expect("missing required gradient")
                    .clone(),
            );
        }
        results.push(result);
    }
    for (a, b) in results[0].iter().zip(&results[1]) {
        close(a, b)?;
    }
    Ok(())
}
#[test]
fn cuda_train_ops_reductions_and_layer_norm_match_cpu() -> TestResult {
    let input = [0.3_f32, -0.7, 2., 0.4, -0.2, 0.9];
    for dim in [0, 1] {
        compare(&[(&input, &[2, 3])], |xs, _| {
            train_ops::softmax(&xs[0], dim)
        })?;
    }
    compare(&[(&input, &[2, 3])], |xs, _| {
        train_ops::softmax(&xs[0].transpose(0, 1)?, 1)
    })?;
    compare(&[(&input, &[2, 3])], |xs, device| {
        train_ops::masked_softmax(
            &xs[0],
            &Tensor::from_slice(&[1_u8, 0, 1], (1, 3), device)?,
            1,
        )
    })?;
    for input in [input, [3.; 6]] {
        for keepdim in [false, true] {
            compare(&[(&input, &[2, 3])], |xs, _| {
                train_ops::pop_var(&xs[0], 1, keepdim)
            })?;
        }
        compare(
            &[
                (&input, &[2, 3]),
                (&[0.8, 1.2, -0.6], &[3]),
                (&[0.1, -0.2, 0.3], &[3]),
            ],
            |xs, _| train_ops::layer_norm(&xs[0], &xs[1], &xs[2], 1e-5),
        )?;
    }
    compare(&[(&[3., -2.], &[2, 1])], |xs, _| {
        train_ops::pop_var(&xs[0], 1, true)
    })?;
    let gpu = Device::new_cuda(0)?;
    let x = Tensor::zeros((2, 3), DType::F32, &gpu)?;
    let mask = Tensor::from_slice(&[1_u8, 0, 1, 0, 0, 0], (2, 3), &gpu)?;
    assert!(matches!(
        train_ops::masked_softmax(&x, &mask, 1),
        Err(train_ops::OpError::FullyMaskedRow)
    ));
    Ok(())
}
#[test]
fn cuda_attention_qkv_gradients_match_cpu() -> TestResult {
    let inputs: &[(&[f32], &[usize])] = &[
        (&[0.3, -0.7, 1.1, 0.2], &[1, 1, 2, 2]),
        (&[0.4, 0.8, -0.2, 0.1, 0.3, -0.9], &[1, 1, 3, 2]),
        (&[0.6, -0.4, 0.2, 0.5, 0.3, -0.8], &[1, 1, 3, 2]),
    ];
    compare(inputs, |xs, _| {
        train_ops::attention(&xs[0], &xs[1], &xs[2], None, None)
    })?;
    compare(inputs, |xs, device| {
        let mask = Tensor::from_slice(&[1_u8, 1, 0, 1, 0, 1], (1, 1, 2, 3), device)?;
        train_ops::attention(
            &xs[0],
            &xs[1],
            &xs[2],
            Some(&mask),
            Some(Dropout {
                probability: 0.2,
                mode: Mode::Train,
                stream: RngStream {
                    seed: 137,
                    run_step: 4,
                    layer_id: 2,
                },
            }),
        )
    })
}
#[test]
fn cuda_dropout_mask_and_gradients_match_cpu() -> TestResult {
    let input = vec![1_f32; 256];
    let stream = RngStream {
        seed: 37,
        run_step: 12,
        layer_id: 3,
    };
    for mode in [Mode::Train, Mode::Eval] {
        compare(&[(&input, &[16, 16])], |xs, _| {
            dropout(&xs[0], 0.25, mode, stream)
        })?;
    }
    let cpu = dropout(
        &Tensor::from_slice(&input, 256, &Device::Cpu)?,
        0.25,
        Mode::Train,
        stream,
    )?
    .to_vec1::<f32>()?;
    let gpu = dropout(
        &Tensor::from_slice(&input, 256, &Device::new_cuda(0)?)?,
        0.25,
        Mode::Train,
        stream,
    )?
    .to_vec1::<f32>()?;
    assert_eq!(
        cpu.iter().map(|x| *x == 0.).collect::<Vec<_>>(),
        gpu.iter().map(|x| *x == 0.).collect::<Vec<_>>()
    );
    assert!(cpu.contains(&0.) && cpu.iter().any(|x| *x != 0.));
    Ok(())
}
