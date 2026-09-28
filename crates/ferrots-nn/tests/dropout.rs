use candle_core::{DType, Device, Tensor, Var};
use ferrots_nn::{Dropout, Mode, RngStream, dropout, train_ops};
type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;
fn stream() -> RngStream {
    RngStream {
        seed: 123,
        run_step: 5,
        layer_id: 8,
    }
}
#[test]
fn dropout_reconstructs_stream_and_changes_each_component() -> TestResult {
    let x = Tensor::ones(256, DType::F32, &Device::Cpu)?;
    let run = |s| {
        dropout(&x, 0.5, Mode::Train, s)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let expected = run(stream());
    assert_eq!(expected, run(stream()));
    for s in [
        RngStream {
            seed: 124,
            ..stream()
        },
        RngStream {
            run_step: 6,
            ..stream()
        },
        RngStream {
            layer_id: 9,
            ..stream()
        },
    ] {
        assert_ne!(expected, run(s));
    }
    assert!(expected.contains(&0.0) && expected.contains(&2.0));
    Ok(())
}
#[test]
fn dropout_mode_probability_and_gradient() -> TestResult {
    let x = Var::ones(64, DType::F64, &Device::Cpu)?;
    let y = dropout(&x, 0.25, Mode::Train, stream())?;
    let grads = y.sum_all()?.backward()?;
    assert_eq!(
        y.to_vec1::<f64>()?,
        grads.get(&x).unwrap().to_vec1::<f64>()?
    );
    assert_eq!(
        dropout(&x, 0.9, Mode::Eval, stream())?.to_vec1::<f64>()?,
        vec![1.; 64]
    );
    assert_eq!(
        dropout(&x, 0., Mode::Train, stream())?.to_vec1::<f64>()?,
        vec![1.; 64]
    );
    for p in [-0.1, 1., f64::NAN, f64::INFINITY] {
        assert!(dropout(&x, p, Mode::Train, stream()).is_err());
    }
    Ok(())
}
#[test]
fn dropout_attention_preserves_gradient() -> TestResult {
    let q = Var::from_slice(&[0.3_f32, -0.2, 0.7, 0.1], (1, 1, 2, 2), &Device::Cpu)?;
    let option = Dropout {
        probability: 0.2,
        mode: Mode::Train,
        stream: stream(),
    };
    let y = train_ops::attention(&q, &q, &q, None, Some(option))?;
    assert!(y.sqr()?.sum_all()?.backward()?.get(&q).is_some());
    Ok(())
}
