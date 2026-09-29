use candle_core::{Device, Tensor, Var};
use ferrots_core::config::{ModelSpec, WindowSpec};
use ferrots_nn::baselines::{Baseline, ModelInput, decompose};
type R = Result<(), Box<dyn std::error::Error>>;
fn devices() -> candle_core::Result<Vec<Device>> {
    #[cfg(feature = "cuda")]
    {
        Ok(vec![Device::Cpu, Device::new_cuda(0)?])
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(vec![Device::Cpu])
    }
}
#[test]
fn naive_shapes_period_gap_and_empty_parameters() -> R {
    for d in devices()? {
        let w = WindowSpec {
            lookback: 4,
            horizon: 5,
            gap: 1,
            stride: 1,
        };
        let input = ModelInput::history(Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4, 1), &d)?);
        let last = Baseline::new(&ModelSpec::LastValue, &w, 1, 1, &d)?;
        assert!(last.named_parameters().is_empty());
        assert_eq!(
            last.forward(&input)?.flatten_all()?.to_vec1::<f32>()?,
            vec![4.; 5]
        );
        let seasonal = Baseline::new(&ModelSpec::SeasonalNaive { period: 2 }, &w, 1, 1, &d)?;
        assert_eq!(
            seasonal.forward(&input)?.flatten_all()?.to_vec1::<f32>()?,
            [4., 3., 4., 3., 4.]
        );
        assert!(Baseline::new(&ModelSpec::SeasonalNaive { period: 5 }, &w, 1, 1, &d).is_err());
    }
    Ok(())
}
#[test]
fn initialization_constants_and_independent_freeze() -> R {
    for d in devices()? {
        for individual in [false, true] {
            let w = WindowSpec {
                lookback: 3,
                horizon: 2,
                gap: 0,
                stride: 1,
            };
            let m = Baseline::new(
                &ModelSpec::Dlinear {
                    individual,
                    moving_avg: 25,
                },
                &w,
                2,
                2021,
                &d,
            )?;
            for (n, v) in m.named_parameters() {
                let vals = v.flatten_all()?.to_vec1::<f32>()?;
                if n.ends_with("weight") {
                    assert!(vals.iter().all(|v| (*v - 1. / 3.).abs() < 1e-7));
                } else {
                    assert!(vals.iter().all(|v| v.abs() <= 1. / 3f32.sqrt()));
                    assert!(vals.iter().any(|v| *v != 0.));
                }
            }
            let x = Var::from_vec(vec![2f32; 6], (1, 3, 2), &d)?;
            let (seasonal, trend) = decompose(&x, 25)?;
            assert!(
                seasonal
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|v| v.abs() < 1e-6)
            );
            assert!(
                trend
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|v| (*v - 2.).abs() < 1e-6)
            );
            let frozen = m.freeze_for_eval()?;
            assert!(frozen.named_parameters().is_empty());
            assert!(frozen.weights().values().all(|v| !v.track_op()));
            let input = ModelInput::history(x.as_tensor().clone());
            let before = frozen.forward(&input)?;
            assert!(!before.track_op());
            for (_, v) in m.named_parameters() {
                v.set(&v.zeros_like()?)?;
            }
            assert_eq!(
                before.flatten_all()?.to_vec1::<f32>()?,
                frozen.forward(&input)?.flatten_all()?.to_vec1::<f32>()?
            );
            let linear = Baseline::new(&ModelSpec::Linear { individual }, &w, 2, 1, &d)?;
            let loss = linear.forward(&input)?.sqr()?.sum_all()?;
            let grad = loss.backward()?;
            for (_, v) in linear.named_parameters() {
                assert!(grad.get(&v).is_some());
            }
        }
    }
    Ok(())
}
