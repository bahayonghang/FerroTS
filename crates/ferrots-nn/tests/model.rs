use candle_core::{Device, Tensor};
use ferrots_core::config::{ModelSpec, WindowSpec};
use ferrots_nn::{
    baselines::Baseline,
    model::{ForecastModel, ModelInput},
    registry,
};

#[test]
fn common_dispatch_preserves_baselines_and_independent_freeze() -> candle_core::Result<()> {
    let window = WindowSpec {
        lookback: 3,
        horizon: 2,
        ..Default::default()
    };
    let input = ModelInput::history(Tensor::from_vec(
        vec![1f32, 3., 2., 5., 4., 7.],
        (1, 3, 2),
        &Device::Cpu,
    )?);
    for spec in [
        ModelSpec::LastValue,
        ModelSpec::SeasonalNaive { period: 2 },
        ModelSpec::Linear { individual: false },
        ModelSpec::Dlinear {
            individual: true,
            moving_avg: 3,
        },
    ] {
        assert!(registry::find(spec.name()).is_some());
        let old = Baseline::new(&spec, &window, 2, 2021, &Device::Cpu)?;
        let model = ForecastModel::new(&spec, &window, 2, 2021, &Device::Cpu)?;
        let actual = model.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            actual,
            old.forward(&input)?.flatten_all()?.to_vec1::<f32>()?
        );
        let frozen = model.freeze_for_eval()?;
        for (_, parameter) in model.named_parameters() {
            parameter.set(&parameter.affine(0.0, 2.0)?)?;
        }
        let output = frozen.forward(&input)?;
        assert!(!output.track_op());
        assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    }
    assert_eq!(
        registry::find("itransformer").unwrap().reproduction_status,
        "p2_cuda_formal_pass"
    );
    Ok(())
}
