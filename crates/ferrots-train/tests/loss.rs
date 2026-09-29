use candle_core::{Device, Tensor, Var};
use ferrots_train::loss::masked_mse;
#[test]
fn invalid_nan_positions_are_zeroed_before_arithmetic() -> Result<(), Box<dyn std::error::Error>> {
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    for d in devices {
        let p = Var::from_vec(vec![2f32, f32::NAN, 3.], 3, &d)?;
        let y = Tensor::from_vec(vec![1f32, f32::NAN, 1.], 3, &d)?;
        let mask = Tensor::from_vec(vec![1u8, 0, 1], 3, &d)?;
        let loss = masked_mse(&p, &y, &mask)?;
        assert_eq!(loss.valid_count, 2);
        assert_eq!(loss.sum.to_scalar::<f32>()?, 5.);
        let g = loss.sum.backward()?;
        assert_eq!(g.get(&p).unwrap().to_vec1::<f32>()?, [2., 0., 4.]);
        assert_eq!(masked_mse(&p, &y, &mask.zeros_like()?)?.valid_count, 0);
        assert!(masked_mse(&p, &y, &mask.ones_like()?).is_err());
    }
    Ok(())
}
