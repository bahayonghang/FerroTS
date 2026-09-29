use candle_core::{Device, Tensor, Var};
use ferrots_core::config::Schedule;
use ferrots_train::{
    NamedParams, ParamGroup, StatefulAdam, accumulation::GradientAccumulator, loss::masked_mse,
    schedule::epoch_end_rate,
};
#[test]
fn unequal_microbatches_match_large_batch_and_empty_does_not_step()
-> Result<(), Box<dyn std::error::Error>> {
    let mut devices = vec![Device::Cpu];
    if cfg!(feature = "cuda") {
        devices.push(Device::new_cuda(0)?);
    }
    for d in devices {
        let mut values = vec![];
        for sizes in [vec![5], vec![2, 3]] {
            let v = Var::from_vec(vec![0.5f32], 1, &d)?;
            let params = NamedParams::new(vec![("w".into(), v.clone())])?;
            let mut opt = StatefulAdam::new(
                params.clone(),
                vec![ParamGroup::adam(vec!["w".into()], 0.01)],
            )?;
            let mut a = GradientAccumulator::new(params);
            let xs = [1f32, 2., 3., 4., 5.];
            let ys = [2f32, 0., 4., 0., 3.];
            let ms = [1u8, 0, 1, 0, 1];
            let mut offset = 0;
            for n in sizes {
                let x = Tensor::from_slice(&xs[offset..offset + n], n, &d)?;
                let y = Tensor::from_slice(&ys[offset..offset + n], n, &d)?;
                let m = Tensor::from_slice(&ms[offset..offset + n], n, &d)?;
                a.add(&masked_mse(&x.broadcast_mul(&v)?, &y, &m)?)?;
                offset += n;
            }
            assert_eq!(a.valid_count, 3);
            assert!(a.step(&mut opt)?);
            assert!(!a.step(&mut opt)?);
            assert_eq!(opt.state("w").unwrap().step, 1);
            let moment = opt.state("w").unwrap().exp_avg.to_vec1::<f32>()?;
            opt.set_learning_rate(0.003)?;
            assert_eq!(moment, opt.state("w").unwrap().exp_avg.to_vec1::<f32>()?);
            assert_eq!(opt.state("w").unwrap().step, 1);
            assert!(opt.set_learning_rate(f64::NAN).is_err());
            assert_eq!(opt.groups()[0].lr, 0.003);
            values.push(v.to_vec1::<f32>()?[0]);
        }
        assert!((values[0] - values[1]).abs() < 1e-6);
    }
    let mut lr = 0.0001;
    let mut actual = vec![];
    for epoch in 1..=4 {
        actual.push(lr);
        lr = epoch_end_rate(0.0001, epoch, Schedule::Type1)?;
    }
    assert_eq!(actual, [0.0001, 0.0001, 0.00005, 0.000025]);
    Ok(())
}
