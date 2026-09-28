use candle_core::{DType, Device, Tensor, Var, backprop::GradStore};
use ferrots_train::{NamedParams, ParamGroup, StatefulOptimizer};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

type TestResult = Result<(), Box<dyn std::error::Error>>;
#[derive(Deserialize)]
struct FixtureMetadata {
    groups: Vec<ParamGroup>,
    steps: usize,
}
struct Fixture {
    tensors: HashMap<String, Tensor>,
    metadata: FixtureMetadata,
}
impl Fixture {
    fn load<const W: bool>(device: &Device) -> Self {
        let kind = if W { "adamw" } else { "adam" };
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../tests/fixtures/optim/{kind}.safetensors"));
        let bytes = fs::read(&path).unwrap();
        let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        let metadata =
            serde_json::from_str(&metadata.metadata().as_ref().unwrap()["fixture"]).unwrap();
        Self {
            tensors: candle_core::safetensors::load(path, device).unwrap(),
            metadata,
        }
    }
    fn optimizer<const W: bool>(&self) -> StatefulOptimizer<W> {
        let params = self
            .metadata
            .groups
            .iter()
            .flat_map(|g| &g.names)
            .map(|name| {
                (
                    name.clone(),
                    Var::from_tensor(&self.tensors[&format!("initial.{name}")]).unwrap(),
                )
            })
            .collect();
        StatefulOptimizer::new(
            NamedParams::new(params).unwrap(),
            self.metadata.groups.clone(),
        )
        .unwrap()
    }
    fn step<const W: bool>(&self, optimizer: &mut StatefulOptimizer<W>, step: usize) {
        let mut gradients = GradStore::default();
        for (name, var) in optimizer.params().iter() {
            if let Some(gradient) = self.tensors.get(&format!("grad.{step:04}.{name}")) {
                gradients.insert(var, gradient.clone());
            }
        }
        optimizer.step(&gradients).unwrap();
    }
}
fn values(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}
fn close(actual: &Tensor, expected: &Tensor, tolerance: f32) {
    assert_eq!(actual.shape(), expected.shape());
    for (a, b) in values(actual).into_iter().zip(values(expected)) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
            "{a} != {b}, tolerance {tolerance}"
        );
    }
}
fn directory(label: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-artifacts")
        .join(format!(
            "{label}-{}-{nanos}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    fs::create_dir_all(&path).unwrap();
    path
}
fn fixture_match<const W: bool>(device: &Device) {
    let fixture = Fixture::load::<W>(device);
    let mut optimizer = fixture.optimizer::<W>();
    for step in 1..=fixture.metadata.steps {
        fixture.step(&mut optimizer, step);
        for (name, var) in optimizer.params().iter() {
            let state = optimizer.state(name).unwrap();
            close(
                var,
                &fixture.tensors[&format!("param.{step:04}.{name}")],
                1e-6,
            );
            close(
                &state.exp_avg,
                &fixture.tensors[&format!("exp_avg.{step:04}.{name}")],
                1e-6,
            );
            close(
                &state.exp_avg_sq,
                &fixture.tensors[&format!("exp_avg_sq.{step:04}.{name}")],
                1e-6,
            );
            assert_eq!(
                state.step,
                fixture.tensors[&format!("step.{step:04}.{name}")]
                    .to_scalar::<i64>()
                    .unwrap() as u64
            );
            assert_eq!(state.exp_avg.dtype(), DType::F32);
            assert!(!state.exp_avg.track_op() && !state.exp_avg_sq.track_op());
        }
    }
}
fn resume<const W: bool>(device: &Device, bitwise: bool) -> TestResult {
    let fixture = Fixture::load::<W>(device);
    let mut continuous = fixture.optimizer::<W>();
    let mut interrupted = fixture.optimizer::<W>();
    for step in 1..=10 {
        fixture.step(&mut continuous, step);
    }
    for step in 1..=5 {
        fixture.step(&mut interrupted, step);
    }
    let path = directory("resume");
    interrupted.params().save(path.join("model.safetensors"))?;
    interrupted.save(&path)?;
    let mut resumed = fixture.optimizer::<W>();
    resumed.params().load(path.join("model.safetensors"))?;
    resumed.load(&path)?;
    for step in 6..=10 {
        fixture.step(&mut resumed, step);
    }
    for (name, var) in continuous.params().iter() {
        let other = resumed.params().get(name).unwrap();
        let a = continuous.state(name).unwrap();
        let b = resumed.state(name).unwrap();
        assert_eq!(a.step, b.step);
        for (left, right) in [
            (var.as_tensor(), other.as_tensor()),
            (&a.exp_avg, &b.exp_avg),
            (&a.exp_avg_sq, &b.exp_avg_sq),
        ] {
            if bitwise {
                assert_eq!(
                    values(left).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    values(right)
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            } else {
                close(left, right, 1e-6);
            }
        }
    }
    Ok(())
}
#[test]
fn adam_matches_torch_ten_steps() {
    fixture_match::<false>(&Device::Cpu);
}
#[cfg(feature = "cuda")]
fn compare_devices<const W: bool>() -> TestResult {
    let cpu = Fixture::load::<W>(&Device::Cpu);
    let gpu = Fixture::load::<W>(&Device::new_cuda(0)?);
    let mut a = cpu.optimizer::<W>();
    let mut b = gpu.optimizer::<W>();
    for step in 1..=10 {
        cpu.step(&mut a, step);
        gpu.step(&mut b, step);
        for (name, var) in a.params().iter() {
            close(var, b.params().get(name).unwrap(), 1e-6);
            let left = a.state(name).unwrap();
            let right = b.state(name).unwrap();
            assert_eq!(left.step, right.step);
            close(&left.exp_avg, &right.exp_avg, 1e-6);
            close(&left.exp_avg_sq, &right.exp_avg_sq, 1e-6);
        }
    }
    Ok(())
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_optimizers_match_cpu_and_torch() -> TestResult {
    compare_devices::<false>()?;
    compare_devices::<true>()?;
    let device = Device::new_cuda(0)?;
    fixture_match::<false>(&device);
    fixture_match::<true>(&device);
    Ok(())
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_resume_matches_continuous_training() -> TestResult {
    let device = Device::new_cuda(0)?;
    resume::<false>(&device, false)?;
    resume::<true>(&device, false)
}
#[test]
fn adamw_matches_torch_ten_steps() {
    fixture_match::<true>(&Device::Cpu);
}
#[test]
fn cpu_resume_is_bitwise_identical() -> TestResult {
    resume::<false>(&Device::Cpu, true)?;
    resume::<true>(&Device::Cpu, true)
}

#[test]
fn constructors_reject_invalid_parameters_and_groups() -> TestResult {
    let var = Var::zeros(2, DType::F32, &Device::Cpu)?;
    assert!(NamedParams::new(vec![]).is_err());
    assert!(NamedParams::new(vec![("".into(), var.clone())]).is_err());
    assert!(NamedParams::new(vec![("a".into(), var.clone()), ("b".into(), var)]).is_err());
    assert!(
        NamedParams::new(vec![("a".into(), Var::zeros(2, DType::F64, &Device::Cpu)?)]).is_err()
    );
    assert!(
        NamedParams::new(vec![
            ("a".into(), Var::zeros(2, DType::F32, &Device::Cpu)?),
            ("a".into(), Var::zeros(2, DType::F32, &Device::Cpu)?)
        ])
        .is_err()
    );
    let fixture = Fixture::load::<true>(&Device::Cpu);
    let params = fixture.optimizer::<true>().params().clone();
    let good = fixture.metadata.groups;
    let mut cases = vec![
        vec![],
        vec![good[0].clone()],
        vec![good[0].clone(), good[0].clone()],
    ];
    for (field, value) in [
        ("lr", f64::NAN),
        ("lr", -0.1),
        ("beta1", 1.0),
        ("beta2", -0.1),
        ("eps", 0.0),
        ("weight_decay", -0.1),
    ] {
        let mut groups = good.clone();
        match field {
            "lr" => groups[0].lr = value,
            "beta1" => groups[0].beta1 = value,
            "beta2" => groups[0].beta2 = value,
            "eps" => groups[0].eps = value,
            _ => groups[0].weight_decay = value,
        }
        cases.push(groups);
    }
    let mut unknown = good.clone();
    unknown[0].names = vec!["missing".into()];
    cases.push(unknown);
    for groups in cases {
        assert!(StatefulOptimizer::<true>::new(params.clone(), groups).is_err());
    }
    Ok(())
}
#[test]
fn missing_or_invalid_gradients_do_not_mutate_state() -> TestResult {
    let fixture = Fixture::load::<true>(&Device::Cpu);
    let mut optimizer = fixture.optimizer::<true>();
    let original: Vec<_> = optimizer.params().iter().map(|(_, v)| values(v)).collect();
    optimizer.step(&GradStore::default())?;
    let mut gradients = GradStore::default();
    for (index, (_, var)) in optimizer.params().iter().enumerate() {
        gradients.insert(
            var,
            if index == 0 {
                var.ones_like()?
            } else {
                Tensor::zeros(99, DType::F32, &Device::Cpu)?
            },
        );
    }
    assert!(optimizer.step(&gradients).is_err());
    for ((name, var), old) in optimizer.params().iter().zip(original) {
        assert_eq!(values(var), old);
        assert_eq!(optimizer.state(name).unwrap().step, 0);
    }
    Ok(())
}
#[test]
fn checkpoint_validation_is_transactional() -> TestResult {
    let fixture = Fixture::load::<true>(&Device::Cpu);
    let mut source = fixture.optimizer::<true>();
    fixture.step(&mut source, 1);
    let path = directory("validation");
    source.save(&path)?;
    let json_path = path.join("optimizer-state.json");
    let original: serde_json::Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let mut destination = fixture.optimizer::<true>();
    let mut cases = Vec::new();
    for (key, value) in [
        ("format_version", serde_json::json!(99)),
        ("kind", serde_json::json!("adam")),
        ("dtype", serde_json::json!("f64")),
    ] {
        let mut bad = original.clone();
        bad[key] = value;
        cases.push(bad);
    }
    let mut bad = original.clone();
    bad["parameters"][0]["name"] = serde_json::json!("unknown");
    cases.push(bad);
    let mut bad = original.clone();
    bad["parameters"].as_array_mut().unwrap().reverse();
    cases.push(bad);
    let mut bad = original.clone();
    bad["parameters"][0]["shape"] = serde_json::json!([99]);
    cases.push(bad);
    let mut bad = original.clone();
    bad["groups"][0]["eps"] = serde_json::json!(0);
    cases.push(bad);
    for bad in cases {
        fs::write(&json_path, serde_json::to_vec(&bad)?)?;
        assert!(destination.load(&path).is_err());
        for (name, var) in destination.params().iter() {
            assert_eq!(destination.state(name).unwrap().step, 0);
            close(var, &fixture.tensors[&format!("initial.{name}")], 0.0);
        }
        assert_eq!(destination.groups(), fixture.metadata.groups);
    }
    fs::write(&json_path, serde_json::to_vec(&original)?)?;
    let tensor_path = path.join("optimizer.safetensors");
    let tensors = candle_core::safetensors::load(&tensor_path, &Device::Cpu)?;
    for mode in 0..5 {
        let mut bad = tensors.clone();
        let key = "encoder.weight.exp_avg_sq";
        match mode {
            0 => {
                bad.remove(key);
            }
            1 => {
                bad.insert("extra".into(), Tensor::zeros(1, DType::F32, &Device::Cpu)?);
            }
            2 => {
                bad.insert(key.into(), Tensor::zeros(99, DType::F32, &Device::Cpu)?);
            }
            3 => {
                bad.insert(key.into(), tensors[key].to_dtype(DType::F64)?);
            }
            _ => {
                let value = bad.remove(key).unwrap();
                bad.insert("wrong-key".into(), value);
            }
        }
        candle_core::safetensors::save(&bad, &tensor_path)?;
        assert!(destination.load(&path).is_err());
        assert!(
            destination
                .params()
                .iter()
                .all(|(n, _)| destination.state(n).unwrap().step == 0)
        );
    }
    candle_core::safetensors::save(&tensors, &tensor_path)?;
    let mut changed = original;
    changed["groups"][0]["lr"] = serde_json::json!(0.25);
    fs::write(&json_path, serde_json::to_vec(&changed)?)?;
    destination.load(&path)?;
    assert_eq!(destination.groups()[0].lr, 0.25);
    assert_eq!(destination.state("encoder.weight").unwrap().step, 1);
    Ok(())
}
#[test]
fn model_load_rejects_invalid_files_before_mutation() -> TestResult {
    let fixture = Fixture::load::<false>(&Device::Cpu);
    let optimizer = fixture.optimizer::<false>();
    let file = directory("model-validation").join("model.safetensors");
    let original: HashMap<String, Tensor> = optimizer
        .params()
        .iter()
        .map(|(n, v)| (n.into(), v.detach().copy().unwrap()))
        .collect();
    for mode in 0..5 {
        let mut bad = original.clone();
        bad.insert(
            "encoder.bias".into(),
            Tensor::zeros(3, DType::F32, &Device::Cpu)?,
        );
        match mode {
            0 => {
                bad.remove("encoder.weight");
            }
            1 => {
                bad.insert("extra".into(), Tensor::zeros(1, DType::F32, &Device::Cpu)?);
            }
            2 => {
                bad.insert(
                    "encoder.weight".into(),
                    Tensor::zeros(99, DType::F32, &Device::Cpu)?,
                );
            }
            3 => {
                bad.insert(
                    "encoder.weight".into(),
                    original["encoder.weight"].to_dtype(DType::F64)?,
                );
            }
            _ => {
                let value = bad.remove("encoder.weight").unwrap();
                bad.insert("unknown".into(), value);
            }
        }
        candle_core::safetensors::save(&bad, &file)?;
        assert!(optimizer.params().load(&file).is_err());
        for (name, var) in optimizer.params().iter() {
            close(var, &original[name], 0.0);
        }
    }
    Ok(())
}
