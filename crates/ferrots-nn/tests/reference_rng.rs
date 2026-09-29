use candle_core::Device;
use ferrots_core::config::{ModelSpec, WindowSpec};
use ferrots_nn::baselines::Baseline;

#[test]
fn pinned_initialization_and_ten_epoch_sampler() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/p1/reference_rng.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let get = |name: &str| case[name].as_u64().unwrap() as usize;
        let model = Baseline::new(
            &if case["kind"] == "dlinear" {
                ModelSpec::Dlinear {
                    individual: case["individual"].as_bool().unwrap(),
                    moving_avg: 25,
                }
            } else {
                ModelSpec::Linear {
                    individual: case["individual"].as_bool().unwrap(),
                }
            },
            &WindowSpec {
                lookback: get("length"),
                horizon: get("horizon"),
                gap: 0,
                stride: 1,
            },
            get("channels"),
            get("seed") as u64,
            &Device::Cpu,
        )
        .unwrap();
        for (name, tensor) in model.weights() {
            let expected = &case["parameters"][name];
            let bits: Vec<i32> = tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(|v| v.to_bits() as i32)
                .collect();
            assert_eq!(bits.len(), expected["count"].as_u64().unwrap() as usize);
            let first: Vec<i32> = serde_json::from_value(expected["first"].clone()).unwrap();
            let last: Vec<i32> = serde_json::from_value(expected["last"].clone()).unwrap();
            assert_eq!(bits[..first.len()], first, "{name} seed {}", get("seed"));
            assert_eq!(bits[bits.len() - last.len()..], last, "{name}");
        }
        for (epoch, order) in case["orders"].as_array().unwrap().iter().enumerate() {
            let expected: Vec<usize> = serde_json::from_value(order.clone()).unwrap();
            assert_eq!(
                model.sampling_rng().training_permutation(137, epoch as u64),
                expected
            );
        }
    }
}
