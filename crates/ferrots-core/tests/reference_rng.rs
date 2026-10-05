use ferrots_core::reference_rng::ReferenceRng;

#[test]
fn pinned_torch_integer_and_permutation_sequences() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/p1/reference_rng.json"
    ))
    .unwrap();
    for case in fixture["primitives"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let mut rng = ReferenceRng::new(seed);
        for value in case["integers"].as_array().unwrap() {
            assert_eq!(rng.seed_draw(), value.as_u64().unwrap());
        }
        for (size, values) in case["permutations"].as_object().unwrap() {
            let expected: Vec<usize> = serde_json::from_value(values.clone()).unwrap();
            assert_eq!(
                ReferenceRng::new(seed).permutation(size.parse().unwrap()),
                expected
            );
        }
    }
}

#[test]
fn versioned_rng_roundtrip_rejects_invalid_states() {
    use ferrots_core::reference_rng::ReferenceRngState;
    for consumed in [0, 1, 623, 624, 625, 10000] {
        let mut rng = ReferenceRng::new(2021);
        for _ in 0..consumed {
            rng.uniform(-1., 1.);
        }
        let saved = rng.snapshot();
        let json = serde_json::to_vec(&saved).unwrap();
        let restored: ReferenceRngState = serde_json::from_slice(&json).unwrap();
        let mut resumed = ReferenceRng::from_state(&restored).unwrap();
        assert_eq!(saved, resumed.snapshot());
        for _ in 0..1000 {
            assert_eq!(rng.seed_draw(), resumed.seed_draw());
        }
    }
    let saved = ReferenceRng::new(2021).snapshot();
    for kind in 0..5 {
        let mut bad = saved.clone();
        match kind {
            0 => bad.format_version = 2,
            1 => bad.algorithm = "unknown".into(),
            2 => {
                bad.words.pop();
            }
            3 => bad.index = 625,
            _ => bad.words.fill(0),
        }
        assert!(ReferenceRng::from_state(&bad).is_err());
    }
    let mut value = serde_json::to_value(saved).unwrap();
    value["extra"] = 1.into();
    assert!(serde_json::from_value::<ReferenceRngState>(value).is_err());
}
