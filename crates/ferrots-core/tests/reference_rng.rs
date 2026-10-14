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

#[test]
fn torch_randint_word_count_and_empty_sites_keep_five_draws() {
    // torch 2.14.0+cu130, git 08187d9e. manual_seed(2022), randint, then one int64 random_().
    let cases = [
        (96_u64, 4_usize, 2_091_559_949_382_189_041_u64),
        (1 << 28, 4, 3_420_166_536_313_309_202),
        ((1 << 28) - 1, 3, 243_636_865_075_689_137),
        (1 << 28, 3, 921_857_968_566_694_839),
    ];
    for (high, count, expected) in cases {
        let mut rng = ReferenceRng::new(2022);
        rng.consume_torch_randint(high, count);
        assert_eq!(rng.seed_draw(), expected, "high={high} count={count}");
    }
    let base = ReferenceRng::new(2022);
    for epoch in 0..6 {
        assert_eq!(
            base.training_permutation(137, epoch),
            base.training_permutation_after_randint(137, epoch, 9, 4, 3, &[])
        );
    }
    let plain = base.training_permutation(32, 1);
    let shifted = base.training_permutation_after_randint(32, 1, 1, 0, 0, &[(7, 3)]);
    assert_ne!(plain, shifted);
}
