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
