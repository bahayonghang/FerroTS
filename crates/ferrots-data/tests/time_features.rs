use ferrots_data::{csv_source::parse_timestamp, time_features::compute};
#[test]
fn calendar_and_real_ett_match_fixed_upstream() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/p1/ett.json")).unwrap();
    let check = |row: &serde_json::Value, freq: &str| {
        let t = parse_timestamp(row["timestamp"].as_str().unwrap(), "%Y-%m-%d %H:%M:%S").unwrap();
        let actual = compute(&[t], freq).unwrap();
        for (a, b) in actual.iter().zip(row["features"].as_array().unwrap()) {
            assert!((f64::from(*a) - b.as_f64().unwrap()).abs() < 1e-6);
        }
    };
    for freq in ["h", "t"] {
        for row in fixture["calendar"][freq].as_array().unwrap() {
            check(row, freq);
        }
    }
    for dataset in fixture["datasets"].as_array().unwrap() {
        for row in dataset["time_rows"].as_array().unwrap() {
            check(row, "h");
        }
    }
    assert!(compute(&[0], "unsupported").is_err());
}
