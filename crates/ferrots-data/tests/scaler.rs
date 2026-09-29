mod common;
use ferrots_core::config::SplitSpec;
use ferrots_data::{prepare, read_csv, scaler::Scaler};
#[test]
fn fits_only_training_observations_and_roundtrips() {
    let mut s = common::csv(
        "date,x,z\n2024-01-01 00:00:00,1,7\n2024-01-01 01:00:00,3,7\n2024-01-01 02:00:00,100,9\n2024-01-01 03:00:00,200,9\n",
        &["x", "z"],
    );
    s.split = SplitSpec::Rows {
        train_end: 2,
        validation_end: 3,
        test_end: 4,
    };
    let p = prepare(read_csv(&s).unwrap(), &s).unwrap();
    let scaler = Scaler::fit(&p, &s).unwrap();
    assert_eq!(
        (scaler.columns[0].mean, scaler.columns[0].scale),
        (2.0, 1.0)
    );
    assert_eq!(scaler.columns[1].scale, 1.0);
    assert!(scaler.columns[1].constant);
    assert_eq!(scaler.inverse(0, f64::from(scaler.transform(0, 3.0))), 3.0);
    let restored: Scaler = serde_json::from_str(&serde_json::to_string(&scaler).unwrap()).unwrap();
    restored.validate_for(&s, &p.columns).unwrap();
    assert!(
        restored
            .validate_for(&s, &["z".into(), "x".into()])
            .is_err()
    );
}
