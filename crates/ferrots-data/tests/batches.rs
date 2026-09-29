mod common;
use ferrots_core::config::{SplitSpec, WindowSpec};
use ferrots_core::reference_rng::ReferenceRng;
use ferrots_data::batch::Dataset;
#[test]
fn deterministic_training_order_fixed_evaluation_and_tail() {
    let mut text = "date,x,y\n".to_string();
    for i in 0..40 {
        let t = chrono::DateTime::from_timestamp(1_704_067_200 + i * 3600, 0).unwrap();
        text.push_str(&format!(
            "{},{},{}\n",
            t.format("%Y-%m-%d %H:%M:%S"),
            i,
            i * 2
        ));
    }
    let mut s = common::csv(&text, &["x", "y"]);
    s.target_columns = vec!["y".into()];
    s.split = SplitSpec::Rows {
        train_end: 24,
        validation_end: 32,
        test_end: 40,
    };
    let w = WindowSpec {
        lookback: 1,
        horizon: 1,
        gap: 0,
        stride: 1,
    };
    let d = Dataset::load(&s, &w, None).unwrap();
    let rng = ReferenceRng::new(2021);
    let a = d.index.training_order(&rng, 0);
    assert_eq!(a, d.index.training_order(&rng, 0));
    assert_ne!(a, d.index.training_order(&rng, 1));
    assert!(d.index.order(1).windows(2).all(|w| w[0] < w[1]));
    assert!(d.index.order(2).windows(2).all(|w| w[0] < w[1]));
    let sizes = a
        .chunks(11)
        .map(|chunk| d.batch(chunk).unwrap().batch_size)
        .collect::<Vec<_>>();
    assert_eq!(sizes, [11, 11, 1]);
    let b = d.batch(&[d.index.order(1)[0]]).unwrap();
    assert_eq!(
        (
            b.batch_size,
            b.lookback,
            b.horizon,
            b.channels,
            b.target_count
        ),
        (1, 1, 1, 2, 1)
    );
    assert_eq!(b.origins[0].row, 24);
    assert_eq!(b.input.history.len(), 2);
    assert_eq!(b.supervision.targets.len(), 1);
    let history_last = d.scaler.inverse(0, f64::from(b.input.history[0]));
    assert!((history_last - 23.0).abs() < 1e-5);
    let label = d.scaler.inverse(1, f64::from(b.supervision.targets[0]));
    assert!((label - 48.0).abs() < 1e-5);
}
