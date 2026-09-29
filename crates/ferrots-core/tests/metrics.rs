use ferrots_core::metrics::{Accumulator, MetricGrid};
#[test]
fn element_weighted_missing_and_constant_errors() {
    let mut a = Accumulator::default();
    a.add(f64::NAN, f64::NAN, false).unwrap();
    assert_eq!(a.finish().mse, None);
    a.add(2.0, 1.0, true).unwrap();
    a.add(6.0, 3.0, true).unwrap();
    assert_eq!(a.finish().mae, Some(2.0));
    assert_eq!(a.finish().mse, Some(5.0));
    assert!(a.add(f64::INFINITY, 0.0, true).is_err());
    assert_eq!(a.finish().count, 2);
    let mut g = MetricGrid::new(vec!["a".into(), "b".into()], 2).unwrap();
    g.add(0, 0, (2., 1.), (20., 10.), true).unwrap();
    g.add(1, 0, (4., 1.), (40., 10.), true).unwrap();
    g.add(0, 1, (5., 5.), (5., 5.), true).unwrap();
    let r = g.finish();
    assert_eq!(r.standardized.overall.mse, Some(10. / 3.));
    assert_eq!(r.original.variables["a"].mae, Some(20.));
    assert_eq!(r.standardized.horizons[1].count, 1);
    assert_eq!(r.original.by_variable_horizon["b"][1].count, 0);
}
