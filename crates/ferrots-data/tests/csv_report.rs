mod common;
use ferrots_core::config::{DuplicatePolicy, NumberType, SortPolicy};
use ferrots_data::read_csv;

#[test]
fn unicode_named_columns_and_microseconds() {
    let mut spec = common::csv(
        "date,p (mbar),合成信号,% WEIGHTED ILI\n2024-02-29T01:02:03.123456,2,3,4\n",
        &["合成信号", "% WEIGHTED ILI", "p (mbar)"],
    );
    spec.timestamp_format = "%Y-%m-%dT%H:%M:%S%.f".into();
    let frame = read_csv(&spec).unwrap();
    assert_eq!(frame.values, [3.0, 4.0, 2.0]);
    assert_eq!(frame.timestamps[0].rem_euclid(1_000_000), 123456);
    assert!(frame.raw.start.ends_with("03.123456"));
    spec.target_columns = vec!["not an input".into()];
    assert!(read_csv(&spec).is_err());
}
#[test]
fn exchange_format_missing_and_error_redaction() {
    let mut spec = common::csv(
        "date,x,z\n1990/1/1 0:00,1,\n1990/1/2 0:00,1,NaN\n",
        &["x", "z"],
    );
    spec.timestamp_format = "%Y/%m/%d %H:%M".into();
    let f = read_csv(&spec).unwrap();
    assert_eq!(f.raw.rows, 2);
    assert!(f.raw.columns[0].constant);
    assert!(!f.raw.columns[1].constant);
    assert_eq!(f.raw.columns[1].missing_rate, 1.0);
    std::fs::write(&spec.path, "date,x,z\n1990/1/1 0:00,SECRET_CELL,2\n").unwrap();
    let error = read_csv(&spec).unwrap_err().to_string();
    assert!(error.contains("line 2"));
    assert!(!error.contains("SECRET_CELL"));
    std::fs::write(&spec.path, "date,x,z\n1990/1/1 0:00,inf,2\n").unwrap();
    assert!(read_csv(&spec).is_err());
    std::fs::write(&spec.path, "date,x,z\n1990/1/1 0:00,1.5,2\n").unwrap();
    spec.column_types.insert("x".into(), NumberType::Integer);
    assert!(read_csv(&spec).is_err());
    spec.path = spec.path.with_file_name("absent.csv");
    assert!(read_csv(&spec).is_err());
}
#[test]
fn duplicate_counts_and_stable_policies() {
    let mut text = "date,x\n".to_string();
    for hour in (0..7).rev() {
        text.push_str(&format!(
            "2024-01-01 {hour:02}:00:00,1\n2024-01-01 {hour:02}:00:00,2\n"
        ));
    }
    let mut spec = common::csv(&text, &["x"]);
    assert!(read_csv(&spec).is_err());
    spec.duplicates = DuplicatePolicy::KeepFirst;
    spec.sort = SortPolicy::Stable;
    let f = read_csv(&spec).unwrap();
    assert_eq!(
        (
            f.raw.rows,
            f.raw.duplicate_groups,
            f.raw.duplicate_rows,
            f.raw.duplicate_removed
        ),
        (14, 7, 14, 7)
    );
    assert_eq!(f.raw.non_increasing, 13);
    assert_eq!(f.values, vec![1.0; 7]);
    assert!(f.timestamps.windows(2).all(|t| t[0] < t[1]));
    spec.duplicates = DuplicatePolicy::KeepLast;
    assert_eq!(read_csv(&spec).unwrap().values, vec![2.0; 7]);
    spec.duplicates = DuplicatePolicy::Preserve;
    assert_eq!(read_csv(&spec).unwrap().rows(), 14);
}
#[test]
fn reordering_and_duplicate_headers() {
    let spec = common::csv("date,b,a\n2024-01-01 00:00:00,2,1\n", &["a", "b"]);
    assert_eq!(read_csv(&spec).unwrap().values, [1.0, 2.0]);
    std::fs::write(&spec.path, "date,a,a\n2024-01-01 00:00:00,1,2\n").unwrap();
    assert!(read_csv(&spec).is_err());
}
