mod common;
use ferrots_core::config::{
    CompareOp, DataSpec, DuplicatePolicy, GridSpec, MissingMode, RunRule, SortPolicy, SplitSpec,
};
use ferrots_data::{prepare, read_csv};

fn synthetic(
    rows: usize,
    missing: impl Fn(usize, usize) -> bool,
    stopped: impl Fn(usize) -> bool,
) -> DataSpec {
    let mut text = "date,state,x\n".to_string();
    for i in 0..rows {
        for sample in 0..3 {
            let t =
                chrono::DateTime::from_timestamp(1_704_067_200 + i as i64 * 60 + sample * 10, 0)
                    .unwrap();
            let state = if missing(i, 0) {
                String::new()
            } else if stopped(i) {
                "0".into()
            } else {
                "10".into()
            };
            let x = if missing(i, 1) {
                String::new()
            } else {
                (i + 1).to_string()
            };
            text.push_str(&format!("{},{state},{x}\n", t.format("%Y-%m-%d %H:%M:%S")));
        }
    }
    let mut spec = common::csv(&text, &["state", "x"]);
    spec.interval_seconds = 60;
    spec.grid = Some(GridSpec { min_samples: 3 });
    spec.sort = SortPolicy::Stable;
    spec.duplicates = DuplicatePolicy::KeepFirst;
    spec.missing.mode = MissingMode::Causal;
    spec.running = Some(RunRule {
        column: "state".into(),
        op: CompareOp::Gt,
        threshold: 5.0,
        min_duration_seconds: 1800,
        max_missing_seconds: 300,
    });
    spec.split = SplitSpec::Segments {
        train: 0.7,
        validation: 0.1,
    };
    spec
}
#[test]
fn unknown_five_bridges_six_breaks_and_edges_are_trimmed() {
    let s = synthetic(
        80,
        |i, j| j == 0 && (i < 2 || (35..40).contains(&i) || i >= 78),
        |_| false,
    );
    let p = prepare(read_csv(&s).unwrap(), &s).unwrap();
    assert_eq!(p.segments.len(), 1);
    assert_eq!((p.segments[0].start, p.segments[0].end), (2, 78));
    assert_eq!(p.imputed[35 * 2], 1);
    let s = synthetic(80, |i, j| j == 0 && (35..41).contains(&i), |_| false);
    let p = prepare(read_csv(&s).unwrap(), &s).unwrap();
    assert_eq!(
        p.segments
            .iter()
            .map(|s| (s.start, s.end))
            .collect::<Vec<_>>(),
        [(0, 35), (41, 80)]
    );
}
#[test]
fn minimum_segment_and_stopped_rows() {
    let s = synthetic(60, |_, _| false, |i| i == 29);
    let p = prepare(read_csv(&s).unwrap(), &s).unwrap();
    assert_eq!(
        p.segments
            .iter()
            .map(|s| (s.start, s.end))
            .collect::<Vec<_>>(),
        [(30, 60)]
    );
    let s = synthetic(29, |_, _| false, |_| false);
    assert!(prepare(read_csv(&s).unwrap(), &s).is_err());
}
#[test]
fn fill_clock_does_not_reset_and_complete_spans_split() {
    let mut s = synthetic(70, |i, j| j == 1 && (11..=21).contains(&i), |_| false);
    s.protocol_id = "cement-twin-v1".into();
    s.missing.complete_windows = true;
    s.missing.filled_targets_valid = true;
    let p = prepare(read_csv(&s).unwrap(), &s).unwrap();
    assert_eq!(p.available[20 * 2 + 1], 1);
    assert_eq!(p.available[21 * 2 + 1], 0);
    assert_eq!(p.observed[20 * 2 + 1], 0);
    assert_eq!(p.values[20 * 2 + 1], 11.0);
    assert_eq!(
        p.spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
        [(0, 21), (22, 70)]
    );
    assert_eq!(p.report.segment_rows, 70);
    assert_eq!(p.report.usable_rows.iter().sum::<usize>(), 69);
}
#[test]
fn split_midpoints_use_full_segment_lengths_before_complete_filter() {
    let mut s = synthetic(
        152,
        |i, j| j == 1 && (5..20).contains(&i),
        |i| i == 50 || i == 101,
    );
    s.missing.max_stale_seconds = 0;
    s.missing.complete_windows = true;
    s.split = SplitSpec::Segments {
        train: 0.5,
        validation: 0.25,
    };
    let p = prepare(read_csv(&s).unwrap(), &s).unwrap();
    assert_eq!(p.report.segment_rows, 150);
    assert_eq!(p.report.split_segments, [1, 1, 1]);
    assert_eq!(p.report.usable_rows, [35, 50, 50]);
    assert_eq!(p.splits[50], 3);
    assert_eq!(p.splits[51], 1);
}
#[test]
fn aggregate_minimum_count_and_date_boundaries() {
    let mut s = common::csv(
        "date,x\n2024-01-01 00:00:01,2\n2024-01-01 00:00:10,4\n2024-01-01 00:01:00,8\n2024-01-01 00:02:00,9\n",
        &["x"],
    );
    s.interval_seconds = 60;
    s.grid = Some(GridSpec { min_samples: 2 });
    s.missing.mode = MissingMode::Causal;
    s.split = SplitSpec::Dates {
        train_end: "2024-01-01 00:01:00".into(),
        validation_end: "2024-01-01 00:02:00".into(),
        test_end: "2024-01-01 00:03:00".into(),
    };
    let p = prepare(read_csv(&s).unwrap(), &s).unwrap();
    assert_eq!(p.observed, [1, 0, 0]);
    assert_eq!(p.values[0], 3.0);
    assert_eq!(p.splits, [0, 1, 2]);
    s.missing.mode = MissingMode::Error;
    assert!(prepare(read_csv(&s).unwrap(), &s).is_err());
}
