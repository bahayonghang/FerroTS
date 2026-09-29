mod common;
use ferrots_core::config::{DataSpec, SplitSpec, WindowSpec};
use ferrots_data::{prepare, read_csv, scaler::Scaler, windows::WindowIndex};
#[test]
fn all_four_ett_datasets_match_official_counts_and_scaler() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/p1/ett.json")).unwrap();
    for reference in fixture["datasets"].as_array().unwrap() {
        let name = reference["name"].as_str().unwrap();
        let mul = if name.starts_with("ETTm") { 4 } else { 1 };
        let cols: Vec<String> = reference["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().into())
            .collect();
        let s = DataSpec {
            path: std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../data/ETT-small/{name}.csv")),
            input_columns: cols.clone(),
            target_columns: cols,
            interval_seconds: 3600 / mul as i64,
            split: SplitSpec::Rows {
                train_end: 8640 * mul,
                validation_end: 11520 * mul,
                test_end: 14400 * mul,
            },
            ..DataSpec::default()
        };
        let p = prepare(read_csv(&s).unwrap(), &s).unwrap();
        let scaler = Scaler::fit(&p, &s).unwrap();
        for (j, col) in scaler.columns.iter().enumerate() {
            assert!((col.mean - reference["mean"][j].as_f64().unwrap()).abs() < 1e-9);
            assert!((col.scale - reference["scale"][j].as_f64().unwrap()).abs() < 1e-9);
        }
        for h in [96, 192, 336, 720] {
            let w = WindowSpec {
                lookback: 96,
                horizon: h,
                gap: 0,
                stride: 1,
            };
            let idx = WindowIndex::build(&p, &w).unwrap();
            let expected = &reference["counts"][h.to_string()];
            assert_eq!(
                idx.counts,
                [
                    expected["train"].as_u64().unwrap() as usize,
                    expected["val"].as_u64().unwrap() as usize,
                    expected["test"].as_u64().unwrap() as usize
                ],
                "{name} H{h}"
            );
            for e in &idx.entries {
                assert!(
                    p.splits[e.origin..e.origin + h]
                        .iter()
                        .all(|&s| s == e.split)
                );
            }
            assert_eq!(
                idx.entries.iter().find(|e| e.split == 1).unwrap().origin,
                8640 * mul
            );
        }
    }
}
#[test]
fn gaps_segments_and_no_windows() {
    let mut s = common::csv(
        "date,x\n2024-01-01 00:00:00,1\n2024-01-01 01:00:00,2\n2024-01-01 02:00:00,3\n2024-01-01 04:00:00,4\n2024-01-01 05:00:00,5\n2024-01-01 06:00:00,6\n",
        &["x"],
    );
    s.split = SplitSpec::Rows {
        train_end: 2,
        validation_end: 4,
        test_end: 6,
    };
    let p = prepare(read_csv(&s).unwrap(), &s).unwrap();
    let w = WindowSpec {
        lookback: 1,
        horizon: 1,
        gap: 1,
        stride: 1,
    };
    let idx = WindowIndex::build(&p, &w).unwrap();
    assert_eq!(
        idx.entries.iter().map(|e| e.origin).collect::<Vec<_>>(),
        [1, 4]
    );
    assert!(WindowIndex::build(&p, &WindowSpec { lookback: 10, ..w }).is_err());
}
