mod common;
use ferrots_data::cache::load_prepared;
#[test]
fn hit_has_zero_parses_and_content_config_changes_invalidate() {
    let mut s = common::csv(
        "date,x\n2024-01-01 00:00:00,1\n2024-01-01 01:00:00,2\n2024-01-01 02:00:00,3\n2024-01-01 03:00:00,4\n2024-01-01 04:00:00,5\n2024-01-01 05:00:00,6\n2024-01-01 06:00:00,7\n2024-01-01 07:00:00,8\n2024-01-01 08:00:00,9\n2024-01-01 09:00:00,10\n",
        &["x"],
    );
    let (p, a) = load_prepared(&s).unwrap();
    let (q, b) = load_prepared(&s).unwrap();
    assert_eq!(
        (a.hit, a.parse_count, b.hit, b.parse_count),
        (false, 1, true, 0)
    );
    assert_eq!(p.values, q.values);
    assert_eq!(p.timestamps, q.timestamps);
    let text = std::fs::read_to_string(&s.path)
        .unwrap()
        .replace(",10", ",11");
    std::fs::write(&s.path, text).unwrap();
    let (_, c) = load_prepared(&s).unwrap();
    assert_ne!(a.key, c.key);
    assert!(!c.hit);
    s.missing.max_stale_seconds += 1;
    let (_, d) = load_prepared(&s).unwrap();
    assert_ne!(c.key, d.key);
    std::fs::write(
        s.cache_dir.join(&d.key).join("table.safetensors"),
        b"corrupt",
    )
    .unwrap();
    assert!(
        load_prepared(&s)
            .unwrap_err()
            .to_string()
            .contains("checksum")
    );
    s.cache_bypass = true;
    let (_, e) = load_prepared(&s).unwrap();
    assert_eq!(e.parse_count, 1);
    assert!(!e.hit);
}
