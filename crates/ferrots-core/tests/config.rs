use ferrots_core::config::{Config, ModelSpec, Overrides};

const BASE: &str = r#"
[data]
path = "synthetic.csv"
input_columns = ["过程量甲", "p (mbar)", "% WEIGHTED ILI"]
target_columns = ["p (mbar)"]
"#;

#[test]
fn defaults_file_and_explicit_cli_have_ordered_precedence() {
    let source = format!("{BASE}\n[train]\nseed=19\nbatch_size=7\ndevice='cpu'\n");
    let a = Config::from_toml(&source, Overrides::default()).unwrap();
    assert_eq!(
        (a.train.seed, a.train.batch_size, a.train.max_epochs),
        (19, 7, 10)
    );
    let b = Config::from_toml(
        &source,
        Overrides {
            seed: Some(0),
            horizon: Some(15),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        (b.train.seed, b.train.batch_size, b.window.horizon),
        (0, 7, 15)
    );
    assert_eq!(b.data.target_columns, ["p (mbar)"]);
    let encoded = toml::to_string(&b).unwrap();
    let restored = Config::from_toml(&encoded, Overrides::default()).unwrap();
    assert_eq!(restored.data.input_columns, b.data.input_columns);
}

#[test]
fn unknown_fields_are_rejected_at_each_boundary() {
    for suffix in [
        "\nextra = 1",
        "\n[train]\nepochz=1",
        "\n[window]\nunknown=1",
        "\n[data.missing]\nmax_stale=1",
        "\n[model]\nname='linear'\nmoving_avg=25",
        "\n[data.split]\nkind='ratio'\ntrain=0.7\ntest=0.2\nunknown=1",
    ] {
        assert!(
            Config::from_toml(&format!("{BASE}{suffix}"), Overrides::default()).is_err(),
            "{suffix}"
        );
    }
    assert!(Config::from_toml(&format!("unknown=1\n{BASE}"), Overrides::default()).is_err());
}

#[test]
fn rejects_invalid_semantics_before_data_io() {
    let base = Config::from_toml(BASE, Overrides::default()).unwrap();
    let mut invalid = vec![];
    let mut c = base.clone();
    c.window.horizon = 0;
    invalid.push(c);
    let mut c = base.clone();
    c.data.target_columns = vec!["missing".into()];
    invalid.push(c);
    let mut c = base.clone();
    c.data.input_columns.push("过程量甲".into());
    invalid.push(c);
    let mut c = base.clone();
    c.train.device = "cuda:-1".into();
    invalid.push(c);
    let mut c = base.clone();
    c.train.dtype = "bf16".into();
    invalid.push(c);
    let mut c = base.clone();
    c.model = ModelSpec::Dlinear {
        individual: false,
        moving_avg: 24,
    };
    invalid.push(c);
    let mut c = base.clone();
    c.model = ModelSpec::SeasonalNaive { period: 97 };
    invalid.push(c);
    let mut c = base.clone();
    c.train.learning_rate = f64::NAN;
    invalid.push(c);
    let mut c = base.clone();
    c.data.missing.filled_targets_valid = true;
    invalid.push(c);
    let mut c = base.clone();
    c.data.time_features = "s".into();
    invalid.push(c);
    let mut c = base.clone();
    c.output.run_id = Some("../existing".into());
    invalid.push(c);
    let mut c = base;
    c.data.split = ferrots_core::config::SplitSpec::Rows {
        train_end: 20,
        validation_end: 10,
        test_end: 30,
    };
    invalid.push(c);
    for config in invalid {
        assert!(config.validate().is_err(), "{config:?}");
    }
}

#[test]
fn future_known_schema_is_readable_but_baselines_reject_it() {
    let mut c = Config::from_toml(BASE, Overrides::default()).unwrap();
    c.data.future_known_columns = vec!["planned".into()];
    c.validate().unwrap();
    assert!(c.validate_model_inputs().is_err());
}
