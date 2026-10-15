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

#[test]
fn itransformer_defaults_and_explicit_parameters_are_strict() {
    use ferrots_core::config::Activation;
    let source = format!("{BASE}\n[model]\nname='itransformer'\n");
    let c = Config::from_toml(&source, Overrides::default()).unwrap();
    assert!(matches!(
        c.model,
        ModelSpec::Itransformer {
            d_model: 512,
            n_heads: 8,
            d_ff: 2048,
            e_layers: 2,
            dropout: 0.1,
            activation: Activation::Gelu,
            use_norm: true
        }
    ));
    assert!(c.model.trainable());
    let explicit = format!(
        "{source}d_model=16\nn_heads=4\nd_ff=32\ne_layers=1\ndropout=0.0\nactivation='relu'\nuse_norm=false\n[train]\ndrop_last=true\n"
    );
    let c = Config::from_toml(&explicit, Overrides::default()).unwrap();
    assert!(c.train.drop_last);
    let encoded = toml::to_string(&c).unwrap();
    assert_eq!(
        toml::to_string(&Config::from_toml(&encoded, Overrides::default()).unwrap()).unwrap(),
        encoded
    );
    for invalid in [
        "d_model=0",
        "n_heads=0",
        "d_ff=0",
        "e_layers=0",
        "n_heads=3",
        "dropout=-0.1",
        "dropout=1.0",
        "dropout=nan",
        "dropout=inf",
        "activation='silu'",
        "label_len=48",
        "d_layers=1",
        "decoder=true",
        "dec_in=7",
        "c_out=7",
        "distil=true",
        "factor=1",
        "unknown=1",
    ] {
        assert!(
            Config::from_toml(&format!("{source}{invalid}\n"), Overrides::default()).is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn itransformer_rejects_future_known_and_reordered_targets() {
    let source = format!("{BASE}\n[model]\nname='itransformer'\n");
    let mut c = Config::from_toml(&source, Overrides::default()).unwrap();
    c.data.future_known_columns.push("planned".into());
    assert!(c.validate().is_err());
    c.data.future_known_columns.clear();
    c.data.target_columns = vec!["% WEIGHTED ILI".into(), "过程量甲".into()];
    assert!(c.validate().is_err());
    c.data.target_columns.reverse();
    c.validate().unwrap();
}

#[test]
fn default_tail_policy_preserves_p1_serialized_identity() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/reference");
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|extension| extension != "toml") {
            continue;
        }
        let c = Config::load(&path, Overrides::default()).unwrap();
        assert!(!c.train.drop_last);
        let encoded = toml::to_string(&c).unwrap();
        assert!(!encoded.contains("drop_last"));
        assert!(!encoded.contains("checkpoint_history"));
        let explicit = encoded.replace("[train]", "[train]\ndrop_last=false");
        let restored = Config::from_toml(&explicit, Overrides::default()).unwrap();
        assert_eq!(
            toml::to_string(&restored).unwrap().as_bytes(),
            encoded.as_bytes()
        );
    }
}

#[test]
fn checkpoint_history_defaults_to_final_and_can_keep_all() {
    let c = Config::from_toml(BASE, Overrides::default()).unwrap();
    assert_eq!(
        c.output.checkpoint_history,
        ferrots_core::config::CheckpointHistory::Final
    );
    assert!(!toml::to_string(&c).unwrap().contains("checkpoint_history"));
    let explicit = format!("{BASE}\n[output]\ncheckpoint_history='final'\n");
    let restored = Config::from_toml(&explicit, Overrides::default()).unwrap();
    assert!(
        !toml::to_string(&restored)
            .unwrap()
            .contains("checkpoint_history")
    );
    let all = format!("{BASE}\n[output]\ncheckpoint_history='all'\n");
    let kept = Config::from_toml(&all, Overrides::default()).unwrap();
    assert_eq!(
        kept.output.checkpoint_history,
        ferrots_core::config::CheckpointHistory::All
    );
    let encoded = toml::to_string(&kept).unwrap();
    assert!(encoded.contains("checkpoint_history = \"all\""));
    assert_eq!(
        toml::to_string(&Config::from_toml(&encoded, Overrides::default()).unwrap()).unwrap(),
        encoded
    );
    assert!(
        Config::from_toml(
            &format!("{BASE}\n[output]\ncheckpoint_history='everything'\n"),
            Overrides::default()
        )
        .is_err()
    );
}

#[test]
fn informer_configuration_rejects_invalid_decoder_and_dimensions() {
    let source = format!("{BASE}\n[model]\nname='informer'\n");
    let c = Config::from_toml(&source, Overrides::default()).unwrap();
    assert_eq!(c.model.name(), "informer");
    assert!(c.model.trainable());
    let encoded = toml::to_string(&c).unwrap();
    assert_eq!(
        toml::to_string(&Config::from_toml(&encoded, Overrides::default()).unwrap()).unwrap(),
        encoded
    );
    for suffix in [
        "d_model=15\nn_heads=3",
        "n_heads=0",
        "d_ff=0",
        "e_layers=0",
        "d_layers=0",
        "factor=0",
        "label_len=0",
        "label_len=97",
        "dropout=1.0",
        "use_norm=true",
        "[window]\ngap=1",
    ] {
        assert!(
            Config::from_toml(&format!("{source}{suffix}\n"), Overrides::default()).is_err(),
            "{suffix}"
        );
    }
    let mut c = c;
    c.data.future_known_columns = vec!["known".into()];
    assert!(c.validate().is_err());
}
