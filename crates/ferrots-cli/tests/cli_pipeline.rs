mod common;
use ferrots_core::config::ModelSpec;
use std::{fs, io::Read};
#[test]
fn four_models_inspect_train_evaluate_predict_and_future() {
    for spec in [
        ModelSpec::LastValue,
        ModelSpec::SeasonalNaive { period: 2 },
        ModelSpec::Linear { individual: true },
        ModelSpec::Dlinear {
            individual: false,
            moving_avg: 25,
        },
    ] {
        let (root, c) = common::fixture(spec);
        let config = root.join("config.toml");
        let config = config.to_str().unwrap();
        let first = common::json(&["data", "inspect", "--config", config]);
        assert_eq!(first["raw"]["rows"], 22);
        assert_eq!(first["cache"]["parse_count"], 1);
        let second = common::json(&["data", "inspect", "--config", config]);
        assert_eq!(second["cache"]["parse_count"], 0);
        let train = common::json(&["train", "--config", config]);
        assert_eq!(train["access"]["windows"][2], 0);
        let best = c.output.root.join("model/best");
        let best = best.to_str().unwrap();
        let metrics = root.join("metrics.json");
        let evaluation = common::json(&[
            "evaluate",
            "--checkpoint",
            best,
            "--output",
            metrics.to_str().unwrap(),
        ]);
        let output = root.join("predictions.csv.gz");
        let prediction = common::json(&[
            "predict",
            "--checkpoint",
            best,
            "--output",
            output.to_str().unwrap(),
            "--columns",
            "x",
        ]);
        assert_eq!(evaluation["metrics"], prediction["metrics"]);
        assert_eq!(prediction["prediction_rows"], 8);
        let mut text = String::new();
        flate2::read::GzDecoder::new(fs::File::open(output).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        let mut reader = csv::Reader::from_reader(text.as_bytes());
        assert_eq!(
            reader.headers().unwrap().iter().collect::<Vec<_>>(),
            [
                "run_id",
                "series_id",
                "forecast_origin",
                "target_time",
                "horizon",
                "target_name",
                "y_true",
                "y_pred",
                "valid"
            ]
        );
        assert_eq!(reader.records().count(), 8);
        let future_data = root.join("tail.csv");
        fs::write(
            &future_data,
            "date,y,x\n2024-01-21 00:00:00,40,20\n2024-01-22 00:00:00,42,21\n",
        )
        .unwrap();
        let future = root.join("future.csv");
        let p = common::json(&[
            "predict",
            "--checkpoint",
            best,
            "--future",
            "--input",
            future_data.to_str().unwrap(),
            "--output",
            future.to_str().unwrap(),
        ]);
        assert_eq!(p["rows"], 4);
        let mut rows = csv::Reader::from_path(future).unwrap();
        for row in rows.records() {
            let row = row.unwrap();
            assert_eq!(&row[6], "");
            assert_eq!(&row[8], "false");
            assert_eq!(&row[2], "2024-01-23 00:00:00.000000");
        }
    }
}
#[test]
fn reordered_input_uses_saved_scaler_and_cli_overrides_are_explicit() {
    let (root, c) = common::fixture(ModelSpec::LastValue);
    let config = root.join("config.toml");
    common::json(&[
        "train",
        "--config",
        config.to_str().unwrap(),
        "--seed",
        "77",
        "--horizon",
        "1",
    ]);
    let best = c.output.root.join("model/best");
    let best = best.to_str().unwrap();
    let expected = common::json(&["evaluate", "--checkpoint", best]);
    assert_eq!(expected["seed"], 77);
    let mut text = "y,date,x\n".to_string();
    for i in 0..22 {
        text.push_str(&format!("{},2024-01-{:02} 00:00:00,{}\n", i * 2, i + 1, i));
    }
    let input = root.join("reordered.csv");
    fs::write(&input, text).unwrap();
    let output = root.join("reordered-pred.csv");
    let actual = common::json(&[
        "predict",
        "--checkpoint",
        best,
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
    ]);
    assert_eq!(actual["metrics"], expected["metrics"]);
    let fail = common::run(&[
        "evaluate",
        "--checkpoint",
        best,
        "--input",
        input.to_str().unwrap(),
    ]);
    assert!(!fail.status.success());
    assert!(String::from_utf8_lossy(&fail.stderr).contains("fingerprint"));
    let missing = common::run(&["data", "inspect", "--config", "missing-config.toml"]);
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("data inspect failed"));
}
#[test]
fn shipped_protocol_configs_parse() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in [
        "reference/ett-h1.toml",
        "reference/ett-h2.toml",
        "reference/ett-m1.toml",
        "reference/ett-m2.toml",
        "reference/exchange.toml",
        "engineering/cement-twin.toml",
        "reference/synthetic.toml",
    ] {
        ferrots_core::config::Config::load(root.join("configs").join(name), Default::default())
            .unwrap()
            .validate_model_inputs()
            .unwrap();
    }
}
