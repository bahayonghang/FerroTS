use clap::{Args, ValueEnum};
use ferrots_core::config::{Config, ModelSpec, Overrides};
use ferrots_data::{batch::Dataset, cache::write_synced};
use ferrots_train::{
    checkpoint::{self, Checkpoint, Provenance},
    evaluation::{self, EvaluationOptions},
    trainer::Trainer,
};
use std::{
    collections::BTreeMap,
    error::Error,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
#[derive(Debug, Clone, ValueEnum)]
pub enum ModelChoice {
    LastValue,
    SeasonalNaive,
    Linear,
    Dlinear,
}
#[derive(Debug, Args)]
pub struct ConfigArgs {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    device: Option<String>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    horizon: Option<usize>,
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long)]
    max_epochs: Option<usize>,
    #[arg(long)]
    learning_rate: Option<f64>,
    #[arg(long, value_enum)]
    model: Option<ModelChoice>,
    #[arg(long)]
    seasonal_period: Option<usize>,
    #[arg(long)]
    output_root: Option<PathBuf>,
    #[arg(long)]
    run_id: Option<String>,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long,num_args=0..=1,default_missing_value="true")]
    cache_bypass: Option<bool>,
}
impl ConfigArgs {
    fn resolve(&self) -> Result<Config> {
        let mut c = Config::load(
            &self.config,
            Overrides {
                device: self.device.clone(),
                seed: self.seed,
                horizon: self.horizon,
                batch_size: self.batch_size,
                max_epochs: self.max_epochs,
                learning_rate: self.learning_rate,
                output_root: self.output_root.clone(),
                run_id: self.run_id.clone(),
                cache_dir: self.cache_dir.clone(),
                input: self.input.clone(),
                ..Overrides::default()
            },
        )?;
        if let Some(choice) = &self.model {
            c.model = match choice {
                ModelChoice::LastValue => ModelSpec::LastValue,
                ModelChoice::SeasonalNaive => ModelSpec::SeasonalNaive {
                    period: self
                        .seasonal_period
                        .ok_or("--seasonal-period is required for the seasonal-naive override")?,
                },
                ModelChoice::Linear => {
                    if matches!(c.model, ModelSpec::Linear { .. }) {
                        c.model
                    } else {
                        ModelSpec::Linear { individual: false }
                    }
                }
                ModelChoice::Dlinear => {
                    if matches!(c.model, ModelSpec::Dlinear { .. }) {
                        c.model
                    } else {
                        ModelSpec::Dlinear {
                            individual: false,
                            moving_avg: 25,
                        }
                    }
                }
            };
        }
        if let Some(period) = self.seasonal_period {
            if let ModelSpec::SeasonalNaive { period: ref mut p } = c.model {
                *p = period;
            } else {
                return Err("--seasonal-period requires seasonal-naive".into());
            }
        }
        if let Some(b) = self.cache_bypass {
            c.data.cache_bypass = b;
        }
        c.validate()?;
        Ok(c)
    }
}
#[derive(Debug, Args)]
pub struct CheckpointArgs {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    device: Option<String>,
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}
fn emit(value: &impl serde::Serialize, path: Option<&Path>) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        write_synced(path, &bytes)?;
    }
    println!("{}", String::from_utf8(bytes)?);
    Ok(())
}
fn provenance() -> Provenance {
    Provenance {
        code_sha: env!("FERROTS_CODE_SHA").into(),
        dirty: env!("FERROTS_DIRTY") == "true",
        source_sha256: env!("FERROTS_SOURCE_SHA256").into(),
        environment: BTreeMap::from([
            ("rustc".into(), env!("FERROTS_RUSTC").into()),
            ("target".into(), env!("FERROTS_TARGET").into()),
            ("features".into(), env!("FERROTS_FEATURES").into()),
            ("candle".into(), "0.11.0".into()),
        ]),
    }
}
pub fn inspect(args: &ConfigArgs, output: Option<&Path>) -> Result<()> {
    let config = args.resolve()?;
    let data = Dataset::load(&config.data, &config.window, None)?;
    let all_same_segment = data.index.entries.iter().all(|w| {
        data.series.spans.iter().any(|s| {
            s.segment == w.segment
                && w.origin - config.window.lookback >= s.start
                && w.origin + config.window.gap + config.window.horizon <= s.end
        })
    });
    let all_target_split = data.index.entries.iter().all(|w| {
        data.series.splits
            [w.origin + config.window.gap..w.origin + config.window.gap + config.window.horizon]
            .iter()
            .all(|&s| s == w.split)
    });
    emit(
        &serde_json::json!({"format_version":1,"raw":data.series.raw,"prepared":data.series.report,"window_counts":data.index.counts,"excluded_missing_history":data.index.excluded_missing_history,"scaler":data.scaler,"cache":data.cache,
        "window_audit":{"checked":data.index.entries.len(),"all_same_segment":all_same_segment,"all_target_split":all_target_split}}),
        output,
    )
}
pub fn train(args: &ConfigArgs) -> Result<()> {
    let config = args.resolve()?;
    config.validate_model_inputs()?;
    let device = ferrots_nn::resolve_device(config.train.device.parse()?)?;
    let data = Dataset::load(&config.data, &config.window, None)?;
    eprintln!(
        "cache_hit={} parse_count={} data_seconds={:.3}",
        data.cache.hit, data.cache.parse_count, data.cache.total_seconds
    );
    let identity = provenance();
    let result = Trainer {
        config: &config,
        data: &data,
        device: &device,
        provenance: &identity,
    }
    .fit()?;
    emit(&result, None)
}
fn open(args: &CheckpointArgs) -> Result<(Checkpoint, candle_core::Device, String, usize)> {
    let directory = checkpoint::resolve(&args.checkpoint)?;
    let config = Config::load(directory.join("resolved-config.toml"), Overrides::default())?;
    let device = ferrots_nn::resolve_device(
        args.device
            .as_deref()
            .unwrap_or(&config.train.device)
            .parse()?,
    )?;
    let loaded = checkpoint::load(&directory, &device)?;
    let run_id = loaded
        .config
        .output
        .run_id
        .clone()
        .or_else(|| {
            loaded
                .directory
                .parent()?
                .parent()?
                .file_name()?
                .to_str()
                .map(str::to_string)
        })
        .unwrap_or_else(|| "run".into());
    let batch_size = args.batch_size.unwrap_or(loaded.config.train.batch_size);
    if batch_size == 0 {
        return Err("batch size must be positive".into());
    }
    Ok((loaded, device, run_id, batch_size))
}
fn dataset(args: &CheckpointArgs, loaded: &Checkpoint) -> Result<Dataset> {
    let mut data = loaded.config.data.clone();
    if let Some(path) = &args.input {
        data.path = path.clone();
    }
    if let Some(path) = &args.cache_dir {
        data.cache_dir = path.clone();
    }
    Ok(Dataset::load(
        &data,
        &loaded.config.window,
        Some(loaded.scaler.clone()),
    )?)
}
pub fn evaluate(args: &CheckpointArgs, output: Option<&Path>) -> Result<()> {
    let (loaded, device, run_id, batch_size) = open(args)?;
    let data = dataset(args, &loaded)?;
    if data.cache.source_sha256 != loaded.manifest.data_sha256 {
        return Err("evaluation source fingerprint differs from the checkpoint".into());
    }
    let result = evaluation::evaluate(
        &loaded,
        &data,
        &device,
        &EvaluationOptions {
            run_id: &run_id,
            split: 2,
            batch_size,
            export: None,
            columns: &[],
        },
    )?;
    emit(&result, output)
}
pub fn predict(
    args: &CheckpointArgs,
    output: &Path,
    future: bool,
    columns: &[String],
) -> Result<()> {
    let (mut loaded, device, run_id, batch_size) = open(args)?;
    if let Some(path) = &args.cache_dir {
        loaded.config.data.cache_dir = path.clone();
    }
    if future {
        emit(
            &evaluation::predict_future(
                &loaded,
                args.input.as_deref(),
                &device,
                &run_id,
                output,
                columns,
            )?,
            None,
        )
    } else {
        let data = dataset(args, &loaded)?;
        let result = evaluation::evaluate(
            &loaded,
            &data,
            &device,
            &EvaluationOptions {
                run_id: &run_id,
                split: 2,
                batch_size,
                export: Some(output),
                columns,
            },
        )?;
        emit(&result, None)
    }
}
