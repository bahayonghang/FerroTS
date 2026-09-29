//! Strict, backend-independent configuration for one forecasting experiment.
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("configuration IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("configuration TOML: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("invalid configuration: {0}")]
    Invalid(String),
}
pub type Result<T> = std::result::Result<T, ConfigError>;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(ConfigError::Invalid(message.into()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub data: DataSpec,
    pub window: WindowSpec,
    pub model: ModelSpec,
    pub train: TrainSpec,
    pub output: OutputSpec,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            schema_version: 1,
            data: DataSpec::default(),
            window: WindowSpec::default(),
            model: ModelSpec::default(),
            train: TrainSpec::default(),
            output: OutputSpec::default(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DataSpec {
    pub protocol_id: String,
    pub path: PathBuf,
    pub timestamp_column: String,
    pub timestamp_format: String,
    pub timezone: String,
    pub interval_seconds: i64,
    pub input_columns: Vec<String>,
    pub target_columns: Vec<String>,
    pub future_known_columns: Vec<String>,
    pub column_types: BTreeMap<String, NumberType>,
    pub sort: SortPolicy,
    pub duplicates: DuplicatePolicy,
    pub grid: Option<GridSpec>,
    pub running: Option<RunRule>,
    pub missing: MissingSpec,
    pub split: SplitSpec,
    pub time_features: String,
    pub cache_dir: PathBuf,
    pub cache_bypass: bool,
}
impl Default for DataSpec {
    fn default() -> Self {
        Self {
            protocol_id: "custom".into(),
            path: PathBuf::new(),
            timestamp_column: "date".into(),
            timestamp_format: "%Y-%m-%d %H:%M:%S".into(),
            timezone: "naive".into(),
            interval_seconds: 3600,
            input_columns: vec![],
            target_columns: vec![],
            future_known_columns: vec![],
            column_types: BTreeMap::new(),
            sort: SortPolicy::Preserve,
            duplicates: DuplicatePolicy::Error,
            grid: None,
            running: None,
            missing: MissingSpec::default(),
            split: SplitSpec::default(),
            time_features: "h".into(),
            cache_dir: "runs/cache/data".into(),
            cache_bypass: false,
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NumberType {
    Float64,
    Integer,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SortPolicy {
    Preserve,
    Stable,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DuplicatePolicy {
    Error,
    KeepFirst,
    KeepLast,
    Preserve,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridSpec {
    pub min_samples: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRule {
    pub column: String,
    pub op: CompareOp,
    pub threshold: f64,
    pub min_duration_seconds: i64,
    pub max_missing_seconds: i64,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}
impl CompareOp {
    pub fn matches(self, value: f64, threshold: f64) -> bool {
        match self {
            Self::Gt => value > threshold,
            Self::Ge => value >= threshold,
            Self::Lt => value < threshold,
            Self::Le => value <= threshold,
            Self::Eq => value == threshold,
            Self::Ne => value != threshold,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MissingSpec {
    pub mode: MissingMode,
    pub tokens: Vec<String>,
    pub max_stale_seconds: i64,
    pub max_missing_seconds: i64,
    pub complete_windows: bool,
    pub filled_targets_valid: bool,
}
impl Default for MissingSpec {
    fn default() -> Self {
        Self {
            mode: MissingMode::Error,
            tokens: vec![String::new(), "NaN".into(), "nan".into()],
            max_stale_seconds: 600,
            max_missing_seconds: 300,
            complete_windows: false,
            filled_targets_valid: false,
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MissingMode {
    Error,
    Causal,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SplitSpec {
    Ratio {
        train: f64,
        test: f64,
    },
    Rows {
        train_end: usize,
        validation_end: usize,
        test_end: usize,
    },
    Dates {
        train_end: String,
        validation_end: String,
        test_end: String,
    },
    Segments {
        train: f64,
        validation: f64,
    },
}
impl Default for SplitSpec {
    fn default() -> Self {
        Self::Ratio {
            train: 0.7,
            test: 0.2,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WindowSpec {
    pub lookback: usize,
    pub horizon: usize,
    pub gap: usize,
    pub stride: usize,
}
impl Default for WindowSpec {
    fn default() -> Self {
        Self {
            lookback: 96,
            horizon: 96,
            gap: 0,
            stride: 1,
        }
    }
}
fn default_kernel() -> usize {
    25
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelSpec {
    LastValue,
    SeasonalNaive {
        period: usize,
    },
    Linear {
        #[serde(default)]
        individual: bool,
    },
    Dlinear {
        #[serde(default)]
        individual: bool,
        #[serde(default = "default_kernel")]
        moving_avg: usize,
    },
}
impl Default for ModelSpec {
    fn default() -> Self {
        Self::Dlinear {
            individual: false,
            moving_avg: 25,
        }
    }
}
impl ModelSpec {
    pub fn name(&self) -> &'static str {
        match self {
            Self::LastValue => "last_value",
            Self::SeasonalNaive { .. } => "seasonal_naive",
            Self::Linear { .. } => "linear",
            Self::Dlinear { .. } => "dlinear",
        }
    }
    pub fn trainable(&self) -> bool {
        matches!(self, Self::Linear { .. } | Self::Dlinear { .. })
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Schedule {
    Constant,
    Type1,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainSpec {
    pub seed: u64,
    pub device: String,
    pub dtype: String,
    pub batch_size: usize,
    pub accumulation_steps: usize,
    pub max_epochs: usize,
    pub patience: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub lr_schedule: Schedule,
}
impl Default for TrainSpec {
    fn default() -> Self {
        Self {
            seed: 2021,
            device: "cpu".into(),
            dtype: "f32".into(),
            batch_size: 32,
            accumulation_steps: 1,
            max_epochs: 10,
            patience: 3,
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            lr_schedule: Schedule::Type1,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputSpec {
    pub root: PathBuf,
    pub run_id: Option<String>,
    pub mape: bool,
}
impl Default for OutputSpec {
    fn default() -> Self {
        Self {
            root: "runs".into(),
            run_id: None,
            mape: false,
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub device: Option<String>,
    pub seed: Option<u64>,
    pub horizon: Option<usize>,
    pub batch_size: Option<usize>,
    pub max_epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub model: Option<ModelSpec>,
    pub output_root: Option<PathBuf>,
    pub run_id: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub input: Option<PathBuf>,
}
impl Config {
    pub fn from_toml(source: &str, overrides: Overrides) -> Result<Self> {
        let mut config: Self = toml::from_str(source)?;
        config.apply(overrides);
        config.validate()?;
        Ok(config)
    }
    pub fn load(path: impl AsRef<Path>, overrides: Overrides) -> Result<Self> {
        Self::from_toml(&std::fs::read_to_string(path)?, overrides)
    }
    pub fn apply(&mut self, o: Overrides) {
        if let Some(v) = o.device {
            self.train.device = v;
        }
        if let Some(v) = o.seed {
            self.train.seed = v;
        }
        if let Some(v) = o.horizon {
            self.window.horizon = v;
        }
        if let Some(v) = o.batch_size {
            self.train.batch_size = v;
        }
        if let Some(v) = o.max_epochs {
            self.train.max_epochs = v;
        }
        if let Some(v) = o.learning_rate {
            self.train.learning_rate = v;
        }
        if let Some(v) = o.model {
            self.model = v;
        }
        if let Some(v) = o.output_root {
            self.output.root = v;
        }
        if let Some(v) = o.run_id {
            self.output.run_id = Some(v);
        }
        if let Some(v) = o.cache_dir {
            self.data.cache_dir = v;
        }
        if let Some(v) = o.input {
            self.data.path = v;
        }
    }
    pub fn validate(&self) -> Result<()> {
        require(self.schema_version == 1, "unsupported schema_version")?;
        let d = &self.data;
        require(!d.path.as_os_str().is_empty(), "data.path is required")?;
        require(!d.protocol_id.is_empty(), "protocol_id is empty")?;
        require(
            !d.timestamp_column.is_empty() && !d.timestamp_format.is_empty(),
            "timestamp column and format are required",
        )?;
        require(d.timezone == "naive", "only naive timestamps are supported")?;
        require(d.interval_seconds > 0, "interval_seconds must be positive")?;
        require(
            matches!(d.time_features.as_str(), "h" | "t"),
            "time_features must be h or t",
        )?;
        let unique = |cols: &[String]| {
            !cols.is_empty()
                && cols.iter().all(|s| !s.is_empty())
                && cols.iter().collect::<BTreeSet<_>>().len() == cols.len()
        };
        require(
            unique(&d.input_columns),
            "input_columns must be nonempty and unique",
        )?;
        require(
            unique(&d.target_columns),
            "target_columns must be nonempty and unique",
        )?;
        require(
            d.future_known_columns.is_empty() || unique(&d.future_known_columns),
            "future_known_columns must be unique",
        )?;
        require(
            !d.input_columns.contains(&d.timestamp_column),
            "timestamp column cannot be a numeric input",
        )?;
        require(
            d.target_columns.iter().all(|c| d.input_columns.contains(c)),
            "targets must be named input columns",
        )?;
        require(
            d.column_types
                .keys()
                .all(|c| d.input_columns.contains(c) || d.future_known_columns.contains(c)),
            "column_types contains an undeclared column",
        )?;
        require(
            d.missing.max_stale_seconds >= 0 && d.missing.max_missing_seconds >= 0,
            "missing durations must be nonnegative",
        )?;
        require(
            !d.missing.filled_targets_valid
                || (d.protocol_id == "cement-twin-v1"
                    && d.missing.complete_windows
                    && d.missing.mode == MissingMode::Causal),
            "filled targets require the complete-window cement-twin-v1 protocol",
        )?;
        if let Some(grid) = &d.grid {
            require(grid.min_samples > 0, "min_samples must be positive")?;
        }
        if let Some(rule) = &d.running {
            require(
                d.input_columns.contains(&rule.column),
                "running column must be an input",
            )?;
            require(
                rule.threshold.is_finite(),
                "running threshold must be finite",
            )?;
            require(
                rule.min_duration_seconds > 0 && rule.max_missing_seconds >= 0,
                "invalid running durations",
            )?;
        }
        match &d.split {
            SplitSpec::Ratio { train, test } => validate_ratios(*train, *test)?,
            SplitSpec::Segments { train, validation } => {
                validate_ratios(*train, *validation)?;
                require(d.running.is_some(), "segment split requires a running rule")?;
            }
            SplitSpec::Rows {
                train_end,
                validation_end,
                test_end,
            } => require(
                *train_end > 0 && train_end < validation_end && validation_end < test_end,
                "row split boundaries must increase",
            )?,
            SplitSpec::Dates {
                train_end,
                validation_end,
                test_end,
            } => require(
                !train_end.is_empty() && !validation_end.is_empty() && !test_end.is_empty(),
                "date split boundaries are required",
            )?,
        }
        require(
            self.window.lookback > 0 && self.window.horizon > 0 && self.window.stride > 0,
            "window lengths and stride must be positive",
        )?;
        require(
            self.window
                .lookback
                .checked_add(self.window.gap)
                .and_then(|n| n.checked_add(self.window.horizon))
                .is_some(),
            "window length overflow",
        )?;
        match self.model {
            ModelSpec::Dlinear { moving_avg, .. } => require(
                moving_avg > 0 && moving_avg % 2 == 1,
                "moving_avg must be a positive odd number",
            )?,
            ModelSpec::SeasonalNaive { period } => require(
                period > 0 && period <= self.window.lookback,
                "seasonal period must be in 1..=lookback",
            )?,
            _ => {}
        }
        let t = &self.train;
        t.device
            .parse::<crate::Device>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        t.dtype
            .parse::<crate::DTypePolicy>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        require(
            t.batch_size > 0 && t.accumulation_steps > 0 && t.max_epochs > 0 && t.patience > 0,
            "training counts must be positive",
        )?;
        require(
            t.learning_rate.is_finite()
                && t.learning_rate > 0.0
                && t.eps.is_finite()
                && t.eps > 0.0,
            "learning rate and eps must be finite and positive",
        )?;
        require(
            (0.0..1.0).contains(&t.beta1)
                && (0.0..1.0).contains(&t.beta2)
                && t.weight_decay.is_finite()
                && t.weight_decay >= 0.0,
            "invalid Adam parameters",
        )?;
        require(!self.output.mape, "MAPE is not supported in P1")?;
        require(
            !self.output.root.as_os_str().is_empty() && !d.cache_dir.as_os_str().is_empty(),
            "output and cache paths must be nonempty",
        )?;
        if let Some(id) = &self.output.run_id {
            require(
                !id.is_empty()
                    && id != "."
                    && id != ".."
                    && id
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
                "run_id must be a simple ASCII identifier",
            )?;
        }
        Ok(())
    }
    pub fn validate_model_inputs(&self) -> Result<()> {
        self.validate()?;
        require(
            self.data.future_known_columns.is_empty(),
            "P1 baselines do not support future-known inputs",
        )
    }
}
fn validate_ratios(a: f64, b: f64) -> Result<()> {
    require(
        a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 && a + b < 1.0,
        "split ratios must be positive and sum to less than one",
    )
}
