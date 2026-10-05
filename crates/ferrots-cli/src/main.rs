mod doctor;
mod pipeline;
use clap::{Parser, Subcommand};
use pipeline::{CheckpointArgs, ConfigArgs};
use std::path::PathBuf;
#[derive(Parser)]
#[command(name = "ferrots", version, about = "FerroTS training tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    /// List model capabilities compiled into this binary.
    Models {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Report build/device information and optionally test training.
    Doctor {
        #[arg(long, default_value = "cpu")]
        device: ferrots_core::Device,
        #[arg(long)]
        training_smoke_test: bool,
    },
    /// Inspect local time-series data.
    Data {
        #[command(subcommand)]
        command: DataCommand,
    },
    /// Train a forecasting model; select a checkpoint using validation only.
    Train {
        #[command(flatten)]
        args: ConfigArgs,
        #[arg(long)]
        stop_after_steps: Option<u64>,
    },
    /// Resume a complete training checkpoint into a new continuation run.
    Resume {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        stop_after_steps: Option<u64>,
    },
    /// Evaluate a checkpoint on its fingerprinted source data.
    Evaluate {
        #[command(flatten)]
        args: CheckpointArgs,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Stream backtest or future-only predictions to CSV or CSV.gz.
    Predict {
        #[command(flatten)]
        args: CheckpointArgs,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        future: bool,
        #[arg(long, value_delimiter = ',')]
        columns: Vec<String>,
    },
}
#[derive(Subcommand)]
enum DataCommand {
    Inspect {
        #[command(flatten)]
        args: ConfigArgs,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}
#[derive(Subcommand)]
enum ModelCommand {
    List {
        #[arg(long, default_value = "forecast")]
        task: String,
        #[arg(long, default_value = "cuda:0")]
        device: ferrots_core::Device,
    },
}
fn main() -> std::process::ExitCode {
    let (label, result) = match Cli::parse().command {
        Commands::Models {
            command: ModelCommand::List { task, device },
        } => ("models list", pipeline::models(&task, device)),
        Commands::Doctor {
            device,
            training_smoke_test,
        } => ("doctor", doctor::run(device, training_smoke_test)),
        Commands::Data {
            command: DataCommand::Inspect { args, output },
        } => ("data inspect", pipeline::inspect(&args, output.as_deref())),
        Commands::Train {
            args,
            stop_after_steps,
        } => ("train", pipeline::train(&args, stop_after_steps)),
        Commands::Resume {
            checkpoint,
            stop_after_steps,
        } => ("resume", pipeline::resume(&checkpoint, stop_after_steps)),
        Commands::Evaluate { args, output } => {
            ("evaluate", pipeline::evaluate(&args, output.as_deref()))
        }
        Commands::Predict {
            args,
            output,
            future,
            columns,
        } => (
            "predict",
            pipeline::predict(&args, &output, future, &columns),
        ),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{label} failed: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
