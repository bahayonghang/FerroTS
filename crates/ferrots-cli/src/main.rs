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
    /// Train a baseline; select a frozen checkpoint using validation only.
    Train {
        #[command(flatten)]
        args: ConfigArgs,
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
fn main() -> std::process::ExitCode {
    let (label, result) = match Cli::parse().command {
        Commands::Doctor {
            device,
            training_smoke_test,
        } => ("doctor", doctor::run(device, training_smoke_test)),
        Commands::Data {
            command: DataCommand::Inspect { args, output },
        } => ("data inspect", pipeline::inspect(&args, output.as_deref())),
        Commands::Train { args } => ("train", pipeline::train(&args)),
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
