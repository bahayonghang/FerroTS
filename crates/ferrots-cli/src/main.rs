mod doctor;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ferrots", version, about = "FerroTS training tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    /// Report build and device information and optionally test training.
    Doctor {
        #[arg(long, default_value = "cpu")]
        device: ferrots_core::Device,
        #[arg(long)]
        training_smoke_test: bool,
    },
}
fn main() -> std::process::ExitCode {
    let result = match Cli::parse().command {
        Commands::Doctor {
            device,
            training_smoke_test,
        } => doctor::run(device, training_smoke_test),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("doctor failed: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
