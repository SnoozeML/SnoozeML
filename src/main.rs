//! Application binary entry point for the SnoozeML inference daemon.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use clap::Parser;
use snooze_ml::cli::{run_cli, Cli};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    if let Err(err) = run_cli(cli).await {
        tracing::error!(error = %err, "SnoozeML daemon terminated with error");
        std::process::exit(1);
    }
}
