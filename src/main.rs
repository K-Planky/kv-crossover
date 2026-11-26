mod audit;

use std::{fs, path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};
use kv_crossover::{TraceRequest, reusable_tokens};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Match an arriving prompt against a resident prompt (two JSONL lines, resident first).
    Prefix { path: PathBuf },
    /// Verify both pinned workloads and summarize input plus output lengths.
    Audit {
        #[arg(long)]
        config: PathBuf,
    },
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Audit { config } => audit::run(&config),
        Command::Prefix { path } => prefix(path),
    }
}

fn prefix(path: PathBuf) -> Result<(), String> {
    let input = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = input.lines();
    let mut record = |number| {
        let result = lines
            .next()
            .ok_or_else(|| "missing request".to_owned())
            .and_then(TraceRequest::parse_line);
        result.map_err(|e| format!("{}:{number}: {e}", path.display()))
    };
    let resident = record(1)?;
    let arriving = record(2)?;
    if lines.next().is_some() {
        return Err(format!(
            "{}:3: expected exactly two requests",
            path.display()
        ));
    }
    let reused = reusable_tokens(&resident, &arriving);
    println!(
        "reusable_tokens={reused} local_tokens={}",
        arriving.input_length - reused
    );
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
