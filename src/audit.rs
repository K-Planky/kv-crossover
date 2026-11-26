use kv_crossover::workload::TraceFile;
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Config {
    mooncake: Workloads,
}
#[derive(Deserialize)]
struct Workloads {
    conversation: TraceFile,
    toolagent: TraceFile,
}

pub fn run(config: &Path) -> Result<(), String> {
    let input = fs::read_to_string(config).map_err(|e| format!("{}: {e}", config.display()))?;
    let files: Config = toml::from_str(&input).map_err(|e| format!("{}: {e}", config.display()))?;
    let mut table = String::from("workload requests min p50 p95 max beyond_8192 share_pct\n");
    for (name, file) in [
        ("conversation", files.mooncake.conversation),
        ("toolagent", files.mooncake.toolagent),
    ] {
        let arrivals = file.load(config.parent().unwrap_or(Path::new(".")))?;
        let mut lengths = Vec::with_capacity(arrivals.len());
        for arrival in arrivals {
            lengths.push(
                arrival
                    .request
                    .input_length
                    .checked_add(arrival.request.output_length)
                    .ok_or_else(|| format!("{name}:{}: total length overflow", arrival.id.0))?,
            );
        }
        lengths.sort_unstable();
        let n = lengths.len();
        let beyond = n - lengths.partition_point(|v| *v <= 8192);
        let percentile = |pct: usize| lengths[(n * pct).div_ceil(100) - 1];
        table.push_str(&format!(
            "{name} {n} {} {} {} {} {beyond} {:.2}\n",
            lengths[0],
            percentile(50),
            percentile(95),
            lengths[n - 1],
            100.0 * beyond as f64 / n as f64
        ));
    }
    print!("{table}");
    Ok(())
}
