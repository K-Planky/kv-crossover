use kv_crossover::{
    BLOCK_TOKENS, TraceRequest,
    profile::{Profile, supported},
    workload::VerifiedFile,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Config {
    mooncake: Workloads,
    vidur_a100: VerifiedFile,
}
#[derive(Deserialize)]
struct Workloads {
    conversation: VerifiedFile,
    toolagent: VerifiedFile,
}

// A node ID denotes the entire path from the root, not just its last hash.
#[derive(Default)]
struct History(HashMap<(usize, u64), usize>);
impl History {
    fn observe(&mut self, request: &TraceRequest) -> usize {
        let mut node = 0;
        let mut matched = 0;
        for hash in request.full_blocks() {
            let next = self.0.len() + 1;
            node = *self.0.entry((node, *hash)).or_insert(next);
            if node != next {
                matched += BLOCK_TOKENS;
            }
        }
        matched
    }
}

pub fn run(config: &Path) -> Result<(), String> {
    let input = fs::read_to_string(config).map_err(|e| format!("{}: {e}", config.display()))?;
    let files: Config = toml::from_str(&input).map_err(|e| format!("{}: {e}", config.display()))?;
    let dir = config.parent().unwrap_or(Path::new("."));
    let _profile = Profile::load(&files.vidur_a100, dir)?;
    let mut qualification = String::from(
        "workload retained_requests all_requests retained_prefix_tokens all_prefix_tokens\n",
    );
    let mut table = String::from("workload requests min p50 p95 max beyond_8192 share_pct\n");
    for (name, file) in [
        ("conversation", files.mooncake.conversation),
        ("toolagent", files.mooncake.toolagent),
    ] {
        let arrivals = file.load_arrivals(dir)?;
        let (mut full, mut subset) = (History::default(), History::default());
        let (mut kept, mut kept_tokens, mut all_tokens) = (0, 0, 0);
        for a in &arrivals {
            all_tokens += full.observe(&a.request);
            if supported(&a.request) {
                kept += 1;
                kept_tokens += subset.observe(&a.request);
            }
        }
        qualification.push_str(&format!(
            "{name} {kept} {} {kept_tokens} {all_tokens}\n",
            arrivals.len()
        ));
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
    print!("{table}{qualification}");
    Ok(())
}
