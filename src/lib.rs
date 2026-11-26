pub mod kv;
mod link;
pub mod profile;
pub mod replay;
pub mod router;
pub mod workload;

use serde::Deserialize;

pub const BLOCK_TOKENS: usize = 512;

#[derive(Deserialize)]
pub struct TraceRequest {
    #[serde(rename = "timestamp")]
    pub arrival_ms: u64,
    pub input_length: usize,
    pub output_length: usize,
    hash_ids: Vec<u64>,
}

impl TraceRequest {
    pub fn parse_line(line: &str) -> Result<Self, String> {
        let request: Self = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let expected = request.input_length.div_ceil(BLOCK_TOKENS);
        if request.hash_ids.len() != expected {
            return Err(format!(
                "expected {expected} block hashes, found {}",
                request.hash_ids.len()
            ));
        }
        Ok(request)
    }

    pub fn full_blocks(&self) -> impl Iterator<Item = &u64> {
        self.hash_ids.iter().take(self.input_length / BLOCK_TOKENS)
    }
}

/// Treat all complete prompt blocks of `resident` as available.
pub fn reusable_tokens(resident: &TraceRequest, arriving: &TraceRequest) -> usize {
    resident
        .full_blocks()
        .zip(arriving.full_blocks())
        .take_while(|(a, b)| a == b)
        .count()
        * BLOCK_TOKENS
}
