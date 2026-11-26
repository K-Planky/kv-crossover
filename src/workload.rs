use crate::TraceRequest;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// One-based source line within a single workload, not an index into a filtered run.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId(pub usize);

pub struct Arrival {
    pub id: RequestId,
    pub request: TraceRequest,
}

#[derive(Deserialize)]
pub struct VerifiedFile {
    path: PathBuf,
    sha256: String,
}

impl VerifiedFile {
    pub fn path(&self, config_dir: &Path) -> PathBuf {
        config_dir.join(&self.path)
    }

    pub fn read(&self, config_dir: &Path) -> Result<String, String> {
        let path = self.path(config_dir);
        let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let actual: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if actual != self.sha256 {
            return Err(format!(
                "{}: expected SHA-256 {}, got {actual}",
                path.display(),
                self.sha256
            ));
        }
        String::from_utf8(bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load_arrivals(&self, dir: &Path) -> Result<Vec<Arrival>, String> {
        parse_arrivals(&self.read(dir)?, &self.path(dir))
    }
}

fn parse_arrivals(input: &str, path: &Path) -> Result<Vec<Arrival>, String> {
    let mut arrivals = Vec::new();
    let mut previous = 0;
    for (index, line) in input.lines().enumerate() {
        let id = RequestId(index + 1);
        let fail = |e: String| format!("{}:{}: {e}", path.display(), id.0);
        let request = TraceRequest::parse_line(line).map_err(fail)?;
        if request.arrival_ms < previous {
            return Err(fail(format!(
                "timestamp {} precedes {previous} ms",
                request.arrival_ms
            )));
        }
        previous = request.arrival_ms;
        arrivals.push(Arrival { id, request });
    }
    if arrivals.is_empty() {
        return Err(format!("{}: empty workload", path.display()));
    }
    Ok(arrivals)
}

#[cfg(test)]
mod tests {
    use super::*;
    const LINE: &str = r#"{"timestamp":7,"input_length":1,"output_length":2,"hash_ids":[0]}"#;
    #[test]
    fn ties_keep_source_ids_and_times() {
        let input = format!(
            "{LINE}\n{}",
            LINE.replace("\"output_length\":2", "\"output_length\":5")
        );
        let rows = parse_arrivals(&input, Path::new("trace")).unwrap();
        assert_eq!(
            rows.iter()
                .map(|r| (r.id.0, r.request.arrival_ms, r.request.output_length))
                .collect::<Vec<_>>(),
            [(1, 7, 2), (2, 7, 5)]
        );
    }

    #[test]
    fn bad_rows_are_not_skipped_or_reordered() {
        for bad in ["{}".to_string(), LINE.replace(":7", ":6")] {
            let result = parse_arrivals(&format!("{LINE}\n{bad}"), Path::new("trace"));
            assert!(result.err().unwrap().starts_with("trace:2:"));
        }
    }
}
