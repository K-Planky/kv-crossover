//! Qualified worker phase costs; padding affects time only, never logical KV.
use crate::{TraceRequest, workload::VerifiedFile};
use std::{collections::HashMap, path::Path};

pub const CONTEXT: usize = 8192;
pub const CHUNK: usize = 128;
pub const BATCH: usize = 8;

pub fn supported(r: &TraceRequest) -> bool {
    r.input_length > 0
        && r.output_length > 0
        && r.input_length
            .checked_add(r.output_length)
            .is_some_and(|n| n <= CONTEXT)
}

pub struct Profile {
    // Only the two selected prefill shapes are needed, indexed by chunk start.
    prefill: [[u64; 2]; CONTEXT / CHUNK],
    decode: [[u64; CONTEXT / 64]; BATCH],
}

impl Profile {
    pub fn load(file: &VerifiedFile, dir: &Path) -> Result<Self, String> {
        let input = file.read(dir)?;
        Self::parse(&input).map_err(|e| format!("{}: {e}", file.path(dir).display()))
    }

    fn parse(input: &str) -> Result<Self, String> {
        let mut lines = input.lines();
        if lines.next()
            != Some("phase,batch_size,cached_bucket_tokens,new_tokens_per_request,predicted_ns")
        {
            return Err("unexpected profile header".into());
        }
        let mut grid = HashMap::new();
        for (index, line) in lines.enumerate() {
            let mut parse = || {
                let mut fields = line.split(',');
                let key = (
                    fields.next()?,
                    fields.next()?.parse::<usize>().ok()?,
                    fields.next()?.parse::<usize>().ok()?,
                    fields.next()?.parse::<usize>().ok()?,
                );
                let ns = fields.next()?.parse::<u64>().ok()?;
                (fields.next().is_none() && ns > 0 && grid.insert(key, ns).is_none()).then_some(())
            };
            parse().ok_or_else(|| format!("line {}: invalid or duplicate timing", index + 2))?;
        }
        let get = |key| {
            grid.get(&key)
                .copied()
                .ok_or_else(|| format!("missing timing {key:?}"))
        };
        let mut p = Self {
            prefill: [[0; 2]; CONTEXT / CHUNK],
            decode: [[0; CONTEXT / 64]; BATCH],
        };
        for (i, row) in p.prefill.iter_mut().enumerate() {
            *row = [
                get(("prefill", 1, i * CHUNK, 64))?,
                get(("prefill", 1, i * CHUNK, 128))?,
            ];
        }
        for (b, row) in p.decode.iter_mut().enumerate() {
            for (c, ns) in row.iter_mut().enumerate() {
                *ns = get(("decode", b + 1, (c + 1) * 64, 1))?;
            }
        }
        Ok(p)
    }

    pub fn prefill_ns(&self, cached: usize, new: usize) -> Result<u64, String> {
        if !cached.is_multiple_of(CHUNK) || cached >= CONTEXT || !(1..=CHUNK).contains(&new) {
            return Err(format!("unsupported prefill: cached={cached}, new={new}"));
        }
        Ok(self.prefill[cached / CHUNK][usize::from(new > 64)])
    }

    pub fn decode_ns(&self, processed: &[usize]) -> Result<u64, String> {
        if !(1..=BATCH).contains(&processed.len())
            || processed.iter().any(|n| !(1..CONTEXT).contains(n))
        {
            return Err(format!("unsupported decode contexts: {processed:?}"));
        }
        let mean = processed.iter().sum::<usize>() / processed.len();
        Ok(self.decode[processed.len() - 1][mean.div_ceil(64) - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qualified_shapes_and_boundaries() {
        let p = Profile::parse(include_str!("../profiles/a100.csv")).unwrap();
        assert_eq!(p.prefill_ns(6144, 1), Ok(14_177_252));
        assert_eq!(p.decode_ns(&[512, 1537]), Ok(10_884_036));
        assert_eq!(p.decode_ns(&[513, 1537]), Ok(10_898_702));
        assert_eq!(p.prefill_ns(6144, 127), Ok(18_696_410));
        assert_eq!(p.prefill_ns(8064, 128), Ok(19_512_860));
        assert_eq!(p.decode_ns(&[512, 1536]), Ok(10_884_036));
        assert_eq!(p.decode_ns(&[8191; 8]), Ok(16_139_675));
        for (cached, new) in [(64, 128), (8192, 1), (0, 0), (0, 129)] {
            assert!(p.prefill_ns(cached, new).is_err());
        }
        for contexts in [&[][..], &[8192, 64], &[10; 9], &[0]] {
            assert!(p.decode_ns(contexts).is_err());
        }
    }
}
