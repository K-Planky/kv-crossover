//! Arrival-time observations and local-only placement rules. No trace rows or event heap.
use crate::{
    profile::{self, Profile},
    replay::{BlockId, WorkerId},
};

#[derive(Debug, PartialEq, Eq)]
pub struct RouterView {
    pub input_tokens: usize,
    pub workers: [WorkerView; 2], // A, B
    pub transfers: Vec<TransferView>,
    pub(crate) mean_output_tokens: usize,
}
#[derive(Debug, PartialEq, Eq)]
pub struct TransferView {
    pub destination: WorkerId,
    pub blocks: Vec<BlockId>,
    pub remaining_bytes: u64,
}
#[derive(Debug, PartialEq, Eq)]
pub struct WorkerView {
    pub cached_tokens: usize,
    pub can_admit_local: bool,
    pub(crate) jobs: Vec<JobView>,
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct JobView {
    pub input: usize,
    pub processed: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct LocalScore {
    pub worker: WorkerId,
    pub queued_and_active: usize,
    pub predicted_backlog_ns: u64,
    pub predicted_ttft_ns: u64,
}
#[derive(Clone, Copy)]
pub enum LocalPolicy {
    ShortestQueue,
    CacheAware,
}
impl LocalPolicy {
    pub fn choose(self, scores: &[LocalScore]) -> Option<WorkerId> {
        match self {
            Self::ShortestQueue => scores
                .iter()
                .min_by_key(|s| (s.queued_and_active, s.worker)),
            Self::CacheAware => scores
                .iter()
                .min_by_key(|s| (s.predicted_ttft_ns, s.worker)),
        }
        .map(|s| s.worker)
    }
}
fn add_ns(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| "predicted duration overflow".into())
}
fn prefill_ns(p: &Profile, mut cached: usize, input: usize) -> Result<u64, String> {
    let mut ns = 0;
    while cached < input {
        let new = (input - cached).min(profile::CHUNK);
        ns = add_ns(ns, p.prefill_ns(cached, new)?)?;
        cached += new;
    }
    Ok(ns)
}
impl RouterView {
    pub fn local_scores(&self, p: &Profile) -> Result<Vec<LocalScore>, String> {
        let mut scores = Vec::new();
        for (worker, view) in [WorkerId::A, WorkerId::B].into_iter().zip(&self.workers) {
            if !view.can_admit_local {
                continue;
            }
            let mut backlog = 0;
            for j in &view.jobs {
                let emitted = j.processed.saturating_sub(j.input);
                let context = j.processed.max(j.input);
                let remaining = self
                    .mean_output_tokens
                    .saturating_sub(emitted)
                    .max(1)
                    .min(profile::CONTEXT - context);
                let decode = p
                    .decode_ns(&[context])?
                    .checked_mul(remaining as u64)
                    .ok_or("predicted decode duration overflow")?;
                backlog = add_ns(
                    add_ns(backlog, prefill_ns(p, j.processed, j.input)?)?,
                    decode,
                )?;
            }
            let service = add_ns(
                prefill_ns(p, view.cached_tokens, self.input_tokens)?,
                p.decode_ns(&[self.input_tokens])?,
            )?;
            scores.push(LocalScore {
                worker,
                queued_and_active: view.jobs.len(),
                predicted_backlog_ns: backlog,
                predicted_ttft_ns: add_ns(backlog, service)?,
            });
        }
        Ok(scores)
    }
}
