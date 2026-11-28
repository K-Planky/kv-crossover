//! Arrival-time observations and placement rules. No trace rows or event heap.
use crate::BLOCK_TOKENS;
use crate::{
    profile::{self, Profile},
    replay::{BlockId, WorkerId},
};
use std::num::NonZeroU64;

#[derive(Debug, PartialEq, Eq)]
pub struct RouterView {
    pub input_tokens: usize,
    pub workers: [WorkerView; 2], // A, B
    pub transfers: Vec<TransferView>,
    pub(crate) link_rate: NonZeroU64,
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
    pub(crate) copy: Option<CopyPlan>,
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CopyPlan {
    pub prefix_blocks: usize,
    pub new_bytes: u64,
    // Each joined physical batch is counted once, even for multiple needed blocks.
    pub joined_remaining: Vec<u64>,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyForecast {
    Nominal,
    ContentionAware,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Wait,
    Recompute,
    Transfer { prefix_blocks: usize },
}
#[derive(Debug, PartialEq, Eq)]
pub struct ActionScore {
    pub action: Action,
    pub placement: LocalScore,
    pub new_copy_bytes: u64,
    pub predicted_blocks_ready_ns: u64,
}
pub fn earliest(scores: &[ActionScore]) -> Option<&ActionScore> {
    scores.iter().min_by_key(|s| {
        (
            s.placement.predicted_ttft_ns,
            s.new_copy_bytes != 0,
            s.placement.worker,
        )
    })
}
impl RouterView {
    fn blocks_ready_ns(&self, plan: &CopyPlan, forecast: CopyForecast) -> Result<u64, String> {
        let required = plan
            .joined_remaining
            .iter()
            .copied()
            .chain([plan.new_bytes]);
        let bytes: u128 = match forecast {
            CopyForecast::Nominal => required.map(u128::from).sum(),
            CopyForecast::ContentionAware => {
                // By the last required finish, every batch has received at most
                // that batch's remaining size in service (equal sharing).
                let last = required.max().unwrap();
                self.transfers
                    .iter()
                    .map(|t| u128::from(last.min(t.remaining_bytes)))
                    .sum::<u128>()
                    + u128::from(plan.new_bytes)
            }
        };
        let ns = bytes
            .checked_mul(1_000_000_000)
            .ok_or("predicted copy duration overflow")?
            .div_ceil(u128::from(self.link_rate.get()));
        u64::try_from(ns).map_err(|_| "predicted copy duration overflow".into())
    }
    pub fn local_scores(&self, p: &Profile) -> Result<Vec<LocalScore>, String> {
        Ok(self
            .action_scores(p, None)?
            .into_iter()
            .map(|s| s.placement)
            .collect())
    }
    pub fn action_scores(
        &self,
        p: &Profile,
        forecast: Option<CopyForecast>,
    ) -> Result<Vec<ActionScore>, String> {
        let mut scores = Vec::new();
        let longest = self.workers.iter().map(|w| w.cached_tokens).max().unwrap();
        for (worker, view) in [WorkerId::A, WorkerId::B].into_iter().zip(&self.workers) {
            if !view.can_admit_local && (forecast.is_none() || view.copy.is_none()) {
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
            let mut score = |action, cached, ready, new_copy_bytes| -> Result<(), String> {
                let service = add_ns(
                    prefill_ns(p, cached, self.input_tokens)?,
                    p.decode_ns(&[self.input_tokens])?,
                )?;
                scores.push(ActionScore {
                    action,
                    new_copy_bytes,
                    predicted_blocks_ready_ns: ready,
                    placement: LocalScore {
                        worker,
                        queued_and_active: view.jobs.len(),
                        predicted_backlog_ns: backlog,
                        predicted_ttft_ns: add_ns(backlog.max(ready), service)?,
                    },
                });
                Ok(())
            };
            if view.can_admit_local {
                let action = if view.cached_tokens > 0 && view.cached_tokens == longest {
                    Action::Wait
                } else {
                    Action::Recompute
                };
                score(action, view.cached_tokens, 0, 0)?;
            }
            if let (Some(f), Some(plan)) = (forecast, &view.copy) {
                score(
                    Action::Transfer {
                        prefix_blocks: plan.prefix_blocks,
                    },
                    plan.prefix_blocks * BLOCK_TOKENS,
                    self.blocks_ready_ns(plan, f)?,
                    plan.new_bytes,
                )?;
            }
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ties_prefer_no_new_batch_then_worker_id() {
        let score = |worker, bytes| ActionScore {
            action: Action::Transfer { prefix_blocks: 1 },
            predicted_blocks_ready_ns: 0,
            new_copy_bytes: bytes,
            placement: LocalScore {
                worker,
                queued_and_active: 0,
                predicted_backlog_ns: 0,
                predicted_ttft_ns: 100,
            },
        };
        use WorkerId::*;
        assert_eq!(
            earliest(&[score(A, 1), score(B, 0)])
                .unwrap()
                .placement
                .worker,
            B
        );
        assert_eq!(
            earliest(&[score(B, 0), score(A, 0)])
                .unwrap()
                .placement
                .worker,
            A
        );
    }
}
