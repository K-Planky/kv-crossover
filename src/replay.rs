//! Single-threaded event clock for both workers and the shared link.
use crate::{
    BLOCK_TOKENS, TraceRequest,
    kv::BYTES_PER_TOKEN,
    kv::{Cache, Lease},
    link::Link,
    profile::{self, Profile},
    workload::{Arrival, RequestId},
};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque},
    num::NonZeroU64,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum WorkerId {
    A,
    B,
}
impl WorkerId {
    fn other(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}
pub use crate::kv::BlockId;
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Phase {
    Prefill { id: RequestId, new_tokens: usize },
    Decode(Vec<RequestId>),
}
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Event {
    WorkDone(WorkerId, Phase),
    BlockReceived(WorkerId, BlockId),
    Arrival(RequestId),
    DecodeCapacityFailure(WorkerId, RequestId),
}
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Tick {
    Publish(Event),
    Dispatch(WorkerId),
}

struct Job {
    id: RequestId,
    processed: usize,
    input: usize,
    outputs_left: usize,
    lease: Lease,
}
struct Worker {
    prompts: VecDeque<Job>,
    responses: VecDeque<Job>,
    cache: Cache,
    busy_or_scheduled: bool,
    prefill_turn: bool,
}
impl Worker {
    fn new(kv_bytes: u64) -> Self {
        Self {
            prompts: VecDeque::new(),
            responses: VecDeque::new(),
            cache: Cache::new(kv_bytes),
            busy_or_scheduled: false,
            prefill_turn: false,
        }
    }
    fn next_phase(&self, p: &Profile) -> Result<Option<(Phase, u64)>, String> {
        if (self.prefill_turn || self.responses.is_empty())
            && let Some(j) = self.prompts.front()
        {
            let new_tokens = (j.input - j.processed).min(profile::CHUNK);
            return Ok(Some((
                Phase::Prefill {
                    id: j.id,
                    new_tokens,
                },
                p.prefill_ns(j.processed, new_tokens)?,
            )));
        }
        let batch = self.responses.iter().take(profile::BATCH);
        if batch.len() == 0 {
            return Ok(None);
        }
        let contexts: Vec<_> = batch.clone().map(|j| j.processed).collect();
        Ok(Some((
            Phase::Decode(batch.map(|j| j.id).collect()),
            p.decode_ns(&contexts)?,
        )))
    }
    fn finish(&mut self, phase: &Phase, now: u64) -> Vec<RequestId> {
        let mut failed = Vec::new();
        match phase {
            Phase::Prefill { id, new_tokens } => {
                let j = self.prompts.front_mut().unwrap();
                debug_assert_eq!(j.id, *id);
                j.processed += new_tokens;
                self.cache.computed(&mut j.lease, j.processed, now);
                if j.processed == j.input {
                    self.responses.push_back(self.prompts.pop_front().unwrap());
                }
                self.prefill_turn = false;
            }
            Phase::Decode(ids) => {
                let mut continuing = Vec::new();
                for id in ids {
                    let mut j = self.responses.pop_front().unwrap();
                    debug_assert_eq!(j.id, *id);
                    j.processed += 1;
                    j.outputs_left -= 1;
                    if j.outputs_left == 0 {
                        self.cache.release(j.lease, now);
                    } else {
                        continuing.push(j);
                    }
                }
                // All departures from this batch free their KV before survivors grow.
                for mut j in continuing {
                    if self.cache.grow(&mut j.lease) {
                        self.responses.push_back(j);
                    } else {
                        failed.push(j.id);
                        self.cache.release(j.lease, now);
                    }
                }
                self.prefill_turn = true;
            }
        }
        failed
    }
}
pub struct Replay<'p> {
    now_ns: u64,
    events: BinaryHeap<Reverse<(u64, u8, u64, Tick)>>,
    sequence: u64,
    requests: HashMap<RequestId, TraceRequest>,
    awaiting_placement: HashSet<RequestId>,
    workers: [Worker; 2],
    profile: &'p Profile,
    link: Link<(WorkerId, Vec<BlockId>)>,
    // One waiter list per physical block; one job may depend on several batches.
    waiters: BTreeMap<(WorkerId, BlockId), Vec<RequestId>>,
    pending: HashMap<RequestId, (Job, usize)>,
}
impl<'p> Replay<'p> {
    pub fn new(
        arrivals: Vec<Arrival>,
        link_bytes_per_second: NonZeroU64,
        profile: &'p Profile,
        kv_bytes_each: u64,
    ) -> Result<Self, String> {
        let mut replay = Self {
            now_ns: 0,
            events: BinaryHeap::new(),
            sequence: 0,
            requests: HashMap::new(),
            awaiting_placement: HashSet::new(),
            workers: std::array::from_fn(|_| Worker::new(kv_bytes_each)),
            profile,
            link: Link::new(link_bytes_per_second),
            waiters: BTreeMap::new(),
            pending: HashMap::new(),
        };
        let mut previous_id = None;
        for Arrival { id, request } in arrivals {
            if id.0 == 0 || previous_id.is_some_and(|last| last >= id) {
                return Err(format!(
                    "request {}: expected increasing source line IDs",
                    id.0
                ));
            }
            previous_id = Some(id);
            let at = request
                .arrival_ms
                .checked_mul(1_000_000)
                .ok_or_else(|| format!("request {}: timestamp exceeds nanosecond range", id.0))?;
            replay.schedule(at, Tick::Publish(Event::Arrival(id)))?;
            replay.requests.insert(id, request);
        }
        Ok(replay)
    }
    fn schedule(&mut self, at: u64, tick: Tick) -> Result<(), String> {
        let priority = match tick {
            Tick::Dispatch(_) => 2,
            Tick::Publish(Event::Arrival(_)) => 1,
            _ => 0,
        };
        let seq = self.sequence;
        self.sequence = seq.checked_add(1).ok_or("event sequence overflow")?;
        self.events.push(Reverse((at, priority, seq, tick)));
        Ok(())
    }
    pub fn now_ns(&self) -> u64 {
        self.now_ns
    }
    pub fn link_bytes_per_second(&self) -> NonZeroU64 {
        self.link.rate
    }
    /// Execution-side trace data, not a router view.
    pub fn request(&self, id: RequestId) -> Option<&TraceRequest> {
        self.requests.get(&id)
    }
    pub fn is_resident(&self, worker: WorkerId, block: &BlockId) -> bool {
        self.workers[worker as usize].cache.contains(block)
    }
    pub fn used_kv_tokens(&self, worker: WorkerId) -> u64 {
        self.workers[worker as usize].cache.used()
    }
    /// Match and pin the actual local prefix, reserving the remaining prompt and first decode slot.
    pub fn enqueue(&mut self, worker: WorkerId, id: RequestId) -> Result<(), String> {
        if !self.awaiting_placement.contains(&id) {
            return Err(format!("request {}: not awaiting placement", id.0));
        }
        let r = &self.requests[&id];
        if !profile::supported(r) {
            return Err(format!("request {}: unsupported lengths", id.0));
        }
        let (cached, lease) = self.workers[worker as usize]
            .cache
            .admit(r, self.now_ns)
            .ok_or_else(|| format!("request {}: insufficient KV on {worker:?}", id.0))?;
        let job = Job {
            lease,
            id,
            processed: cached,
            input: r.input_length,
            outputs_left: r.output_length,
        };
        self.awaiting_placement.remove(&id);
        self.ready(worker, job)
    }
    fn ready(&mut self, worker: WorkerId, job: Job) -> Result<(), String> {
        self.wake(worker)?;
        let cached = job.processed;
        let w = &mut self.workers[worker as usize];
        if cached == job.input {
            w.responses.push_back(job)
        } else {
            w.prompts.push_back(job)
        }
        Ok(())
    }
    fn wake(&mut self, worker: WorkerId) -> Result<(), String> {
        if !self.workers[worker as usize].busy_or_scheduled {
            self.schedule(self.now_ns, Tick::Dispatch(worker))?;
            self.workers[worker as usize].busy_or_scheduled = true;
        }
        Ok(())
    }
    pub fn delivered_bytes(&self) -> u128 {
        self.link.delivered_bytes
    }

    /// Place a request using the first `prefix_blocks` complete blocks on `dst`.
    /// Join existing deliveries; batch only newly missing blocks into one copy.
    pub fn transfer(
        &mut self,
        dst: WorkerId,
        id: RequestId,
        prefix_blocks: usize,
    ) -> Result<(), String> {
        let src = dst.other();
        if !self.awaiting_placement.contains(&id) {
            return Err(format!("request {}: not awaiting placement", id.0));
        }
        let r = &self.requests[&id];
        let blocks = BlockId::prompt(r);
        if !profile::supported(r) || prefix_blocks == 0 || prefix_blocks > blocks.len() {
            return Err(format!(
                "request {}: unsupported transfer prefix/lengths",
                id.0
            ));
        }
        let mut missing = Vec::new();
        let mut new = Vec::new();
        for block in &blocks[..prefix_blocks] {
            if self.is_resident(dst, block) {
                continue;
            }
            missing.push(block.clone());
            if self.waiters.contains_key(&(dst, block.clone())) {
                continue;
            }
            if !self.is_resident(src, block) || self.workers[dst as usize].cache.is_incoming(block)
            {
                return Err(format!(
                    "request {}: block {block:?} unavailable for transfer to {dst:?}",
                    id.0
                ));
            }
            new.push(block.clone());
        }
        let (processed, lease) = self.workers[dst as usize]
            .cache
            .admit_prefix(r, blocks, prefix_blocks, &new, self.now_ns)
            .ok_or_else(|| format!("request {}: insufficient KV on {dst:?}", id.0))?;
        let job = Job {
            id,
            processed,
            lease,
            input: r.input_length,
            outputs_left: r.output_length,
        };
        self.awaiting_placement.remove(&id);
        if missing.is_empty() {
            return self.ready(dst, job);
        }
        self.pending.insert(id, (job, missing.len()));
        for block in missing {
            self.waiters.entry((dst, block)).or_default().push(id);
        }
        if !new.is_empty() {
            for block in &new {
                self.workers[src as usize].cache.pin(block, self.now_ns);
            }
            let bytes = new.len() as u64 * BLOCK_TOKENS as u64 * BYTES_PER_TOKEN;
            self.link.start(self.now_ns, bytes, (dst, new));
        }
        Ok(())
    }
    /// Synthetic cache-warming delivery for controlled initial states, not a physical copy.
    pub fn receive_after(
        &mut self,
        worker: WorkerId,
        block: BlockId,
        delay_ns: NonZeroU64,
    ) -> Result<(), String> {
        let at = self
            .now_ns
            .checked_add(delay_ns.get())
            .ok_or_else(|| format!("block {block:?}: delivery time overflow"))?;
        self.workers[worker as usize].cache.reserve(&block)?;
        self.schedule(at, Tick::Publish(Event::BlockReceived(worker, block)))
    }
    /// Apply completions before arrivals, and dispatch after all arrivals at this tick.
    /// A clock/scheduling error is fatal to this run.
    pub fn step(&mut self) -> Result<Option<Event>, String> {
        loop {
            // Compare the next physical finish directly with the heap. A new
            // copy changes this deadline without leaving obsolete heap ticks.
            if let Some(end) = self.link.next_finish()?
                && self.events.peek().is_none_or(|e| end <= e.0.0)
            {
                for (dst, blocks) in self.link.finish(end) {
                    self.link.delivered_bytes +=
                        blocks.len() as u128 * BLOCK_TOKENS as u128 * u128::from(BYTES_PER_TOKEN);
                    for block in blocks {
                        self.schedule(end, Tick::Publish(Event::BlockReceived(dst, block)))?;
                    }
                }
            }
            let Some(Reverse((at, _, _, tick))) = self.events.pop() else {
                return Ok(None);
            };
            self.now_ns = at;
            match tick {
                Tick::Dispatch(worker) => {
                    let w = &mut self.workers[worker as usize];
                    if let Some((phase, ns)) = w.next_phase(self.profile)? {
                        let end = at.checked_add(ns).ok_or_else(|| {
                            format!("worker {worker:?}: {phase:?} finish time overflow")
                        })?;
                        self.schedule(end, Tick::Publish(Event::WorkDone(worker, phase)))?;
                    } else {
                        w.busy_or_scheduled = false;
                        w.prefill_turn = false;
                    }
                }
                Tick::Publish(event) => {
                    match &event {
                        Event::WorkDone(worker, phase) => {
                            let w = &mut self.workers[*worker as usize];
                            let failed = w.finish(phase, at);
                            w.busy_or_scheduled = false;
                            for id in failed {
                                self.schedule(
                                    at,
                                    Tick::Publish(Event::DecodeCapacityFailure(*worker, id)),
                                )?;
                            }
                            self.wake(*worker)?;
                        }
                        Event::BlockReceived(worker, block) => {
                            self.workers[*worker as usize].cache.receive(block, at);
                            if let Some(ids) = self.waiters.remove(&(*worker, block.clone())) {
                                self.workers[worker.other() as usize].cache.unpin(block, at);
                                for id in ids {
                                    self.workers[*worker as usize].cache.pin(block, at);
                                    let (_, left) = self.pending.get_mut(&id).unwrap();
                                    *left -= 1;
                                    if *left == 0 {
                                        let (job, _) = self.pending.remove(&id).unwrap();
                                        self.ready(*worker, job)?;
                                    }
                                }
                            }
                        }
                        Event::DecodeCapacityFailure(_, _) => (),
                        Event::Arrival(id) => {
                            self.awaiting_placement.insert(*id);
                        }
                    }
                    return Ok(Some(event));
                }
            }
        }
    }
}
