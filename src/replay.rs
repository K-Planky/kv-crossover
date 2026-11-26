//! Single-threaded replay clock. Load one trace in source order, then drain `step`.
//! Handling an arrival is synchronous: the next tied arrival sees its placement.

use crate::{
    TraceRequest,
    workload::{Arrival, RequestId},
};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    num::NonZeroU64,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum WorkerId {
    A,
    B,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct BlockId(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Event {
    WorkDone(WorkerId, RequestId),
    BlockReceived(WorkerId, BlockId),
    Arrival(RequestId),
}

#[derive(Default)]
struct Worker {
    ready: VecDeque<RequestId>,
    running: Option<RequestId>,
    received: HashSet<BlockId>,
}

pub struct Replay {
    now_ns: u64,
    // Sequence is unique, so payload ordering never decides a tie.
    events: BinaryHeap<Reverse<(u64, u8, u64, Event)>>,
    sequence: u64,
    requests: HashMap<RequestId, TraceRequest>,
    workers: [Worker; 2],
    link_bytes_per_second: NonZeroU64,
}

impl Replay {
    pub fn new(arrivals: Vec<Arrival>, link_bytes_per_second: NonZeroU64) -> Result<Self, String> {
        let mut replay = Self {
            now_ns: 0,
            events: BinaryHeap::new(),
            sequence: 0,
            requests: HashMap::new(),
            workers: Default::default(),
            link_bytes_per_second,
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
            replay.schedule(at, Event::Arrival(id))?;
            replay.requests.insert(id, request);
        }
        Ok(replay)
    }

    fn schedule(&mut self, at: u64, event: Event) -> Result<(), String> {
        let seq = self.sequence;
        self.sequence = seq.checked_add(1).ok_or("event sequence overflow")?;
        let priority = u8::from(matches!(event, Event::Arrival(_)));
        self.events.push(Reverse((at, priority, seq, event)));
        Ok(())
    }

    pub fn now_ns(&self) -> u64 {
        self.now_ns
    }
    pub fn link_bytes_per_second(&self) -> NonZeroU64 {
        self.link_bytes_per_second
    }
    /// Execution-side trace data, not a router view.
    pub fn request(&self, id: RequestId) -> Option<&TraceRequest> {
        self.requests.get(&id)
    }
    pub fn running(&self, worker: WorkerId) -> Option<RequestId> {
        self.workers[worker as usize].running
    }
    pub fn has_received(&self, worker: WorkerId, block: BlockId) -> bool {
        self.workers[worker as usize].received.contains(&block)
    }

    /// Place an arrival in worker FIFO order. Called once by the arrival handler.
    pub fn enqueue(&mut self, worker: WorkerId, id: RequestId) -> Result<(), String> {
        if !self.requests.contains_key(&id) {
            return Err(format!("unknown request {}", id.0));
        }
        self.workers[worker as usize].ready.push_back(id);
        Ok(())
    }

    /// Start one queued unit of work if the worker is idle. The scheduler supplies its cost.
    pub fn start_next(
        &mut self,
        worker: WorkerId,
        duration_ns: NonZeroU64,
    ) -> Result<Option<RequestId>, String> {
        let state = &self.workers[worker as usize];
        let Some(&id) = state.ready.front() else {
            return Ok(None);
        };
        if state.running.is_some() {
            return Ok(None);
        }
        let at = self
            .now_ns
            .checked_add(duration_ns.get())
            .ok_or_else(|| format!("request {}: work finish time overflow", id.0))?;
        self.schedule(at, Event::WorkDone(worker, id))?;
        let state = &mut self.workers[worker as usize];
        state.ready.pop_front();
        state.running = Some(id);
        Ok(Some(id))
    }

    /// Link service supplies the delivery delay; scheduling does not publish the block.
    pub fn receive_after(
        &mut self,
        worker: WorkerId,
        block: BlockId,
        delay_ns: NonZeroU64,
    ) -> Result<(), String> {
        let at = self
            .now_ns
            .checked_add(delay_ns.get())
            .ok_or_else(|| format!("block {}: delivery time overflow", block.0))?;
        self.schedule(at, Event::BlockReceived(worker, block))
    }

    /// Advance to one event and apply its completion effects before returning it.
    pub fn step(&mut self) -> Option<Event> {
        let Reverse((at, _, _, event)) = self.events.pop()?;
        self.now_ns = at;
        match event {
            Event::WorkDone(worker, _) => self.workers[worker as usize].running = None,
            Event::BlockReceived(worker, block) => {
                self.workers[worker as usize].received.insert(block);
            }
            Event::Arrival(_) => (),
        }
        Some(event)
    }
}
