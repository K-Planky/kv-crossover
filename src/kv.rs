//! Local KV, charged in logical token slots (not profiler padding).
use crate::{BLOCK_TOKENS, TraceRequest};
use std::collections::{BTreeMap, BTreeSet};

pub const BYTES_PER_TOKEN: u64 = 2 * 32 * 8 * 128 * 2;
const BLOCK: u64 = BLOCK_TOKENS as u64;

/// Include ancestry: the same hash after a divergence is not the same KV.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct BlockId(pub Vec<u64>);
impl BlockId {
    pub fn prompt(r: &TraceRequest) -> Vec<Self> {
        let mut prefix = Vec::new();
        r.full_blocks()
            .map(|h| {
                prefix.push(*h);
                Self(prefix.clone())
            })
            .collect()
    }
}
#[derive(Default)]
struct Resident {
    pins: usize,
    last_use_ns: u64,
}

pub(crate) struct Cache {
    capacity: u64,
    used: u64,
    resident: BTreeMap<BlockId, Resident>,
    incoming: BTreeSet<BlockId>,
}
/// Ownership of pinned complete blocks and private (or reserved) token slots.
pub(crate) struct Lease {
    blocks: Vec<BlockId>,
    pinned: usize,
    private: u64,
}
impl Cache {
    pub fn new(bytes: u64) -> Self {
        Self {
            capacity: bytes / BYTES_PER_TOKEN,
            used: 0,
            resident: BTreeMap::new(),
            incoming: BTreeSet::new(),
        }
    }
    pub fn used(&self) -> u64 {
        self.used
    }
    pub fn contains(&self, block: &BlockId) -> bool {
        self.resident.contains_key(block)
    }
    fn allocate(&mut self, slots: u64, protect: &[BlockId]) -> bool {
        let needed = slots
            .saturating_sub(self.capacity - self.used)
            .div_ceil(BLOCK);
        if needed > 0 {
            let mut victims: Vec<_> = self
                .resident
                .iter()
                .filter(|(k, v)| v.pins == 0 && !protect.contains(k))
                .map(|(k, v)| (v.last_use_ns, k.clone()))
                .collect();
            if (victims.len() as u64) < needed {
                return false;
            }
            victims.sort_unstable();
            for (_, k) in victims.into_iter().take(needed as usize) {
                self.resident.remove(&k);
                self.used -= BLOCK;
            }
        }
        self.used += slots;
        true
    }
    pub fn admit(&mut self, r: &TraceRequest, now: u64) -> Option<(usize, Lease)> {
        let blocks = BlockId::prompt(r);
        let pinned = blocks.iter().take_while(|b| self.contains(b)).count();
        let cached = pinned * BLOCK_TOKENS;
        let private = (r.input_length - cached) as u64 + 1;
        if !self.allocate(private, &blocks[..pinned]) {
            return None;
        }
        for k in &blocks[..pinned] {
            let v = self.resident.get_mut(k).unwrap();
            v.pins += 1;
            v.last_use_ns = now;
        }
        Some((
            cached,
            Lease {
                blocks,
                pinned,
                private,
            },
        ))
    }
    pub fn computed(&mut self, l: &mut Lease, processed: usize, now: u64) {
        while l.pinned < processed / BLOCK_TOKENS {
            let k = &l.blocks[l.pinned];
            if self.contains(k) {
                self.used -= BLOCK;
            } // discard redundant private computation
            let v = self.resident.entry(k.clone()).or_default();
            v.pins += 1;
            v.last_use_ns = now;
            l.private -= BLOCK;
            l.pinned += 1;
        }
    }
    pub fn grow(&mut self, l: &mut Lease) -> bool {
        if !self.allocate(1, &[]) {
            return false;
        }
        l.private += 1;
        true
    }
    pub fn release(&mut self, l: Lease, now: u64) {
        self.used -= l.private;
        for k in &l.blocks[..l.pinned] {
            let v = self.resident.get_mut(k).unwrap();
            v.pins -= 1;
            v.last_use_ns = now;
        }
    }
    pub fn reserve(&mut self, k: &BlockId) -> Result<(), String> {
        if k.0.is_empty() || self.contains(k) || self.incoming.contains(k) {
            return Err(format!("invalid or duplicate incoming block {k:?}"));
        }
        if !self.allocate(BLOCK, &[]) {
            return Err(format!("insufficient KV for incoming block {k:?}"));
        }
        self.incoming.insert(k.clone());
        Ok(())
    }
    pub fn receive(&mut self, k: &BlockId, now: u64) {
        assert!(self.incoming.remove(k));
        if self.contains(k) {
            self.used -= BLOCK;
        }
        self.resident.entry(k.clone()).or_default().last_use_ns = now;
    }
}
