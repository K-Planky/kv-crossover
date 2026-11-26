//! Equal sharing among batches of newly missing blocks (not among logical waiters).
use std::num::NonZeroU64;

const NS_PER_SECOND: u128 = 1_000_000_000;

#[derive(Clone)]
pub(crate) struct Link<T> {
    pub rate: NonZeroU64,
    at_ns: u64,
    // Remaining work in billionths of a byte, in deterministic start order.
    active: Vec<(u128, T)>,
    pub delivered_bytes: u128,
}
impl<T> Link<T> {
    pub fn new(rate: NonZeroU64) -> Self {
        Self {
            rate,
            at_ns: 0,
            active: Vec::new(),
            delivered_bytes: 0,
        }
    }
    fn advance(&mut self, now: u64) {
        if !self.active.is_empty() {
            let mut budget = u128::from(now - self.at_ns) * u128::from(self.rate.get());
            // If a copy finishes partway through the last tick, give unused
            // service to survivors instead of charging them an extra tick.
            loop {
                let count = self.active.iter().filter(|(w, _)| *w > 0).count();
                if count == 0 {
                    break;
                }
                let share = budget / count as u128;
                if share == 0 {
                    break;
                }
                for (work, _) in &mut self.active {
                    let sent = share.min(*work);
                    *work -= sent;
                    budget -= sent;
                }
            }
        }
        self.at_ns = now;
    }
    pub fn start(&mut self, now: u64, bytes: u64, batch: T) {
        self.advance(now);
        self.active.push((u128::from(bytes) * NS_PER_SECOND, batch));
    }
    pub fn observed(&self, now: u64) -> Vec<(u64, T)>
    where
        T: Clone,
    {
        let mut snapshot = self.clone();
        snapshot.advance(now);
        snapshot
            .active
            .into_iter()
            .map(|(w, t)| (w.div_ceil(NS_PER_SECOND) as u64, t))
            .collect()
    }
    pub fn next_finish(&self) -> Result<Option<u64>, String> {
        let Some(shortest) = self.active.iter().map(|(work, _)| *work).min() else {
            return Ok(None);
        };
        let delay = (shortest * self.active.len() as u128).div_ceil(u128::from(self.rate.get()));
        Ok(Some(
            u64::try_from(u128::from(self.at_ns) + delay)
                .map_err(|_| "physical link finish time overflow")?,
        ))
    }
    pub fn finish(&mut self, now: u64) -> Vec<T> {
        self.advance(now);
        self.active
            .extract_if(.., |(work, _)| *work == 0)
            .map(|(_, batch)| batch)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn equal_share(s in 1u64..1<<30, r in 1u64..1<<30, rate in 1u64..1<<36) {
            let mut l = Link::new(NonZeroU64::new(rate).unwrap());
            l.start(0,s,0); l.start(0,r,1);
            loop {
                let at = l.next_finish().unwrap().unwrap();
                if l.finish(at).contains(&0) {
                    let expected = u128::from(s+s.min(r))*NS_PER_SECOND;
                    prop_assert!((u128::from(at)*u128::from(rate)).abs_diff(expected) <= u128::from(rate));
                    break;
                }
            }
        }
    }
    #[test]
    fn stagger_and_tie() {
        let mut l = Link::new(NonZeroU64::new(1_000_000_000).unwrap());
        l.start(0, 100, 'a');
        l.start(30, 70, 'b');
        l.start(30, 10, 'c');
        assert_eq!(l.next_finish().unwrap(), Some(60));
        assert_eq!(l.finish(60), ['c']);
        assert_eq!(l.next_finish().unwrap(), Some(180));
        assert_eq!(l.finish(180), ['a', 'b']);
        assert_eq!(l.next_finish().unwrap(), None);
    }
}
