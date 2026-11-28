use super::*;
use crate::router::{Action::*, CopyForecast::*, earliest};
use WorkerId::{A, B};
const MS: u64 = 1_000_000;
const TRANSFER: Action = Transfer { prefix_blocks: 12 };

// At 1 ms the warmup deliveries have landed; one cached decode on A
// (optionally one on B) models an already committed compute interval.
fn trial(a_ms: u64, b_busy: bool, contested: bool, worker: WorkerId, forced: Action) -> u64 {
    let p = Profile::hand_oracle();
    let row = |id, input: usize, hash| Arrival {
        id: RequestId(id),
        request: TraceRequest {
            arrival_ms: 1,
            input_length: input,
            output_length: 1,
            hash_ids: vec![hash; input.div_ceil(BLOCK_TOKENS)],
        },
    };
    let a_input = match a_ms {
        20 => 1024,
        100 => 512,
        _ => unreachable!(),
    };
    let mut rows = vec![row(1, a_input, 1)];
    if b_busy {
        rows.push(row(2, 1536, 2));
    }
    let target = row(3, 6472, 3);
    let shared = BlockId::prompt(&target.request);
    let bytes = (shared.len() * BLOCK_TOKENS) as u64 * BYTES_PER_TOKEN;
    assert_eq!(bytes, 768 << 20);
    rows.push(target);
    let mut r = Replay::new(rows, NonZeroU64::new(25 << 29).unwrap(), &p, 48 << 30).unwrap();
    for (dst, blocks) in [
        (A, shared),
        (A, BlockId::prompt(r.request(RequestId(1)).unwrap())),
    ] {
        for b in blocks {
            r.receive_after(dst, b, NonZeroU64::MIN).unwrap();
        }
    }
    if b_busy {
        for b in BlockId::prompt(r.request(RequestId(2)).unwrap()) {
            r.receive_after(B, b, NonZeroU64::MIN).unwrap();
        }
    }
    while let Some(event) = r.step().unwrap() {
        match event {
            Event::Arrival(id @ RequestId(1)) => r.enqueue(A, id).unwrap(),
            Event::Arrival(id @ RequestId(2)) => r.enqueue(B, id).unwrap(),
            Event::Arrival(id) => {
                if contested {
                    // Independent background delivery: owns destination reservations,
                    // but no logical waiter, so its finish creates no compute work.
                    let blocks: Vec<_> = (1..=12).map(|n| BlockId(vec![9; n])).collect();
                    for b in &blocks {
                        r.workers[A as usize].cache.reserve(b).unwrap();
                    }
                    r.link.start(r.now_ns(), bytes, (A, blocks));
                }
                let view = r.router_view(id).unwrap();
                for forecast in [Nominal, ContentionAware] {
                    let scores = view.action_scores(&p, Some(forecast)).unwrap();
                    let transfer = scores.iter().find(|s| s.action == TRANSFER).unwrap();
                    let slow = contested && forecast == ContentionAware;
                    assert_eq!(
                        transfer.placement.predicted_ttft_ns,
                        if slow { 130 } else { 70 } * MS
                    );
                    let best = earliest(&scores).unwrap();
                    let (action, ms) = if a_ms == 20 {
                        (Wait, 30)
                    } else if slow {
                        (Recompute, 100)
                    } else {
                        (TRANSFER, 70)
                    };
                    assert_eq!(
                        (best.action, best.placement.predicted_ttft_ns),
                        (action, ms * MS)
                    );
                }
                r.place(worker, id, forced).unwrap();
            }
            Event::WorkDone(_, Phase::Decode(ids)) if ids.contains(&RequestId(3)) => {
                return r.now_ns() - MS;
            }
            _ => (),
        }
    }
    panic!("no first token")
}
#[test]
fn measured_hand_crossover() {
    for contested in [false, true] {
        let observed = [(A, Wait), (B, TRANSFER), (B, Recompute)]
            .map(|(w, a)| trial(100, false, contested, w, a));
        assert_eq!(
            observed,
            [110, if contested { 130 } else { 70 }, 100].map(|ms| ms * MS)
        );
    }
    assert_eq!(trial(20, false, false, A, Wait), 30 * MS);
    assert_eq!(trial(100, true, false, B, TRANSFER), 70 * MS);
}
