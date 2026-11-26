use kv_crossover::{
    TraceRequest,
    replay::{BlockId, Event, Replay, WorkerId},
    workload::{Arrival, RequestId},
};
use std::num::NonZeroU64;
fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}
fn row(line: usize, ms: u64) -> Arrival {
    Arrival {
        id: RequestId(line),
        request: TraceRequest::parse_line(&format!(
            r#"{{"timestamp":{ms},"input_length":1,"output_length":1,"hash_ids":[9]}}"#
        ))
        .unwrap(),
    }
}
#[test]
fn completions_are_visible_to_tied_arrivals() {
    use Event::*;
    use WorkerId::*;
    let run = || {
        let mut replay = Replay::new(vec![row(2, 0), row(5, 1), row(7, 1)], nz(1024)).unwrap();
        let mut log = Vec::new();
        while let Some(event) = replay.step() {
            log.push((replay.now_ns(), event));
            if let Arrival(id) = event {
                if id == RequestId(2) {
                    replay.receive_after(B, BlockId(9), nz(1_000_000)).unwrap();
                    replay.enqueue(A, id).unwrap();
                    replay.start_next(A, nz(1_000_000)).unwrap();
                    assert!(!replay.has_received(B, BlockId(9)));
                    assert_eq!(replay.running(A), Some(id));
                } else {
                    assert!(replay.has_received(B, BlockId(9)));
                    assert_eq!(replay.running(A), None);
                    replay.enqueue(B, id).unwrap();
                    assert_eq!(
                        replay.start_next(B, nz(10)).unwrap(),
                        (id == RequestId(5)).then_some(id)
                    );
                }
            } else if event == WorkDone(B, RequestId(5)) {
                replay.start_next(B, nz(1)).unwrap();
            }
        }
        log
    };
    let log = run();
    assert_eq!(
        log,
        [
            (0, Arrival(RequestId(2))),
            (1_000_000, BlockReceived(B, BlockId(9))),
            (1_000_000, WorkDone(A, RequestId(2))),
            (1_000_000, Arrival(RequestId(5))),
            (1_000_000, Arrival(RequestId(7))),
            (1_000_010, WorkDone(B, RequestId(5))),
            (1_000_011, WorkDone(B, RequestId(7)))
        ]
    );
    assert_eq!(log, run());
}
#[test]
fn rejects_invalid_source_order_and_time_overflow() {
    assert!(
        Replay::new(vec![row(2, 1), row(1, 1)], nz(1))
            .err()
            .unwrap()
            .contains("source line")
    );
    assert!(
        Replay::new(vec![row(1, u64::MAX)], nz(1))
            .err()
            .unwrap()
            .contains("timestamp")
    );
    let mut replay = Replay::new(vec![row(1, 1)], nz(1)).unwrap();
    replay.step();
    replay.enqueue(WorkerId::A, RequestId(1)).unwrap();
    assert!(replay.start_next(WorkerId::A, nz(u64::MAX)).is_err());
    assert_eq!(
        replay.start_next(WorkerId::A, nz(1)).unwrap(),
        Some(RequestId(1))
    );
}
