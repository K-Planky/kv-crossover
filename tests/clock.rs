use Event::{Arrival as Arr, BlockReceived, WorkDone};
use WorkerId::{A, B};
use kv_crossover::{
    TraceRequest,
    profile::Profile,
    replay::{BlockId, Event, Phase, Replay, WorkerId},
    workload::{Arrival, RequestId, VerifiedFile},
};
use std::{num::NonZeroU64, path::Path};
fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}
fn profile() -> Profile {
    #[derive(serde::Deserialize)]
    struct Manifest {
        vidur_a100: VerifiedFile,
    }
    let m: Manifest = toml::from_str(include_str!("../experiments/paper.toml")).unwrap();
    Profile::load(&m.vidur_a100, Path::new("experiments")).unwrap()
}
fn row(line: usize, ms: u64, input: usize, output: usize) -> Arrival {
    let hashes = vec![0; input.div_ceil(512)];
    Arrival{id:RequestId(line),request:TraceRequest::parse_line(&format!(r#"{{"timestamp":{ms},"input_length":{input},"output_length":{output},"hash_ids":{hashes:?}}}"#)).unwrap()}
}
#[test]
fn completions_are_visible_to_tied_arrivals() {
    let p = profile();
    let run = || {
        let mut r = Replay::new(
            vec![row(2, 0, 512, 1), row(5, 20, 1, 1), row(7, 20, 1, 1)],
            nz(1024),
            &p,
        )
        .unwrap();
        let mut log = Vec::new();
        while let Some(event) = r.step().unwrap() {
            match event {
                Arr(RequestId(2)) => {
                    r.receive_after(A, BlockId(1), nz(20_000_000 - 10_770_595))
                        .unwrap();
                    r.receive_after(B, BlockId(9), nz(20_000_000)).unwrap();
                }
                BlockReceived(A, _) => r.enqueue(A, RequestId(2), 512).unwrap(),
                Arr(_) => assert!(r.has_received(B, BlockId(9))),
                _ => (),
            }
            log.push((r.now_ns(), event));
        }
        log
    };
    let log = run();
    assert_eq!(log, run());
    assert_eq!(
        &log[2..],
        &[
            (20_000_000, BlockReceived(B, BlockId(9))),
            (20_000_000, WorkDone(A, Phase::Decode(vec![RequestId(2)]))),
            (20_000_000, Arr(RequestId(5))),
            (20_000_000, Arr(RequestId(7)))
        ]
    );
}
#[test]
fn rejects_invalid_inputs_and_clock_overflow() {
    let p = profile();
    let new = |rows| Replay::new(rows, nz(1), &p);
    assert!(
        new(vec![row(2, 1, 1, 1), row(1, 1, 1, 1)])
            .err()
            .unwrap()
            .contains("source line")
    );
    assert!(
        new(vec![row(1, u64::MAX, 1, 1)])
            .err()
            .unwrap()
            .contains("timestamp")
    );
    let mut r = new(vec![row(1, u64::MAX / 1_000_000, 1, 1)]).unwrap();
    let id = RequestId(1);
    assert!(r.enqueue(A, id, 0).is_err());
    r.step().unwrap();
    assert!(r.enqueue(A, id, 1).is_err());
    r.enqueue(B, id, 0).unwrap();
    assert!(r.enqueue(A, id, 0).is_err());
    assert!(r.step().unwrap_err().contains("finish time overflow"));
}
#[test]
fn phase_elapsed_times_match_independent_vidur_checks() {
    let p = profile();
    let mut r = Replay::new(
        vec![row(1, 0, 6272, 1), row(2, 0, 512, 1), row(3, 0, 1536, 1)],
        nz(1),
        &p,
    )
    .unwrap();
    let mut done = Vec::new();
    while let Some(event) = r.step().unwrap() {
        match event {
            Arr(id) => r.enqueue(A, id, [6144, 512, 1536][id.0 - 1]).unwrap(),
            WorkDone(A, phase) => done.push((r.now_ns(), phase)),
            _ => unreachable!(),
        }
    }
    // Absolute nanosecond ticks; both intervals exactly match the rounded predictor (0 ticks tolerance).
    assert_eq!(done.len(), 3);
    assert_eq!(
        done[0],
        (10_884_036, Phase::Decode(vec![RequestId(2), RequestId(3)]))
    );
    assert_eq!(done[1].0 - done[0].0, 18_696_410);
    assert_eq!(
        done[1].1,
        Phase::Prefill {
            id: RequestId(1),
            new_tokens: 128
        }
    );
    assert_eq!(done[2].1, Phase::Decode(vec![RequestId(1)]));
}
#[test]
fn mixed_round_delays_tokens_and_respects_caps() {
    let p = profile();
    let mut rows: Vec<_> = (1..=9).map(|i| row(i, 0, 512, 2)).collect();
    rows.extend([row(10, 0, 129, 1), row(11, 0, 1, 1)]);
    let mut r = Replay::new(rows, nz(1), &p).unwrap();
    let mut done = Vec::new();
    while let Some(e) = r.step().unwrap() {
        match e {
            Arr(id) => r.enqueue(B, id, if id.0 <= 9 { 512 } else { 0 }).unwrap(),
            WorkDone(B, phase) => done.push((r.now_ns(), phase)),
            _ => unreachable!(),
        }
    }
    assert_eq!(done[0].1, Phase::Decode((1..=8).map(RequestId).collect()));
    assert_eq!(
        done[1].1,
        Phase::Prefill {
            id: RequestId(10),
            new_tokens: 128
        }
    );
    assert_eq!(
        done[2].1,
        Phase::Decode([9, 1, 2, 3, 4, 5, 6, 7].map(RequestId).to_vec())
    );
    // Request 1's token gap includes a cold 128-token chunk plus one whole mixed decode step.
    assert_eq!(
        done[2].0 - done[0].0,
        15_441_210
            + p.decode_ns(&[512, 513, 513, 513, 513, 513, 513, 513])
                .unwrap()
    );
    assert_eq!(
        done[3].1,
        Phase::Prefill {
            id: RequestId(10),
            new_tokens: 1
        }
    );
    assert_eq!(
        done[5].1,
        Phase::Prefill {
            id: RequestId(11),
            new_tokens: 1
        }
    );
    let emitted: usize = done
        .iter()
        .map(|(_, p)| match p {
            Phase::Decode(ids) => ids.len(),
            _ => 0,
        })
        .sum();
    assert_eq!(emitted, 20);
}
