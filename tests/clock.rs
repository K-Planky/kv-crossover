use Event::{Arrival as Arr, BlockReceived, WorkDone};
use WorkerId::{A, B};
use kv_crossover::kv::BYTES_PER_TOKEN;
use kv_crossover::{
    TraceRequest,
    profile::Profile,
    replay::{BlockId, Event, Phase, Replay, WorkerId},
    workload::{Arrival, RequestId, VerifiedFile},
};
use std::{num::NonZeroU64, path::Path};
const BUDGET: u64 = 48 << 30;
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
            BUDGET,
        )
        .unwrap();
        let mut log = Vec::new();
        while let Some(event) = r.step().unwrap() {
            match event {
                Arr(RequestId(2)) => {
                    r.receive_after(A, BlockId(vec![0]), nz(20_000_000 - 10_770_595))
                        .unwrap();
                    r.receive_after(B, BlockId(vec![9]), nz(20_000_000))
                        .unwrap();
                }
                BlockReceived(A, _) => r.enqueue(A, RequestId(2)).unwrap(),
                Arr(_) => assert!(r.is_resident(B, &BlockId(vec![9]))),
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
            (20_000_000, BlockReceived(B, BlockId(vec![9]))),
            (20_000_000, WorkDone(A, Phase::Decode(vec![RequestId(2)]))),
            (20_000_000, Arr(RequestId(5))),
            (20_000_000, Arr(RequestId(7)))
        ]
    );
}
#[test]
fn rejects_invalid_inputs_and_clock_overflow() {
    let p = profile();
    let new = |rows| Replay::new(rows, nz(1), &p, BUDGET);
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
    assert!(r.enqueue(A, id).is_err());
    r.step().unwrap();
    r.enqueue(B, id).unwrap();
    assert!(r.enqueue(A, id).is_err());
    assert!(r.step().unwrap_err().contains("finish time overflow"));
}
#[test]
fn phase_elapsed_times_match_independent_vidur_checks() {
    let p = profile();
    let mut r = Replay::new(
        vec![row(1, 1, 6272, 1), row(2, 1, 512, 1), row(3, 1, 1536, 1)],
        nz(1),
        &p,
        BUDGET,
    )
    .unwrap();
    for n in 1..=12 {
        r.receive_after(A, BlockId(vec![0; n]), nz(1)).unwrap();
    }
    let mut done = Vec::new();
    while let Some(event) = r.step().unwrap() {
        match event {
            Arr(id) => r.enqueue(A, id).unwrap(),
            WorkDone(A, phase) => done.push((r.now_ns(), phase)),
            BlockReceived(_, _) => (),
            _ => unreachable!(),
        }
    }
    // Absolute nanosecond ticks; both intervals exactly match the rounded predictor (0 ticks tolerance).
    assert_eq!(done.len(), 3);
    assert_eq!(
        done[0],
        (11_884_036, Phase::Decode(vec![RequestId(2), RequestId(3)]))
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
    let mut rows: Vec<_> = (1..=9).map(|i| row(i, 1, 512, 2)).collect();
    rows.extend([row(10, 1, 129, 1), row(11, 1, 1, 1)]);
    let mut r = Replay::new(rows, nz(1), &p, BUDGET).unwrap();
    r.receive_after(B, BlockId(vec![0]), nz(1)).unwrap();
    let mut done = Vec::new();
    while let Some(e) = r.step().unwrap() {
        match e {
            Arr(id) => r.enqueue(B, id).unwrap(),
            WorkDone(B, phase) => done.push((r.now_ns(), phase)),
            BlockReceived(_, _) => (),
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

#[test]
fn published_blocks_stay_pinned_until_decode_failure() {
    let p = profile();
    let mut r = Replay::new(
        vec![row(1, 0, 513, 3), row(2, 0, 515, 1), row(3, 1000, 3, 1)],
        nz(1),
        &p,
        515 * BYTES_PER_TOKEN,
    )
    .unwrap();
    let prefix = BlockId(vec![0]);
    let other = BlockId(vec![9]);
    let mut processed = 0;
    let mut used_after_decode = Vec::new();
    let mut failed = Vec::new();
    while let Some(event) = r.step().unwrap() {
        match event {
            Arr(RequestId(2)) => {
                assert!(
                    r.enqueue(A, RequestId(2))
                        .unwrap_err()
                        .contains("insufficient KV")
                );
                assert_eq!(r.used_kv_tokens(A), 514);
            }
            Arr(id) => {
                r.enqueue(A, id).unwrap();
            }
            WorkDone(
                A,
                Phase::Prefill {
                    id: RequestId(1),
                    new_tokens,
                },
            ) => {
                processed += new_tokens;
                assert_eq!(r.is_resident(A, &prefix), processed >= 512);
                assert!(!r.is_resident(A, &BlockId(vec![0, 0])));
                assert!(r.receive_after(A, other.clone(), nz(1)).is_err());
            }
            WorkDone(A, Phase::Decode(ids)) => {
                used_after_decode.push((ids[0], r.used_kv_tokens(A)))
            }
            Event::DecodeCapacityFailure(A, id) => failed.push(id),
            WorkDone(A, Phase::Prefill { .. }) => (),
            _ => unreachable!(),
        }
        assert!(r.used_kv_tokens(A) <= 515);
    }
    assert_eq!(processed, 513);
    assert_eq!(failed, [RequestId(1)]);
    assert_eq!(
        used_after_decode,
        [(RequestId(1), 515), (RequestId(1), 512), (RequestId(3), 0)]
    );
}

#[test]
fn incoming_space_and_deterministic_eviction() {
    let p = profile();
    let mut r = Replay::new(vec![row(1, 1, 1024, 1)], nz(1), &p, 1536 * BYTES_PER_TOKEN).unwrap();
    // Same timestamp for root and descendant; root wins the eviction tie.
    let [root, child, older, newer, extra] =
        [vec![0], vec![0, 0], vec![9], vec![7], vec![8]].map(BlockId);
    for (key, tick) in [(&root, 2), (&child, 2), (&older, 1)] {
        r.receive_after(A, key.clone(), nz(tick)).unwrap();
    }
    assert!(!r.is_resident(A, &older));
    assert!(r.receive_after(A, extra.clone(), nz(1)).is_err());
    for _ in 0..3 {
        assert!(matches!(r.step().unwrap(), Some(BlockReceived(_, _))));
    }
    r.receive_after(A, newer, nz(1)).unwrap();
    assert!(!r.is_resident(A, &older));
    r.receive_after(A, extra, nz(1)).unwrap();
    assert!(!r.is_resident(A, &root));
    assert!(r.is_resident(A, &child));
    for _ in 0..2 {
        r.step().unwrap();
    }
    assert_eq!(r.used_kv_tokens(A), 1536);
    assert_eq!(r.step().unwrap(), Some(Arr(RequestId(1))));
    r.enqueue(A, RequestId(1)).unwrap();
    // Missing root: all 1024 tokens must be computed despite the resident child.
    let mut computed = 0;
    while let Some(e) = r.step().unwrap() {
        if let WorkDone(_, Phase::Prefill { new_tokens, .. }) = e {
            computed += new_tokens;
        }
    }
    assert_eq!(computed, 1024);
    assert_eq!(r.used_kv_tokens(A), 1024);
    r.receive_after(A, older, nz(1)).unwrap(); // previous reservation was cleared
}
