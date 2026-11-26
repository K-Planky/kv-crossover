use kv_crossover::{TraceRequest, reusable_tokens};

#[test]
fn mooncake_guide_pair() {
    let pair: Vec<_> = include_str!("../examples/mooncake-pair.jsonl")
        .lines()
        .map(|line| TraceRequest::parse_line(line).unwrap())
        .collect();
    let [resident, arriving] = pair.as_slice() else {
        panic!("expected a pair")
    };
    assert_eq!((arriving.arrival_ms, arriving.output_length), (30535, 26));
    let reused = reusable_tokens(resident, arriving);
    assert_eq!(reused, 6144);
    assert_eq!(arriving.input_length - reused, 328);
}

fn prompt(tokens: usize, hashes: &[u64]) -> TraceRequest {
    TraceRequest::parse_line(
        &serde_json::json!({
            "timestamp": 0, "input_length": tokens, "output_length": 1, "hash_ids": hashes
        })
        .to_string(),
    )
    .unwrap()
}

#[test]
fn never_resumes_after_divergence() {
    assert_eq!(
        reusable_tokens(&prompt(1536, &[1, 2, 3]), &prompt(1536, &[1, 4, 3])),
        512
    );
}

#[test]
fn hash_must_cover_a_full_block_in_both_prompts() {
    let short = prompt(513, &[1, 2]);
    let full = prompt(1024, &[1, 2]);
    assert_eq!(reusable_tokens(&short, &full), 512);
    assert_eq!(reusable_tokens(&full, &short), 512);
    assert_eq!(reusable_tokens(&full, &full), 1024);
}
