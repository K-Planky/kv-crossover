# KV crossover

The first experiment measures how many prompt tokens a second request can reuse
if the first prompt is resident. The two records in
[`examples/mooncake-pair.jsonl`](examples/mooncake-pair.jsonl) are the example
from the [Mooncake FAST ’25 trace guide](https://github.com/kvcache-ai/Mooncake/blob/df19e4eeb8bf49836f3612d8eb7b5b76fa757ee2/FAST25-release/README.md),
serialized one request per line. They are not a workload replay.

From the project root with [rustup](https://rustup.rs/) installed:

```sh
cargo run --locked -- prefix examples/mooncake-pair.jsonl
# reusable_tokens=6144 local_tokens=328
```

The matcher compares hashes in order until the first difference. Only full
512-token blocks from both prompts qualify. Here 12 blocks match; the remaining
328 of the arriving prompt’s 6,472 tokens need local processing.

Development checks (rustup selects the pinned compiler and components):

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```
