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

## FAST ’25 workload audit

```sh
cargo run --locked -- audit --config experiments/paper.toml
```

This verifies SHA-256 over the original JSONL bytes before parsing each file.
Missing or changed inputs fail the audit; restore them from Git or from the
pinned [release directory](https://github.com/kvcache-ai/Mooncake/tree/df19e4eeb8bf49836f3612d8eb7b5b76fa757ee2/FAST25-release/traces)
and rerun. Both full releases fit in ordinary Git (3,029,533 and 4,415,857
bytes respectively). `data/mooncake/` contains the unmodified conversation
and tool/agent files from Mooncake and its [Apache 2.0 license](data/mooncake/LICENSE-APACHE);
the pinned release has no separate trace license or NOTICE. Provenance and
digests are in [`experiments/paper.toml`](experiments/paper.toml).

| Workload | Requests | Min | p50 | p95 | Max | Above 8,192 | Share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Conversation | 12,031 | 895 | 7,255 | 40,056 | 126,527 | 5,570 | 46.30% |
| Tool/agent | 23,608 | 895 | 6,403 | 26,446 | 126,527 | 5,282 | 22.37% |

Lengths are **input plus output** tokens, with nearest-rank percentiles over
all arrivals in each file. Many requests exceed the original Llama 3 8B
context limit; this is a coverage check, not a decision to filter the replay.
The loader keeps the one-based JSONL line as the request ID within its trace,
preserves relative millisecond timestamps and does not reorder ties. Invalid
records (including blank lines) or decreasing timestamps fail rather than
being skipped. Both releases satisfy the guide example’s hash-count rule:
one hash per 512-token block, including a final partial block if present.

Development checks (rustup selects the pinned compiler and components):

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```
