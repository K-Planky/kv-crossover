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

All four routers will use two identical single-A100 workers. The
[hardware qualification](profiles/qualification.md) explains why the H200
attention measurements are insufficient, which arrivals qualify and how
much potential prefix reuse is lost. The [profile guide](profiles/README.md)
documents the predictor and how to regenerate its timing table.

## Local router baselines

At arrival, `Replay::router_view` snapshots both workers' complete resident
prefixes, local admission feasibility, assigned jobs (including copy waiters),
and active physical copies with remaining bytes rounded up. The local policies
ignore the link. Shortest queue minimizes assigned request count; the
[Preble-inspired](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html) rule minimizes
predicted backlog + uncached chunk times + one decode step. Both skip infeasible
workers, break ties toward A and reuse the chosen worker's local prefix.

Backlog is an approximation, not a forward run of the scheduler. Charge each
assigned job its uncompleted prefill chunks and estimated remaining tokens at
the single-response decode cost for its current (or post-prefill) context.
Estimate total output as the rounded-up mean emitted by successfully finished
responses across both workers, initially one. Subtract tokens already emitted,
leaving at least one for an unfinished job, bounded by the context limit.
This serial-work proxy ignores batching, copy delays and fractional progress
inside a phase: a long response can make a worker look busier than it is.
Neither scoring nor history estimation reads recorded output lengths.

## Worker rounds

Both workers use the same nonpreemptive scheduler. At each clock tick the
replay applies completions and block deliveries, handles tied arrivals in
file order, then dispatches idle workers. A round decodes one token each for
up to eight responses, then prefills up to 128 tokens for the oldest ready
prompt. An empty phase costs nothing. An unfinished prompt stays at the
head; responses rotate to the tail after their batch. Admission order is
when work becomes ready, not the original arrival time. A fully cached
prompt joins the response queue directly.

In this model prefill **only** builds prompt KV. Even the first output token
costs one decode step. At decode completion every member emits simultaneously,
adds one logical token to its processed context, and leaves if its execution-side
output count is exhausted. Profile padding never adds logical tokens. The
single-worker clock tests (`cargo test --locked --test clock`) check elapsed
phase times against the [independent predictor checks](profiles/README.md)
and show a response's token gap lengthening by an intervening prefill chunk.
The tests warm the cache with scheduled block deliveries, not a supplied hit count.

## Finite worker KV

Each worker gets its own byte budget (48 GiB in the selected profile), rounded
down to whole 131,072-byte token slots. At admission, scan complete blocks in
order until the first miss; a key includes all preceding hashes. Pin the hits
and reserve private slots for the uncached prompt (including its partial tail)
and one output token. A failed admission changes no cache state. A prefill
completion publishes each newly finished full block, merging identical KV if
another job or delivery already published it. Partial blocks and output KV
stay private. A scheduled incoming block holds 512 slots but is not a hit until
delivery. `Replay::transfer` admits the private suffix and first output slot
before starting a copy. All newly needed remote blocks for that placement
travel as one batch and become available together. An overlapping request
joins by (destination, full block path); only its additional missing blocks
start another batch. Source blocks stay pinned through delivery; each waiter
then holds destination pins until its own job releases them. Jobs enter the
ready queue only when all required batches have arrived.

Both directions share one configured bytes/second rate equally per active
batch, independently of worker compute. The clock recomputes the next finish
on active-set changes, publishes deliveries before tied arrivals, and counts
payload bytes once per completed batch. Service is accounted in billionths
of a byte (discarding less than one such unit per active batch per update);
completion rounds up to a nanosecond. `receive_after` is a synthetic fixture
for seeding controlled states; it does not contribute link traffic.

After a decode batch emits, release finished jobs first. In batch order, each
survivor reserves one more slot. If it cannot, report a decode capacity failure
at that tick and release its pins and private slots; already emitted tokens
still count. No job reserves its recorded future output length. For space,
evict unpinned resident blocks by oldest last use (pin, publication or release),
then lexicographic hash path on a tied tick. Eviction need not preserve parents:
a resident descendant past a gap is not a prefix hit.

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
context limit. The audit also verifies the selected profile and prints the
request and prefix-token counts for the fixed replay subset.
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
