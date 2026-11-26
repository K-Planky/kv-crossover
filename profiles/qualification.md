# Hardware qualification: select the A100, restrict the workload

Use two identical single-A100 workers and the original Llama 3 8B for all four
routers. This is a **subset experiment**, especially for conversation; it
does not estimate goodput for the full releases.

| Per worker | A100 / Llama 3 8B | H200 / Llama 3.1 8B |
| --- | --- | --- |
| Model context (input + output) | [8,192](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) | [131,072](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json); both trace maxima (126,527) fit |
| Device memory | [80 GB Vidur SKU](https://github.com/microsoft/vidur/blob/abae7f63aa857300f5cdc6f5e0d27860cd24721b/vidur/config/device_sku_config.py) | [141 GB](https://www.nvidia.com/en-us/data-center/h200/) |
| Modeled KV budget | 48 GiB = 393,216 tokens | 96 GiB = 786,432 tokens (comparison only) |
| Service-cost source | Vidur transformer phase predictor with published attention **and** MLP measurements | Published FlashInfer *attention kernel* times, not a worker phase |

For budgeting, interpret GB as decimal; allow 17 GB for roughly 8 billion
two-byte weights and reserve 8 GB for runtime/workspace on either GPU.
Round down further to the KV budgets above (not measurements of free HBM).
Both models use 2 × 32 layers × 8 KV heads × 128 elements × 2 bytes =
131,072 bytes/token. Even the largest single model-supported request fits
its worker; simultaneous requests and retained prefixes still share one
finite budget. Memory cannot extend model context or create a timing sample.

**Why not use H200 for all arrivals?** In ServingStudio's H200 rows filtered
to BF16 queries/KV, 32 query heads, 8 KV heads and head dimension 128,
[prefill](https://servingstudio.cs.washington.edu/kernels/flashinfer_attn_prefill.md)
has 270 rows across backends, minimum append 128, maximum prefix + append
81,920 (137 conversation and 114 tool prompts alone exceed this).
[Decode](https://servingstudio.cs.washington.edu/kernels/flashinfer_attn_decode.md)
has 486 rows with batch sizes in powers of two from 1 to 256, not measurements
of arbitrary skewed context mixes: each request gets floor(total/batch).
These are sparse, cold-cache CUPTI launch timings; planning is excluded.
Prefill uses contiguous KV for one request. Even multiplying attention by 32
would omit projections, normalization and feed-forward work; see the
[single-GPU deployment tree](https://servingstudio.cs.washington.edu/models/Llama-3.1-8B/llama3_dense.md).
A separately validated composite H200 profile could change this decision.

## Selected lookup domain

Use the [pinned table and its regeneration checks](README.md); the table's
revision and SHA-256 are in [paper.toml](../experiments/paper.toml). One
verified table is loaded into memory, not reopened per phase. Its entries
are predicted nanoseconds, not end-to-end service measurements.

- Prefill in 128-token chunks from a 512-token reuse boundary (or zero).
  Only the last chunk may be shorter. **Charge** 1–64 new tokens as 64 and
  65–128 as 128, without adding padding to logical progress or KV. Both
  charged shapes at every reachable chunk start (0, 128, …, 8064) occur
  in the pinned TP=1 attention inputs. This explicit timing approximation
  avoids treating very small exported tail predictions as measured shapes;
  it can overcharge short tails. Misaligned starts and invalid sizes fail.
- Decode one token per request, batch 1–8. Require each processed context
  in 1–8191, floor the batch mean, then round it up to a multiple of 64.
  TP=1 attention samples span 32–16,128 cached tokens for each selected
  batch size. The lookup uses Vidur's predictions **inside** those sampled
  ranges, not a claim every key or unequal mix was measured. MLP inputs
  span 1–32,768 tokens, including both charged chunks but not every small
  decode batch. Mean context hides skew. No fallback to the last row beyond
  coverage; sensitivity to modeled costs matters for interpreting outcomes.

## Fixed cohort and loss of prefix reuse

Keep positive input/output lengths with input + output ≤ 8192 in file order,
without renumbering lines, compressing gaps or changing tied timestamps.
Apply this offline before any router sees a request, never as a backlog
estimate. To reproduce both denominators, run the root README's audit command.

| Workload | Requests retained / original | Reusable tokens retained / original |
| --- | ---: | ---: |
| Conversation | 6,461 / 12,031 (53.70%) | 8,080,384 / 54,063,104 (14.95%) |
| Tool/agent | 18,326 / 23,608 (77.63%) | 71,506,432 / 115,741,696 (61.78%) |

For each arrival count its longest ordered prefix of **full** 512-token blocks
shared with any earlier prompt in that stream. Sum over arrivals. Compute
this twice from empty history: excluded requests are neither consumers nor
donors in the numerator. This counts potential reuse with unlimited history,
not realized finite-cache hits or physical bytes transferred.
