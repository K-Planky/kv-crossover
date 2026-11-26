# A100 pilot: cost of one worker phase

`a100.csv` is a table of **predictions**, not end-to-end measurements. It uses
[Vidur's published MLP and attention profiles](https://github.com/microsoft/vidur/tree/abae7f63aa857300f5cdc6f5e0d27860cd24721b/data/profiling/compute/a100/meta-llama/Meta-Llama-3-8B)
for the original Llama 3 8B on **one A100** (TP=PP=1). Vidur fits per-operation
models to medians, then combines transformer block costs across 32 layers.
The export calls `get_execution_time` on single-phase batches and rounds its
seconds to the nearest nanosecond. It excludes CPU scheduling and sampling,
embedding and final softmax, queueing, and transfers. The 16-token attention
kernel block is not the trace's 512-token unit of reuse.

| Phase | Batch and new tokens | Cached-context key (tokens already processed) |
| --- | --- | --- |
| Prefill | One request, exact chunk of 1–128 tokens | Round context **up** to a multiple of 64 (zero stays zero); exported only if bucket + chunk ≤ 8192. |
| Decode | 1–8 requests, one token each | Take the integer floor of the mean processed tokens across the batch, then round **up** to a multiple of 64; buckets 64–8192. |

For instance, active contexts 512 and 1536 share the same prediction as two
1024-token contexts. This loses information about imbalance; it is Vidur's
feature, not evidence that arbitrary mixes run equally fast. The value is the
elapsed time for the **whole batch**, not per request. Respect individual
model context limits too. Missing keys have no pilot value. The grid is a
predictor domain, not proof of measurement coverage: for example, the released
attention prefill samples start at 32 new tokens. Hardware/subset qualification
must check coverage before using these predictions for workload comparisons.

## Regenerate and check (CPU only)

From the project root, with Git, `uv` and CPython 3.12 (tested on Linux x86-64
with 3.12.3):

```sh
work=$(mktemp -d)
git clone https://github.com/microsoft/vidur "$work/vidur"
git -C "$work/vidur" checkout --detach abae7f63aa857300f5cdc6f5e0d27860cd24721b
uv venv --python 3.12 "$work/env"
uv pip install --python "$work/env/bin/python" -r profiles/requirements.txt
"$work/env/bin/python" profiles/export_a100.py "$work/vidur" "$work/a100.csv"
cmp profiles/a100.csv "$work/a100.csv"
"$work/env/bin/python" -B profiles/check_a100.py "$work/vidur"
```

After setup the exporter needs neither network access nor model weights. The
[manifest](../experiments/paper.toml) records source and table digests. Each
run fits from scratch in a temporary cache. To bound regeneration cost, we
use Vidur's random forest with one fixed configuration (32 trees, depth 16,
minimum split 2, two folds, serial fits, NumPy seed 42), not its full tuning
grid. `check_a100.py` calls the predictor on a 128-token chunk after 6144
processed tokens and on the unequal-context decode pair above. The shipped
rows are **18,696,410 ns** and **10,884,036 ns**, within half a nanosecond of
the unrounded predictions. These are predictor consistency checks, not GPU
validation or routing outcomes.
