# Ouro on Ascend

Run Ouro-1.4B on one Ascend NPU with BF16, greedy decoding and four recurrent
loops. The Python runtime uses Torch's `scaled_dot_product_attention` through
the `ascend_eager` backend, matching HF's SDPA entry point. Torch-NPU selects
the device kernel according to the input shape, mask and hardware.
The example enables the optional asynchronous engine plan.

## Prepare the environment and model

Use `main`, which includes Ascend support added after `v0.2.0`. Install CANN and
the matching driver and firmware for your Ascend hardware first.

### Install PyTorch and Torch-NPU

The following commands use CANN 9.0.0 with PyTorch 2.10.0 and Torch-NPU 2.10.0.
For other versions, consult the
[Ascend compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md#torchnpu-version-mapping-table).
Activate your Python environment and load the CANN environment.

```bash
source /path/to/npu-environment/bin/activate
source /path/to/cann/set_env.sh
```

On x86-64, install PyTorch from the CPU wheel index.

```bash
python -m pip install 'torch==2.10.0+cpu' --index-url https://download.pytorch.org/whl/cpu
```

On Arm64 (`aarch64`), use the following PyTorch command.

```bash
python -m pip install 'torch==2.10.0'
```

On either architecture, install `torch-npu` and check that PyTorch can see the NPU.
The package is imported as `torch_npu` in Python.

```bash
python -m pip install 'torch-npu==2.10.0' pyyaml setuptools
python -c 'import torch, torch_npu; assert torch.npu.is_available(), "Ascend NPU unavailable"'
```

### Install vllm-rlt and set the checkpoint path

Install the project dependencies in that environment. Use the checkpoint's
config and byte-level tokenizer.

```bash
git clone --branch main https://github.com/InferMatrixLab/vllm-rlt.git
cd vllm-rlt
python -m pip install --no-deps --no-build-isolation -e .
export MODEL=/path/to/prepared/Ouro-1.4B
export ASCEND_RT_VISIBLE_DEVICES=0  # Replace 0 with your assigned NPU.
```

## Serve fixed-depth requests

This example sizes KV for 32 resident requests with 128 input and 128 output
tokens each. Set `--num-blocks` explicitly and adjust it for the expected
sequence lengths, resident request count and available device memory. For
whole-cohort prefill, the chunk size must fit each prompt and the batched token
budget must fit the sum of prompt lengths in the cohort.

```bash
TASK_QUEUE_ENABLE=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m vllm_rlt.entrypoints.serve \
  --model "$MODEL" --tokenizer "$MODEL" --served-model-name ouro \
  --device npu --dtype bfloat16 --attention-backend ascend_eager \
  --mode refill --exit-mode ouro_delayed \
  --async-scheduling --single-stream --no-uva \
  --block-size 16 --num-blocks 2064 --max-num-seqs 32 \
  --max-num-batched-tokens 8192 --prefill-chunk-size 1024 \
  --max-prefill-batches-before-decode 32 --host 127.0.0.1 --port 8123
```

For synchronous eager execution, omit `--async-scheduling`.

Wait for `/health` to return HTTP 200, then send a completion. Add
`"stream": true` for server-sent events.

```bash
curl --fail http://127.0.0.1:8123/health
curl --fail http://127.0.0.1:8123/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ouro","prompt":"The capital of France is","max_tokens":8,"temperature":0,"min_loops":4,"max_loops":4,"exit_threshold":1}'
```

## Reuse shared prompt prefixes on a single NPU

For repeated system prompts or documents, opt in with
`--enable-prefix-caching`. Ouro can reuse complete KV pages from a previous
request, including a request that has already finished. A continuation
computes the remaining prompt tokens. Each of the four loops uses its own KV
pages. Prefix caching is off by default.

Use synchronous, single-stream `ascend_eager` execution with BF16 weights,
the `ouro` exit mode, ordinary prefill and fully reserved `last_exited` KV.
The following example allows two concurrent requests. Keep each prompt within
128 tokens for this example and send the fixed-depth greedy sampling parameters
shown above.

```bash
python -m vllm_rlt.entrypoints.serve \
  --model "$MODEL" --tokenizer "$MODEL" --served-model-name ouro \
  --device npu --dtype bfloat16 --attention-backend ascend_eager \
  --mode refill --exit-mode ouro --single-stream --no-uva \
  --enable-prefix-caching --block-size 16 --num-blocks 128 \
  --max-num-seqs 2 --max-num-batched-tokens 256 --prefill-chunk-size 128 \
  --host 127.0.0.1 --port 8123
```

**BF16 output can change on a cache hit.** Tests on Ascend 910B3 and 910B4
found prompts for which enabling reuse changed a greedy output token, even
though page and mask checks passed. Keep caching disabled when the application
requires token-for-token agreement with uncached inference. Latency depends on
the suffix length and on the other requests in the same batch.

<a id="6-fixed-depth-asynchronous-engine-plan"></a>
## Fixed-depth asynchronous engine plan

- **Scheduling.** Stream events track completion and protect buffer and request
  lifetimes. Decode PRELUDE and each recurrent loop are separate submissions,
  so the scheduler can select a new batch at each boundary.
- **Reuse and batching.** Ascend uses the shared packed-prefill metadata by
  default with `ascend_eager`. Use `--no-packed-prefill` to select the per-query
  layout. Greedy sampling and compatible prefill are batched. Full prefill
  uses causal SDPA.
  Cached chunks and ragged decode pass boolean masks to preserve each row's
  visible KV range.
- **Paged KV.** Each loop writes to the page pool through PyTorch indexed
  assignment using its own block table.
  The Ascend backend gathers the visible KV rows for Torch SDPA.

The example uses private, fully reserved `last_exited` KV. Keep graphs,
static buffers, padding, prefix caching, incremental KV allocation, preemption,
speculation and prefill/decode disaggregation disabled in this configuration.
Greedy parity remains unqualified for arbitrary chunked or mixed-length inputs
and adaptive exit.

<a id="8-profile-guided-high-batch-scheduling"></a>
## Performance and correctness

A higher prefill quota lets more requests reach decode together and can delay
decode for existing requests. The scheduler default is 1.

### Full unified W1-W7 validation

This historical run used the C++ host executor and explicit eager attention
on both sides of the comparison. For the SDPA configuration above, compare
against HF SDPA with the same weights, input IDs, output lengths and concurrency.
Check the actual prefill and decode kernels on both sides with a short profile.
Measure throughput, TTFT and TPOT with the profiler disabled. Separate
synchronous and asynchronous vllm-rlt runs to measure the effect of scheduling.

Runtime `7b4537f` used the same budget 8192, chunk 1024 and prefill quota 32 for
all seven [unified execution workloads](../unified_execution.md#protocol-and-comparison-rules).
The comparison ran Ouro-1.4B BF16 with four fixed loops and greedy decoding on
one 64 GiB Ascend 910B3 per run, with CANN 9, Torch/Torch-NPU 2.10 and
Transformers 4.54.1. The reference used official HF eager at revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1` with a DynamicCache adapter.

All 42 runs completed with 1,422 successful requests and zero runtime failures.
Each case used three unprofiled repetitions per arm on the same card and CPU
affinity. Each repetition admitted closed-loop traffic for 60 seconds and then
fully drained. Throughput includes drain and excludes startup, compilation and
warmup. Timing uses the in-process engine actor. W1–W5 ignore EOS. W6/W7 use
natural EOS with a 512-token cap and a fixed ShareGPT selection.

| Workload | Input / output / clients | Residents per arm | HF eager tok/s | Engine tok/s | vs HF | Complete output matches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| W1 | 128 / 128 / 1 | 1 | 6.334 [5.284–6.496] | 9.464 [9.403–10.394] | +49.42% | 15/15 |
| W2 | 128 / 128 / 32 | 32 | 201.314 [200.099–204.437] | 305.496 [293.376–316.662] | +51.75% | 480/480 |
| W3 | 4096 / 128 / 1 | 1 | 6.254 [6.206–6.268] | 7.569 [6.110–9.215] | +21.03% | 12/12 |
| W4 | 4096 / 128 / 32 | 4 | 21.211 [21.012–21.580] | 31.312 [31.135–31.405] | +47.62% | 120/132 |
| W5 | 1024 / 512 / 8 | 8 | 50.885 [50.234–51.362] | 79.791 [76.843–80.666] | +56.80% | 48/48 |
| W6 | ShareGPT / ≤512 / 8 | 8 | 26.343 [26.060–26.385] | 37.992 [37.906–38.879] | +44.22% | 18/27 |
| W7 | ShareGPT / ≤512 / 32 | 8 | 36.341 [35.576–36.931] | 36.194 [35.808–36.679] | -0.41% | 45/99 |

Rates are medians [min–max]. Resident limits reflect single-device memory
capacity and apply equally to both arms. Complete output checks compare token IDs
and finish reasons against timed or supplemental HF references.

**W1, W2, W3 and W5 passed the covered-input output checks.** W4, W6 and W7
still differ from HF, so their throughput figures are diagnostic. W1–W5 used
owned KV and native dispatch throughout timing. W6/W7 used paged KV and Python
execution. W7 throughput ranges overlap, and its peak allocated memory was
30.06 GiB for the engine versus 19.54 GiB for HF. At W2, mean TPOT fell from
157.04 to 102.73 ms while TTFT p95 rose from 315.98 to 616.33 ms.

## Run regression checks

With an assigned NPU visible and `MODEL` set as above:

```bash
VLLM_RLT_OURO_CHECKPOINT="$MODEL" python -m pytest -q --run-npu \
  tests/test_ascend_eager.py tests/test_ascend_kv.py \
  tests/test_ascend_decode_scheduling.py \
  tests/test_ascend_checkpoint.py tests/test_ascend_serving.py \
  tests/test_ascend_prefix_cache_contract.py
```

For broader evaluation, see [accuracy](../accuracy.md) and
[serving benchmarks](../serving.md).

## Experimental 16-token paged prefill

Add `--ascend-paged-prefill` to the synchronous prefix-cache serving command above
to opt in on Ouro-1.4B. When a packed prefill batch has multiple requests and at
least one multi-token query, cold requests of different lengths and requests
with cached prefixes can share one TND fused attention call per model layer and
loop. Equal-length cold prompts use the existing batched SDPA path. Single
requests and all-one-token batches also use the existing backend.

For example, for one 128-token hit in a batch of four 144-token prompts, TND
takes 448 query rows with cumulative ends `[16, 160, 304, 448]` and KV lengths
`[144, 144, 144, 144]`. Each request reads its own table at its current
recurrence depth. Right-aligned causal masking lets the first suffix row see
the 128 cached tokens and itself. A 16-token page is the allocation unit.
A request can attend to its entire causal history across many pages.

The fused path follows the Q-boundary and page-table convention of
[vLLM-Ascend's cache-hit attention](https://github.com/vllm-project/vllm-ascend/blob/4098f7a51440bc4e3384cf5a1ff0fc7eba4067a4/vllm_ascend/attention/attention_v1.py#L903-L909).
RLT stores each K and V pool as `[block, layer, token, head, channel]`. A view
exposes the original pool storage and page `block * num_layers + layer`
selects the required layer. The cache manager validates ownership and the
continuous written prefix before dispatch.

### Run on 910B3 and 910B4

With the original CANN 9.0.0 environment active, run the following from this
checkout on each machine. Set `MODEL` to the local Ouro-1.4B checkpoint.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest --run-npu -m npu -q \
  tests/test_ascend_varlen_prefill.py

ASCEND_RT_VISIBLE_DEVICES=0 python -m benchmarks.ascend_varlen_prefill \
  --checkpoint "$MODEL" --cases c4_h1 mixed --rounds 1 --with-cache-off \
  --output results/varlen-smoke.json
```

For an eight-card sweep, run one scenario per card. Each comparison stays on
one physical card, mapped to process-local `npu:0`. Run inside Bash.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cases=(c4_h1 mixed cold_ragged one_token_suffix w2_shape cold_uniform single long_suffix)
mkdir -p results/varlen
pids=()
for i in "${!cases[@]}"; do
  ASCEND_RT_VISIBLE_DEVICES="$i" python -m benchmarks.ascend_varlen_prefill \
    --checkpoint "$MODEL" --cases "${cases[i]}" --rounds 3 --max-tokens 16 \
    --seed 797543261 --with-cache-off --output "results/varlen/${cases[i]}.json" \
    >"results/varlen/${cases[i]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for i in "${!pids[@]}"; do
  wait "${pids[i]}" || { echo "Review ${cases[i]} on card $i"; status=1; }
done
test "$status" -eq 0
```

`legacy` and `paged` both enable prefix caching. They differ only in the
attention routing switch. The optional `off` arm disables both features.
Each arm builds a fresh engine, warms it with disjoint inputs, primes the same
donor prompts, and times the target batch through final token delivery.
Arm order rotates across rounds. Warmup, donor generation and model loading
are excluded. `w2_shape` uses 32 prompts of 128 tokens with 96-token hits;
the default generates 16 tokens per request in one burst. The W2 throughput
table above uses a 60-second closed-loop workload and 128 output tokens.

Read `prefill_batches`, `hits`, `prefill_sdpa_calls` and `fia_calls` first.
The counts refer to framework attention API calls. In the one-hot/three-cold
case, a single target batch on a 24-layer, four-loop model should give
384 legacy SDPA calls and 96 paged FIA calls. Compare the paired TTFT P50
saving in milliseconds and the percentage change in output tokens/s across
rounds. TPOT P50, per-arm output IDs and the first mismatch per request are
also recorded. A positive TTFT saving means a shorter wait for the first token.
A positive throughput change means more output tokens delivered per second.
A mismatch or a dispatch/hit-count failure gives a nonzero exit after saving
the results. Errors during execution identify the interrupted arm in the JSON.

### Measure prefix-cache benefit on a fixed backend

Both arms of this experiment use synchronous BF16 Ouro, 16-token pages and
`ascend_paged_prefill=True`. Only `enable_prefix_caching` changes. The same
dispatch policy can select different kernels as a consequence of cache hits.
In particular, equal-length cold W2 prompts take batched SDPA, while multiple
hot suffixes can take paged FIA. The earlier `legacy` versus `paged` comparison
instead measured the effect of changing the attention implementation with the
cache already enabled.

This covers ordinary packed prefill on the RLT single-stream engine. In
vLLM-Ascend, the corresponding backend also has separate cold, cache-hit,
decode and chunked-prefill states, including dispatch for a mixed prefill/decode
step and graph replay. Its declared kernel page size at the inspected revision
is 128. Those additional paths have independent execution requirements.
[Backend states and page size](https://github.com/vllm-project/vllm-ascend/blob/4098f7a51440bc4e3384cf5a1ff0fc7eba4067a4/vllm_ascend/attention/attention_v1.py#L141-L150)
· [Mixed-step dispatch](https://github.com/vllm-project/vllm-ascend/blob/4098f7a51440bc4e3384cf5a1ff0fc7eba4067a4/vllm_ascend/attention/attention_v1.py#L1038-L1061)
· [Graph replay](https://github.com/vllm-project/vllm-ascend/blob/4098f7a51440bc4e3384cf5a1ff0fc7eba4067a4/vllm_ascend/attention/attention_v1.py#L619-L662).

Run in the original CANN environment on each host, using an idle physical card
and its local Ouro-1.4B checkpoint. Use a new output filename for each run.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ASCEND_RT_VISIBLE_DEVICES=0 python -m benchmarks.ascend_prefix_benefit \
  --checkpoint "$MODEL" --cases W2-cold W2-quarter W2 --rounds 4 \
  --seed 797543261 --output results/prefix-benefit-w2.json
```

| Case | Requests hitting 96 tokens | Target hit tokens / 4096 | Expected target prefill rows with cache |
| --- | ---: | ---: | ---: |
| W2-cold | 0 / 32 | 0 | 4096 |
| W2-quarter | 8 / 32 | 768 | 3328 |
| W2 | 32 / 32 | 3072 | 1024 |

All three cases generate exactly 4096 output tokens per arm. The default is a
fixed cohort, timed through drain, with four alternating same-card pairs. The
engine and KV pool are recreated for each arm, then warmed with disjoint inputs.
The first pair warms each arm for the complete output length. Later pairs use
a two-token warmup per engine.
Both arms run the same donor before measuring the targets. The measured ON
primer must start cold. A trial checks actual per-request hits, scheduled
prefill rows and output lengths. Different generated token IDs are counted as
an observation. OOMs and incomplete requests leave the experiment incomplete.

The JSON contains the per-arm throughput, TTFT P50/P95, TPOT P50, wall time and
attention API call counts. For each case, read the median of the round-by-round
ON/OFF throughput changes, its range and the number of faster rounds. The
primary measurement covers the targets with a resident donor. The companion
`on_with_primer_vs_off_pct` also charges the ON arm for the complete measured
donor run, once per cohort. A TTFT improvement can coexist with unchanged or
lower output throughput.

The same command accepts `--cases W1 W2 W3 W4 W5`. This uses the historical
PR1 W-series input/output lengths, client counts, resident limits and nominal
75% shared prefix, with a fixed cohort and an explicit prefill chunk limit
(default 256). W4 submits 32 clients and admits at most four at a time. To use
five idle cards concurrently, launch one case per card with its own result file.

```bash
cases=(W1 W2 W3 W4 W5)
mkdir -p results/prefix-benefit
pids=()
for i in "${!cases[@]}"; do
  ASCEND_RT_VISIBLE_DEVICES="$i" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m benchmarks.ascend_prefix_benefit --checkpoint "$MODEL" \
    --cases "${cases[i]}" --rounds 4 --seed 797543261 \
    --output "results/prefix-benefit/${cases[i]}.json" \
    >"results/prefix-benefit/${cases[i]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for i in "${!pids[@]}"; do
  wait "${pids[i]}" || { echo "Review ${cases[i]} on card $i"; status=1; }
done
test "$status" -eq 0
```
