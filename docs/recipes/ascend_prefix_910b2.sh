#!/usr/bin/env bash
# Run from the vllm-rlt repository root in the existing CANN/Torch-NPU environment.
set -euo pipefail

port=${PORT:-8123}
case "${1:-}" in
    serve|test)
        : "${MODEL:?Set MODEL to the local Ouro-1.4B checkpoint directory}"
        if [[ ! -d "$MODEL" ]]; then
            echo "Local checkpoint directory missing: $MODEL" >&2
            exit 2
        fi
        ;;
    smoke) ;;
    *)
        echo "Usage: bash docs/recipes/ascend_prefix_910b2.sh {test|serve|smoke}" >&2
        exit 2
        ;;
esac

case "$1" in
    test)
        python -m pytest -q --run-npu -m npu tests/test_ascend_prefix_cache.py
        exec python -m benchmarks.ascend_prefix_cache --model "$MODEL" \
            --prefix-tokens 128 --output-tokens 16 --repeats 5
        ;;
    serve)
        exec python -m vllm_rlt.entrypoints.serve \
            --model "$MODEL" --tokenizer "$MODEL" --served-model-name ouro \
            --device npu --dtype bfloat16 --attention-backend ascend_eager \
            --enable-prefix-caching --kv-layout last_exited --num-blocks 128 --block-size 16 \
            --mode refill --max-num-seqs 1 --max-num-batched-tokens 256 --prefill-chunk-size 256 \
            --host 127.0.0.1 --port "$port"
        ;;
    smoke)
        exec python - "$port" <<'PY'
import json
import sys
from urllib.request import Request, urlopen

base = f"http://127.0.0.1:{int(sys.argv[1])}"
with urlopen(f"{base}/health", timeout=10) as response:
    health = json.load(response)
if health.get("status") != "ready":
    sys.exit(f"Unexpected health response: {health}")

prefix = "Paris is in France. Rome is in Italy. Tokyo is in Japan. " * 2
for question in ("Which country contains Rome?", "Which country contains Tokyo?"):
    body = {
        "model": "ouro",
        "prompt": prefix + question,
        "max_tokens": 8,
        "temperature": 0,
        "min_loops": 4,
        "max_loops": 4,
        "exit_threshold": 1,
        "ignore_eos": True,
    }
    request = Request(
        f"{base}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    choice = result["choices"][0]
    usage = result["usage"]
    if choice["finish_reason"] != "length" or usage["completion_tokens"] != 8:
        sys.exit(f"Unexpected completion: {result}")
    print(f"HTTP OK prompt_tokens={usage['prompt_tokens']} completion_tokens=8")
PY
        ;;
esac
