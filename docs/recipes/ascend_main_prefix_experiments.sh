#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
experiment=${1:-}
if [[ $# -gt 0 ]]; then shift; fi
model=${MODEL:-/models/Ouro-1.4B}
case "$experiment" in
  fp32-all)
    if [[ $# -ne 0 ]]; then
      echo "Use the individual experiments for custom CLI arguments." >&2
      exit 2
    fi
    rows=${CHUNK_ROWS:-256}
    pairs=${BENEFIT_REPEATS:-6}
    if [[ ! "$rows" =~ ^[1-9][0-9]*$ || ! "$pairs" =~ ^[1-9][0-9]*$ || ! -d "$model" ]]; then
      echo "Set MODEL to a local checkpoint and CHUNK_ROWS and BENEFIT_REPEATS to positive integers." >&2
      exit 2
    fi
    mkdir -p results
    folder=$(mktemp -d "$repo_root/results/main-prefix-fp32-r$rows-XXXXXXXX")
    echo "RESULTS $folder"
    printf 'stage\texit\n' > "$folder/stages.tsv"
    status=0
    stage() {
      local name=$1 code=0
      shift
      "$@" 2>&1 | tee "$folder/$name.log" || code=$?
      printf '%s\t%s\n' "$name" "$code" | tee -a "$folder/stages.tsv"
      if (( code >= 128 )); then exit "$code"; fi
      if (( code > status )); then status=$code; fi
    }
    common=(--model "$model" --linear-fp32-chunk "$rows")
    stage combo python -m benchmarks.ascend_main_prefix_combo "${common[@]}" --repeats 2 --out "$folder/combo.jsonl"
    stage full python -m benchmarks.ascend_main_prefix_matrix fp32 "${common[@]}" --repeats 3 --out-dir "$folder/full"
    stage benefit python -m benchmarks.ascend_main_prefix_matrix benefit "${common[@]}" --all-cases --repeats "$pairs" --out-dir "$folder/benefit"
    stage burst python -m benchmarks.ascend_main_prefix_service "${common[@]}" --out-dir "$folder/burst"
    stage paced python -m benchmarks.ascend_main_prefix_service "${common[@]}" --rps 3 --out-dir "$folder/paced"
    exit "$status"
    ;;
  combo)
    exec python -m benchmarks.ascend_main_prefix_combo --model "$model" "$@"
    ;;
  hot-long)
    exec python -m benchmarks.ascend_main_hot_long --model "$model" "$@"
    ;;
  bf16|fp32|benefit)
    exec python -m benchmarks.ascend_main_prefix_matrix "$experiment" --model "$model" "$@"
    ;;
  service)
    exec python -m benchmarks.ascend_main_prefix_service --model "$model" "$@"
    ;;
  *)
    echo "Usage: bash docs/recipes/ascend_main_prefix_experiments.sh {fp32-all|combo|fp32|bf16|benefit|service|hot-long} [options]" >&2
    exit 2
    ;;
esac
