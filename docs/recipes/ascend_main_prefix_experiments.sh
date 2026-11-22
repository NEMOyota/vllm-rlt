#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
experiment=${1:-}
if [[ $# -gt 0 ]]; then shift; fi
case "$experiment" in
  bf16|benefit)
    exec python -m benchmarks.ascend_main_prefix_matrix "$experiment" --model "${MODEL:-/models/Ouro-1.4B}" "$@"
    ;;
  service)
    exec python -m benchmarks.ascend_main_prefix_service --model "${MODEL:-/models/Ouro-1.4B}" "$@"
    ;;
  *)
    echo "Usage: bash docs/recipes/ascend_main_prefix_experiments.sh {bf16|benefit|service} [options]" >&2
    exit 2
    ;;
esac
