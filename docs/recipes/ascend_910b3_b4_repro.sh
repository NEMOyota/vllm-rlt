#!/usr/bin/env bash
# Replay the four historical 910B4 token mismatches on one assigned 910B3.
# Set MODEL and ASCEND_RT_VISIBLE_DEVICES. Use --out DIR for a new result
# directory, --fp16-only to run the native FP16 supplement, or --list to preview.
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stages=(--only b4-cases-bf16 --only b4-cases-fp32)
options=()
for option in "$@"; do
  case "$option" in
    --fp16-only) stages=(--only b4-cases-fp16) ;;
    *) options+=("$option") ;;
  esac
done
exec "${PYTHON:-python}" "$recipe_dir/../../benchmarks/ascend_prefix_pr1_suite.py" npu \
  "${options[@]}" --chip 910B3 --only environment "${stages[@]}"
