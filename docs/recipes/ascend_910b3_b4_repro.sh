#!/usr/bin/env bash
# Replay the four historical 910B4 token mismatches on one assigned 910B3.
# Set MODEL and ASCEND_RT_VISIBLE_DEVICES. Use --out DIR for a new result
# directory, or --list to inspect the two workers without starting the NPU.
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python}" "$recipe_dir/../../benchmarks/ascend_prefix_pr1_suite.py" npu "$@" \
  --chip 910B3 --only environment --only b4-cases-bf16 --only b4-cases-fp32
