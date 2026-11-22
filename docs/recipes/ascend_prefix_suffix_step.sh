#!/usr/bin/env bash
# Export the framework and three-way c4_h2 driver beside the server's locally patched FP32 matrix.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
rev=$(git rev-parse --verify "${STEP_REV:-HEAD}^{commit}")
stage=$(mktemp -d "${TMPDIR:-/tmp}/rlt-suffix-step2-XXXXXXXX")
git archive "$rev" vllm_rlt \
    benchmarks/ascend_prefix_mixed_gather.py benchmarks/ascend_prefix_three_plus_one.py \
    benchmarks/ascend_prefix_cold_step.py benchmarks/ascend_prefix_suffix_step.py | tar -x -C "$stage"
# The exported benchmarks directory has no __init__.py. Imports of the matrix and
# workload package resolve to the live server checkout; vllm_rlt resolves to stage.
export PYTHONPATH="$stage:$PWD${PYTHONPATH:+:$PYTHONPATH}"
echo "FRAMEWORK $rev $stage"
echo "MATRIX_CHECKOUT $PWD"
exec python "$stage/benchmarks/ascend_prefix_suffix_step.py" \
    --model "${MODEL:-/models/Ouro-1.4B}" \
    --reference "${REFERENCE:-results/fp32pad-full-20261121T193410.json}" \
    --chunk-rows "${CHUNK_ROWS:-256}" --repeats "${REPEATS:-3}" "$@"
