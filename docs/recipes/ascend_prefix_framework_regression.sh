#!/usr/bin/env bash
# Export the framework and regression driver beside the server's locally patched FP32 matrix.
# With "miss-control", run ten paired c4_h2 misses; pass --help for the trace option.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
rev=$(git rev-parse --verify "${STEP_REV:-HEAD}^{commit}")
driver=ascend_prefix_framework_regression
default_repeats=3
if [[ ${1:-} == miss-control ]]; then
    shift
    driver=ascend_prefix_miss_control
    default_repeats=10
fi
stage=$(mktemp -d "${TMPDIR:-/tmp}/rlt-framework-regression-XXXXXXXX")
git archive "$rev" vllm_rlt \
    benchmarks/ascend_prefix_mixed_gather.py benchmarks/ascend_prefix_three_plus_one.py \
    benchmarks/ascend_prefix_cold_step.py benchmarks/ascend_prefix_scenarios.py \
    benchmarks/ascend_prefix_framework_regression.py "benchmarks/$driver.py" | tar -x -C "$stage"
# The exported benchmarks directory has no __init__.py. Imports of the matrix and
# workload package resolve to the live server checkout; vllm_rlt resolves to stage.
export RLT_FRAMEWORK_REVISION="$rev"
export PYTHONPATH="$stage:$PWD${PYTHONPATH:+:$PYTHONPATH}"
echo "FRAMEWORK $rev $stage"
echo "MATRIX_CHECKOUT $PWD"
exec python "$stage/benchmarks/$driver.py" \
    --model "${MODEL:-/models/Ouro-1.4B}" \
    --reference "${REFERENCE:-results/fp32pad-full-20261121T193410.json}" \
    --chunk-rows "${CHUNK_ROWS:-256}" --repeats "${REPEATS:-$default_repeats}" "$@"
