#!/usr/bin/env bash
# Run from the experiment checkout with the usual Ascend environment active.
set -euo pipefail

usage() {
    echo "Usage: bash docs/recipes/ascend_matmul_v3.sh {source|probe|failures|full}"
    echo "FIXTURE defaults to results/linear-inputs-20261119T092602.pt. Set MODEL for matrix runs."
    echo "Optional: RESULTS_ROOT (default results), REPEATS=5, WARMUPS=1, OUTPUT_TOKENS=16."
}
stage=${1:-full}
case "$stage" in
    -h|--help) usage; exit 0 ;;
    source|probe|failures|full) ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

root=${RESULTS_ROOT:-results}
mkdir -p -- "$root"
output=$(mktemp -d "$(realpath -- "$root")/matmul-v3-$stage-XXXXXXXX")
echo "RESULTS $output"
if [[ $stage == source ]]; then
    sha=8ac78fb1700262c7c7a97f36b58fe587b9917c8a
    url=${V3_SOURCE_REMOTE:-https://github.com/hicann/ops-nn.git}
    git -C "$output" init -q
    git -C "$output" remote add origin "$url"
    git -C "$output" sparse-checkout set matmul/mat_mul_v3 matmul/common
    git -C "$output" fetch --filter=blob:none --depth=1 origin "$sha"
    git -C "$output" checkout --detach FETCH_HEAD
    echo "SOURCE $output/matmul/mat_mul_v3"
    exit 0
fi

fixture=$(realpath -e -- "${FIXTURE:-results/linear-inputs-20261119T092602.pt}")
if [[ $stage != probe && ! -d ${MODEL:-} ]]; then
    echo "Set MODEL to the local Ouro checkpoint directory." >&2
    exit 2
fi
command -v msprof >/dev/null || { echo "Activate CANN so msprof is in PATH." >&2; exit 2; }
python_bin=$(python -c 'import sys; print(sys.executable)')
module=benchmarks.ascend_matmul_v3
library="$output/rlt_matmul_v3.so"
exec > >(tee "$output/run.log") 2>&1
"$python_bin" -m "$module" build "$library"
profile() {
    local capture_dir=$1
    shift
    mkdir "$capture_dir"
    msprof --output="$capture_dir" --task-time=l1 --ai-core=on --aic-mode=task-based \
        --aic-metrics=PipeUtilization "$python_bin" -m "$module" "$@"
    if [[ -z $(find "$capture_dir" -type f -name 'op_summary*.csv' -print -quit) ]]; then
        local -a profiles
        local profile_dir
        mapfile -t profiles < <(find "$capture_dir" -type d -name 'PROF_*')
        for profile_dir in "${profiles[@]}"; do
            msprof --export=on --type=text --output="$profile_dir"
        done
    fi
}
for arm in native v3; do
    capture="$output/$arm-profile"
    op_args=()
    verify_args=()
    if [[ $arm == v3 ]]; then op_args=(--library "$library"); verify_args=(--v3); fi
    profile "$capture" probe "$fixture" "${op_args[@]}" --out "$output/$arm-probe.json"
    [[ -s "$output/$arm-probe.json" ]] || { echo "Missing $arm numeric probe output" >&2; exit 2; }
    "$python_bin" -m "$module" verify "$capture" "${verify_args[@]}" --out "$output/$arm-dispatch.json"
done

if [[ $stage != probe ]]; then
    selection=(--suite full)
    if [[ $stage == failures ]]; then
        selection=(--case p1024_s8 --case total512_p384 --case c4_h4)
    fi
    for arm in native v3; do
        op_args=()
        if [[ $arm == v3 ]]; then op_args=(--kv-v3-library "$library"); fi
        status=0
        "$python_bin" -m benchmarks.ascend_prefix_matrix --model "$MODEL" \
            --repeats "${REPEATS:-5}" --warmups "${WARMUPS:-1}" --output-tokens "${OUTPUT_TOKENS:-16}" \
            "${selection[@]}" "${op_args[@]}" --out "$output/$arm-matrix.json" || status=$?
        # A completed matrix returns 1 for numerical mismatches, also on the native baseline.
        if (( status != 0 )); then
            if (( status != 1 )); then exit "$status"; fi
            "$python_bin" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["state"] != "complete")' \
                "$output/$arm-matrix.json"
        fi
    done
    capture="$output/v3-matrix-shapes-profile"
    profile "$capture" replay-shapes "$library" "$output/v3-matrix.json"
    "$python_bin" -m "$module" verify "$capture" --v3 --matrix "$output/v3-matrix.json" \
        --out "$output/v3-matrix-dispatch.json"
fi
"$python_bin" -m benchmarks.ascend_matmul_v3_compare "$output" "$stage"
