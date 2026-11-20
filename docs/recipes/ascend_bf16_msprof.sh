#!/usr/bin/env bash
# Run from the repository root after activating the Ascend Python/CANN environment.
set -euo pipefail

if (( $# > 2 )); then
    echo "Usage: bash $0 [input-fixture.pt] [new-output-directory]" >&2
    exit 2
fi
fixture=$(realpath -e -- "${1:-results/linear-inputs-20261119T092602.pt}")
recipe="$(dirname "$(realpath -- "${BASH_SOURCE[0]}")")/ascend_bf16_matmul.py"
python_bin=$(python -c 'import sys; print(sys.executable)')
if ! command -v msprof >/dev/null; then
    echo "msprof is missing from PATH in this CANN environment" >&2
    exit 1
fi
if (( $# == 2 )); then
    mkdir -p -- "$(dirname -- "$2")"
    mkdir -- "$2"
    output=$(realpath -- "$2")
else
    mkdir -p results
    output=$(mktemp -d "$(pwd)/results/bf16-msprof-XXXXXXXX")
fi
echo "Saving results in $output"

status=0
for rows in 384 512; do
    run="$output/m$rows"
    capture="$run/capture"
    mkdir -p -- "$capture"
    "$python_bin" "$recipe" "$fixture" --rows "$rows" --result "$run/baseline.json" \
        2>&1 | tee "$run/baseline.log"
    msprof --output="$capture" --task-time=l1 --ai-core=on \
        --aic-mode=task-based --aic-metrics=PipeUtilization \
        "$python_bin" "$recipe" "$fixture" --rows "$rows" --result "$run/profiled.json" \
        2>&1 | tee "$run/msprof.log"

    if [[ ! -s "$run/profiled.json" || ! -s "$run/baseline.json" ]]; then
        echo "M=$rows is missing a numeric result. See $run/*.log" >&2
        status=1
    elif cmp -s "$run/baseline.json" "$run/profiled.json"; then
        echo "M=$rows baseline and profiled values match"
    else
        echo "M=$rows numerical results differ. Compare the two JSONs in $run" >&2
        status=1
    fi

    if [[ -z $(find "$capture" -name 'op_summary_*.csv' -type f -print -quit) ]]; then
        mapfile -t profiles < <(find "$capture" -name 'PROF_*' -type d -print)
        if (( ${#profiles[@]} == 1 )); then
            msprof --export=on --type=text --output="${profiles[0]}" 2>&1 | tee "$run/export.log"
        fi
    fi
    mapfile -t summaries < <(find "$capture" -name 'op_summary_*.csv' -type f -print)
    if (( ${#summaries[@]} )); then
        printf 'OP_SUMMARY %s\n' "${summaries[@]}"
    else
        echo "M=$rows has no op_summary CSV. See $run/msprof.log" >&2
        status=1
    fi
done
exit "$status"
