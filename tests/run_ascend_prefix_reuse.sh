#!/usr/bin/env bash
# Run one mechanism at a time inside the existing Ascend test container.
set -eu
stage=${1:?usage: bash tests/run_ascend_prefix_reuse.sh tail|eviction|generated}
: "${VLLM_RLT_OURO_CHECKPOINT:?set VLLM_RLT_OURO_CHECKPOINT to the local Ouro checkpoint directory}"
[[ -f "$VLLM_RLT_OURO_CHECKPOINT/config.json" ]] || { echo "Checkpoint config.json missing" >&2; exit 2; }
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
IFS=, read -ra devices <<< "${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
declare -A seen=()
for card in "${devices[@]}"; do
    [[ "$card" =~ ^[0-9]+$ && ! -v "seen[$card]" ]] || { echo "Invalid or duplicate device: $card" >&2; exit 2; }
    seen[$card]=1
done
case "$stage" in
    tail) cases=(test_aligned_donor_tail_on_ascend) ;;
    eviction) cases=(test_leaf_eviction_on_ascend) ;;
    generated)
        cases=()
        for seed in 114281796 273728779 355234819 421559278 641794792 770929007 797543261 1179450028 1560366951 1727668039; do
            cases+=("test_generated_prefix_two_turn_on_ascend[$seed]")
        done
        ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
mkdir -p "${RLT_RESULTS_DIR:-results-prefix-reuse}"
out=$(mktemp -d "${RLT_RESULTS_DIR:-results-prefix-reuse}/${stage}.XXXXXXXX")
git rev-parse HEAD > "$out/commit.txt"
echo "Results: $out"
workers=${#devices[@]}
if (( workers > ${#cases[@]} )); then workers=${#cases[@]}; fi
pids=()
for ((i=0; i<workers; i++)); do
    card=${devices[i]}
    nodes=()
    for ((j=i; j<${#cases[@]}; j+=workers)); do
        nodes+=("tests/test_ascend_prefix_cache_contract.py::${cases[j]}")
    done
    printf '%s\n' "${nodes[@]}" > "$out/card-$card.cases"
    (
        export ASCEND_RT_VISIBLE_DEVICES=$card
        python -c 'import torch, torch_npu; assert torch.npu.is_available() and torch.npu.device_count() == 1, "expected one visible NPU"'
        python -m pytest --run-npu -q -o junit_family=legacy --junitxml="$out/card-$card.xml" "${nodes[@]}"
    ) > "$out/card-$card.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for log in "$out"/*.log; do echo "$log"; tail -n 2 "$log"; done
exit "$status"
