#!/usr/bin/env bash
set -euo pipefail
usage() { echo "Usage: run.sh prepare|baseline|scan|compare|profile|window|dump|replay|op|pack CHIP CASE [arm] [precision]" >&2; exit 2; }
stage=${1:-}; chip=${2:-}; name=${3:-}; arm=${4:-all}; precision=${5:-bf16}
d=$(cd "$(dirname "$0")" && pwd)
: "${OP_WORK:?set OP_WORK to a dedicated absolute experiment directory}"
mkdir -p "$OP_WORK"
work=$(cd "$OP_WORK" && pwd)
tree="$work/runtime"
out="$work/evidence"
if [[ "$stage" == prepare ]]; then exec python "$d/prepare.py" "$tree"; fi
[[ "$chip" == 910B3 || "$chip" == 910B4 ]] || usage
case "$name" in c4_h1) card=3 ;; total512_p256) card=0 ;; *) usage ;; esac
[[ "$precision" == bf16 || ( "$precision" == fp32-linear && "$name" == total512_p256 ) ]] || usage
export ASCEND_RT_VISIBLE_DEVICES="$card" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$tree${PYTHONPATH:+:$PYTHONPATH}"
cd "$tree"
base="$out/$chip/$name/$precision"
report=(python "$d/report.py" --out "$out" --chip "$chip" --case "$name")
trap '"${report[@]}"' EXIT
if [[ "$stage" == compare ]]; then
    if [[ -n "${OP_SELECT_FROM:-}" ]]; then cp "$OP_SELECT_FROM" "$base/requested.json"; fi
    exit 0
fi
if [[ "$stage" == pack ]]; then
    "${report[@]}"
    python "$d/package.py" "$out/$chip/$name" "$work/$chip-$name-evidence.tar.gz"
    exit 0
fi
if [[ "$stage" == replay || "$stage" == op ]]; then
    replay=(python "$d/replay_op.py" "$base/dump/off/operator.pt" "$base/dump/reuse/operator.pt")
    if [[ "$stage" == replay ]]; then
        python - "$base" <<'PYGATE'
import json, pathlib, sys
base=pathlib.Path(sys.argv[1])
for arm in ("off","reuse"):
    if not json.loads((base/"dump"/arm/"result.json").read_text())["matches_baseline"]:
        raise SystemExit("dump changed the uninstrumented result")
PYGATE
        "${replay[@]}" --out "$base/replay.json"; exit
    fi
    [[ "$arm" == off || "$arm" == reuse ]] || usage
    python "$d/collect_msprof.py" op --base "$base" --arm "$arm" --folder "$base/op-$arm" -- "${replay[@]}" --only "$arm"
    exit
fi
: "${MODEL:?set MODEL to absolute original checkpoint}" "${ORIGINAL_RESULTS:?set ORIGINAL_RESULTS to original results-910Bx directory}"
runner=(python "$d/capture.py" --chip "$chip" --case "$name" --model "$MODEL" --report "$ORIGINAL_RESULTS" --out "$out" --precision "$precision" --cann "${OP_CANN_VERSION:-9.0.0}")
if [[ -n "${OP_INPUT_IDS:-}" ]]; then runner+=(--input-ids "$OP_INPUT_IDS"); fi
if [[ "$stage" == baseline ]]; then "${runner[@]}" --mode baseline; exit; fi
case "$stage" in scan|profile|window|dump) ;; *) usage ;; esac
arms=("$arm")
if [[ "$arm" == all ]]; then
    if [[ "$stage" == dump ]]; then arms=(off reuse); else arms=(off miss reuse); fi
fi
for a in "${arms[@]}"; do
    if [[ "$stage" == profile ]]; then
        python "$d/collect_msprof.py" app --base "$base" --arm "$a" --folder "$base/msprof-$a" -- "${runner[@]}" --mode profile --arm "$a"
    else
        "${runner[@]}" --mode "$stage" --arm "$a"
    fi
done
