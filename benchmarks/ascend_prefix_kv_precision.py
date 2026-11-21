"""Compare native and selectively FP32 Ouro K/V on the same prefix-cache requests."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.ascend_grouped_kv_single import check_trial, write_record
from benchmarks.ascend_prefix_projection import fp32_kv_projections
from benchmarks.ascend_prefix_scenarios import ARMS, suites
from benchmarks.ascend_prefix_workload import run_trial
from vllm_rlt.models import OuroForCausalLM

# The three failures reported for the K/V-only FP32 experiment.
DEFAULT_CASES = ("p1024_s8", "p128_s128", "c4_h4")


def run_precision_arms(model, backend, case, output_tokens, seed, *, work=None, arm_offset=0):
    """Start a fresh engine for each arm and snapshot the active projection counters."""
    counters = {} if work is None else work
    trial, work_by_arm = {}, {}
    offset = arm_offset % len(ARMS)
    for arm in ARMS[offset:] + ARMS[:offset]:
        for counts in counters.values():
            counts.clear()
        trial[arm] = run_trial(model, backend, case, arm, output_tokens, seed)
        for name, counts in counters.items():
            if not counts["calls"]:
                raise RuntimeError(f"{case.name}/{arm}: {name} did not run in FP32")
        work_by_arm[arm] = {name: dict(counts) for name, counts in counters.items()}
    return trial, work_by_arm


def run_pair(model, backend, case, output_tokens, seed, projection, layers, chunk_rows):
    native, _ = run_precision_arms(model, backend, case, output_tokens, seed)
    with fp32_kv_projections(model, projection, layers, chunk_rows) as work:
        candidate, work_by_arm = run_precision_arms(
            model, backend, case, output_tokens, seed, work=work
        )
    return {"native": native, "kv_fp32": candidate}, work_by_arm


def run_cases(model, backend, cases, args, stream):
    failed, cold_changed, invalid = set(), set(), set()
    for case in cases:
        reference = None
        print(f"Starting {case.name}", flush=True)
        for repeat in range(1, args.repeats + 1):
            try:
                trials, work = run_pair(
                    model,
                    backend,
                    case,
                    args.output_tokens,
                    args.seed,
                    args.projection,
                    args.layers,
                    args.chunk_rows,
                )
                if reference is None:
                    reference = trials["native"]["off"]
                checks = check_trial(case, trials, reference, candidate="kv_fp32")
            except (Exception, KeyboardInterrupt) as exc:
                write_record(
                    stream, kind="incomplete", case=case.name, repeat=repeat, error=repr(exc)
                )
                raise
            if checks["controls"]:
                parity = "INVALID"
                invalid.add(case.name)
            elif checks["kv_fp32_parity_diffs"]:
                parity = "FAIL"
                failed.add(case.name)
            else:
                parity = "PASS"
            if checks["native_off_vs_kv_fp32_off"]:
                cold_changed.add(case.name)
            info = dict(case=case.name, repeat=repeat, cache_parity=parity, **checks)
            print("KV_FP32_CASE", json.dumps(info), flush=True)
            write_record(stream, kind="trial", **info, projection_work=work, trials=trials)
    review = dict(
        cases=len(cases),
        repeats=args.repeats,
        cache_parity_failed=sorted(failed),
        native_cold_changed=sorted(cold_changed),
        invalid=sorted(invalid),
    )
    write_record(stream, kind="review", **review)
    print("KV_FP32_REVIEW", json.dumps(review), flush=True)
    return review


@torch.inference_mode()
def main():
    presets = {case.name: case for case in suites()["full"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="local Ouro checkpoint")
    parser.add_argument("--projection", choices=("kv", "k", "v"), default="kv")
    parser.add_argument(
        "--layers", type=int, nargs="+", help="zero-based decoder layers (default: all)"
    )
    parser.add_argument(
        "--chunk-rows", type=int, help="pad every FP32 tile to this many rows (default: native M)"
    )
    parser.add_argument(
        "--case", action="append", choices=presets, help="repeat to select multiple cases"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=168)
    parser.add_argument(
        "--out", type=Path, help="new JSONL file (default: timestamped file in results/)"
    )
    args = parser.parse_args()
    if not args.model.is_dir() or min(args.repeats, args.output_tokens) < 1:
        parser.error("use a local checkpoint and positive repeats and output length")
    if args.chunk_rows is not None and args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    import torch_npu

    torch.npu.set_device(0)
    cases = [presets[name] for name in dict.fromkeys(args.case or DEFAULT_CASES)]
    model = OuroForCausalLM.from_pretrained(
        args.model, device=torch.device("npu:0"), dtype=torch.bfloat16
    )
    model.eval()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.out or Path(f"results/prefix-kv-fp32-{stamp}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        print("RESULTS", path.resolve(), flush=True)
        write_record(
            stream,
            kind="config",
            model=str(args.model.resolve()),
            device=torch.npu.get_device_name(0),
            torch=str(torch.__version__),
            torch_npu=torch_npu.__version__,
            native_allow_hf32=getattr(getattr(torch.npu, "matmul", None), "allow_hf32", None),
            projection=args.projection,
            layers=(list(range(len(model.model.layers))) if args.layers is None else args.layers),
            chunk_rows=args.chunk_rows,
            output_tokens=args.output_tokens,
            repeats=args.repeats,
            seed=args.seed,
            cases=[asdict(case) for case in cases],
        )
        review = run_cases(model, "ascend_eager", cases, args, stream)
    return int(
        any(review[key] for key in ("cache_parity_failed", "native_cold_changed", "invalid"))
    )


if __name__ == "__main__":
    raise SystemExit(main())
