"""Screen three fixed-row FP32 decoder projection tiers on the three Ouro prefix failures."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.ascend_grouped_kv_single import check_trial, write_record
from benchmarks.ascend_prefix_kv_precision import DEFAULT_CASES, run_precision_arms
from benchmarks.ascend_prefix_projection import fp32_decoder_projections
from benchmarks.ascend_prefix_scenarios import suites
from benchmarks.preemption import checkpoint_identity
from vllm_rlt.device import resolve_device
from vllm_rlt.models import OuroForCausalLM

QKV = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
ATTENTION = (*QKV, "self_attn.o_proj")
TIER_PATHS = {
    2: QKV,
    3: ATTENTION,
    4: (*ATTENTION, "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
}
OUTPUT_TOKENS = 16
SEED = 168


def _run_tier(model, backend, cases, native, tier, chunk_rows, stream):
    failed, invalid, cold_changed = set(), set(), set()
    # Keep one FP32 weight copy per selected module across arms, cases and repeats.
    with fp32_decoder_projections(model, TIER_PATHS[tier], chunk_rows=chunk_rows) as work:
        for case in cases:
            first_off = None
            for repeat, baseline in enumerate(native[case.name], 1):
                print(f"Starting tier {tier}, {case.name}, repeat {repeat}", flush=True)
                trial, counters = run_precision_arms(
                    model, backend, case, OUTPUT_TOKENS, SEED, work=work
                )
                checks = check_trial(
                    case,
                    {"native": baseline, "fp32": trial},
                    native[case.name][0]["off"],
                    candidate="fp32",
                    candidate_reference=first_off,
                )
                if first_off is None:
                    first_off = trial["off"]
                if checks["controls"]:
                    status = "INVALID"
                    invalid.add(case.name)
                elif checks["fp32_parity_diffs"]:
                    status = "FAIL"
                    failed.add(case.name)
                else:
                    status = "PASS"
                if checks["native_off_vs_fp32_off"]:
                    cold_changed.add(case.name)
                row = dict(
                    tier=tier,
                    case=case.name,
                    repeat=repeat,
                    cache_parity=status,
                    parity_diffs=checks["fp32_parity_diffs"],
                    reuse_hits=checks["fp32_reuse_hits"],
                    native_off_vs_fp32_off=checks["native_off_vs_fp32_off"],
                    native_off_vs_reuse=checks["native_off_vs_reuse"],
                    controls=checks["controls"],
                )
                write_record(stream, kind="trial", **row, arms=trial, projection_work=counters)
                print("TIER_CASE", json.dumps(row), flush=True)
    return dict(
        cache_parity_failed=sorted(failed),
        invalid=sorted(invalid),
        native_cold_changed=sorted(cold_changed),
    )


def run_sweep(model, backend, chunk_rows, repeats, stream):
    if chunk_rows < 1 or repeats < 1:
        raise ValueError("the tier screen requires positive chunk_rows and repeats")
    presets = {c.name: c for c in suites()["full"]}
    cases = [presets[name] for name in DEFAULT_CASES]
    native = {}
    for case in cases:
        native[case.name] = []
        for repeat in range(1, repeats + 1):
            print(f"Starting native BF16, {case.name}, repeat {repeat}", flush=True)
            trial, _ = run_precision_arms(model, backend, case, OUTPUT_TOKENS, SEED)
            native[case.name].append(trial)
            write_record(stream, kind="native", case=case.name, repeat=repeat, arms=trial)

    tiers = {}
    for tier in TIER_PATHS:
        tiers[tier] = _run_tier(model, backend, cases, native, tier, chunk_rows, stream)
    passing = [
        tier for tier, r in tiers.items() if not r["cache_parity_failed"] and not r["invalid"]
    ]
    review = dict(
        state="complete",
        backend=backend,
        cases=list(DEFAULT_CASES),
        repeats=repeats,
        chunk_rows=chunk_rows,
        tiers=tiers,
        passing_tiers=passing,
        lowest_passing_tier=min(passing, default=None),
        lowest_passing_tier_matching_native_off=min(
            (t for t in passing if not tiers[t]["native_cold_changed"]), default=None
        ),
    )
    write_record(stream, kind="review", **review)
    print("TIER_REVIEW", json.dumps(review), flush=True)
    return review


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="local Ouro checkpoint directory")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=64,
        help="fixed rows per FP32 tile, including the padded tail",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--out", type=Path, help="new JSONL report (default: timestamped file in results/)"
    )
    parser.add_argument(
        "--device",
        choices=("npu:0", "cpu"),
        default="npu:0",
        help="use cpu for a portable engine smoke test",
    )
    args = parser.parse_args()
    if not args.model.is_dir() or min(args.chunk_rows, args.repeats) < 1:
        parser.error("provide a local --model, positive --chunk-rows and positive --repeats")

    device = resolve_device(args.device)
    backend, device_name, npu_version, hf32 = "torch", "cpu", None, None
    if device.type == "npu":
        import torch_npu

        torch.npu.set_device(device.index)
        backend, device_name = "ascend_eager", torch.npu.get_device_name(device.index)
        npu_version = torch_npu.__version__
        hf32 = getattr(getattr(torch.npu, "matmul", None), "allow_hf32", None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.out or Path(f"results/prefix-fp32-tiers-{stamp}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        print("RESULTS", path.resolve(), flush=True)
        try:
            write_record(
                stream,
                kind="config",
                model=str(args.model.resolve()),
                checkpoint=checkpoint_identity(args.model),
                device=args.device,
                device_name=device_name,
                backend=backend,
                torch=str(torch.__version__),
                torch_npu=npu_version,
                native_allow_hf32=hf32,
                chunk_rows=args.chunk_rows,
                repeats=args.repeats,
                output_tokens=OUTPUT_TOKENS,
                seed=SEED,
                loops=4,
                projection_paths_by_tier=TIER_PATHS,
                cases=[asdict(c) for c in suites()["full"] if c.name in DEFAULT_CASES],
            )
            model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
            review = run_sweep(model.eval(), backend, args.chunk_rows, args.repeats, stream)
        except (Exception, KeyboardInterrupt) as exc:
            write_record(stream, kind="incomplete", error=repr(exc))
            raise
    return int(
        review["lowest_passing_tier"] is None or any(r["invalid"] for r in review["tiers"].values())
    )


if __name__ == "__main__":
    raise SystemExit(main())
