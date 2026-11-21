"""Screen Ouro projection tiers or run the full prefix-cache matrix with tier 3."""

import argparse
import csv
import json
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.ascend_grouped_kv_single import check_trial, write_record
from benchmarks.ascend_prefix_kv_precision import DEFAULT_CASES, run_precision_arms
from benchmarks.ascend_prefix_projection import fp32_decoder_projections
from benchmarks.ascend_prefix_scenarios import ARMS, suites, summarize
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


def _selection(tier3_full):
    full = suites()["full"]
    if tier3_full:
        return full, (3,)
    presets = {c.name: c for c in full}
    return [presets[name] for name in DEFAULT_CASES], tuple(TIER_PATHS)


def _case_row(case, measured, status, changed):
    metrics = summarize(case, measured)["arms"]
    reuse = metrics["reuse"]
    row = dict(
        case=case.name,
        cache_parity=status,
        native_cold_changed=changed,
        repeats=len(measured),
        reuse_hit_tokens_per_request=round(reuse["hit_tokens"] / reuse["request_count"], 2),
    )
    for arm in ARMS:
        for key in ("ttft_p50_ms", "ttft_p95_ms", "output_tok_s"):
            row[f"{arm}_{key}"] = round(metrics[arm][key], 2)
    row["reuse_vs_miss_ttft_p50_saved_ms"] = (
        round(metrics["miss"]["ttft_p50_ms"] - reuse["ttft_p50_ms"], 2)
        if status == "PASS"
        else None
    )
    return row


def _run_tier(model, backend, cases, native, tier, chunk_rows, stream, *, tier3_full, csv_stream):
    failed, invalid, cold_changed = set(), set(), set()
    csv_writer = None
    # Keep one FP32 weight copy per selected module across arms, cases and repeats.
    with fp32_decoder_projections(model, TIER_PATHS[tier], chunk_rows=chunk_rows) as work:
        for case in cases:
            first_off = None
            measured = []
            for index, record in enumerate(native[case.name]):
                warmup, repeat = record["warmup"], record["repeat"]
                phase = "warmup" if warmup else "repeat"
                print(f"Starting tier {tier}, {case.name}, {phase} {repeat}", flush=True)
                trial, counters = run_precision_arms(
                    model,
                    backend,
                    case,
                    OUTPUT_TOKENS,
                    SEED,
                    work=work,
                    arm_offset=index if tier3_full else 0,
                )
                checks = check_trial(
                    case,
                    {"native": record["arms"], "fp32": trial},
                    native[case.name][0]["arms"]["off"],
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
                    warmup=warmup,
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
                if tier3_full and not warmup:
                    measured.append(trial)
            if tier3_full:
                status = "PASS"
                if case.name in invalid:
                    status = "INVALID"
                elif case.name in failed:
                    status = "FAIL"
                row = _case_row(case, measured, status, case.name in cold_changed)
                write_record(stream, kind="case_summary", tier=tier, **row)
                print("TIER3_FULL_CASE", json.dumps(row), flush=True)
                if csv_stream is not None:
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_stream, fieldnames=list(row))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    csv_stream.flush()
    return dict(
        passed_cases=len(cases) - len(failed | invalid),
        cache_parity_failed=sorted(failed),
        invalid=sorted(invalid),
        native_cold_changed=sorted(cold_changed),
    )


def run_sweep(model, backend, chunk_rows, repeats, stream, *, tier3_full=False, csv_stream=None):
    if chunk_rows < 1 or repeats < 1:
        raise ValueError("the tier screen requires positive chunk_rows and repeats")
    cases, selected = _selection(tier3_full)
    warmups = 1 if tier3_full else 0
    native = {}
    for case in cases:
        native[case.name] = []
        for index in range(warmups + repeats):
            warmup = index < warmups
            repeat = index + 1 if warmup else index + 1 - warmups
            phase = "warmup" if warmup else "repeat"
            print(f"Starting native BF16, {case.name}, {phase} {repeat}", flush=True)
            trial, _ = run_precision_arms(
                model, backend, case, OUTPUT_TOKENS, SEED, arm_offset=index if tier3_full else 0
            )
            record = dict(case=case.name, warmup=warmup, repeat=repeat, arms=trial)
            native[case.name].append(record)
            write_record(stream, kind="native", **record)

    tiers = {}
    for tier in selected:
        tiers[tier] = _run_tier(
            model,
            backend,
            cases,
            native,
            tier,
            chunk_rows,
            stream,
            tier3_full=tier3_full,
            csv_stream=csv_stream,
        )
    passing = [
        tier for tier, r in tiers.items() if not r["cache_parity_failed"] and not r["invalid"]
    ]
    review = dict(
        state="complete",
        backend=backend,
        suite="full" if tier3_full else "screen",
        cases=[c.name for c in cases],
        warmups=warmups,
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
    parser.add_argument(
        "--tier3-full",
        action="store_true",
        help="test only tier 3 on the full matrix (one warmup, five measured repeats by default)",
    )
    parser.add_argument("--repeats", type=int, help="measured repeats (screen: 2, tier 3 full: 5)")
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
    if args.repeats is None:
        args.repeats = 5 if args.tier3_full else 2
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
    prefix = "prefix-tier3-full" if args.tier3_full else "prefix-fp32-tiers"
    path = args.out or Path(f"results/{prefix}-{stamp}.jsonl")
    csv_path = path.with_suffix(".csv") if args.tier3_full else None
    if csv_path is not None and (csv_path == path or csv_path.exists()):
        parser.error("choose a new JSONL report path with an unused companion CSV path")
    path.parent.mkdir(parents=True, exist_ok=True)
    cases, selected = _selection(args.tier3_full)
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
                suite="full" if args.tier3_full else "screen",
                warmups=1 if args.tier3_full else 0,
                chunk_rows=args.chunk_rows,
                repeats=args.repeats,
                output_tokens=OUTPUT_TOKENS,
                seed=SEED,
                loops=4,
                projection_paths_by_tier={t: TIER_PATHS[t] for t in selected},
                cases=[asdict(c) for c in cases],
            )
            model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
            csv_output = csv_path.open("x", newline="") if csv_path is not None else nullcontext()
            with csv_output as csv_stream:
                if csv_path is not None:
                    print("CSV", csv_path.resolve(), flush=True)
                review = run_sweep(
                    model.eval(),
                    backend,
                    args.chunk_rows,
                    args.repeats,
                    stream,
                    tier3_full=args.tier3_full,
                    csv_stream=csv_stream,
                )
        except (Exception, KeyboardInterrupt) as exc:
            write_record(stream, kind="incomplete", error=repr(exc))
            raise
    return int(
        review["lowest_passing_tier"] is None or any(r["invalid"] for r in review["tiers"].values())
    )


if __name__ == "__main__":
    raise SystemExit(main())
