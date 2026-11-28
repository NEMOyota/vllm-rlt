"""Same-card, paired Ouro benchmark for length-aware paged prefill on Ascend."""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestPhase

# Pairs of (cached prefix length, query length). W2 reproduces the batch shape.
CASES = {
    "c4_h1": [(128, 16), (0, 144), (0, 144), (0, 144)],
    "mixed": [(128, 16), (0, 96), (160, 32)],
    "cold_ragged": [(0, 15), (0, 16), (0, 17)],
    "one_token_suffix": [(128, 1), (0, 33)],
    "w2_shape": [(96, 32)] * 32,
    "cold_uniform": [(0, 64)] * 4,
    "long_suffix": [(2048, 16), (64, 16)],
    "single": [(128, 16)],
}


def inputs(case, seed, vocab):
    rng = random.Random(seed)

    def tokens(n):
        return [rng.randrange(3, vocab) for _ in range(n)]

    shared = {start: tokens(start) for start in sorted({s for s, _ in CASES[case] if s})}
    donors = [prefix + tokens(1) for prefix in shared.values()]
    targets = [shared.get(start, []) + tokens(q) for start, q in CASES[case]]
    return donors, targets


def batch_run(engine, prompts, params, *, tag):
    engine.model_runner.synchronize()
    cache = engine.cache_manager
    strategy = cache.attention.packed_pool_prefill
    hits_before = cache.prefix_hits
    calls_before = strategy.calls if strategy else 0
    last_sdpa = cache.attention.kernel.sdpa_calls
    prefill_sdpa = 0
    prefill_batches = []
    submitted, first, finished, starts, outputs, issues = {}, {}, {}, {}, {}, []
    start = perf_counter()
    for i, tokens in enumerate(prompts):
        rid = f"{tag}-{i}"
        submitted[rid] = perf_counter()
        engine.add_request(rid, tokens, params)

    while engine.has_unfinished_requests():
        result = engine.step()
        visible = perf_counter()
        schedule = engine.last_schedule
        sdpa = cache.attention.kernel.sdpa_calls
        if schedule is not None and schedule.phase == RequestPhase.PREFILL:
            prefill_sdpa += sdpa - last_sdpa
            prefill_batches.append(
                [
                    (item.request.request_id, item.token_start, item.token_count)
                    for item in schedule.requests
                ]
            )
            for item in schedule.requests:
                starts.setdefault(item.request.request_id, item.token_start)
        last_sdpa = sdpa
        for out in result:
            if out.token_ids:
                first.setdefault(out.request_id, visible)
            if out.finished:
                finished[out.request_id] = visible
                outputs[out.request_id] = out.token_ids
                if (
                    out.finish_reason != "length"
                    or len(out.token_ids) != params.max_tokens
                    or out.exit_depths != [4] * params.max_tokens
                ):
                    issues.append(
                        f"{out.request_id}: unexpected finish, output length or exit depth"
                    )
    engine.model_runner.synchronize()
    wall = perf_counter() - start
    if not (first.keys() == finished.keys() == submitted.keys()):
        raise RuntimeError("missing token delivery or completion")
    if cache.num_used_blocks:
        issues.append("live KV allocations after drain")
    ttft = [1000 * (first[rid] - submitted[rid]) for rid in submitted]
    tpot = [
        1000 * (finished[rid] - first[rid]) / (len(outputs[rid]) - 1)
        for rid in submitted
        if len(outputs[rid]) > 1
    ]
    return dict(
        wall_s=wall,
        output_tok_s=sum(map(len, outputs.values())) / wall,
        ttft_ms=ttft,
        ttft_p50_ms=median(ttft),
        tpot_p50_ms=median(tpot) if tpot else None,
        hits=[starts[rid] for rid in submitted],
        hit_tokens=cache.prefix_hits - hits_before,
        prefill_sdpa_calls=prefill_sdpa,
        prefill_batches=prefill_batches,
        fia_calls=strategy.calls - calls_before if strategy else 0,
        tokens=[outputs[rid] for rid in submitted],
        issues=issues,
    )


def exercise(engine, fixture, params, tag):
    donors, targets = fixture
    if donors:
        batch_run(engine, donors, replace(params, max_tokens=1), tag=tag + "-donor")
    return batch_run(engine, targets, params, tag=tag)


def run_arm(model, case, seed, arm, params):
    spans = CASES[case]
    # Reserve sufficient pages for the full uncached batch in every arm.
    blocks = 4 * sum(math.ceil((s + q + params.max_tokens) / 16) for s, q in spans) + 128
    engine = StageEngine(
        model,
        attention_backend="ascend_eager",
        cache_config=CacheConfig(
            block_size=16, num_blocks=blocks, enable_prefix_caching=arm != "off"
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=len(spans), max_num_batched_tokens=8192, prefill_chunk_size=256
        ),
        execution_config=ExecutionConfig(
            packed_prefill=True,
            multi_stream=False,
            use_uva=False,
            ascend_paged_prefill=arm == "paged",
        ),
    )
    try:
        vocab = model.config.vocab_size
        exercise(engine, inputs(case, ~seed, vocab), params, "warm")
        fixture = inputs(case, seed, vocab)
        row = exercise(engine, fixture, params, "target")
        expected = [s if arm != "off" else 0 for s, _ in spans]
        if row["hits"] != expected or row["hit_tokens"] != sum(expected):
            row["issues"].append(f"expected cached tokens {expected}")
        # Target prefill fits one scheduling batch. Decode must contribute no FIA calls.
        calls = (
            engine.cache_manager.num_layers * engine.cache_manager.storage_depths
            if arm == "paged" and case not in ("cold_uniform", "single")
            else 0
        )
        if row["fia_calls"] != calls:
            row["issues"].append(f"expected {calls} paged FIA calls, saw {row['fia_calls']}")
        row["input_sha256"] = hashlib.sha256(json.dumps(fixture).encode()).hexdigest()
        row["num_blocks"] = blocks
        return row
    finally:
        engine.close()


def compare(a, b):
    differences = []
    for i, (old, new) in enumerate(zip(a["tokens"], b["tokens"], strict=True)):
        if old == new:
            continue
        j = next((j for j, (x, y) in enumerate(zip(old, new)) if x != y), min(len(old), len(new)))
        differences.append(
            dict(target=i, position=j + 1, before=old[j : j + 1], after=new[j : j + 1])
        )
    return dict(
        same_inputs=a["input_sha256"] == b["input_sha256"],
        first_differences=differences,
        ttft_saved_ms=a["ttft_p50_ms"] - b["ttft_p50_ms"],
        output_tok_s_change_pct=100 * (b["output_tok_s"] / a["output_tok_s"] - 1),
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=797543261)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--with-cache-off", action="store_true", help="add an uncached legacy arm")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.rounds, args.max_tokens) < 1 or not args.checkpoint.is_dir():
        p.error("supply a local checkpoint directory, positive rounds and max tokens")

    import torch_npu

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    torch.npu.matmul.allow_hf32 = False
    model = OuroForCausalLM.from_pretrained(
        args.checkpoint, device=device, dtype=torch.bfloat16
    ).eval()
    params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        min_loops=4,
        max_loops=4,
        exit_threshold=1.0,
        ignore_eos=True,
    )
    base_arms = ["legacy", "paged"] + (["off"] if args.with_cache_off else [])
    report = dict(
        git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip(),
        checkpoint=str(args.checkpoint.resolve()),
        card=torch.npu.get_device_name(0),
        visible=os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__),
        seed=args.seed,
        max_tokens=args.max_tokens,
        page_size=16,
        allow_hf32=bool(torch.npu.matmul.allow_hf32),
        runs=[],
        summary={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.output)

    errors = False
    for case in args.cases:
        pairs = []
        for r in range(args.rounds):
            offset = r % len(base_arms)
            arms = base_arms[offset:] + base_arms[:offset]
            result = dict(case=case, round=r + 1, order=arms, arms={})
            report["runs"].append(result)
            for arm in arms:
                gc.collect()
                torch.npu.empty_cache()
                try:
                    row = run_arm(model, case, args.seed, arm, params)
                except Exception as exc:
                    result["error"] = dict(arm=arm, message=repr(exc))
                    save()
                    raise
                result["arms"][arm] = row
                errors |= bool(row["issues"])
                save()
            data = result["arms"]
            pair = compare(data["legacy"], data["paged"])
            pairs.append(pair)
            result["legacy_to_paged"] = pair
            if args.with_cache_off:
                result["off_to_legacy"] = compare(data["off"], data["legacy"])
                result["off_to_paged"] = compare(data["off"], data["paged"])
            comparisons = [
                result[key]
                for key in ("legacy_to_paged", "off_to_legacy", "off_to_paged")
                if key in result
            ]
            errors |= any(not p["same_inputs"] or p["first_differences"] for p in comparisons)
            save()
        report["summary"][case] = dict(
            median_paired_ttft_saved_ms=median(p["ttft_saved_ms"] for p in pairs),
            median_paired_output_tok_s_change_pct=median(
                p["output_tok_s_change_pct"] for p in pairs
            ),
            legacy_to_paged_divergent_rounds=sum(bool(p["first_differences"]) for p in pairs),
            rounds=len(pairs),
        )
        print(case, json.dumps(report["summary"][case]), flush=True)
        save()
    report["status"] = "FAIL" if errors else "PASS"
    save()
    return int(errors)


if __name__ == "__main__":
    sys.exit(main())
