"""Measure engine TTFT and greedy parity for a pair of prompts with a shared prefix."""

import argparse
import gc
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import torch

from benchmarks.preemption import percentile
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM

BLOCK_SIZE = 16


class Sample(NamedTuple):
    tokens: list[int]
    hit_tokens: int
    prefill_start: int
    ttft_ms: float


def generate(engine, request_id, tokens, params):
    hits_before = engine.cache_manager.prefix_hits
    started = time.perf_counter()
    engine.add_request(request_id, tokens, params)
    prefill_start = None
    ttft_ms = None
    while engine.has_unfinished_requests():
        outputs = engine.step()
        if prefill_start is None:
            prefill_start = engine.last_schedule.requests[0].token_start
        for output in outputs:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - started) * 1000
            if output.finished:
                if (
                    output.finish_reason != "length"
                    or len(output.token_ids) != params.max_tokens
                    or output.exit_depths != [4] * params.max_tokens
                ):
                    raise RuntimeError(
                        f"{request_id}: finish={output.finish_reason}, "
                        f"tokens={len(output.token_ids)}, depths={set(output.exit_depths)}"
                    )
                hits = engine.cache_manager.prefix_hits - hits_before
                return Sample(output.token_ids, hits, prefill_start, ttft_ms)
    raise RuntimeError(f"{request_id} drained without an output")


def measure(model, backend, primer, target, params, *, cache_enabled, chunk_size):
    # Each pair starts with an empty cache. Retire the previous engine's Python
    # reference cycles before allocating a new device pool; share the model weights.
    gc.collect()
    pages = math.ceil((len(target) + params.max_tokens - 1) / BLOCK_SIZE)
    engine = StageEngine(
        model,
        attention_backend=backend,
        cache_config=CacheConfig(
            num_blocks=2 * model.config.total_ut_steps * pages,
            block_size=BLOCK_SIZE,
            enable_prefix_caching=cache_enabled,
        ),
        execution_config=ExecutionConfig(multi_stream=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=len(target), prefill_chunk_size=chunk_size
        ),
    )
    try:
        generate(engine, "primer", primer, replace(params, max_tokens=1))
        return generate(engine, "target", target, params)
    finally:
        engine.close()


@torch.inference_mode()
def experiment(model, backend, prefix_tokens, output_tokens, repeats):
    if prefix_tokens < BLOCK_SIZE or output_tokens < 1 or repeats < 1:
        raise ValueError("require at least one full prefix page, one output token and one repeat")
    vocab_size = model.config.vocab_size
    target = random.Random(168).choices(range(vocab_size), k=prefix_tokens + 8)
    shared = target.copy()
    shared[prefix_tokens] = (shared[prefix_tokens] + 1) % vocab_size
    unrelated = shared.copy()
    unrelated[0] = (unrelated[0] + 1) % vocab_size
    hit_length = prefix_tokens // BLOCK_SIZE * BLOCK_SIZE
    params = SamplingParams(
        max_tokens=output_tokens, min_loops=4, max_loops=4, temperature=0, ignore_eos=True
    )
    split = measure(
        model, backend, shared, target, params, cache_enabled=False, chunk_size=hit_length
    )
    arms = ("off", "miss", "hit")
    timings = {arm: [] for arm in arms}
    first_diff = {"off_miss": None, "off_hit": None, "split_hit": None}
    for trial in range(repeats + 1):
        samples = {}
        offset = trial % len(arms)
        for arm in arms[offset:] + arms[:offset]:
            sample = measure(
                model,
                backend,
                unrelated if arm == "miss" else shared,
                target,
                params,
                cache_enabled=arm != "off",
                chunk_size=len(target),
            )
            expected = hit_length if arm == "hit" else 0
            if (sample.hit_tokens, sample.prefill_start) != (expected, expected):
                return {
                    "status": "HIT_MISMATCH",
                    "backend": backend,
                    "arm": arm,
                    "expected_tokens": expected,
                    "hit_tokens": sample.hit_tokens,
                    "prefill_start": sample.prefill_start,
                }
            samples[arm] = sample
            if trial:  # Exclude one warmup per arm.
                timings[arm].append(sample.ttft_ms)
        samples["split"] = split
        for pair in first_diff:
            left, right = pair.split("_")
            for i, (a, b) in enumerate(
                zip(samples[left].tokens, samples[right].tokens, strict=True), 1
            ):
                if a != b:
                    previous = first_diff[pair]
                    first_diff[pair] = i if previous is None else min(previous, i)
                    break
    return {
        "status": "PASS" if all(i is None for i in first_diff.values()) else "DIFF",
        "backend": backend,
        "prefix_tokens": prefix_tokens,
        "output_tokens": output_tokens,
        "hit_tokens": samples["hit"].hit_tokens,
        "prefill_start": samples["hit"].prefill_start,
        "first_diff_1based": first_diff,
        "repeats": repeats,
        "ttft_p50_ms": {arm: round(percentile(times, 0.5), 2) for arm, times in timings.items()},
        "ttft_p95_ms": {arm: round(percentile(times, 0.95), 2) for arm, times in timings.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Prepared local Ouro checkpoint")
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("--model must be a local checkpoint directory")
    device = resolve_device("npu:0")
    torch.npu.set_device(0)
    model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    report = experiment(model, "ascend_eager", args.prefix_tokens, args.output_tokens, args.repeats)
    print("PREFIX", json.dumps(report, separators=(",", ":")), flush=True)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
