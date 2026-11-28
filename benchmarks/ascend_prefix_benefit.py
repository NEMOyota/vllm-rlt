"""Measure the prefix-cache switch on the same 16-token Ascend attention backend."""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median

import torch

from benchmarks.ascend_varlen_prefill import batch_run
from benchmarks.preemption import percentile
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM


@dataclass(frozen=True)
class Workload:
    length: int
    outputs: int
    clients: int
    residents: int
    prefix: int
    hot_clients: int


# The first five reproduce the fixed-cohort PR1 W-series lengths and concurrency.
W2 = Workload(128, 128, 32, 32, 96, 32)
WORKLOADS = {
    "W1": Workload(128, 128, 1, 1, 96, 1),
    "W2": W2,
    "W3": Workload(4096, 128, 1, 1, 3072, 1),
    "W4": Workload(4096, 128, 32, 4, 3072, 32),
    "W5": Workload(1024, 512, 8, 8, 768, 8),
    "W2-cold": replace(W2, hot_clients=0),
    "W2-quarter": replace(W2, hot_clients=8),
}
DEFAULT_CASES = ["W2-cold", "W2-quarter", "W2"]


def workload_inputs(w, seed, vocab):
    if w.clients + 2 >= vocab:
        raise ValueError("too few distinct tokens for the request markers")
    # Same deterministic donor and target stream as PR1 performance_inputs.
    rng = random.Random(f"ascend-pr1/{seed}/W/{w.length}")
    shared = rng.choices(range(vocab), k=w.prefix)
    marker = rng.randrange(vocab)
    donor = shared + [(marker - 1) % vocab] + rng.choices(range(vocab), k=w.length - w.prefix - 1)
    targets = [
        shared + [(marker + i) % vocab] + rng.choices(range(vocab), k=w.length - w.prefix - 1)
        for i in range(w.clients)
    ]
    # Distribute hits through the arrival order. Give each cold prompt a private
    # first token so it cannot accidentally hit another target in a later wave.
    hits = []
    for i, prompt in enumerate(targets):
        hot = (i + 1) * w.hot_clients // w.clients > i * w.hot_clients // w.clients
        hits.append(w.prefix if hot else 0)
        if not hot:
            prompt[0] = (shared[0] + i + 1) % vocab
    return donor, targets, hits


def measure_arm(model, w, seed, chunk, enabled, *, full_warmup=False):
    blocks = (w.residents + 1) * 4 * math.ceil((w.length + w.outputs - 1) / 16)
    engine = StageEngine(
        model,
        attention_backend="ascend_eager",
        cache_config=CacheConfig(block_size=16, num_blocks=blocks, enable_prefix_caching=enabled),
        execution_config=ExecutionConfig(
            multi_stream=False, use_uva=False, packed_prefill=True, ascend_paged_prefill=True
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=w.residents,
            prefill_chunk_size=chunk,
            max_num_batched_tokens=min(chunk, w.length) * w.residents,
        ),
    )
    try:
        if engine.cache_manager.attention.packed_pool_prefill is None:
            raise RuntimeError("paged prefill dispatch is unavailable")
        params = SamplingParams(
            temperature=0,
            min_loops=4,
            max_loops=4,
            max_tokens=w.outputs,
            ignore_eos=True,
            exit_threshold=1.0,
        )
        prime = replace(params, max_tokens=1)
        warm_donor, warm_targets, _ = workload_inputs(w, ~seed, model.config.vocab_size)
        donor, targets, expected = workload_inputs(w, seed, model.config.vocab_size)
        warm_params = params if full_warmup else replace(params, max_tokens=min(2, w.outputs))
        for prompts, sample, tag in (
            ([warm_donor], prime, "warm-donor"),
            (warm_targets[: w.residents], warm_params, "warm"),
            ([donor], prime, "donor"),
        ):
            run = batch_run(engine, prompts, sample, tag=tag)
            if run["issues"]:
                raise RuntimeError(f"{tag}: {run['issues']}")
        if run["hit_tokens"]:
            raise RuntimeError("measured primer hit the warmup corpus")
        primer_s = run["wall_s"]
        row = batch_run(engine, targets, params, tag="target")
        expected = expected if enabled else [0] * w.clients
        if row["hits"] != expected or row["hit_tokens"] != sum(expected):
            raise RuntimeError(f"observed hits {row['hits']}, expected {expected}")
        if row["issues"]:
            raise RuntimeError(f"invalid timing cohort: {row['issues']}")
        # The scheduler can split prefill into multiple chunks. Count the logical
        # input spans it actually executed, including later waves of W4 clients.
        prefill_rows = sum(n for batch in row["prefill_batches"] for _, _, n in batch)
        if prefill_rows != w.length * w.clients - sum(expected):
            raise RuntimeError(f"unexpected number of scheduled prompt rows: {prefill_rows}")
        return dict(
            enable_prefix_caching=enabled,
            warmup_output_length=warm_params.max_tokens,
            ascend_paged_prefill=True,
            block_size=16,
            num_blocks=blocks,
            primer_s=primer_s,
            prefill_rows=prefill_rows,
            hits=row["hits"],
            hit_tokens=row["hit_tokens"],
            hit_fraction=sum(expected) / (w.length * w.clients),
            input_sha256=hashlib.sha256(json.dumps([donor, targets]).encode()).hexdigest(),
            generated_ids_sha256=[
                hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in row["tokens"]
            ],
            output_tokens=sum(map(len, row["tokens"])),
            wall_s=row["wall_s"],
            output_tok_s=row["output_tok_s"],
            ttft_p50_ms=row["ttft_p50_ms"],
            ttft_p95_ms=percentile(row["ttft_ms"], 0.95),
            tpot_p50_ms=row["tpot_p50_ms"],
            sdpa_prefill_calls=row["prefill_sdpa_calls"],
            fia_calls=row["fia_calls"],
        )
    finally:
        engine.close()


def paired(off, on):
    if off["input_sha256"] != on["input_sha256"] or off["output_tokens"] != on["output_tokens"]:
        raise RuntimeError("cannot pair different inputs or output token counts")
    # Charge only ON for filling the prefix in this conservative one-cohort view.
    inclusive_rate = on["output_tokens"] / (on["primer_s"] + on["wall_s"])
    return dict(
        output_tok_s_change_pct=100 * (on["output_tok_s"] / off["output_tok_s"] - 1),
        ttft_p50_saved_ms=off["ttft_p50_ms"] - on["ttft_p50_ms"],
        ttft_p95_saved_ms=off["ttft_p95_ms"] - on["ttft_p95_ms"],
        on_with_primer_vs_off_pct=100 * (inclusive_rate / off["output_tok_s"] - 1),
        differing_outputs=sum(
            a != b
            for a, b in zip(off["generated_ids_sha256"], on["generated_ids_sha256"], strict=True)
        ),
    )


def summary(rounds):
    metrics = (
        "output_tok_s_change_pct",
        "ttft_p50_saved_ms",
        "ttft_p95_saved_ms",
        "on_with_primer_vs_off_pct",
    )
    gains = [r["pair"]["output_tok_s_change_pct"] for r in rounds]
    return dict(
        medians={m: median(r["pair"][m] for r in rounds) for m in metrics},
        arm_medians={
            arm: {
                m: median(r[arm][m] for r in rounds)
                for m in ("output_tok_s", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms")
            }
            for arm in ("off", "on")
        },
        faster_rounds=sum(g > 0 for g in gains),
        rounds=len(rounds),
        min_gain_pct=min(gains),
        max_gain_pct=max(gains),
        on_hit_fraction=rounds[0]["on"]["hit_fraction"],
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", choices=WORKLOADS, nargs="+", default=DEFAULT_CASES)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=797543261)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--plan", action="store_true")
    args = p.parse_args()
    if args.rounds < 1 or args.chunk_size < 1 or len(set(args.cases)) != len(args.cases):
        p.error("rounds and chunk size must be positive and case names unique")
    cases = {k: asdict(WORKLOADS[k]) for k in args.cases}
    if args.plan:
        print(json.dumps(cases, indent=2))
        return
    if not args.checkpoint or not args.checkpoint.is_dir() or not args.output:
        p.error("supply a local --checkpoint directory and an --output JSON path")
    if args.output.exists():
        p.error(f"output already exists: {args.output}")
    import torch_npu
    from torch_npu.utils.collect_env import get_cann_version

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    torch.npu.matmul.allow_hf32 = False
    model = OuroForCausalLM.from_pretrained(
        args.checkpoint, device=device, dtype=torch.bfloat16
    ).eval()
    report = dict(
        status="INCOMPLETE",
        seed=args.seed,
        rounds=args.rounds,
        chunk_size=args.chunk_size,
        cases=cases,
        checkpoint=str(args.checkpoint.resolve()),
        torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__),
        cann=str(get_cann_version()),
        card=torch.npu.get_device_name(0),
        physical_visible=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        allow_hf32=bool(torch.npu.matmul.allow_hf32),
        git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip(),
        runs={},
        summary={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.output)

    save()
    for case in args.cases:
        rounds = report["runs"][case] = []
        for i in range(args.rounds):
            order = ("off", "on") if i % 2 == 0 else ("on", "off")
            row = dict(round=i + 1, order=order)
            rounds.append(row)
            for arm in order:
                gc.collect()
                torch.npu.empty_cache()
                try:
                    row[arm] = measure_arm(
                        model,
                        WORKLOADS[case],
                        args.seed,
                        args.chunk_size,
                        arm == "on",
                        full_warmup=i == 0,
                    )
                except Exception as exc:
                    report["error"] = dict(case=case, round=i + 1, arm=arm, message=repr(exc))
                    save()
                    raise
                save()
            row["pair"] = paired(row["off"], row["on"])
            save()
        report["summary"][case] = summary(rounds)
        print(case, json.dumps(report["summary"][case]), flush=True)
        save()
    report["status"] = "COMPLETE"
    save()


if __name__ == "__main__":
    main()
