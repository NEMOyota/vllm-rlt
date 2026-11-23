"""Check synchronous Ouro prefix reuse on the mainline Ascend execution path.

Run from the experiment worktree with one visible NPU:
    python -m benchmarks.ascend_main_prefix_combo --model /models/Ouro-1.4B
    python -m benchmarks.ascend_main_prefix_combo --model /models/Ouro-1.4B --linear-fp32-chunk 256
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.ascend_main_prefix_linear import (
    add_precision_arg,
    benchmark_dtype,
    check_linear_usage,
    linear_usage,
    padded_fp32_linears,
    precision_settings,
)
from benchmarks.ascend_prefix_combo_checks import PrefillAudit, page_oracle, require
from benchmarks.preemption import checkpoint_identity
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.attention import create_backend
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestPhase

BLOCK = 16
SEED = 168
OUTPUTS = 16
ARMS = ("off", "miss", "reuse")


@dataclass(frozen=True)
class Case:
    prefix: int
    suffix: int
    count: int = 1
    hot: int = 1
    tight_pool: bool = False
    churn: int = 0

    def starts(self, arm):
        reusable = min(self.prefix, self.prefix + self.suffix - 1) // BLOCK * BLOCK
        return {
            f"target-{i}": reusable if arm == "reuse" and i < self.hot and not self.churn else 0
            for i in range(self.count)
        }


CASES = {
    "c4_h2": Case(128, 16, 4, 2),
    "c4_h0": Case(128, 16, 4, 0),
    "c4_h1": Case(128, 16, 4, 1),
    "c4_h3": Case(128, 16, 4, 3),
    "c4_h4": Case(128, 16, 4, 4),
    "total512_p384": Case(384, 128),
    "p1024_s8": Case(1024, 8),
    "edge_p17": Case(17, 8),
    "repeat_p128": Case(128, 0),
}


def workload(case, vocab):
    """Keep the seed-168 targets and donor mutations of ascend_prefix_workload._prompts."""
    rng = random.Random(SEED)
    target = rng.choices(range(vocab), k=case.prefix + case.suffix)
    donor = target.copy()
    if case.suffix:
        donor[case.prefix] = (target[case.prefix] + 1) % vocab
    miss = donor.copy()
    miss[0] = (donor[0] + 1) % vocab
    reserved = {target[0], donor[0], miss[0]}
    if case.suffix:
        reserved.update((target[case.prefix], donor[case.prefix]))
    candidates = rng.sample(range(vocab), case.count + len(reserved))
    markers = [t for t in candidates if t not in reserved][: case.count]
    targets = {}
    for i in range(case.count):
        tokens = target.copy()
        if i and case.suffix:
            tokens[case.prefix :] = rng.choices(range(vocab), k=case.suffix)
            tokens[case.prefix] = markers[i]
        if i >= case.hot:
            tokens[0] = markers[i]
        targets[f"target-{i}"] = tokens
    fillers = []
    if case.churn:
        reserved.update(markers)
        count = case.count * case.churn
        candidates = rng.sample(range(vocab), count + len(reserved))
        for marker in [t for t in candidates if t not in reserved][:count]:
            tokens = rng.choices(range(vocab), k=len(donor) + OUTPUTS - 1)
            tokens[0] = marker
            fillers.append(tokens)
    return donor, miss, targets, fillers


def prompts(case, vocab):
    donor, miss, targets, _ = workload(case, vocab)
    return donor, miss, targets


def open_engine(model, case, arm, *, token_budget=None):
    capacity = case.prefix + case.suffix + OUTPUTS - 1
    if token_budget is None:
        token_budget = case.count * capacity
    blocks = 4 * math.ceil(capacity / BLOCK) * (case.count if case.tight_pool else case.count + 1)
    return StageEngine(
        model,
        attention_backend="ascend_eager",
        cache_config=CacheConfig(
            num_blocks=blocks, block_size=BLOCK, enable_prefix_caching=arm != "off"
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=case.count,
            prefill_chunk_size=capacity,
            max_num_batched_tokens=token_budget,
        ),
        execution_config=ExecutionConfig(multi_stream=False, use_uva=False),
    )


def drain(engine, expected_ids, output_count, first_outputs=()):
    finished = {}
    outputs = first_outputs
    while True:
        for out in outputs:
            if out.finished:
                require(
                    out.request_id not in finished and out.finish_reason == "length",
                    f"{out.request_id} duplicate finish or wrong finish reason {out.finish_reason}",
                )
                require(
                    len(out.token_ids) == output_count and out.exit_depths == [4] * output_count,
                    f"{out.request_id} unexpected token count or exit depths",
                )
                finished[out.request_id] = dict(tokens=out.token_ids, exit_depths=out.exit_depths)
        if not engine.has_unfinished_requests():
            break
        outputs = engine.step()
    require(set(finished) == set(expected_ids), "missing or unexpected completed request")
    require(engine.cache_manager.num_used_blocks == 0, "finished requests retained live KV")
    return finished


def check_pages(cache, starts, donor_prefix):
    # The pool fits the primer and the entire target burst without eviction.
    donor_pages = {block for group in donor_prefix for block in group}
    private = []
    for rid, start in starts.items():
        for depth, table in enumerate(cache.plane_block_tables(rid)):
            n = start // BLOCK
            expected = tuple(group[depth] for group in donor_prefix[:n])
            require(
                len(expected) == n and table[:n] == expected,
                f"{rid} wrong donor pages at depth {depth}",
            )
            private.extend(table[n:])
    require(
        len(private) == len(set(private)) and donor_pages.isdisjoint(private),
        "live private pages overlap another request or a shared prefix",
    )


@torch.inference_mode()
def run_arm(model, case, arm, primer, targets):
    gc.collect()
    engine = open_engine(model, case, arm)
    cache = engine.cache_manager
    try:
        require(
            engine.execution_config.packed_prefill and cache.backend == "ascend_eager",
            "expected mainline packed ascend_eager",
        )
        require(
            cache._prepared_coverage is not None
            and cache.dtype
            == cache.key_cache.dtype
            == cache.value_cache.dtype
            == model.lm_head.weight.dtype,
            "expected model dtype in KV and Ascend range bookkeeping",
        )
        params = SamplingParams(
            max_tokens=OUTPUTS,
            temperature=0,
            exit_threshold=1.0,
            min_loops=4,
            max_loops=4,
            ignore_eos=True,
        )
        engine.add_request("donor", primer, replace(params, max_tokens=1))
        drain(engine, ["donor"], 1)
        donor_prefix = cache.lookup_prefix(primer)
        expected_published = (len(primer) - 1) // BLOCK if arm != "off" else 0
        require(len(donor_prefix) == expected_published, "incorrect published donor page count")

        before = cache.prefix_hits
        starts = case.starts(arm)
        length = len(primer)
        for rid, prompt in targets.items():
            engine.add_request(rid, prompt, params)
        audit = PrefillAudit(cache, starts, length)
        with audit.observe():
            first = engine.step()
        batch = engine.last_schedule
        require(batch.phase == RequestPhase.PREFILL, "expected an ordinary prefill")
        origins = {s.request.request_id: s.request.prefill_window_origin for s in batch.requests}
        actual_starts = {s.request.request_id: s.token_start for s in batch.requests}
        require(
            actual_starts == starts and origins == starts,
            f"target starts={actual_starts}, hit tokens={origins}, expected={starts}",
        )
        require(cache.prefix_hits - before == sum(starts.values()), "wrong aggregate hit tokens")
        check_pages(cache, starts, donor_prefix)
        path = audit.result()
        outputs = drain(engine, targets, OUTPUTS, first)
        return dict(
            hit_tokens=origins,
            prefill_start=actual_starts,
            prefill_rows=batch.num_tokens,
            audit=path,
            requests=outputs,
        )
    finally:
        engine.close()


def token_diffs(reference, actual, comparison):
    diffs = []
    for rid, expected in reference["requests"].items():
        for pos, (a, b) in enumerate(
            zip(expected["tokens"], actual["requests"][rid]["tokens"], strict=True), 1
        ):
            if a != b:
                diffs.append(
                    dict(
                        comparison=comparison,
                        request=rid,
                        first_diff_1based=pos,
                        expected=a,
                        actual=b,
                    )
                )
                break
    return diffs


def environment(device, *, dtype=torch.bfloat16):
    import torch_npu

    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    from vllm_rlt import __file__ as runtime_file

    require(
        Path(runtime_file).resolve().parent.parent == root,
        "benchmark and runtime are from different trees",
    )
    try:
        from torch_npu.utils.collect_env import get_cann_version

        cann = str(get_cann_version())
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        cann = None
    return dict(
        commit=git("rev-parse", "HEAD"),
        tracked_changes=git("status", "--porcelain", "-uno"),
        torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__),
        cann=cann,
        chip=torch.npu.get_device_name(device.index),
        visible=os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        seed=SEED,
        block_size=BLOCK,
        output_tokens=OUTPUTS,
        loops=4,
        dtype=str(dtype).removeprefix("torch.") if dtype is not None else None,
        execution="sync ascend_eager, mainline default packing, no experimental subgroup batching",
    )


def run_model_cases(model, args, names, record, location, oracle_failed):
    settings = precision_settings(args.linear_fp32_chunk, dtype=model.lm_head.weight.dtype)
    parity_failed, controls_failed, repeat_drift = set(), set(), set()
    location.update(phase="install_linears", **settings)
    with padded_fp32_linears(model, args.linear_fp32_chunk) as work:
        if work is not None:
            installed, calls = work
            record(
                "COMBO_LINEAR",
                modules=len(installed),
                **settings,
                has_exit_gate="model.early_exit_gate" in installed,
                has_lm_head="lm_head" in installed,
                output_dtype="bfloat16",
            )
            record("COMBO_LINEAR_MODULES", echo=False, names=installed)
        for name in names:
            case = CASES[name]
            donor, miss, targets = prompts(case, model.config.vocab_size)
            hashes = {
                rid: hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()
                for rid, ids in targets.items()
            }
            record("COMBO_INPUTS", echo=False, case=name, prompt_sha256=hashes)
            first = {}
            for repeat in range(1, args.repeats + 1):
                arms, controls, diffs = {}, {}, []
                for arm in ARMS:
                    location.update(phase="targets", case=name, repeat=repeat, arm=arm)
                    fields = dict(**settings, case=name, repeat=repeat, arm=arm)
                    try:
                        if work is not None:
                            calls.clear()
                        result = run_arm(
                            model, case, arm, miss if arm == "miss" else donor, targets
                        )
                        if work is not None:
                            result["linear_work"] = linear_usage(work)
                            check_linear_usage(result["linear_work"])
                    except AssertionError as exc:
                        controls[arm] = str(exc)
                        record("COMBO_CONTROL", **fields, detail=str(exc))
                        continue
                    arms[arm] = result
                    record("COMBO_ARM", echo=False, **fields, **result)
                    if repeat == 1:
                        first[arm] = result
                    elif arm in first:
                        changes = token_diffs(first[arm], result, f"{arm}_vs_repeat1")
                        if changes:
                            repeat_drift.add(name)
                            diffs.extend(changes)
                for arm in ("miss", "reuse"):
                    if "off" in arms and arm in arms:
                        changes = token_diffs(arms["off"], arms[arm], f"off_vs_{arm}")
                        if changes:
                            parity_failed.add(name)
                            diffs.extend(changes)
                if controls:
                    controls_failed.add(name)
                reuse = arms.get("reuse", {})
                record(
                    "COMBO_CASE",
                    **settings,
                    case=name,
                    repeat=repeat,
                    status="FAIL" if diffs or controls else "PASS",
                    hit_tokens=reuse.get("hit_tokens"),
                    audit=reuse.get("audit"),
                    prefill_rows={a: r["prefill_rows"] for a, r in arms.items()},
                    controls=controls,
                    first_diffs=diffs,
                )
    failed = parity_failed | controls_failed | repeat_drift
    if oracle_failed:
        failed.add("page_oracle")
    record(
        "COMBO_REVIEW",
        **settings,
        status="FAIL" if failed else "PASS",
        cases=names,
        repeats=args.repeats,
        failed=sorted(failed),
        parity_failed=sorted(parity_failed),
        controls_failed=sorted(controls_failed),
        repeat_drift=sorted(repeat_drift),
    )
    return int(bool(failed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Ouro-1.4B checkpoint")
    parser.add_argument("--case", choices=CASES, action="append", help="Repeat to select cases")
    parser.add_argument("--repeats", type=int, default=2)
    add_precision_arg(parser)
    parser.add_argument(
        "--out", type=Path, help="New JSONL file (default: results/main-prefix-combo-<UTC>.jsonl)"
    )
    args = parser.parse_args(argv)
    if args.repeats < 1 or not args.model.is_dir():
        parser.error("provide a local Ouro checkpoint and positive --repeats")
    names = list(dict.fromkeys(args.case or CASES))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.out or Path("results") / f"main-prefix-combo-{stamp}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    failed_so_far = set()
    with path.open("x", encoding="utf-8") as stream:
        print("RESULTS", path.resolve(), flush=True)

        def record(tag, *, echo=True, **data):
            if data.get("status") == "FAIL" and tag in ("COMBO_CASE", "COMBO_ORACLE"):
                failed_so_far.add(data.get("case", "page_oracle"))
            line = json.dumps(dict(kind=tag, **data), ensure_ascii=False)
            print(line, file=stream, flush=True)
            if echo:
                print(tag, json.dumps(data, ensure_ascii=False), flush=True)

        location = dict(phase="setup")
        dtype = benchmark_dtype()
        try:
            device = resolve_device("npu:0")
            torch.npu.set_device(device)
            matmul = getattr(torch.npu, "matmul", None)
            record(
                "COMBO_ENV",
                **environment(device, dtype=dtype),
                cases=names,
                repeats=args.repeats,
                **precision_settings(args.linear_fp32_chunk, dtype=dtype),
                native_allow_hf32=getattr(matmul, "allow_hf32", None),
            )
            location = dict(phase="page_oracle")
            attention = create_backend(
                "ascend_eager", device, dtype, 64, BLOCK, prefer_packed_prefill=True
            )
            probe = KVCacheManager(
                2,
                2,
                64,
                32,
                BLOCK,
                4,
                device=device,
                dtype=dtype,
                attention=attention,
                enable_prefix_caching=True,
            )
            oracle_failed = False
            try:
                page_oracle(probe)
                record("COMBO_ORACLE", status="PASS", layers=2, depths=4)
            except AssertionError as exc:
                oracle_failed = True
                record("COMBO_ORACLE", status="FAIL", detail=str(exc))
            del probe
            location = dict(phase="checkpoint")
            record(
                "COMBO_CHECKPOINT",
                echo=False,
                files=json.loads(Path(identity).read_text())
                if (identity := os.getenv("RLT_BENCH_CHECKPOINT_JSON"))
                else checkpoint_identity(args.model),
            )
            model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=dtype).eval()
            return run_model_cases(model, args, names, record, location, oracle_failed)
        except Exception as exc:
            record("COMBO_ERROR", **location, error=type(exc).__name__, detail=str(exc))
            record(
                "COMBO_REVIEW",
                status="INCOMPLETE",
                stopped_at=location,
                **precision_settings(args.linear_fp32_chunk, dtype=dtype),
                failed=sorted(failed_so_far),
            )
            traceback.print_exc()
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
