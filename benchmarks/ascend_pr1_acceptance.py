#!/usr/bin/env python3
r"""Seven-category, BF16-first Ouro prefix-cache acceptance for Ascend 910B3/B4.

Run with Python 3.10+ in the project's CANN 9.0.0 / torch-npu environment.
The CLI for seeds, plan and report needs only Python's standard library.
W1-W5 reuse the lengths, concurrency, resident limits and measurement windows
from docs/recipes/ouro_ascend.md at 6e5392d. PR1 uses ordinary prefill and 16-token
KV pages. Fixed cold/hit cohorts and the self-warming closed loop are recorded
separately. The ten fixed seeds select the prompt token IDs for both chips.

  git clone --single-branch -b test/ascend-pr1-classified https://github.com/NEMOyota/vllm-rlt.git
  cd vllm-rlt
  export MODEL=/models/Ouro-1.4B
  python benchmarks/ascend_pr1_acceptance.py plan --chip 910B4
  python benchmarks/ascend_pr1_acceptance.py run --chip 910B4 --categories 3 --cases W1,W2,W3,W4,W5
  python benchmarks/ascend_pr1_acceptance.py run --chip 910B4
  python benchmarks/ascend_pr1_acceptance.py report results-910B3 results-910B4

Both machines use the same built-in seed list. To export it, run
`python benchmarks/ascend_pr1_acceptance.py seeds pr1-fixed-seeds.json`.
On either chip the seeds rotate across the selected cards in their listed
order. All cache arms, repeats and precision replays for a seed stay on its
assigned card. Fixed checks run once on the first card.
The checkout includes benchmarks/ascend_pr1_package/pr1-fixed10-tests.bundle with
candidate 606a60c, its regression base 4821d47 and the pinned harness. The runner
reads missing Git objects from this bundle into a temporary checkout. To test
another local checkout or bundle, use --repo or --candidate-bundle together
with --candidate. The report records the full SHA used for inference.
Select categories with --categories 1,2,4. Report paths are relative to the run.
The W1-W5 phase starts each workload after the preceding workload's workers exit
on all selected cards. Each workload/seed has a fresh process. Within that
process each replay releases its engine and KV pool outside the measured window.
Exit codes: 0 selected checks passed, 1 measured failures, 2 incomplete execution.
"""

import argparse
import asyncio
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import weakref
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree

HARNESS_SHA = "136a775e02c8c63ac5335970189a1d50b6c3d4f2"
HARNESS_URL = "https://github.com/NEMOyota/vllm-rlt.git"
CANDIDATE_SHA = "606a60ce9bacdd5df1d25168d38665b3a0ef106d"
UPSTREAM_SHA = "4821d47cec92aa1d7d2ec4be5f589d19b2794ae3"
FIXED_SEEDS = (
    114281796,
    273728779,
    355234819,
    421559278,
    641794792,
    770929007,
    797543261,
    1179450028,
    1560366951,
    1727668039,
)
SCRIPT = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT.parent / "ascend_pr1_package"
CANDIDATE_BUNDLE_NAME = "ascend-ouro-prefix-cache.bundle"
MODULE = "ascend_pr1_acceptance"
CATEGORIES = {
    1: "功能正确性",
    2: "数值正确性",
    3: "性能",
    4: "边界与异常",
    5: "兼容性",
    6: "回归",
    7: "并发与稳定性",
}
ARMS = ("off", "miss", "reuse")
DIAGNOSTICS = ("fp16", "fp32-linear")
BLOCK = 16
DELAYED = (
    "tests/test_npu_async.py::test_npu_delayed_delivery_consumes_device_token_before_host_result"
)
MATRIX = (
    *[f"p{p}_s8" for p in (128, 256, 384, 512, 1024)],
    *[f"edge_p{p}" for p in (0, 15, 16, 17, 127, 129)],
    "repeat_p128",
    *[f"total512_p{p}" for p in (0, 128, 256, 384, 512)],
    "p128_s1",
    "p128_s32",
    "p128_s128",
    "c1_h1",
    "c2_h1",
    "c2_h2",
    "c4_h0",
    "c4_h1",
    "c4_h2",
    "c4_h4",
    "resident_p64",
    "evicted_p64",
    "c4_h3",
)
EDGES = {n for n in MATRIX if n.startswith("edge_")} | {
    "repeat_p128",
    "p128_s1",
    "resident_p64",
    "evicted_p64",
}
# input, output, clients, maximum residents, shared prefix
WORKLOADS = {
    "W1": (128, 128, 1, 1, 96),
    "W2": (128, 128, 32, 32, 96),
    "W3": (4096, 128, 1, 1, 3072),
    "W4": (4096, 128, 32, 4, 3072),
    "W5": (1024, 512, 8, 8, 768),
}
STATIC = {
    "lint",
    "cpu-suite",
    "baseline-delayed",
    "candidate-delayed",
    "npu-suite",
    "config-guards",
    "http-disconnect",
}
HELPERS = (
    "ascend_main_prefix_combo",
    "ascend_main_prefix_grouping",
    "ascend_main_prefix_linear",
    "ascend_main_prefix_matrix",
    "ascend_main_prefix_results",
    "ascend_main_prefix_service",
    "ascend_main_prefix_service_worker",
    "ascend_prefix_combo_checks",
    "ascend_prefix_opportunities",
    "ascend_prefix_pr1_hardware",
    "ascend_prefix_pr1_diagnostics",
    "preemption",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    os.replace(tmp, path)


def digest(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *map(str, args)], text=True).strip()


def commit_sha(repo, ref):
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "--quiet",
            "--verify",
            "--end-of-options",
            ref + "^{commit}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def seed_list(path):
    seeds = list(FIXED_SEEDS)
    if path is not None:
        manifest = read(path)
        values = manifest.get("seeds") if isinstance(manifest, dict) else None
        if values != seeds or any(type(n) is not int for n in values):
            raise ValueError(f"{path} must contain the ten fixed PR1 integer seeds in order")
    return seeds


@dataclass(frozen=True)
class Job:
    groups: tuple
    case: str
    seed: int | None
    card: int

    @property
    def key(self):
        return f"{self.case}/" + ("fixed" if self.seed is None else f"seed-{self.seed}")

    @property
    def kind(self):
        if self.case in STATIC:
            return "pytest"
        return "http" if self.case.startswith("http-") else "engine"


def make_jobs(seeds, cards, selected):
    jobs = []

    def add(groups, case, seed, card):
        groups = tuple(g for g in groups if g in selected)
        if groups:
            jobs.append(Job(groups, case, seed, card))

    add((5,), "config-guards", None, cards[0])
    for name in ("lint", "cpu-suite", "baseline-delayed", "candidate-delayed", "npu-suite"):
        add((6,), name, None, cards[0])
    add((7,), "http-disconnect", None, cards[0])
    add((1,), "page-oracle", None, cards[0])
    for i, seed in enumerate(seeds):
        card = cards[i % len(cards)]
        add((1,), "shared-pages", seed, card)
        for name in MATRIX:
            add((2, 4) if name in EDGES else (2,), name, seed, card)
        for name in WORKLOADS:
            add((3,), name, seed, card)
        for name in ("http-burst", "http-paced"):
            add((3,), name, seed, card)
        for name in ("one-token", "failed-admission"):
            add((4,), name, seed, card)
        for name in ("packed", "per-row"):
            add((5,), name, seed, card)
        add((7,), "cancel-reuse-evict", seed, card)
    add((3,), "W3-trace", seeds[0], cards[0])
    return jobs


def rng_for(seed, label):
    return random.Random(f"ascend-pr1/{seed}/{label}")


def params(count):
    from vllm_rlt import SamplingParams

    return SamplingParams(
        max_tokens=count,
        temperature=0,
        min_loops=4,
        max_loops=4,
        exit_threshold=1.0,
        ignore_eos=True,
    )


def new_engine(model, enabled, blocks, residents, width, packed=True):
    from vllm_rlt import CacheConfig, ExecutionConfig, SchedulerConfig
    from vllm_rlt.engine.stage_engine import StageEngine

    return StageEngine(
        model,
        attention_backend="ascend_eager",
        cache_config=CacheConfig(
            num_blocks=blocks, block_size=BLOCK, enable_prefix_caching=enabled
        ),
        execution_config=ExecutionConfig(multi_stream=False, use_uva=False, packed_prefill=packed),
        scheduler_config=SchedulerConfig(
            max_num_seqs=residents,
            prefill_chunk_size=width,
            max_num_batched_tokens=width * residents,
        ),
    )


def occupancy(cache):
    # num_free_blocks includes evictable, cache-only blocks; _free_blocks excludes them.
    free = cache.num_free_blocks
    return {"active": cache.num_blocks - free, "cache_only": free - len(cache._free_blocks)}


def memory(model, *, reset=False):
    device = model.lm_head.weight.device
    if device.type == "cpu":  # Allows tiny CPU regression fixtures to exercise the driver.
        return {}
    import torch

    if reset:
        torch.npu.synchronize(device)
        torch.npu.reset_peak_memory_stats(device)
    return dict(
        peak_allocated=torch.npu.max_memory_allocated(device),
        peak_reserved=torch.npu.max_memory_reserved(device),
        allocated=torch.npu.memory_allocated(device),
        reserved=torch.npu.memory_reserved(device),
    )


def release_replay_memory(model):
    import torch

    device = model.lm_head.weight.device
    if device.type == "npu":
        torch.npu.synchronize(device)
    # StageEngine and PreemptionManager reference one another even after close().
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()


def generate(engine, rid, tokens, count):
    from benchmarks.ascend_main_prefix_combo import drain

    before = engine.cache_manager.prefix_hits
    engine.add_request(rid, tokens, params(count))
    row = drain(engine, [rid], count)[rid]
    return row["tokens"], engine.cache_manager.prefix_hits - before


def ids_fingerprint(inputs):
    from benchmarks.ascend_main_prefix_results import ids_digest

    donor, miss, targets, fillers = inputs
    return dict(
        donor=ids_digest(donor),
        miss=ids_digest(miss),
        targets={rid: ids_digest(ids) for rid, ids in targets.items()},
        fillers=[ids_digest(ids) for ids in fillers],
    )


def record_inputs(folder, data):
    if folder is not None:
        write(folder / "inputs.json", data)
    return data


def first_difference(left, right):
    for i, (a, b) in enumerate(zip(left, right), 1):
        if a != b:
            return dict(position=i, expected=a, actual=b)
    if len(left) != len(right):
        i = min(len(left), len(right))
        return dict(
            position=i + 1,
            expected=left[i] if i < len(left) else None,
            actual=right[i] if i < len(right) else None,
        )
    return None


# 1. Page publication, ownership and real-weight suffix isolation.
def functional(model, job, folder=None):
    from benchmarks import ascend_main_prefix_combo as combo

    if job.case == "page-oracle":
        from benchmarks.ascend_prefix_combo_checks import page_oracle
        from vllm_rlt.attention import create_backend
        from vllm_rlt.core.kv_cache_manager import KVCacheManager

        weights = model.lm_head.weight
        backend = create_backend(
            "ascend_eager", weights.device, weights.dtype, 64, BLOCK, prefer_packed_prefill=True
        )
        cache = KVCacheManager(
            2,
            2,
            64,
            32,
            BLOCK,
            4,
            device=weights.device,
            dtype=weights.dtype,
            attention=backend,
            enable_prefix_caching=True,
        )
        page_oracle(cache)
        return dict(layers=2, depths=4, blocks=occupancy(cache))
    case = combo.CASES["c4_h3"]
    inputs = combo.workload(case, model.config.vocab_size, seed=job.seed)
    fingerprint = record_inputs(folder, ids_fingerprint(inputs))
    donor, miss, targets, _ = inputs
    arms = {
        arm: combo.run_arm(model, case, arm, miss if arm == "miss" else donor, targets)
        for arm in ARMS
    }
    require([arms[a]["prefill_rows"] for a in ARMS] == [576, 576, 192], "c4_h3 prefill rows")
    return dict(inputs=fingerprint, arms=arms)


# 2. Complete, same-precision greedy replay and bounded same-position attribution.
@contextmanager
def observe_rope(targets, starts, values):
    """Read post-RoPE K/Q at the actual KV/attention boundary on aligned suffix rows."""
    from vllm_rlt.core.kv_cache_manager import KVCacheManager

    write_k, attend = KVCacheManager._write_prepared, KVCacheManager._attend_prepared

    def record(cache, layer, batch, tensor, name):
        names = {id(alloc): rid for rid, alloc in batch.allocations}
        for i, (alloc, depth, pos) in enumerate(batch.rows):
            rid = names[id(alloc)]
            if rid not in targets or pos not in (starts[rid], len(targets[rid]) - 1):
                continue
            key = (rid, depth, layer, pos, name)
            values.setdefault(key, tensor[i].detach().cpu().clone())
            if name == "K" and starts[rid]:
                last = starts[rid] - 1
                block = cache.get_block_table(rid, depth)[last // BLOCK]
                values.setdefault(
                    (rid, depth, layer, last, "prefix_K"),
                    cache.key_cache[block, layer, last % BLOCK].detach().cpu().clone(),
                )

    def save_k(cache, layer, batch, k, v, *args, **kw):
        result = write_k(cache, layer, batch, k, v, *args, **kw)
        record(cache, layer, batch, k, "K")
        return result

    def save_q(cache, layer, batch, q, *args, **kw):
        record(cache, layer, batch, q, "Q")
        return attend(cache, layer, batch, q, *args, **kw)

    with (
        patch.object(KVCacheManager, "_write_prepared", save_k),
        patch.object(KVCacheManager, "_attend_prepared", save_q),
    ):
        yield


def diagnose(model, case, inputs, work):
    from benchmarks import ascend_prefix_pr1_diagnostics as diag
    from benchmarks.ascend_main_prefix_combo import token_diffs

    starts = case.starts("reuse")
    reference = {}
    with observe_rope(inputs[2], starts, reference):
        baseline, scores = diag.trace_logits(model, case, inputs, "off", linear_work=work)
    expected = {
        (rid, depth, layer, pos, axis)
        for rid in starts
        for depth in range(4)
        for layer in range(model.config.num_hidden_layers)
        for pos in {starts[rid], len(inputs[2][rid]) - 1}
        for axis in ("Q", "K")
    }
    expected.update(
        (rid, depth, layer, start - 1, "prefix_K")
        for rid, start in starts.items()
        if start
        for depth in range(4)
        for layer in range(model.config.num_hidden_layers)
    )
    require(set(reference) == expected, "incomplete reference post-RoPE observations")
    result = {}
    for arm in ("miss", "reuse"):
        values = {}
        with observe_rope(inputs[2], starts, values):
            current, observed = diag.trace_logits(model, case, inputs, arm, linear_work=work)
        require(values.keys() == reference.keys(), "post-RoPE sample keys differ")
        deltas = []
        for key, left in reference.items():
            delta = float((left.float() - values[key].float()).abs().max())
            if delta != 0:
                deltas.append(dict(key=key, max_abs=delta if math.isfinite(delta) else "nonfinite"))
        issues = token_diffs(baseline, current, f"off_vs_{arm}")
        for issue in issues:
            rid, step = issue["request"], issue["first_diff_1based"] - 1
            pair = [issue["expected"], issue["actual"]]
            issue["off_logits"] = diag.margin(scores[rid][step][1], pair)
            issue["arm_logits"] = diag.margin(observed[rid][step][1], pair)
        result[arm] = dict(sampled=len(reference), unequal=deltas, token_differences=issues)
    return result


def numerical(model, job, folder, work):
    import torch

    from benchmarks import ascend_main_prefix_matrix as matrix
    from benchmarks.ascend_main_prefix_results import Report

    require(set(MATRIX) == set(matrix.FULL), "historical 30-case matrix differs from the manifest")
    case = matrix.FULL[job.case]
    inputs = matrix.combo.workload(case, model.config.vocab_size, seed=job.seed)
    fingerprint = record_inputs(folder, ids_fingerprint(inputs))
    chunk = 256 if work is not None else None
    report = Report("CORRECTNESS", folder / "matrix", linear_fp32_chunk=chunk)
    finite = []
    with ExitStack() as stack:
        stack.enter_context(closing(report.stream))
        if model.lm_head.weight.dtype == torch.float16:

            def check_logits(_module, _inputs, logits):
                finite.append(torch.isfinite(logits).all())

            stack.enter_context(model.lm_head.register_forward_hook(check_logits))
        matrix.run_cases(
            SimpleNamespace(experiment="correctness", warmups=0, repeats=2, correctness_rounds=2),
            model,
            report,
            [job.case],
            linear_work=work,
            modes=("main",),
            seed=job.seed,
        )
    events = [json.loads(line) for line in (report.folder / "raw.jsonl").read_text().splitlines()]
    issues = next(e["all_issues"] for e in events if e["kind"] == "CASE_DETAILS")
    if finite and not torch.stack(finite).all().item():
        issues.append({"precision": "non-finite FP16 logits in a formal trial"})
    trials = {(e["round"], e["arm"]) for e in events if e["kind"] in ("TRIAL", "CONTROL")}
    if trials != {(n, a) for n in (1, 2) for a in ARMS}:
        raise RuntimeError("missing formal trial or control record")
    verdicts = {g: "PASS" for g in job.groups}
    if issues and 2 in verdicts:
        verdicts[2] = "FAIL"
    if any("control" in e for e in issues) and 4 in verdicts:
        verdicts[4] = "FAIL"
    data = dict(inputs=fingerprint, issues=issues, verdicts=verdicts)
    if model.lm_head.weight.dtype == torch.float16:
        if not finite and not issues:
            raise RuntimeError("FP16 replay produced no observable logits")
        data["checked_logits_batches"] = len(finite)
    if issues and 2 in verdicts and not any("control" in e for e in issues):
        try:
            write(folder / "same-history.json", diagnose(model, case, inputs, work))
        except Exception as exc:
            data["attribution_incomplete"] = f"{type(exc).__name__}: {exc}"
    return data


# 3. Fixed-shape controlled hits, closed-loop engine timing, HTTP and W3 traces.
def performance_inputs(model, seed, shape, count):
    length, _, _, _, prefix = shape
    vocab = model.config.vocab_size
    if count + 2 >= vocab:
        raise ValueError("performance stream needs more distinct suffix markers")
    rng = rng_for(seed, f"W/{length}")
    shared = rng.choices(range(vocab), k=prefix)
    # Distinct first suffix tokens stop later targets from extending a real hit.
    marker = rng.randrange(vocab)
    donor = shared + [(marker - 1) % vocab] + rng.choices(range(vocab), k=length - prefix - 1)
    targets = [
        shared + [(marker + index) % vocab] + rng.choices(range(vocab), k=length - prefix - 1)
        for index in range(count)
    ]
    miss = donor.copy()
    miss[0] = (miss[0] + 1) % vocab
    return donor, miss, targets


def percentile(values, fraction):
    ordered = sorted(values)
    x = (len(ordered) - 1) * fraction
    lo = int(x)
    return ordered[lo] + (ordered[math.ceil(x)] - ordered[lo]) * (x - lo)


def time_targets(engine, targets, count, clients, *, seconds=None, fixed=None, audit_kv=False):
    """Closed loop with an external client limit, followed by a complete drain."""
    rows = {}
    started = {}
    first = {}
    hits = {}
    index = 0
    kv_peak = {"active": 0, "cache_only": 0}
    begin = time.perf_counter()
    deadline = begin + seconds if seconds is not None else None
    while True:
        while len(started) - len(rows) < clients:
            if fixed is not None and index >= fixed:
                break
            if deadline is not None and index >= clients and time.perf_counter() >= deadline:
                break
            if index == len(targets):
                raise RuntimeError("exhausted the frozen input stream; increase --stream-size")
            rid = str(index)
            started[rid] = time.perf_counter()
            engine.add_request(rid, targets[index], params(count))
            index += 1
        if not engine.has_unfinished_requests():
            break
        outputs = engine.step()
        visible = time.perf_counter()
        schedule = engine.last_schedule
        if schedule:
            for span in schedule.requests:
                r = span.request
                if r.request_id in started:
                    hits.setdefault(r.request_id, r.prefill_window_origin)
        if audit_kv:
            for name, n in occupancy(engine.cache_manager).items():
                kv_peak[name] = max(kv_peak[name], n)
        for out in outputs:
            rid = out.request_id
            if out.token_ids:
                first.setdefault(rid, visible)
            if out.finished:
                require(
                    rid not in rows
                    and len(out.token_ids) == count
                    and out.exit_depths == [4] * count
                    and out.finish_reason == "length",
                    f"{rid}: incorrect completion, depth or output length",
                )
                rows[rid] = dict(
                    tokens=list(out.token_ids),
                    hits=hits[rid],
                    ttft_ms=1000 * (first[rid] - started[rid]),
                    tpot_ms=1000 * (visible - first[rid]) / (count - 1),
                )
    engine.model_runner.synchronize()
    elapsed = time.perf_counter() - begin
    require(
        index > 0 and len(rows) == index and engine.cache_manager.num_used_blocks == 0,
        "timed stream failed to drain",
    )
    return dict(
        requests=rows,
        wall_s=elapsed,
        output_tok_s=count * index / elapsed,
        requests_s=index / elapsed,
        sampled_kv_peaks=kv_peak if audit_kv else None,
    )


def engine_performance(model, job, cfg, folder):
    from benchmarks.ascend_main_prefix_results import ids_digest

    length, outputs, clients, residents, prefix = shape = WORKLOADS[job.case]
    donor, miss, targets = performance_inputs(model, job.seed, shape, cfg["stream_size"])
    fingerprint = record_inputs(
        folder, digest([ids_digest(donor), ids_digest(miss), [ids_digest(t) for t in targets]])
    )
    # Enough reserved blocks for the live cohort plus its primer in all arms.
    blocks = (residents + 1) * 4 * math.ceil((length + outputs - 1) / BLOCK)
    width = length + outputs - 1
    issues, measurements = [], []
    reference, first_by_arm = {}, {}
    largest = 0

    def replay(arm, phase, round_number, *, seconds=None, fixed=None, resource=False):
        def record(stage, **extra):
            entry = dict(phase=phase, round=round_number, arm=arm, stage=stage, pid=os.getpid())
            with (folder / "replay-memory.jsonl").open("a") as stream:
                print(json.dumps(dict(entry, **memory(model), **extra)), file=stream)

        record("before_create")
        with closing(new_engine(model, arm != "off", blocks, residents, width)) as engine:
            pools = (
                weakref.ref(engine.cache_manager.key_cache),
                weakref.ref(engine.cache_manager.value_cache),
            )
            record("after_create")
            generate(engine, "donor", miss if arm == "miss" else donor, 1)
            engine.model_runner.synchronize()
            before = occupancy(engine.cache_manager)
            initial_memory = memory(model, reset=True)
            run = time_targets(
                engine,
                targets,
                outputs,
                residents if resource else clients,
                seconds=seconds,
                fixed=fixed,
                audit_kv=resource,
            )
            run.update(
                after_prime=before,
                after_drain=occupancy(engine.cache_manager),
                initial_memory=initial_memory,
                memory=memory(model),
            )
            for rid, row in run["requests"].items():
                actual = row["hits"]
                allowed = {0} if arm == "off" else {prefix}
                if arm == "miss":
                    # An initially cold enabled engine publishes the measured prompts.
                    allowed = {0} if resource or rid == "0" else {0, prefix}
                if actual not in allowed:
                    issues.append(
                        dict(
                            arm=arm,
                            request=rid,
                            actual_hit=actual,
                            expected_hits=sorted(allowed),
                            controlled=resource,
                        )
                    )
        del engine
        release_replay_memory(model)
        released = all(ref() is None for ref in pools)
        record("after_cleanup", kv_released=released)
        if not released:
            raise RuntimeError(f"{phase} round {round_number} {arm}: previous KV pool still live")
        return run

    # Each controlled cohort is freshly primed. No client waits behind another
    # cohort and turns an intended miss into a hit. These also warm each shape.
    release_replay_memory(model)
    controlled_refs, controlled = {}, []
    for index in range(3):
        for arm in ARMS[index:] + ARMS[:index]:
            run = replay(arm, "controlled", index + 1, fixed=residents, resource=True)
            write(folder / f"controlled-{index + 1}-{arm}.json", run)
            batch = run["requests"].values()
            controlled.append(
                dict(
                    arm=arm,
                    round=index + 1,
                    hits=dict(Counter(r["hits"] for r in batch)),
                    ttft_p50_ms=percentile([r["ttft_ms"] for r in batch], 0.5),
                    tpot_p50_ms=percentile([r["tpot_ms"] for r in batch], 0.5),
                    memory=run["memory"],
                    sampled_kv_peaks=run["sampled_kv_peaks"],
                    after_drain=run["after_drain"],
                )
            )
            controlled_refs[index, arm] = run["requests"]
    for (index, arm), rows in controlled_refs.items():
        for rid, row in rows.items():
            diff = first_difference(controlled_refs[index, "off"][rid]["tokens"], row["tokens"])
            if diff:
                issues.append(dict(round=index + 1, arm=arm, request=rid, controlled=True, **diff))
            if index and (
                diff := first_difference(controlled_refs[0, arm][rid]["tokens"], row["tokens"])
            ):
                issues.append(
                    dict(
                        round=index + 1, arm=arm, request=rid, controlled=True, repeat=True, **diff
                    )
                )

    for index in range(3):
        for arm in ARMS[index:] + ARMS[:index]:
            run = replay(arm, "window", index + 1, seconds=cfg["window"])
            rows = run["requests"]
            largest = max(largest, len(rows))
            for rid, row in rows.items():
                tokens = row["tokens"]
                first_tokens = first_by_arm.setdefault((arm, rid), tokens)
                if diff := first_difference(first_tokens, tokens):
                    issues.append(dict(round=index + 1, arm=arm, request=rid, repeat=True, **diff))
                if arm == "off":
                    reference.setdefault(rid, tokens)
            ttft = [r["ttft_ms"] for r in rows.values()]
            tpot = [r["tpot_ms"] for r in rows.values()]
            measures = dict(
                round=index + 1,
                arm=arm,
                count=len(rows),
                wall_s=run["wall_s"],
                output_tok_s=run["output_tok_s"],
                requests_s=run["requests_s"],
                ttft_p50_ms=percentile(ttft, 0.5),
                ttft_p95_ms=percentile(ttft, 0.95),
                tpot_p50_ms=percentile(tpot, 0.5),
                tpot_p95_ms=percentile(tpot, 0.95),
                observed_hits=dict(Counter(r["hits"] for r in rows.values())),
                memory=run["memory"],
                after_drain=run["after_drain"],
            )
            measurements.append(measures)
            write(folder / f"window-{index + 1}-{arm}.json", run)
    if len(reference) < largest:
        extra = replay("off", "supplemental", 1, fixed=largest)
        write(folder / "supplemental-off.json", extra)
        for rid, row in extra["requests"].items():
            reference.setdefault(rid, row["tokens"])
    for (arm, rid), tokens in first_by_arm.items():
        if diff := first_difference(reference[rid], tokens):
            issues.append(dict(comparison=f"off_vs_{arm}", request=rid, **diff))
    rates = {}
    for arm in ARMS:
        samples = [row["output_tok_s"] for row in measurements if row["arm"] == arm]
        rates[arm] = dict(median=statistics.median(samples), low=min(samples), high=max(samples))
    return dict(
        inputs=fingerprint,
        shape=shape,
        kv_blocks=blocks,
        window_seconds=cfg["window"],
        controlled_clients=residents,
        controlled_target_count=3 * residents,
        window_miss_semantics="cold start; see per-request hits after publication",
        output_tok_s=rates,
        controlled=controlled,
        measurements=measurements,
        issues=issues,
    )


def w3_trace(model, job, folder):
    from contextlib import nullcontext

    from benchmarks import ascend_main_prefix_combo as combo
    from benchmarks import ascend_prefix_pr1_diagnostics as diag

    donor, miss, rows = performance_inputs(model, job.seed, WORKLOADS["W3"], 1)
    inputs = (donor, miss, {"target-0": rows[0]}, [])
    fingerprint = record_inputs(folder, ids_fingerprint(inputs))
    case = combo.Case(3072, 1024)
    traces = {}
    for arm in ("off", "reuse"):
        release_replay_memory(model)
        diag.prefill_window(model, case, inputs, arm, nullcontext())
        release_replay_memory(model)
        path = folder / arm
        path.mkdir()
        result = diag.prefill_window(model, case, inputs, arm, diag.capture_context(path, True))
        traces[arm] = dict(prefill=result, files=diag.checked_profile_files(path, True))
    return dict(inputs=fingerprint, traces=traces)


def http_worker(mode, audit, *args, **kwargs):
    import torch
    import torch_npu  # noqa: F401

    from benchmarks import _pr1_pinned_http_worker
    from vllm_rlt.engine.stage_engine import StageEngine

    torch.npu.matmul.allow_hf32 = False
    path = Path(audit).with_suffix(".precision.json")
    metrics = {"allow_hf32": bool(torch.npu.matmul.allow_hf32)}
    write(path, metrics)
    stage_init, stage_close = StageEngine.__init__, StageEngine.close

    def initialize(engine, *params, **options):
        stage_init(engine, *params, **options)
        metrics["before_traffic"] = memory(engine.model, reset=True)

    def close(engine):
        metrics.update(memory=memory(engine.model), after_drain=occupancy(engine.cache_manager))
        write(path, metrics)
        return stage_close(engine)

    with (
        patch.object(StageEngine, "__init__", initialize),
        patch.object(StageEngine, "close", close),
    ):
        return _pr1_pinned_http_worker.observed_worker(mode, audit, *args, **kwargs)


def http_main():
    from vllm_rlt.engine import worker
    from vllm_rlt.entrypoints import serve

    rows = os.getenv("RLT_BENCH_LINEAR_FP32_CHUNK")
    target = partial(
        http_worker,
        os.environ["RLT_BENCH_ATTENTION"],
        os.environ["RLT_BENCH_AUDIT"],
        linear_fp32_chunk=int(rows) if rows else None,
    )
    with patch.object(worker, "worker_main", target):
        serve.main()


def http_test(job, cfg, folder, precision):
    from benchmarks import ascend_main_prefix_service as service

    opts = [
        "--model",
        cfg["model"],
        "--out-dir",
        str(folder / "http"),
        "--main-only",
        "--seed",
        str(job.seed),
        "--concurrency",
        "4",
        "--requests",
        "32",
        "--rps",
        "0" if job.case == "http-burst" else "3",
        "--rounds",
        "3",
        "--correctness-rounds",
        "2",
    ]
    if precision == "fp32-linear":
        opts += ["--linear-fp32-chunk", "256"]
    if precision != "bf16":
        opts += [
            "--reference-traffic",
            str(Path(cfg["out"]) / "bf16" / job.key / "http" / "traffic.json"),
        ]
    code = service.main(opts)
    base = folder / "http"
    review = read(base / "review.json")
    if code == 2 or review["status"] == "INCOMPLETE":
        raise RuntimeError(f"HTTP replay incomplete: {review}")
    resources = []
    for n in range(1, 4):
        for arm in ("off", "main"):
            path = base / f"round-{n}-{arm}"
            usage = read(path / "worker.precision.json")
            require(usage["allow_hf32"] is False, "worker HF32 enabled")
            require(usage["after_drain"]["active"] == 0, "HTTP worker retained active blocks")
            resources.append(dict(round=n, arm=arm, **usage))
    with (base / "service.csv").open(newline="", encoding="utf-8") as f:
        totals = [row for row in csv.DictReader(f) if row["cohort"] == "all"]
    if len(totals) != 6:
        raise RuntimeError("missing HTTP round totals")
    return dict(
        inputs=digest(read(base / "traffic.json")),
        review=review,
        resources=resources,
        rounds=totals,
        issues=[] if code == 0 else [review],
    )


# 4. Smallest prompt and atomic failure with fully reserved multi-depth KV.
def boundaries(model, job, folder=None):
    import torch

    from benchmarks import ascend_main_prefix_combo as combo

    rng = rng_for(job.seed, job.case)
    vocab = model.config.vocab_size
    if job.case == "one-token":
        tokens = [rng.randrange(vocab)]
        record_inputs(folder, digest(tokens))
        arms = {}
        for arm in ARMS:
            with closing(new_engine(model, arm != "off", 12, 1, 32)) as engine:
                primer = [(tokens[0] + 1) % vocab] if arm == "miss" else tokens
                generate(engine, "prime", primer, 1)
                ids, hits = generate(engine, "same", tokens, 16)
                require(hits == 0, "one-token target had a prefix hit")
                arms[arm] = ids
        issues = [
            dict(arm=a, **diff)
            for a in ("miss", "reuse")
            if (diff := first_difference(arms["off"], arms[a]))
        ]
        return dict(inputs=digest(tokens), arms=arms, issues=issues)

    donor, _, targets, _ = combo.workload(combo.Case(16, 2), vocab, seed=job.seed)
    target = targets["target-0"]
    old = rng.choices(range(vocab), k=18)
    old[0] = (donor[0] + 1) % vocab
    record_inputs(folder, digest([donor, target, old]))
    # One cache-only page (four planes) and an unrelated live eight-block request.
    # Claiming 33 tokens fits a free pool of 12, but cannot succeed in this state.
    with closing(new_engine(model, True, 12, 2, 64)) as engine:
        cache = engine.cache_manager
        generate(engine, "donor", donor, 1)
        engine.add_request("old", old, params(4))
        first = engine.step()
        shared = cache.lookup_prefix(target)
        before = occupancy(cache)
        require(
            len(shared) == 1 and before == {"active": 8, "cache_only": 4},
            f"pool precondition: {before}",
        )
        table = cache.plane_block_tables("old")
        pages = list(shared[0])
        keys, values = cache.key_cache[pages].clone(), cache.value_cache[pages].clone()
        require(
            not cache.allocate("overflow", 33, prefix=shared),
            "expected reserved allocation failure",
        )
        require(
            not cache.has_allocation("overflow")
            and cache.plane_block_tables("old") == table
            and cache.lookup_prefix(target) == shared
            and occupancy(cache) == before,
            "failed claim changed prefix ownership",
        )
        require(
            torch.equal(keys, cache.key_cache[pages])
            and torch.equal(values, cache.value_cache[pages]),
            "failed claim changed shared KV",
        )
        old_ids = combo.drain(engine, ["old"], 4, first)["old"]["tokens"]
        hot, hits = generate(engine, "next", target, 4)
        require(hits == 16, "old prefix unavailable after failure")
    with closing(new_engine(model, False, 12, 2, 64)) as engine:
        expected_old, _ = generate(engine, "old", old, 4)
        expected_hot, _ = generate(engine, "next", target, 4)
    issues = [
        dict(request=rid, **delta)
        for rid, a, b in (("old", expected_old, old_ids), ("next", expected_hot, hot))
        if (delta := first_difference(a, b))
    ]
    return dict(
        inputs=digest([donor, target, old]),
        before=before,
        reused_tokens=hits,
        old_output=old_ids,
        hot_output=hot,
        issues=issues,
    )


# 5. Positive packed/per-row entry points; allocation-time rejections use pinned pytest guards.
def compatible(model, job, folder=None):
    from benchmarks import ascend_main_prefix_combo as combo

    donor, miss, targets, _ = combo.workload(
        combo.Case(16, 2), model.config.vocab_size, seed=job.seed
    )
    target = targets["target-0"]
    record_inputs(folder, digest([donor, miss, target]))
    packed = job.case == "packed"
    arms = {}
    for arm in ARMS:
        with closing(new_engine(model, arm != "off", 24, 1, 48, packed=packed)) as engine:
            require(engine.execution_config.packed_prefill is packed, "packed selection differs")
            generate(engine, "id", miss if arm == "miss" else donor, 1)
            ids, hits = generate(engine, "id", target, 16)
            require(hits == (16 if arm == "reuse" else 0), f"{arm} hit {hits} tokens")
            arms[arm] = ids
    return dict(
        inputs=digest([donor, miss, target]),
        arms=arms,
        issues=[
            dict(arm=a, **diff)
            for a in ("miss", "reuse")
            if (diff := first_difference(arms["off"], arms[a]))
        ],
    )


# 6. Candidate's pristine CPU/NPU suite and the same upstream/candidate delayed-delivery test.
def pytest_case(job, cfg, folder):
    if job.case == "lint":
        commands = [
            [sys.executable, "-m", "ruff", "check", "."],
            [sys.executable, "-m", "ruff", "format", "--check", "."],
        ]
        codes = []
        with (folder / "ruff.log").open("w") as log:
            for cmd in commands:
                codes.append(
                    subprocess.run(
                        cmd, cwd=cfg["pristine"], stdout=log, stderr=subprocess.STDOUT
                    ).returncode
                )
        if (
            any(n not in (0, 1) for n in codes)
            or "No module named ruff" in (folder / "ruff.log").read_text()
        ):
            raise RuntimeError("Ruff unavailable or could not execute; see ruff.log")
        return dict(commands=commands, issues=[cmd for cmd, n in zip(commands, codes) if n])
    if job.case == "config-guards":
        from vllm_rlt import CacheConfig

        require(CacheConfig().enable_prefix_caching is False, "prefix cache default must be off")
    args = {
        "cpu-suite": ["tests"],
        "baseline-delayed": ["--run-npu", DELAYED],
        "candidate-delayed": ["--run-npu", DELAYED],
        "npu-suite": ["--run-npu", "-m", "npu", "tests"],
        "config-guards": ["-m", "not npu", "tests/test_pr1_classified_guards.py"],
        "http-disconnect": [
            "--run-npu",
            "tests/test_pr1_classified_http.py"
            "::test_checkpoint_http_completions_stream_and_disconnect[ascend-prefix]",
        ],
    }[job.case]
    target = Path.cwd()
    if job.case == "baseline-delayed":
        if not cfg.get("upstream_tree"):
            raise RuntimeError(f"upstream commit unavailable: {cfg['upstream']}")
        target = Path(cfg["upstream_tree"])
    elif job.case not in ("config-guards", "http-disconnect"):
        target = Path(cfg["pristine"])
    xml = folder / "pytest.xml"
    env = dict(
        os.environ,
        VLLM_RLT_ASCEND_RESULTS=str(folder / "service"),
        VLLM_RLT_OURO_CHECKPOINT=cfg["model"],
        OMP_NUM_THREADS="1",
    )
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONPATH"] = str(target)
    if "--run-npu" in args:
        entry = [
            "-c",
            "import torch, torch_npu, pytest; torch.npu.matmul.allow_hf32=False; "
            "raise SystemExit(pytest.main())",
        ]
    else:
        entry = ["-m", "pytest"]
    command = [sys.executable, *entry, "-q", "-ra", f"--junitxml={xml}", *args]
    with (folder / "pytest.log").open("w") as log:
        code = subprocess.run(
            command, cwd=target, stdout=log, stderr=subprocess.STDOUT, env=env
        ).returncode
    counts = Counter()
    failures = []
    for node in ElementTree.parse(xml).iter("testcase"):
        outcome = next(
            (t for t in ("error", "failure", "skipped") if node.find(t) is not None), "passed"
        )
        counts[outcome] += 1
        if outcome in ("error", "failure"):
            failures.append(
                dict(
                    test=node.get("classname") + "." + node.get("name"),
                    message=node.find(outcome).get("message"),
                )
            )
    if code not in (0, 1) or not (counts["passed"] + counts["failure"]) or counts["error"]:
        raise RuntimeError(f"pytest incomplete ({code}, {dict(counts)}); see pytest.log and XML")
    if job.case not in ("cpu-suite", "npu-suite") and counts["skipped"]:
        raise RuntimeError(f"targeted tests skipped: {dict(counts)}")
    return dict(
        revision=git(target, "rev-parse", "HEAD"),
        command=command,
        counts=dict(counts),
        issues=failures,
    )


# 7. One persistent synchronous serving worker per arm, real cancellation and request-ID reuse.
class TokenInputs:
    """Feed exact input IDs into the worker while preserving its real output tokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = {}

    def encode(self, prompt):
        return self.inputs[prompt]

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def stress_inputs(vocab, seed, cycles):
    inputs = []
    for i in range(cycles):
        rng = rng_for(seed, f"stability/{i}")
        tags = rng.sample(range(vocab), 9)

        def prompt(tag, n):
            return [tag, *rng.choices(range(vocab), k=n - 1)]

        donor = prompt(tags[0], 18)
        target = donor[:16] + tags[1:3]
        keep, replacement = prompt(tags[3], 18), prompt(tags[4], 18)
        fillers = [prompt(t, 49) for t in tags[5:9]]
        inputs.append((donor, target, keep, replacement, fillers))
    return inputs


async def stress_arm(model, tokenizer, inputs, folder, enabled):
    import torch

    from vllm_rlt.serving.protocol import CompletionRequest
    from vllm_rlt.serving.worker import EngineWorker

    feeder = TokenInputs(tokenizer)
    worker = EngineWorker(lambda: (new_engine(model, enabled, 20, 2, 64), feeder))
    cycle_rows = []
    active_inputs, finished = {}, {}
    # Gate the owner ticks so cancellation always lands after the first real
    # delivery, including on slower cards. Admission, release and event delivery
    # are the ordinary production paths. Only the generated UUID is prescribed.
    permit, progress = asyncio.Semaphore(0), asyncio.Queue()
    owner_call = worker._call
    stepping = True
    owner_issue = None

    async def gated_call(fn, *args):
        nonlocal owner_issue
        is_tick = fn == worker._tick
        if is_tick and stepping:
            await permit.acquire()
        try:
            result = await owner_call(fn, *args)
        except AssertionError as exc:
            owner_issue = exc
            raise
        if is_tick:
            progress.put_nowait(None)
        return result

    async def until(predicate):
        while not predicate():
            permit.release()
            tick_done = asyncio.create_task(progress.get())
            try:
                done, _ = await asyncio.wait(
                    [tick_done, worker.task],
                    timeout=worker.limits.request_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if worker.task in done:
                    if owner_issue is not None:
                        raise owner_issue
                    raise RuntimeError(worker.failure or "serving worker exited unexpectedly")
                if not done:
                    raise TimeoutError("serving worker did not deliver a scheduled tick")
            finally:
                tick_done.cancel()

    def submit(name, tokens, count):
        prompt = name + "/" + digest(tokens)
        feeder.inputs[prompt] = tokens
        with patch("vllm_rlt.serving.worker.uuid4", return_value=SimpleNamespace(hex=name)):
            ch = worker.submit(CompletionRequest(prompt, params(count)))
        active_inputs[ch.request_id] = tokens
        return ch

    async def finish(channels):
        await until(
            lambda: all(ch.error or (ch.events and ch.events[-1].finish_reason) for ch in channels)
        )
        for ch in channels:
            require(ch.error is None, f"{ch.request_id}: {ch.error}")
            count = ch.spec.params.max_tokens
            for index in range(1, count + 1):
                event = await ch.receive()
                require(
                    event.prompt_tokens == len(active_inputs[ch.request_id])
                    and event.completion_tokens == index
                    and event.finish_reason == ("length" if index == count else None),
                    f"{ch.request_id}: stale or missing channel event {index}",
                )
            require(not ch.events and len(finished[ch.request_id]) == count, "duplicate delivery")
            worker.release(ch)
        await until(lambda: all(ch.request_id not in worker.channels for ch in channels))
        require(
            not worker.decoders and worker.engine.cache_manager.num_used_blocks == 0,
            "requests or KV survived drain",
        )

    worker._call = gated_call
    worker.start()
    try:
        while not worker.ready:
            if worker.task.done():
                raise RuntimeError(worker.failure or "serving worker failed to start")
            await asyncio.sleep(0.001)
        engine, cache = worker.engine, worker.engine.cache_manager
        original = engine.step

        @torch.inference_mode()
        def observed_step():
            outputs = original()
            for out in outputs:
                require(list(out.prompt_token_ids) == active_inputs[out.request_id], "stale prompt")
                if out.finished:
                    require(
                        out.request_id not in finished
                        and out.finish_reason == "length"
                        and out.exit_depths == [4] * len(out.token_ids),
                        "stale or duplicate completion",
                    )
                    finished[out.request_id] = list(out.token_ids)
            return outputs

        with patch.object(engine, "step", observed_step):
            memory(model, reset=True)
            for i, (donor, target, keep, replacement, fillers) in enumerate(inputs):
                finished.clear()
                feeder.inputs.clear()
                await finish([submit("donor", donor, 1)])
                before = cache.prefix_hits
                old, live = submit("same", target, 4), submit("keep", keep, 8)
                await until(lambda: old.events or old.error)
                require(
                    old.error is None and (await old.receive()).completion_tokens == 1,
                    "initial stream failed",
                )
                require(
                    cache.prefix_hits - before == (16 if enabled else 0), "initial hit mismatch"
                )
                require(
                    engine.has_request(old.request_id) and engine.has_request(live.request_id),
                    "missing live pair",
                )
                worker.release(old)
                await until(lambda: old.request_id not in worker.channels)
                require(
                    not engine.has_request(old.request_id)
                    and old.request_id not in worker.decoders
                    and engine.has_request(live.request_id),
                    "cancel not acknowledged",
                )
                queued = len(old.events)
                new = submit("same", replacement, 4)
                require(new.request_id == old.request_id and new is not old, "ID was not reused")
                await finish([new, live])
                require(len(old.events) == queued, "cancelled channel received replacement output")
                for n, tokens in enumerate(fillers):
                    await finish([submit(f"filler-{n}", tokens, 1)])
                require(not cache.lookup_prefix(target), "pressure failed to evict old prefix")
                await finish([submit("rewarm", donor, 1)])
                before = cache.prefix_hits
                await finish([submit("rehit", target, 4)])
                require(cache.prefix_hits - before == (16 if enabled else 0), "rehit mismatch")
                row = dict(
                    cycle=i + 1,
                    outputs={r: finished["cmpl-" + r] for r in ("same", "keep", "rehit")},
                    blocks=occupancy(cache),
                    memory=memory(model),
                )
                cycle_rows.append(row)
                with (folder / f"cycles-{'on' if enabled else 'off'}.jsonl").open("a") as log:
                    print(json.dumps(row), file=log, flush=True)
    finally:
        stepping = False
        permit.release()
        await worker.close()
    return cycle_rows


def stability(model, tokenizer, job, cfg, folder):
    inputs = stress_inputs(model.config.vocab_size, job.seed, cfg["cycles"])
    fingerprint = record_inputs(folder, digest(inputs))
    off = asyncio.run(stress_arm(model, tokenizer, inputs, folder, False))
    on = asyncio.run(stress_arm(model, tokenizer, inputs, folder, True))
    issues = []
    for a, b in zip(off, on, strict=True):
        for rid, ids in a["outputs"].items():
            if delta := first_difference(ids, b["outputs"][rid]):
                issues.append(dict(cycle=a["cycle"], request=rid, **delta))
    trends = {}
    for arm, cycles in (("off", off), ("reuse", on)):
        trends[arm] = {}
        for metric in cycles[0]["memory"]:
            values = [r["memory"][metric] for r in cycles]
            trends[arm][metric] = dict(
                first=values[0],
                last=values[-1],
                maximum=max(values),
                increases=sum(a < b for a, b in zip(values, values[1:])),
                last_growth_cycle=max(
                    (i for i in range(1, len(values)) if values[i] > values[i - 1]), default=-1
                )
                + 1,
            )
    return dict(
        inputs=fingerprint,
        cycles=len(on),
        issues=issues,
        first=on[0],
        last=on[-1],
        memory_trend=trends,
    )


def setup_device(chip, card):
    import torch
    import torch_npu
    from torch_npu.utils.collect_env import get_cann_version

    from benchmarks.ascend_prefix_pr1_hardware import identify

    hw = identify(torch_npu.npu, str(card), expected=chip)
    if hw["verification"] != "matched":
        raise RuntimeError(hw["detail"] or "unverified chip")
    torch.npu.matmul.allow_hf32 = False
    cann = str(get_cann_version())
    if cann != "9.0.0" or torch.npu.matmul.allow_hf32:
        raise RuntimeError(f"expected CANN 9.0.0 and HF32 disabled, got {cann}")
    return dict(
        hardware=hw,
        cann=cann,
        torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__),
        allow_hf32=False,
    )


def run_one(model, tokenizer, job, cfg, folder, precision, linear):
    if job.kind == "pytest":
        return pytest_case(job, cfg, folder)
    if job.kind == "http":
        return http_test(job, cfg, folder, precision)
    if job.case in MATRIX:
        return numerical(model, job, folder, linear)
    if job.case in WORKLOADS:
        return engine_performance(model, job, cfg, folder)
    if job.case == "W3-trace":
        return w3_trace(model, job, folder)
    if job.case in ("shared-pages", "page-oracle"):
        return functional(model, job, folder)
    if job.case in ("one-token", "failed-admission"):
        return boundaries(model, job, folder)
    if job.case in ("packed", "per-row"):
        return compatible(model, job, folder)
    if job.case == "cancel-reuse-evict":
        return stability(model, tokenizer, job, cfg, folder)
    raise ValueError(job.case)


def job_result(root, precision, job):
    path = root / precision / job.key / "result.json"
    return read(path) if path.is_file() else {}


def worker(cfg_path, lane, precision, progress):
    # This process runs from the disposable candidate checkout. Never resolve
    # the runtime from the user's installed editable checkout or the script path.
    sys.path.insert(0, str(Path.cwd()))
    cfg = read(cfg_path)
    jobs = [Job(**x) for x in read(lane)]
    root = Path(cfg["out"])
    card = jobs[0].card
    affinity = cfg.get("cpu_affinity", {}).get(str(card))
    if affinity:
        os.sched_setaffinity(0, affinity)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(card)
    os.environ["RLT_BENCH_DTYPE"] = "float16" if precision == "fp16" else "bfloat16"
    import torch

    import vllm_rlt
    from benchmarks.ascend_main_prefix_linear import (
        benchmark_dtype,
        check_linear_usage,
        linear_usage,
        padded_fp32_linears,
    )

    if Path(vllm_rlt.__file__).resolve().parents[1] != Path.cwd().resolve():
        raise RuntimeError("candidate runtime import escaped its checkout")
    if git(Path.cwd(), "rev-parse", "HEAD") != cfg["candidate_sha"]:
        raise RuntimeError("unexpected worker runtime commit")
    dtype = benchmark_dtype()
    model = tokenizer = linear = device_info = None
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        for job in jobs:
            folder = root / precision / job.key
            folder.mkdir(parents=True, exist_ok=True)
            write(progress, {"key": job.key, "started": time.time()})
            res = {}
            try:
                if job.case not in ("lint", "cpu-suite", "config-guards") and device_info is None:
                    device_info = setup_device(cfg["chip"], card)
                    device_info.update(
                        cpu_affinity=sorted(os.sched_getaffinity(0)),
                        omp_threads=os.environ.get("OMP_NUM_THREADS"),
                        mkl_threads=os.environ.get("MKL_NUM_THREADS"),
                    )
                    write(lane.with_suffix(".environment.json"), device_info)
                if job.kind == "engine" and model is None:
                    from vllm_rlt.models import OuroForCausalLM

                    model = OuroForCausalLM.from_pretrained(
                        cfg["model"], device=torch.device("npu:0"), dtype=dtype
                    ).eval()
                    require(model.lm_head.weight.dtype == dtype, "loaded model dtype mismatch")
                    linear = stack.enter_context(
                        padded_fp32_linears(model, 256 if precision == "fp32-linear" else None)
                    )
            except Exception as exc:
                print(traceback.format_exc(), flush=True)
                if job.case in ("cpu-suite", "config-guards"):
                    raise
                # Setup failure affects all remaining jobs of this worker.
                write(progress, {"setup_error": f"{type(exc).__name__}: {exc}"})
                return 2
            try:
                if job.case == "cancel-reuse-evict" and tokenizer is None:
                    from vllm_rlt.models import load_tokenizer

                    tokenizer = load_tokenizer(cfg["model"])
                if linear is not None:
                    linear[1].clear()
                res = run_one(model, tokenizer, job, cfg, folder, precision, linear)
                verdicts = res.pop(
                    "verdicts", {g: "FAIL" if res.get("issues") else "PASS" for g in job.groups}
                )
            except (AssertionError, FloatingPointError) as exc:
                verdicts = {g: "FAIL" for g in job.groups}
                res = {"issues": [{"error": str(exc)}], "traceback": traceback.format_exc()}
            except Exception as exc:
                verdicts = {g: "INCOMPLETE" for g in job.groups}
                res = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            if linear is not None and job.case != "page-oracle":
                res["linear"] = linear_usage(linear)
                try:
                    check_linear_usage(res["linear"])
                except AssertionError as exc:
                    verdicts = {g: "INCOMPLETE" for g in job.groups}
                    res["precision_error"] = str(exc)
            if (folder / "inputs.json").exists():
                res.setdefault("inputs", read(folder / "inputs.json"))
            if job.kind == "http" and (folder / "http/traffic.json").is_file():
                res.setdefault("inputs", digest(read(folder / "http/traffic.json")))
            if precision in DIAGNOSTICS:
                bf = job_result(root, "bf16", job)
                if not bf.get("inputs") or res.get("inputs") != bf["inputs"]:
                    verdicts = {g: "INCOMPLETE" for g in job.groups}
                    res["comparison_error"] = (
                        "input fingerprints differ or the replay is incomplete"
                    )
            write(
                folder / "result.json",
                dict(
                    **res,
                    verdicts=verdicts,
                    card=card,
                    seed=job.seed,
                    precision=precision,
                    case=job.case,
                    candidate=cfg["candidate_sha"],
                    harness=HARNESS_SHA,
                ),
            )
            print(f"{job.key}: {verdicts}", flush=True)
    return 0


def unavailable(root, precision, job, reason):
    folder = root / precision / job.key
    if not (folder / "result.json").exists():
        write(
            folder / "result.json",
            {
                "case": job.case,
                "seed": job.seed,
                "card": job.card,
                "precision": precision,
                "error": reason,
                "verdicts": {g: "INCOMPLETE" for g in job.groups},
            },
        )


def stop_worker(pid):
    # The pinned HTTP harness creates a separate process group for its server.
    # Stop this worker tree before restarting another lane on the same NPU.
    try:
        os.kill(pid, signal.SIGSTOP)
    except ProcessLookupError:
        return
    for entry in Path(f"/proc/{pid}/task").glob("*/children"):
        try:
            children = entry.read_text().split()
        except FileNotFoundError:
            continue
        for child in children:
            stop_worker(int(child))
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_lane(cfg, tree, precision, jobs):
    root = Path(cfg["out"])
    remaining = list(jobs)
    first = jobs[0]
    label = first.kind
    if first.case in WORKLOADS or first.case == "W3-trace":
        label = f"{first.case}-seed-{first.seed}"
    attempt = 0
    # Restart after a hard timeout or a device-process crash. Completed jobs
    # survive, the interrupted item stays incomplete, other seeds still run.
    while remaining:
        attempt += 1
        lane = root / "workers" / f"{precision}-{label}-{first.card}-{attempt}.json"
        progress = lane.with_suffix(".current.json")
        write(lane, [asdict(j) for j in remaining])
        env = dict(os.environ, PYTHONPATH=str(tree), ASCEND_RT_VISIBLE_DEVICES=str(jobs[0].card))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, "1")
        for name in (
            "RLT_BENCH_PORT",
            "RLT_BENCH_CHECKPOINT_JSON",
            "RLT_BENCH_LINEAR_FP32_CHUNK",
            "RLT_BENCH_PREFILL_STEPS",
            "PYTEST_ADDOPTS",
        ):
            env.pop(name, None)
        cmd = [
            sys.executable,
            str(SCRIPT),
            "_worker",
            str(root / "run.json"),
            str(lane),
            precision,
            str(progress),
        ]
        with lane.with_suffix(".log").open("w") as log:
            proc = subprocess.Popen(
                cmd, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            start = time.time()
            timed_out = False
            try:
                while proc.poll() is None:
                    marker = read(progress) if progress.exists() else {}
                    if time.time() - marker.get("started", start) > cfg["timeout"]:
                        raise TimeoutError("per-case timeout")
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException as exc:
                stop_worker(proc.pid)
                proc.wait()
                if not isinstance(exc, TimeoutError):
                    raise
                timed_out = True
                print(f"Timed out: {lane.name}", flush=True)
        pending = [j for j in remaining if not job_result(root, precision, j)]
        if not pending:
            break
        marker = read(progress) if progress.exists() else {}
        interrupted = next((j for j in pending if j.key == marker.get("key")), None)
        outcome = (
            f"{cfg['timeout']} s per-case timeout"
            if timed_out
            else f"worker exit {proc.returncode}"
        )
        reason = marker.get("setup_error", f"{outcome}; see workers/{lane.stem}.log")
        if interrupted is None:
            for job in pending:
                unavailable(root, precision, job, reason)
            break
        unavailable(root, precision, interrupted, reason)
        remaining = [j for j in pending if j != interrupted]


def execute(cfg, tree, precision, jobs):
    phases = defaultdict(lambda: defaultdict(list))
    for job in jobs:
        phase = job.case if job.case in WORKLOADS or job.case == "W3-trace" else job.kind
        phases[phase][job.card].append(job)

    def per_card(phase, items):
        if phase in WORKLOADS:
            for job in items:
                run_lane(cfg, tree, precision, [job])
        else:
            run_lane(cfg, tree, precision, items)

    with ThreadPoolExecutor(max_workers=max(1, len({j.card for j in jobs}))) as pool:
        for phase in ("pytest", "engine", *WORKLOADS, "W3-trace", "http"):
            # Wait for every worker in this workload to exit before starting the next.
            list(pool.map(partial(per_card, phase), phases[phase].values()))


def failed_jobs(root, jobs):
    selected = []
    for job in jobs:
        if job.kind == "pytest" or job.case in ("page-oracle", "W3-trace"):
            continue
        states = job_result(root, "bf16", job).get("verdicts", {})
        groups = tuple(g for g in job.groups if states.get(str(g)) == "FAIL")
        if groups:
            selected.append(replace(job, groups=groups))
    return selected


def harness_repo(repo, override, tmp):
    for source in (override,) if override else (repo, SCRIPT.parents[1]):
        if commit_sha(source, HARNESS_SHA):
            return source
    if override:
        raise ValueError(f"{override} does not contain pinned harness {HARNESS_SHA}")
    source = tmp / "harness"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    git(source, "fetch", "-q", "--depth=1", HARNESS_URL, HARNESS_SHA)
    return source


def prepare_harness(source, tree):
    def extract(old, new):
        (tree / new).write_bytes(
            subprocess.check_output(["git", "-C", str(source), "show", f"{HARNESS_SHA}:{old}"])
        )

    for name in HELPERS:
        path = f"benchmarks/{name}.py"
        extract(path, path)
    for suffix in ("replay", "http"):
        path = f"benchmarks/ascend_prefix_fp16_{suffix}.patch"
        extract(path, path)
    extract("tests/test_ascend_prefix_cache_contract.py", "tests/test_pr1_classified_guards.py")
    extract("tests/test_ascend_serving.py", "tests/test_pr1_classified_http.py")
    # The pinned HTTP harness spawns its own NPU process. Set/record HF32 in that
    # child too, including the FP16 and fixed-row FP32 diagnostic replays.
    http = tree / "benchmarks/ascend_main_prefix_service_worker.py"
    http.rename(tree / "benchmarks/_pr1_pinned_http_worker.py")
    http.write_text(
        "from benchmarks.ascend_pr1_acceptance import http_main\n"
        "if __name__ == '__main__':\n    http_main()\n"
    )
    shutil.copyfile(SCRIPT, tree / "benchmarks" / (MODULE + ".py"))


def first_issue(result):
    if issues := result.get("issues"):
        return issues[0]
    for key in ("comparison_error", "precision_error", "error"):
        if key in result:
            return result[key]
    return "no completed record"


def cross_check_inputs(roots, reports):
    fingerprints = {}
    for root, (cfg, _, _) in zip(roots, reports, strict=True):
        for spec in cfg["jobs"]:
            job = Job(**spec)
            actual = job_result(root, "bf16", job).get("inputs")
            if actual is None:
                continue
            key = job.key
            if fingerprints.setdefault(key, actual) != actual:
                raise ValueError(f"{cfg['chip']} {key}: actual BF16 inputs differ across cards")


def summary(root, *, save=True):
    cfg = read(root / "run.json")
    jobs = [Job(**row) for row in cfg["jobs"]]
    eligible = {j.key: set(j.groups) for j in failed_jobs(root, jobs)}
    rows, sections, perf, controls, http, regressions, notes = [], [], [], [], [], [], []
    for job in jobs:
        bf = job_result(root, "bf16", job)
        for g in job.groups:
            base = bf.get("verdicts", {}).get(str(g), "INCOMPLETE")
            row = dict(
                chip=cfg["chip"],
                category=CATEGORIES[g],
                card=job.card,
                case=job.case,
                seed=job.seed,
                bf16=base,
            )
            row["evidence"] = f"bf16/{job.key}/result.json"
            row["bf16_first_issue"] = (
                json.dumps(first_issue(bf), ensure_ascii=False) if base != "PASS" else ""
            )
            for p in DIAGNOSTICS:
                needed = g in eligible.get(job.key, set())
                if needed:
                    rec = job_result(root, p, job)
                    row[p] = rec.get("verdicts", {}).get(str(g), "INCOMPLETE")
                elif base == "INCOMPLETE":
                    row[p] = "BLOCKED"
                else:
                    row[p] = "NOT_REQUIRED" if base == "PASS" else "N/A"
                row[p + "_evidence"] = f"{p}/{job.key}/result.json" if needed else ""
                row[p + "_first_issue"] = (
                    json.dumps(first_issue(rec), ensure_ascii=False)
                    if needed and row[p] != "PASS"
                    else ""
                )
            if base != "PASS":
                notes.append((job, base, first_issue(bf), row["evidence"]))
            rows.append(row)
        if job.case in WORKLOADS and "output_tok_s" in bf:
            rates = bf["output_tok_s"]
            perf.append(
                f"| {job.case} | {job.card} | {job.seed} | "
                + " | ".join(
                    f"{rates[a]['median']:.2f} [{rates[a]['low']:.2f}, {rates[a]['high']:.2f}]"
                    for a in ARMS
                )
                + f" | {'诊断' if bf.get('issues') else '有效'} |"
            )
            for arm in ARMS:
                runs = [s for s in bf.get("controlled", []) if s["arm"] == arm]
                if not runs:
                    continue
                allocated = max((r["memory"].get("peak_allocated", 0) for r in runs), default=0)
                reserved = max((r["memory"].get("peak_reserved", 0) for r in runs), default=0)
                hits = sorted({int(h) for r in runs for h in r["hits"]})
                controls.append(
                    f"| {job.case} | {job.card} | {job.seed} | {arm} | "
                    f"{','.join(map(str, hits))} | "
                    f"{statistics.median(r['ttft_p50_ms'] for r in runs):.2f} | "
                    f"{statistics.median(r['tpot_p50_ms'] for r in runs):.2f} | "
                    f"{allocated / 2**30:.2f} / {reserved / 2**30:.2f} | "
                    f"{max(r['sampled_kv_peaks']['active'] for r in runs)} / "
                    f"{max(r['sampled_kv_peaks']['cache_only'] for r in runs)} | "
                    f"{max(r['after_drain']['active'] for r in runs)} / "
                    f"{max(r['after_drain']['cache_only'] for r in runs)} |"
                )
        if job.kind == "http" and bf.get("rounds"):
            pairs = bf.get("review", {}).get("paired_ttft", {}).get("off_to_main", {})
            deltas = pairs.get("pairs_ms", [])
            for mode in ("off", "main"):
                totals = [r for r in bf["rounds"] if r["mode"] == mode]
                times = [float(r["ttft_p50_ms"]) for r in totals if r["ttft_p50_ms"]]
                ttft = f"{statistics.median(times):.2f}" if times else "—"
                slower = (
                    f"{sum(d < 0 for d in deltas)}/{len(deltas)}"
                    if deltas and mode == "main"
                    else "—"
                )
                http.append(
                    f"| {job.case} | {job.card} | {job.seed} | {mode} | "
                    f"{sum(int(r['completed']) for r in totals)} | "
                    f"{sum(int(r['http_errors']) for r in totals)} | {ttft} | {slower} |"
                )
        if 6 in job.groups:
            counts = bf.get("counts", {})
            regressions.append(
                f"| {job.case} | {job.card} | {bf.get('revision', '')[:12] or '—'} | "
                f"{bf.get('verdicts', {}).get('6', 'INCOMPLETE')} | {counts.get('passed', '—')} | "
                f"{counts.get('failure', '—')} | {counts.get('skipped', '—')} |"
            )
    for g in cfg["categories"]:
        group = [r for r in rows if r["category"] == CATEGORIES[g]]
        base = Counter(r["bf16"] for r in group)
        supplemental = []
        for p in DIAGNOSTICS:
            counts = Counter(r[p] for r in group)
            needed = sum(counts[s] for s in ("PASS", "FAIL", "INCOMPLETE"))
            supplemental.append(
                "无需补测"
                if not needed and not base["INCOMPLETE"] and not base["FAIL"]
                else (
                    "—"
                    if not needed
                    else f"{counts['PASS']}/{needed} "
                    f"(FAIL {counts['FAIL']}, 未完成 {counts['INCOMPLETE']})"
                )
            )
        sections.append(
            f"| {g}. {CATEGORIES[g]} | {len(group)} | {base['PASS']} | {base['FAIL']} | "
            f"{base['INCOMPLETE']} | " + " | ".join(supplemental) + " |"
        )
    states = {r[k] for r in rows for k in ("bf16", *DIAGNOSTICS)}
    status = (
        "INCOMPLETE"
        if cfg.get("setup_error") or "INCOMPLETE" in states
        else ("FAIL" if "FAIL" in states else "PASS")
    )
    lines = [
        f"## {cfg['chip']} · {status}",
        "",
        f"候选 `{cfg['candidate_sha']}`，回归基线 `{cfg.get('upstream_sha') or cfg['upstream']}`，"
        f"测试工具 `{HARNESS_SHA[:12]}`。脚本 SHA-256 `{cfg['script_sha256']}`。",
        "",
        "输入种子 " + ", ".join(map(str, cfg["seeds"])) + "。",
        "",
        *(
            ["仅运行指定用例 " + ", ".join(cfg["case_filter"]) + "。", ""]
            if cfg.get("case_filter")
            else []
        ),
        "| 类别 | 计划项 | 主测通过 | 主测失败 | 未完成 | FP16 通过/应补 | FP32 Linear 通过/应补 |",
        "|---|---:|---:|---:|---:|---|---|",
        *sections,
        "",
        "主测包括原生 BF16 推理及独立的配置和回归检查。FP32 Linear 固定为 256 行，KV/SDPA 为 BF16。"
        "同一用例可同时提供边界证据。共用项的页、命中和释放断言计入边界类，"
        "逐 token 比对计入数值类，各类分母独立。",
        "",
    ]
    if cfg.get("setup_error"):
        lines += ["环境或数据完整性检查未完成：" + cfg["setup_error"], ""]
    shapes = cfg.get("workloads", WORKLOADS)
    used = {j.case for j in jobs}
    if any(w in used for w in shapes):
        lines += [
            "### 引擎性能负载",
            "",
            "| 负载 | 输入 token | 输出 token | 客户端并发 | 驻留上限 | 共享前缀 token |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                "| " + " | ".join(map(str, (w, *spec))) + " |"
                for w, spec in shapes.items()
                if w in used
            ],
            "",
            f"持续闭环每轮接收 {cfg['window']:g} 秒，随后排空。每个缓存状态各三轮。"
            "受控批次的并发等于驻留上限。W3 的单独 profiling 不计入时延。",
            "",
        ]
    if perf:
        lines += [
            "### 持续闭环输出吞吐（tok/s，三轮中位数 [最小, 最大]）",
            "",
            "| 负载 | 卡 | 种子 | 关闭 | 开启、冷启动 | 开启、预热命中 | 样本 |",
            "|---|---:|---:|---:|---:|---:|---|",
            *perf,
            "",
            "每轮接收时窗结束后排空。逐请求实际命中、TTFT、TPOT、显存及 KV 块见对应目录。"
            "controlled 文件另列同一驻留批次的纯 off/miss/reuse 测量。",
            "",
        ]
    if controls:
        lines += [
            "### 同卡受控批次（每轮独立预热，三轮）",
            "",
            "| 负载 | 卡 | 种子 | 缓存 | 实际命中 token/请求 | TTFT P50 ms | TPOT P50 ms | "
            "峰值分配/预留 GiB | 活跃/仅缓存块采样峰值 | 排空后活跃/仅缓存块 |",
            "|---|---:|---:|---|---|---:|---:|---|---|---|",
            *controls,
            "",
        ]
    if http:
        lines += [
            "### HTTP（每轮 32 请求，并发上限 4）",
            "",
            "| 负载 | 卡 | 种子 | 缓存 | 完成请求 | HTTP 错误 | "
            "三轮 TTFT P50 中位数 ms | 开缓存后上升轮次 |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
            *http,
            "",
        ]
    if regressions:
        lines += [
            "### 回归（同卡运行基线与候选的 delayed-delivery）",
            "",
            "| 项目 | 卡 | 提交 | 状态 | 通过 | 失败 | 跳过 |",
            "|---|---:|---|---|---:|---:|---:|",
            *regressions,
            "",
        ]
    if notes:
        lines += ["### 主测失败和未完成项", ""]
        for job, state, issue, path in notes[:60]:
            description = issue if isinstance(issue, str) else json.dumps(issue, ensure_ascii=False)
            description = " ".join(description.split())[:240]
            lines.append(f"- {job.key}（卡 {job.card}）{state}，{description}。证据 `{path}`")
        if len(notes) > 60:
            lines.append("其余条目见 summary.csv 和各用例 result.json。")
        lines.append("")
    if 2 in cfg["categories"]:
        chip = cfg["chip"]
        previous = {"910B3": 25, "910B4": 24}[chip]
        lines += [
            "### 历史反例（交接记录，单列）",
            "",
            f"在旧候选 `dca7633`、seed 168 的 30 个场景中，{chip} 原生 BF16 "
            f"通过 {previous}/30，实验 FP16 和固定 256 行 FP32 Linear 均通过 30/30。"
            "在 `c4_h3` 的 target-2 上，第 7 个生成 token 为 302 → 726。"
            + (
                "该卡的 `p1024_s8` 在第 2 个生成 token 出现 1116 → 198。" if chip == "910B4" else ""
            ),
            "",
        ]
    lines += ["逐项结果和各精度的首个问题见 `summary.csv`。原始数据按精度、用例和种子保存。", ""]
    md = "\n".join(lines)
    if save:
        with (root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        write(
            root / "summary.json",
            {"status": status, "setup_error": cfg.get("setup_error"), "rows": rows},
        )
        (root / "PR_RESULTS.md").write_text("# PR1 Ascend Prefix Cache\n\n" + md)
    return cfg, status, md


def runtime_checkout(args, tmp):
    repo = args.repo.resolve()
    pristine = tmp / "pristine"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(repo), str(pristine)], check=True)

    def local_revision(ref):
        sha = commit_sha(repo, ref)
        # Detached refs in linked worktrees can be absent from clone's refs.
        if sha and not commit_sha(pristine, sha):
            git(pristine, "fetch", "-q", "--no-tags", repo, sha)
        return sha or commit_sha(pristine, ref)

    candidate, upstream = local_revision(args.candidate), local_revision(args.upstream)
    kit = PACKAGE_DIR / "pr1-fixed10-tests.bundle"
    if kit.is_file() and not (candidate and upstream and commit_sha(pristine, HARNESS_SHA)):
        git(pristine, "bundle", "unbundle", kit)
        candidate = candidate or commit_sha(pristine, args.candidate)
        upstream = upstream or commit_sha(pristine, args.upstream)
    bundle = args.candidate_bundle
    if bundle is None and not candidate:
        for directory in (repo, PACKAGE_DIR, Path.cwd()):
            path = directory / CANDIDATE_BUNDLE_NAME
            if path.is_file():
                bundle = path
                break
    if bundle is not None:
        git(pristine, "bundle", "unbundle", bundle.resolve())
        candidate = candidate or commit_sha(pristine, args.candidate)
        upstream = upstream or commit_sha(pristine, args.upstream)
    if not candidate:
        raise ValueError(
            f"candidate {args.candidate} unavailable in {repo} and packaged objects. "
            "Pass --repo or --candidate-bundle with the requested commit."
        )
    git(pristine, "checkout", "-q", "--detach", candidate)
    return pristine, candidate, upstream


def run(args, cfg):
    model = args.model.resolve()
    if not model.is_dir():
        raise ValueError(f"checkpoint directory missing: {model}")
    cases = cfg["case_filter"]
    suffix = ""
    if len(cases) == 1:
        suffix = "-" + cases[0]
    elif cases or len(cfg["categories"]) < len(CATEGORIES):
        suffix = "-selected"
    out = (args.out or Path(f"results-{cfg['chip']}{suffix}")).resolve()
    if out.exists():
        raise FileExistsError(f"{out} already exists; pass --out with a new directory")
    with tempfile.TemporaryDirectory(prefix="rlt-pr1-") as scratch:
        tmp = Path(scratch)
        pristine, candidate, upstream = runtime_checkout(args, tmp)
        out.mkdir(parents=True, exist_ok=False)
        cfg.update(
            candidate_sha=candidate,
            upstream_sha=upstream,
            upstream=args.upstream,
            model=str(model),
            out=str(out),
            pristine=str(pristine),
            script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        )
        cpus = sorted(os.sched_getaffinity(0))
        cards = cfg["cards"]
        cfg["cpu_affinity"] = {
            str(card): cpus[i :: len(cards)] or [cpus[i % len(cpus)]]
            for i, card in enumerate(cards)
        }
        write(out / "run.json", cfg)
        print("RESULTS", out, flush=True)
        jobs = [Job(**j) for j in cfg["jobs"]]
        try:
            source = harness_repo(pristine, args.harness_repo, tmp)
            if upstream and 6 in cfg["categories"]:
                up = tmp / "upstream"
                git(pristine, "worktree", "add", "-q", "--detach", up, upstream)
                cfg["upstream_tree"] = str(up)
            native = tmp / "native"
            git(pristine, "worktree", "add", "-q", "--detach", native, candidate)
            prepare_harness(source, native)
            # Hash real file contents once per machine, before any workers start.
            from importlib.util import module_from_spec, spec_from_file_location

            spec = spec_from_file_location(
                "pr1_checkpoint_manifest", native / "benchmarks/preemption.py"
            )
            util = module_from_spec(spec)
            spec.loader.exec_module(util)
            cfg["checkpoint"] = util.checkpoint_identity(model)
            write(out / "run.json", cfg)
            execute(cfg, native, "bf16", jobs)
            replay = failed_jobs(out, jobs)
            write(out / "followups.json", [asdict(j) for j in replay])
            if replay:
                try:
                    fp = tmp / "fp16"
                    git(pristine, "worktree", "add", "-q", "--detach", fp, candidate)
                    prepare_harness(source, fp)
                    for patch_name in ("replay", "http"):
                        if patch_name == "http" and not any(j.kind == "http" for j in replay):
                            continue
                        path = fp / "benchmarks" / f"ascend_prefix_fp16_{patch_name}.patch"
                        shutil.copyfile(path, out / path.name)
                        git(fp, "apply", path)
                    execute(cfg, fp, "fp16", replay)
                except (OSError, subprocess.CalledProcessError) as e:
                    for job in replay:
                        unavailable(out, "fp16", job, str(e))
                execute(cfg, native, "fp32-linear", replay)
            if util.checkpoint_identity(model) != cfg["checkpoint"]:
                cfg["setup_error"] = "checkpoint contents changed during the run"
                write(out / "run.json", cfg)
        except Exception as exc:
            cfg["setup_error"] = f"{type(exc).__name__}: {exc}"
            write(out / "run.json", cfg)
            print(traceback.format_exc(), file=sys.stderr)
    _, status, _ = summary(out)
    print(status, out / "PR_RESULTS.md", flush=True)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[status]


def cli(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subs = parser.add_subparsers(dest="mode", required=True)
    seeds_cmd = subs.add_parser("seeds", help="export the ten fixed PR1 prompt seeds")
    seeds_cmd.add_argument("path", type=Path)
    for cmd in ("plan", "run"):
        p = subs.add_parser(cmd)
        p.add_argument(
            "--seeds", type=Path, help="optional manifest matching the built-in PR1 seeds in order"
        )
        p.add_argument("--chip", choices=("910B3", "910B4"), required=True)
        p.add_argument(
            "--cards",
            default=os.getenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
            help="ordered physical card IDs, comma-separated "
            "(default: ASCEND_RT_VISIBLE_DEVICES if set, otherwise 0 through 7)",
        )
        p.add_argument(
            "--categories", default="1,2,3,4,5,6,7", help="comma-separated category numbers"
        )
        p.add_argument(
            "--cases", help="optional comma-separated case names within selected categories"
        )
        p.add_argument("--cycles", type=int, default=100)
        p.add_argument("--window", type=float, default=60, help="closed-loop admission seconds")
        p.add_argument(
            "--stream-size", type=int, default=512, help="size of deterministic W input corpus"
        )
        p.add_argument(
            "--timeout", type=float, default=7200, help="hard timeout in seconds per case"
        )
        if cmd == "run":
            p.add_argument(
                "--repo",
                type=Path,
                default=os.getenv("CANDIDATE_REPO", Path.cwd()),
                help="source repository (CANDIDATE_REPO or current checkout)",
            )
            p.add_argument("--candidate", default=CANDIDATE_SHA)
            p.add_argument("--upstream", default=UPSTREAM_SHA)
            p.add_argument(
                "--candidate-bundle",
                type=Path,
                default=os.getenv("CANDIDATE_BUNDLE"),
                help="optional additional Git bundle containing the requested candidate",
            )
            p.add_argument("--model", type=Path, default=os.getenv("MODEL"))
            p.add_argument(
                "--harness-repo", type=Path, help="local git repo containing pinned helpers"
            )
            p.add_argument(
                "--out",
                type=Path,
                help="new directory (default: results-CHIP, with a suffix for a subset)",
            )
    rep = subs.add_parser("report", help="render one or more matching chip runs as Markdown")
    rep.add_argument("roots", nargs="+", type=Path)
    worker_cmd = subs.add_parser("_worker", help=argparse.SUPPRESS)
    worker_cmd.add_argument("cfg", type=Path)
    worker_cmd.add_argument("lane", type=Path)
    worker_cmd.add_argument("precision", choices=("bf16", *DIAGNOSTICS))
    worker_cmd.add_argument("progress", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "_worker":
        return worker(args.cfg, args.lane, args.precision, args.progress)
    if args.mode == "seeds":
        with args.path.open("x", encoding="utf-8") as f:
            json.dump({"seeds": FIXED_SEEDS}, f, indent=2)
            f.write("\n")
        print(args.path, list(FIXED_SEEDS))
        return 0
    if args.mode == "report":
        reports = [summary(root, save=False) for root in args.roots]
        identity = (
            "candidate_sha",
            "upstream_sha",
            "harness",
            "script_sha256",
            "seeds",
            "checkpoint",
            "cycles",
            "window",
            "stream_size",
            "categories",
            "case_filter",
        )
        first = reports[0][0]
        if len({cfg["chip"] for cfg, _, _ in reports}) != len(reports):
            raise ValueError("supply at most one run per chip in a combined report")
        if any(cfg.get(k) != first.get(k) for cfg, _, _ in reports for k in identity):
            raise ValueError(
                "refusing to combine mismatched candidates, inputs, weights or protocol"
            )
        if len(reports) > 1:
            cross_check_inputs(args.roots, reports)
        print("# PR1 Ascend Prefix Cache\n")
        for _, _, md in reports:
            print(md)
        return max({"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[s] for _, s, _ in reports)
    cards = [int(x) for x in args.cards.split(",")]
    cats = sorted(set(int(x) for x in args.categories.split(",")))
    if len(set(cards)) != len(cards) or min(cards) < 0 or len(cards) > 10:
        parser.error("select distinct nonnegative card indices (up to ten)")
    if not set(cats) <= CATEGORIES.keys() or not cats:
        parser.error("categories must be selected from 1..7")
    if (
        not math.isfinite(args.window)
        or not math.isfinite(args.timeout)
        or args.window <= 0
        or args.timeout <= 0
        or args.cycles < 1
        or args.stream_size < 32
    ):
        parser.error("positive windows, timeout and cycles, and at least 32 stream inputs required")
    seeds = seed_list(args.seeds)
    jobs = make_jobs(seeds, cards, cats)
    case_filter = sorted(set(args.cases.split(","))) if args.cases else []
    if case_filter:
        missing = set(case_filter) - {j.case for j in jobs}
        if missing:
            parser.error(f"unknown cases in selected categories: {', '.join(sorted(missing))}")
        jobs = [j for j in jobs if j.case in case_filter]
        cats = sorted({g for j in jobs for g in j.groups})
    cfg = dict(
        harness=HARNESS_SHA,
        chip=args.chip,
        cards=cards,
        categories=cats,
        case_filter=case_filter,
        seeds=seeds,
        cycles=args.cycles,
        window=args.window,
        stream_size=args.stream_size,
        timeout=args.timeout,
        workloads=WORKLOADS,
        jobs=[asdict(j) for j in jobs],
    )
    if args.mode == "plan":
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return 0
    if not args.model:
        parser.error("run requires --model or MODEL pointing to local Ouro weights")
    return run(args, cfg)


if __name__ == "__main__":
    try:
        sys.exit(cli())
    except (OSError, ValueError, subprocess.CalledProcessError, KeyError) as exc:
        print(f"PR1: {exc}", file=sys.stderr)
        sys.exit(2)
