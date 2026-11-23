"""Replay one real 242-token HTTP prompt and its 256/257-row controls on one Ascend engine."""

import argparse
import gc
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import median

import torch

from benchmarks import ascend_main_prefix_combo as combo
from benchmarks.ascend_main_hot_long_audit import PrefillOps
from benchmarks.ascend_main_prefix_linear import (
    check_linear_usage,
    linear_usage,
    padded_fp32_linears,
)
from benchmarks.ascend_main_prefix_results import Report, ids_digest, quantiles, savings, write_csv
from benchmarks.ascend_main_prefix_service import shared_tokens
from benchmarks.ascend_prefix_combo_checks import require
from vllm_rlt import SamplingParams
from vllm_rlt.models import OuroForCausalLM, load_tokenizer
from vllm_rlt.request import RequestPhase

HIT = 208
CHUNK = 256
LENGTHS = (242, 256, 257)
ARMS = combo.ARMS


def arm_order(index):
    # Three rotations and their reverses balance each position over six measured rounds.
    rotation = (index // 2) % len(ARMS)
    order = ARMS[rotation:] + ARMS[:rotation]
    return order if index % 2 == 0 else order[::-1]


def service_input(path, tokenizer, request_id):
    with path.open(encoding="utf-8-sig") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    manifests = [r for r in records if r["kind"] == "MANIFEST"]
    require(len(manifests) == 1, "expected one service MANIFEST in the JSONL")
    manifest = manifests[0]
    candidates = [
        r
        for r in manifest["requests"]
        if r["cohort"] == "hot_long"
        and r["prompt_tokens"] == LENGTHS[0]
        and r["min_primed"] == HIT
        and (request_id is None or r["id"] == request_id)
    ]
    require(bool(candidates), "no matching 242-token hot_long request with 208 primed tokens")
    target = candidates[0]
    primers = [r for r in manifest["primers"] if r["id"] == "primer_long"]
    require(len(primers) == 1, "expected the service's primer_long")

    def encode(row):
        ids = list(tokenizer.encode(row["prompt"]))
        require(
            len(ids) == row["prompt_tokens"] and ids_digest(ids) == row["prompt_sha256"],
            f"tokenizer disagrees with archived token IDs for {row['id']}",
        )
        return ids

    base, donor = encode(target), encode(primers[0])
    require(shared_tokens(base, donor) == HIT, "archived primer does not yield 208 complete tokens")
    first = set()
    for run in (r for r in records if r["kind"] == "SERVICE_RUN"):
        observed = [r for r in run["requests"] if r["id"] == target["id"]]
        require(len(observed) == 1, "service replay has a missing or duplicated target")
        row = observed[0]
        require(
            row["ok"]
            and row["prompt_sha256"] == target["prompt_sha256"]
            and row["tokens"]
            and row["cached_tokens"] == (0 if run["mode"] == "off" else HIT),
            "archived service target failed its identity or hit check",
        )
        first.add(row["tokens"][0])
    require(len(first) == 1, "expected a single consistent archived first token")
    # Extend token IDs directly so 242 is unchanged and 257 is exactly 256 plus one ID.
    extended = base + base[-(LENGTHS[-1] - len(base)) :]
    return {n: extended[:n] for n in LENGTHS}, donor, target["id"], first.pop()


@torch.inference_mode()
def trial(model, target, primer, arm, work, diagnostic=False):
    gc.collect()
    # Use exactly the same cache capacity and scheduler limits at all three lengths.
    engine = combo.open_engine(model, combo.Case(HIT, LENGTHS[-1] - HIT), arm)
    cache = engine.cache_manager
    try:
        require(
            engine.execution_config.packed_prefill
            and cache.backend == "ascend_eager"
            and cache.dtype == torch.bfloat16
            and cache.enable_prefix_caching == (arm != "off"),
            "wrong mainline config",
        )
        params = SamplingParams(
            max_tokens=1,
            temperature=0,
            min_loops=4,
            max_loops=4,
            exit_threshold=1.0,
            ignore_eos=True,
        )
        engine.add_request("primer", primer, params)
        combo.drain(engine, ["primer"], 1)
        engine.model_runner.synchronize()
        n = len(target)
        hit = HIT if arm == "reuse" else 0
        before_hits, before_queries = cache.prefix_hits, cache.prefix_queries
        kernel = cache.attention.kernel
        before_batched = kernel.batched_prefill_calls
        work[1].clear()
        ops = PrefillOps(cache.device, CHUNK, n, hit) if diagnostic else None
        schedule = None
        with ops.observe(model) if ops is not None else nullcontext():
            start = time.perf_counter_ns()
            engine.add_request("target", target, params)
            while engine.has_unfinished_requests():
                outputs = engine.step()
                visible = time.perf_counter_ns()
                if schedule is None:
                    schedule = engine.last_schedule
                if outputs and outputs[0].token_ids:
                    break
            else:
                raise AssertionError("target finished without a visible token")
        require(
            len(outputs) == 1
            and outputs[0].request_id == "target"
            and outputs[0].finished
            and outputs[0].exit_depths == [4]
            and len(outputs[0].token_ids) == 1
            and outputs[0].finish_reason == "length",
            "invalid target completion",
        )
        require(
            schedule.phase == RequestPhase.PREFILL and len(schedule.requests) == 1,
            "expected one ordinary target prefill",
        )
        scheduled = schedule.requests[0]
        require(
            scheduled.token_start == scheduled.request.prefill_window_origin == hit
            and schedule.num_tokens == n - hit
            and cache.prefix_hits - before_hits == hit
            and cache.prefix_queries - before_queries == 1
            and not engine.has_unfinished_requests()
            and cache.num_used_blocks == 0,
            f"unexpected target origin, query rows, counters or unreleased KV (expected {hit=})",
        )
        expected_calls = cache.num_layers * cache.storage_depths
        batched = kernel.batched_prefill_calls - before_batched
        require(batched == (0 if hit else expected_calls), "unexpected packed attention path")
        usage = linear_usage(work)
        check_linear_usage(usage)
        result = dict(
            first_token=outputs[0].token_ids[0],
            hit_tokens=hit,
            prefill_rows=schedule.num_tokens,
            batched_prefill_calls=batched,
            linear_work=usage,
        )
        if ops is None:
            result["ttft_ms"] = (visible - start) / 1e6
        else:
            result["ops"] = ops.result(expected_calls)
        return result
    finally:
        engine.close()


def run_cases(model, targets, donor, archive_token, args, report, work):
    miss = [(donor[0] + 1) % model.config.vocab_size, *donor[1:]]
    require(shared_tokens(miss, targets[LENGTHS[0]]) == 0, "miss primer unexpectedly shares a page")
    report.record(
        "INPUTS",
        archived_first=archive_token,
        target_sha256={n: ids_digest(t) for n, t in targets.items()},
        donor_sha256=ids_digest(donor),
        miss_donor_sha256=ids_digest(miss),
    )
    measured, audits = defaultdict(dict), defaultdict(list)
    tokens = {n: set() for n in LENGTHS}
    issues = {n: [] for n in LENGTHS}
    phases = (("warmup", 1), ("timing", args.repeats), ("audit", args.audit_repeats))
    for phase, rounds in phases:
        for i in range(rounds):
            offset = i % len(LENGTHS)
            for n in LENGTHS[offset:] + LENGTHS[:offset]:
                length_index = LENGTHS.index(n)
                order_index = 2 * i if phase == "audit" else i
                for arm in arm_order(order_index + 2 * length_index):
                    where = dict(phase=phase, round=i + 1, length=n, arm=arm)
                    report.location = where
                    try:
                        result = trial(
                            model,
                            targets[n],
                            miss if arm == "miss" else donor,
                            arm,
                            work,
                            diagnostic=phase == "audit",
                        )
                    except AssertionError as exc:
                        problem = dict(**where, reason=str(exc))
                        issues[n].append(problem)
                        report.record("CONTROL", echo=True, **problem)
                        continue
                    report.record("TARGET", **where, **result)
                    tokens[n].add(result["first_token"])
                    if phase == "timing":
                        measured[n, arm][i] = result["ttft_ms"]
                    if phase == "audit":
                        audits[n, arm].append(result["ops"])
            print(f"HOT_LONG_PROGRESS {phase} {i + 1}/{rounds}", flush=True)

    cases, csv_rows = {}, []
    for n in LENGTHS:
        if len(tokens[n]) != 1:
            issues[n].append(
                dict(reason="first-token parity across arms or replays", tokens=sorted(tokens[n]))
            )
        archive_match = tokens[n] == {archive_token} if n == LENGTHS[0] else None
        if archive_match is False:
            issues[n].append(
                dict(
                    reason="242-token first output differs from archived HTTP",
                    expected=archive_token,
                    actual=sorted(tokens[n]),
                )
            )
        for arm in ARMS:
            counts = {(r["linear_tiles"], r["q_proj_tiles"]) for r in audits[n, arm]}
            if len(counts) > 1:
                issues[n].append(dict(reason="linear tile count changed between audits", arm=arm))
        ok = not issues[n]
        arms, paired, paired_ops = {}, {}, {}
        if ok:
            times = {arm: [measured[n, arm][i] for i in range(args.repeats)] for arm in ARMS}
            for arm in ARMS:
                samples = audits[n, arm]
                times_summary = quantiles(times[arm])
                summary = dict(
                    ttft_p50_ms=times_summary["p50"],
                    ttft_p95_ms=times_summary["p95"],
                    hit_tokens=HIT if arm == "reuse" else 0,
                    prefill_rows=n - HIT if arm == "reuse" else n,
                    clock=samples[0]["clock"],
                    linear_tiles=samples[0]["linear_tiles"],
                    q_proj_tiles=samples[0]["q_proj_tiles"],
                    kv_read_p50_ms=round(median(s["kv_read_ms"] for s in samples), 3),
                    sdpa_p50_ms=round(median(s["sdpa_ms"] for s in samples), 3),
                )
                arms[arm] = summary
                csv_rows.append(dict(length=n, arm=arm, **summary))
            for left, right in (("off", "miss"), ("off", "reuse"), ("miss", "reuse")):
                paired[f"{left}_to_{right}"] = savings(times[left], times[right])
            for key in ("kv_read_ms", "sdpa_ms"):
                paired_ops[key] = savings(
                    [r[key] for r in audits[n, "miss"]], [r[key] for r in audits[n, "reuse"]]
                )
        cases[n] = dict(
            status="PASS" if ok else "FAIL",
            first_tokens=sorted(tokens[n]),
            archived_first_match=archive_match,
            arms=arms,
            paired_ttft=paired,
            audit_miss_to_reuse=paired_ops,
            issues=issues[n],
        )
        report.record("HOT_LONG_CASE", echo=True, length=n, **cases[n])
    write_csv(report.folder / "cases.csv", csv_rows)
    failed = [n for n in LENGTHS if issues[n]]
    return report.review(
        "FAIL" if failed else "PASS",
        cases=cases,
        failed=failed,
        repeats=args.repeats,
        audit_repeats=args.audit_repeats,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--service-raw", type=Path, required=True, help="Night run paced/raw.jsonl")
    p.add_argument("--request-id", help="Default: first matching 242-token hot_long request")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--audit-repeats", type=int, default=3)
    p.add_argument(
        "--out-dir", type=Path, help="New directory for raw.jsonl, cases.csv and review.json"
    )
    args = p.parse_args(argv)
    if (
        not args.model.is_dir()
        or not args.service_raw.is_file()
        or min(args.repeats, args.audit_repeats) < 1
    ):
        p.error("provide a local model, archived service raw.jsonl and positive repeat counts")
    report = Report("HOT_LONG", args.out_dir, linear_fp32_chunk=CHUNK)

    def run(args, device, report):
        report.location = {"phase": "restore_prompt"}
        tokenizer = load_tokenizer(str(args.model.resolve()))
        targets, donor, rid, archived = service_input(args.service_raw, tokenizer, args.request_id)
        report.record(
            "SOURCE",
            echo=True,
            service_raw=str(args.service_raw.resolve()),
            request_id=rid,
            base_sha256=ids_digest(targets[LENGTHS[0]]),
            archived_first=archived,
        )
        report.location = {"phase": "load_model"}
        model = OuroForCausalLM.from_pretrained(
            args.model, device=device, dtype=torch.bfloat16
        ).eval()
        with padded_fp32_linears(model, CHUNK) as work:
            return run_cases(model, targets, donor, archived, args, report, work)

    return report.execute(args, run)


if __name__ == "__main__":
    raise SystemExit(main())
